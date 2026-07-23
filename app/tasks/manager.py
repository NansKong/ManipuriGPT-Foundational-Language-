from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from app.utils.logger import logger

class BaseTask(ABC):
    """
    Abstract base class for all training tasks (e.g., translation, instruction, chat, pretraining).
    Defines modular lifecycle methods: prepare, validate, format, tokenize, build_labels, collator, metrics.
    Supports optional PipelineContext across lifecycle stages.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def prepare(self, example: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Normalizes/extracts raw dataset row into clean structured dictionary representing the task data.
        """
        pass

    def validate(self, prepared: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> bool:
        """
        Validates the prepared data structure before formatting.
        Returns True if valid, False to skip this sample cleanly during pipeline execution.
        """
        return True

    @abstractmethod
    def format(self, prepared: Dict[str, Any], formatter: Any, ctx: Optional[Any] = None, **kwargs) -> Union[str, List[Dict[str, str]]]:
        """
        Renders the prepared data using the decoupled PromptFormatter into prompt string or message dicts.
        """
        pass

    @abstractmethod
    def tokenize(self, formatted: Union[str, List[Dict[str, str]]], tokenizer: Any, max_length: int, ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Tokenizes the formatted prompt or message structure using the provided tokenizer instance.
        Returns dict with at least 'input_ids' and 'attention_mask'.
        """
        pass

    @abstractmethod
    def build_labels(self, tokenized: Dict[str, Any], prepared: Dict[str, Any], tokenizer: Any, max_length: int, ctx: Optional[Any] = None, **kwargs) -> Dict[str, List[int]]:
        """
        Constructs target labels from tokenized input and prepared data, applying label masking (-100) where appropriate.
        Returns complete dict containing 'input_ids', 'attention_mask', and 'labels'.
        """
        pass

    @abstractmethod
    def collator(self, tokenizer: Any, ctx: Optional[Any] = None, **kwargs) -> Any:
        """
        Returns the appropriate data collator instance or collator identifier for this task.
        """
        pass

    @abstractmethod
    def metrics(self, ctx: Optional[Any] = None) -> Dict[str, Any]:
        """
        Returns metric names, tracking functions, or evaluation configurations relevant to this task.
        """
        pass

    def requires_chat_template(self) -> bool:
        """Returns True if this task outputs conversational message dicts needing a chat template."""
        return False

    def get_collator_type(self) -> str:
        """Legacy compatibility helper."""
        return "causal_lm"

    def prepare_example(self, example: Dict[str, Any], tokenizer: Any, max_length: int, formatter: Optional[Any] = None, ctx: Optional[Any] = None, **kwargs) -> Dict[str, List[int]]:
        """
        Orchestrates the complete 5-step processing lifecycle: prepare -> validate -> format -> tokenize -> build_labels.
        """
        if formatter is None:
            from app.tokenizer.formatter import PromptFormatter
            formatter = PromptFormatter(task_name=self.name)

        prepared = self.prepare(example, ctx=ctx, **kwargs)
        if not self.validate(prepared, ctx=ctx, **kwargs):
            return {"input_ids": [], "attention_mask": [], "labels": []}

        formatted = self.format(prepared, formatter, ctx=ctx, **kwargs)
        tokenized = self.tokenize(formatted, tokenizer, max_length, ctx=ctx, **kwargs)
        return self.build_labels(tokenized, prepared, tokenizer, max_length, ctx=ctx, **kwargs)


class TaskManager:
    """
    Manager registry for registering and resolving task handlers.
    Replaces TaskRegistry while keeping full backward compatibility.
    """
    def __init__(self):
        self._tasks: Dict[str, BaseTask] = {}

    def register(self, name: str, task_instance: BaseTask) -> None:
        """Registers a task handler under the specified name."""
        self._tasks[name.lower()] = task_instance
        logger.debug(f"TaskManager: Registered task '{name}'")

    def get(self, name: str) -> BaseTask:
        """Retrieves a task handler by name."""
        key = name.lower()
        if key not in self._tasks:
            available = list(self._tasks.keys())
            raise KeyError(f"Task '{name}' not found in TaskManager. Available tasks: {available}")
        return self._tasks[key]

    def list_tasks(self) -> List[str]:
        """Returns a list of registered task names."""
        return list(self._tasks.keys())

# Singleton instances and backward-compatible aliases
task_manager = TaskManager()
task_registry = task_manager
TaskRegistry = TaskManager
