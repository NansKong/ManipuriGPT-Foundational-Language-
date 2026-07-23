import os
from typing import Dict, Any, Optional, Union
from app.configs.settings import settings
from app.utils.logger import logger as default_logger

class PipelineContext:
    """
    Unified context object carrying all runtime dependencies, configuration settings,
    paths, logger, and dynamic stage metadata across the tokenization pipeline.
    Replaces loose parameter lists and enables clean hooks and future extensions.
    """
    def __init__(
        self,
        dataset_name: str = "default_dataset",
        model: Union[str, Any] = "qwen2.5",
        task: Union[str, Any] = "translation",
        config: Optional[Any] = None,
        artifact_dir: str = "artifacts/",
        logger: Optional[Any] = None,
        num_proc: Optional[int] = None,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self.model_obj = model if not isinstance(model, str) else None
        self.model_name = getattr(model, "name", model) if not isinstance(model, str) else model
        self.model_short_name = getattr(model, "short_name", str(model))
        
        self.task_obj = task if not isinstance(task, str) else None
        self.task_name = getattr(task, "name", task) if not isinstance(task, str) else task
        
        self.config = config or settings
        self.artifact_dir = artifact_dir
        self.logger = logger or default_logger
        self.num_proc = num_proc
        
        # Extract common tokenizer parameters from config/kwargs
        settings_dict = {}
        if hasattr(self.config, "tokenizer") and hasattr(self.config.tokenizer, "to_dict"):
            settings_dict = self.config.tokenizer.to_dict()
        elif isinstance(self.config, dict):
            settings_dict = self.config
            
        self.max_length = kwargs.get("max_length", settings_dict.get("max_length", 2048))
        self.packing = kwargs.get("packing", settings_dict.get("packing", True))
        
        # Dynamic storage for stage outputs, hooks communication, and statistics
        self.metadata: Dict[str, Any] = kwargs
        self.stats: Dict[str, Any] = {}
        self.stage_history: list = []

    def log(self, message: str, level: str = "info") -> None:
        """Convenience method for logging via context logger."""
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(f"[{self.task_name}|{self.model_short_name}] {message}")

    def get_artifact_path(self, sub_path: str) -> str:
        """Returns normalized path within the context artifact directory."""
        return os.path.join(self.artifact_dir, sub_path)
