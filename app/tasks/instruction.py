from typing import Dict, Any, List, Union, Optional
from app.tasks.manager import BaseTask, task_manager

class InstructionTask(BaseTask):
    """
    Task handler for Instruction Tuning (Alpaca style).
    Implements full lifecycle: prepare -> validate -> format -> tokenize -> build_labels.
    """
    def __init__(self):
        super().__init__(name="instruction")

    def prepare(self, example: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        response = example.get("response") or example.get("output") or ""
        return {
            "instruction": str(instruction).strip(),
            "input": str(input_text).strip(),
            "output": str(response).strip()
        }

    def validate(self, prepared: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> bool:
        """Checks instruction and output exist before formatting."""
        is_valid = bool(prepared.get("instruction")) and bool(prepared.get("output"))
        if not is_valid and ctx and hasattr(ctx, "log"):
            ctx.log("InstructionTask: Skipped sample due to missing instruction or output.", level="debug")
        return is_valid

    def format(self, prepared: Dict[str, Any], formatter: Any, ctx: Optional[Any] = None, **kwargs) -> str:
        if hasattr(formatter, "render"):
            return formatter.render(prepared, format_type="instruction", **kwargs)

        instruction = prepared["instruction"]
        input_text = prepared["input"]
        response = prepared["output"]

        if input_text:
            prompt = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n"
        else:
            prompt = f"### Instruction\n{instruction}\n\n### Response\n"

        if kwargs.get("prompt_only", False):
            return prompt
        return f"{prompt}{response}"

    def tokenize(self, formatted: Union[str, List[Dict[str, str]]], tokenizer: Any, max_length: int, ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        full_str = formatted if isinstance(formatted, str) else str(formatted)
        tokens = tokenizer(
            full_str,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if len(input_ids) < max_length and input_ids[-1] != tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)
                attention_mask.append(1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def build_labels(self, tokenized: Dict[str, Any], prepared: Dict[str, Any], tokenizer: Any, max_length: int, ctx: Optional[Any] = None, **kwargs) -> Dict[str, List[int]]:
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        prompt_str = self.format(prepared, formatter=kwargs.get("formatter", self), prompt_only=True)
        prompt_tokens = tokenizer(
            prompt_str,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )["input_ids"]

        prompt_len = min(len(prompt_tokens), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def collator(self, tokenizer: Any, ctx: Optional[Any] = None, **kwargs) -> Any:
        from app.tokenizer.collator import DataCollatorManager
        return DataCollatorManager(tokenizer).get_collator("causal_lm")

    def metrics(self, ctx: Optional[Any] = None) -> Dict[str, Any]:
        return {
            "evaluation_metric": "rouge_l",
            "loss_type": "cross_entropy"
        }

    def format_prompt(self, example: Dict[str, Any], **kwargs) -> str:
        prepared = self.prepare(example, **kwargs)
        return self.format(prepared, formatter=self, **kwargs)

task_manager.register("instruction", InstructionTask())
