from typing import Dict, Any, List, Union, Optional
from app.tasks.manager import BaseTask, task_manager
from app.utils.logger import logger

class TranslationTask(BaseTask):
    """
    Task handler for English <-> Manipuri translation.
    Implements full lifecycle: prepare -> validate -> format -> tokenize -> build_labels -> collator -> metrics.
    """
    def __init__(self, mode: str = "causal_lm"):
        super().__init__(name="translation")
        self.mode = mode

    def prepare(self, example: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        source = example.get("en") or example.get("source") or example.get("english") or ""
        target = example.get("mni") or example.get("target") or example.get("manipuri") or ""
        return {
            "source": str(source).strip(),
            "target": str(target).strip()
        }

    def validate(self, prepared: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> bool:
        """Checks source and target strings exist before formatting."""
        is_valid = bool(prepared.get("source")) and bool(prepared.get("target"))
        if not is_valid and ctx and hasattr(ctx, "log"):
            ctx.log("TranslationTask: Skipped sample due to empty source or target.", level="debug")
        return is_valid

    def format(self, prepared: Dict[str, Any], formatter: Any, ctx: Optional[Any] = None, **kwargs) -> str:
        if hasattr(formatter, "render"):
            return formatter.render(prepared, format_type="translation", **kwargs)
        source = prepared["source"]
        target = prepared["target"]
        if kwargs.get("prompt_only", False):
            return f"Translate English to Manipuri.\n\nEnglish:\n{source}\n\nManipuri:\n"
        return f"Translate English to Manipuri.\n\nEnglish:\n{source}\n\nManipuri:\n{target}"

    def tokenize(self, formatted: Union[str, List[Dict[str, str]]], tokenizer: Any, max_length: int, ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        if self.mode == "seq2seq" and isinstance(formatted, dict):
            source_str = formatted.get("source", "")
            target_str = formatted.get("target", "")
        else:
            if isinstance(formatted, str):
                full_str = formatted
            else:
                full_str = str(formatted)

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

        if self.mode == "seq2seq":
            target_tokens = tokenizer(
                prepared["target"],
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": target_tokens["input_ids"]
            }
        else:
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
        return DataCollatorManager(tokenizer).get_collator("seq2seq" if self.mode == "seq2seq" else "causal_lm")

    def metrics(self, ctx: Optional[Any] = None) -> Dict[str, Any]:
        return {
            "evaluation_metric": "bleu" if self.mode == "seq2seq" else "perplexity",
            "loss_type": "cross_entropy",
            "track_accuracy": True
        }

    def format_prompt(self, example: Dict[str, Any], **kwargs) -> str:
        prepared = self.prepare(example, **kwargs)
        return self.format(prepared, formatter=self, **kwargs)

task_manager.register("translation", TranslationTask())
