from typing import Dict, Any, List, Union, Optional
from app.tasks.manager import BaseTask, task_manager
from app.tokenizer.normalizer import ConversationNormalizer
from app.utils.logger import logger

class ChatTask(BaseTask):
    """
    Task handler for conversational chat.
    Utilizes ConversationNormalizer to normalize any input structure into the mandatory
    Conversation Schema: [{'role': '...', 'content': '...'}] and implements validate().
    """
    def __init__(self):
        super().__init__(name="chat")

    def prepare(self, example: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        messages = ConversationNormalizer.normalize(example)
        return {"messages": messages}

    def validate(self, prepared: Dict[str, Any], ctx: Optional[Any] = None, **kwargs) -> bool:
        """Validates that at least one message with non-empty content exists."""
        messages = prepared.get("messages", [])
        is_valid = isinstance(messages, list) and len(messages) > 0 and any(bool(m.get("content")) for m in messages)
        if not is_valid and ctx and hasattr(ctx, "log"):
            ctx.log("ChatTask: Skipped sample due to empty or invalid conversation messages.", level="debug")
        return is_valid

    def format(self, prepared: Dict[str, Any], formatter: Any, ctx: Optional[Any] = None, **kwargs) -> Union[str, List[Dict[str, str]]]:
        messages = prepared.get("messages", [])
        if hasattr(formatter, "render"):
            return formatter.render(prepared, format_type="chat", **kwargs)
        return messages

    def tokenize(self, formatted: Union[str, List[Dict[str, str]]], tokenizer: Any, max_length: int, ctx: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        if isinstance(formatted, str):
            text = formatted
        else:
            if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
                try:
                    text = tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=False)
                except Exception as e:
                    logger.debug(f"ChatTask: apply_chat_template failed, using fallback: {e}")
                    text = self._fallback_chat_format(formatted)
            else:
                text = self._fallback_chat_format(formatted)

        tokens = tokenizer(
            text,
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
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": list(input_ids)
        }

    def _fallback_chat_format(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role in ["user", "human"]:
                lines.append(f"<user>\n{content}\n")
            elif role in ["assistant", "gpt"]:
                lines.append(f"<assistant>\n{content}\n")
            else:
                lines.append(f"<{role}>\n{content}\n")
        return "".join(lines).strip()

    def collator(self, tokenizer: Any, ctx: Optional[Any] = None, **kwargs) -> Any:
        from app.tokenizer.collator import DataCollatorManager
        return DataCollatorManager(tokenizer).get_collator("causal_lm")

    def metrics(self, ctx: Optional[Any] = None) -> Dict[str, Any]:
        return {
            "evaluation_metric": "perplexity",
            "loss_type": "cross_entropy"
        }

    def requires_chat_template(self) -> bool:
        return True

    def format_prompt(self, example: Dict[str, Any], **kwargs) -> Union[str, List[Dict[str, str]]]:
        prepared = self.prepare(example, **kwargs)
        return self.format(prepared, formatter=self, **kwargs)

task_manager.register("chat", ChatTask())
