from typing import Dict, Any, Union, List, Optional
from app.tokenizer.templates import apply_chat_template
from app.utils.logger import logger

class PromptFormatter:
    """
    Pure, decoupled formatter for rendering prepared structured dictionaries or normalized message lists
    into final prompt strings ready for tokenization. Does not depend on specific task classes.
    """
    def __init__(self, task_name: str = "translation", template_name: str = "auto"):
        self.task_name = task_name
        self.template_name = template_name

    def render(
        self,
        structured_data: Dict[str, Any],
        format_type: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Renders structured dictionary into prompt string based on format_type.
        """
        fmt = (format_type or self.task_name).lower()
        prompt_only = kwargs.get("prompt_only", False)

        if fmt == "translation":
            source = structured_data.get("source") or structured_data.get("en") or ""
            target = structured_data.get("target") or structured_data.get("mni") or ""
            if prompt_only:
                return f"Translate English to Manipuri.\n\nEnglish:\n{source}\n\nManipuri:\n"
            return f"Translate English to Manipuri.\n\nEnglish:\n{source}\n\nManipuri:\n{target}"

        elif fmt == "instruction":
            instruction = structured_data.get("instruction", "")
            input_text = structured_data.get("input", "")
            response = structured_data.get("output") or structured_data.get("response") or ""
            if input_text and str(input_text).strip():
                prompt = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n"
            else:
                prompt = f"### Instruction\n{instruction}\n\n### Response\n"
            if prompt_only:
                return prompt
            return f"{prompt}{response}"

        elif fmt == "chat":
            messages = structured_data.get("messages", [])
            if isinstance(messages, list):
                if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
                    try:
                        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=prompt_only)
                    except Exception as e:
                        logger.debug(f"PromptFormatter: apply_chat_template failed or missing, fallback: {e}")
                return apply_chat_template(messages, tokenizer=tokenizer, template_name=self.template_name, add_generation_prompt=prompt_only)
            return str(messages)

        elif fmt == "pretraining":
            return str(structured_data.get("text") or structured_data.get("content") or "")

        else:
            logger.warning(f"PromptFormatter: Unknown format_type '{fmt}', returning raw text representation.")
            return str(structured_data.get("text") or structured_data)

    def format(self, example: Dict[str, Any], task_override: Optional[str] = None, tokenizer: Optional[Any] = None, **kwargs) -> str:
        """
        Legacy helper for direct prompt formatting.
        """
        from app.tasks.manager import task_manager
        task_key = task_override or self.task_name
        try:
            task = task_manager.get(task_key)
            prepared = task.prepare(example, **kwargs)
            rendered = self.render(prepared, format_type=task_key, tokenizer=tokenizer, **kwargs)
            if isinstance(rendered, list):
                return apply_chat_template(rendered, tokenizer=tokenizer, template_name=self.template_name)
            return rendered
        except KeyError:
            return self.render(example, format_type=task_key, tokenizer=tokenizer, **kwargs)
