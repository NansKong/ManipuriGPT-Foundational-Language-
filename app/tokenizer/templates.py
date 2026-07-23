from typing import Dict, Any, List, Optional
from app.utils.logger import logger

CHAT_TEMPLATES: Dict[str, str] = {
    "qwen": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
    "llama": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    ),
    "gemma": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
        "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    ),
    "mistral": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %} {{ message['content'] }}</s>"
        "{% endif %}{% endfor %}"
    )
}

def apply_chat_template(
    messages: List[Dict[str, str]], 
    tokenizer: Optional[Any] = None, 
    template_name: Optional[str] = None, 
    add_generation_prompt: bool = False
) -> str:
    """
    Applies a chat template to a list of message dicts.
    If tokenizer has built-in apply_chat_template and template_name is not explicitly overriding, uses native method.
    Otherwise applies the designated fallback template from CHAT_TEMPLATES.
    """
    # Try native tokenizer template if template_name is 'auto' or unspecified and available
    if tokenizer is not None and (template_name is None or template_name == "auto"):
        if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception as e:
                logger.debug(f"apply_chat_template: Native tokenizer template failed ({e}), falling back to heuristic matching.")

    # Determine family/template_name
    family = "qwen"
    if template_name and template_name != "auto" and template_name.lower() in CHAT_TEMPLATES:
        family = template_name.lower()
    elif tokenizer is not None:
        model_str = getattr(tokenizer, "name_or_path", "").lower()
        if "llama" in model_str:
            family = "llama"
        elif "gemma" in model_str:
            family = "gemma"
        elif "mistral" in model_str:
            family = "mistral"
        else:
            family = "qwen"

    # Manual string formatting if jinja2 rendering isn't applied directly
    # To keep simple and robust without requiring heavy jinja template parsing engines when offline/simple:
    return _render_heuristic_template(family, messages, add_generation_prompt)


def _render_heuristic_template(family: str, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    lines = []
    if family == "qwen":
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            lines.append("<|im_start|>assistant\n")
        return "".join(lines)

    elif family == "llama":
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        if add_generation_prompt:
            lines.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(lines)

    elif family == "gemma":
        for msg in messages:
            role = "model" if msg.get("role") in ["assistant", "model"] else "user"
            content = msg.get("content", "")
            lines.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        if add_generation_prompt:
            lines.append("<start_of_turn>model\n")
        return "".join(lines)

    elif family == "mistral":
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"[INST] {content} [/INST]")
            else:
                lines.append(f" {content}</s>")
        return "".join(lines)

    else:
        # Generic fallback
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"<{role}>\n{content}\n")
        return "".join(lines)
