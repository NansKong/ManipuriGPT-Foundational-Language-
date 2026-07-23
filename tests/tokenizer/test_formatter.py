import pytest
from app.tokenizer.formatter import PromptFormatter
from app.tokenizer.templates import apply_chat_template

def test_prompt_formatter_translation():
    formatter = PromptFormatter(task_name="translation")
    example = {"en": "How are you?", "mni": "Chat"}
    formatted = formatter.format(example)
    assert "Translate English to Manipuri." in formatted
    assert "English:\nHow are you?" in formatted
    assert "Manipuri:\nChat" in formatted

def test_prompt_formatter_instruction():
    formatter = PromptFormatter(task_name="instruction")
    example = {"instruction": "Summarize this", "input": "Long text", "output": "Short text"}
    formatted = formatter.format(example)
    assert "### Instruction\nSummarize this" in formatted
    assert "### Input\nLong text" in formatted
    assert "### Response\nShort text" in formatted

def test_apply_chat_template_qwen():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    rendered = apply_chat_template(messages, template_name="qwen", add_generation_prompt=False)
    assert "<|im_start|>user\nHello<|im_end|>" in rendered
    assert "<|im_start|>assistant\nHi there<|im_end|>" in rendered

def test_apply_chat_template_llama():
    messages = [{"role": "user", "content": "Hello"}]
    rendered = apply_chat_template(messages, template_name="llama", add_generation_prompt=True)
    assert "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>" in rendered
    assert "<|start_header_id|>assistant<|end_header_id|>" in rendered
