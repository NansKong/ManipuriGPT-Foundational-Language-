import pytest
from app.tasks.registry import task_registry, BaseTask
from app.tokenizer.context import PipelineContext
from app.tokenizer.normalizer import ConversationNormalizer

class MockTokenizerForTasks:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.mask_token_id = 4
        self.eos_token = "</s>"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        tokens = [ord(c) % 40 + 5 for c in text]
        if kwargs.get("truncation") and kwargs.get("max_length"):
            max_l = kwargs["max_length"]
            if len(tokens) > max_l:
                tokens = tokens[:max_l]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

def test_task_registry_lookup():
    tasks = task_registry.list_tasks()
    assert "translation" in tasks
    assert "instruction" in tasks
    assert "chat" in tasks
    assert "pretraining" in tasks

    with pytest.raises(KeyError):
        task_registry.get("non_existent_task")

def test_translation_task_with_validate_and_ctx():
    task = task_registry.get("translation")
    mock_tok = MockTokenizerForTasks()
    ctx = PipelineContext(task="translation", max_length=128)
    
    valid_example = {"en": "Hello", "mni": "Chat"}
    out = task.prepare_example(valid_example, mock_tok, max_length=128, ctx=ctx)
    assert len(out["input_ids"]) > 0
    assert -100 in out["labels"]

    # Invalid example missing source should return empty arrays due to validate()
    invalid_example = {"en": "", "mni": "Chat"}
    out_invalid = task.prepare_example(invalid_example, mock_tok, max_length=128, ctx=ctx)
    assert out_invalid["input_ids"] == []

def test_instruction_task():
    task = task_registry.get("instruction")
    mock_tok = MockTokenizerForTasks()
    example = {"instruction": "Translate", "input": "Hello", "output": "Chat"}
    out = task.prepare_example(example, mock_tok, max_length=128)
    assert "input_ids" in out
    assert -100 in out["labels"]

def test_chat_task_and_normalizer():
    task = task_registry.get("chat")
    assert task.requires_chat_template()
    
    # Test normalization from ShareGPT format
    sharegpt_example = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"}
        ]
    }
    prepared = task.prepare(sharegpt_example)
    assert prepared["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    
    mock_tok = MockTokenizerForTasks()
    out = task.prepare_example(sharegpt_example, mock_tok, max_length=128)
    assert len(out["input_ids"]) > 0
    assert out["labels"] == out["input_ids"]

def test_pretraining_task():
    task = task_registry.get("pretraining")
    mock_tok = MockTokenizerForTasks()
    example = {"text": "Raw Manipuri text for continued pretraining."}
    out = task.prepare_example(example, mock_tok, max_length=128)
    assert out["labels"] == out["input_ids"]
