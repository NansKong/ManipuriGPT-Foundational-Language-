import pytest
from app.tokenizer.backends import (
    CustomTokenizerBackend,
    TikTokenTokenizerBackend,
    SentencePieceTokenizerBackend,
    HFTokenizerBackend
)
from app.tokenizer.tokenizer_manager import TokenizerManager

def test_todo_backends_raise_not_implemented():
    """Verifies that non-HuggingFace backends cleanly raise NotImplementedError per TODO specs."""
    with pytest.raises(NotImplementedError, match="TODO"):
        backend = CustomTokenizerBackend()
        backend.encode("Hello world")

    with pytest.raises(NotImplementedError, match="TODO"):
        backend = TikTokenTokenizerBackend()
        backend.load_tokenizer("gpt-4")

    with pytest.raises(NotImplementedError, match="TODO"):
        backend = SentencePieceTokenizerBackend()
        backend.encode("Test")

def test_hf_tokenizer_backend():
    class DummyTok:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 20, 2], "attention_mask": [1, 1, 1]}
        def decode(self, ids, **kwargs):
            return "decoded"
            
    backend = HFTokenizerBackend(DummyTok())
    assert backend.encode("Hello")["input_ids"] == [10, 20, 2]
    assert backend.decode([10, 20]) == "decoded"
    assert backend.count_tokens("Hello") == 3

def test_tokenizer_manager_backend_selection():
    class DummyTok:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}
            
    manager = TokenizerManager(tokenizer_instance=DummyTok(), backend="custom")
    assert manager.encode("Hello")["input_ids"] == [1, 2]
