import pytest
from app.tokenizer.tokenizer_manager import TokenizerManager

class MockTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.mask_token = "[MASK]"
        self.eos_token_id = 2
        self.pad_token_id = None
        self.mask_token_id = 4
        self.padding_side = "right"

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {
                "input_ids": [[10, 20, 2] for _ in text],
                "attention_mask": [[1, 1, 1] for _ in text]
            }
        # simple mock encoding
        tokens = [ord(c) % 100 for c in text] + [self.eos_token_id]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }

    def decode(self, token_ids, **kwargs):
        return "mock decoded string"

def test_tokenizer_manager_initialization():
    mock_tok = MockTokenizer()
    manager = TokenizerManager(config={"max_length": 512, "padding_side": "left"}, tokenizer_instance=mock_tok)

    # Check pad_token fallback setup from eos_token
    assert manager.tokenizer.pad_token == "</s>"
    assert manager.max_length == 512
    assert manager.tokenizer.padding_side == "left"

def test_tokenizer_manager_encode_decode():
    mock_tok = MockTokenizer()
    manager = TokenizerManager(tokenizer_instance=mock_tok)

    encoded = manager.encode("Hello")
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert manager.decode(encoded["input_ids"]) == "mock decoded string"
    assert manager.count_tokens("Hello") == len(encoded["input_ids"])
