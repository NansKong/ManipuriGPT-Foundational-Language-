import pytest
from app.tokenizer.registry import TokenizerRegistry, tokenizer_registry, TOKENIZERS

def test_shortcut_mappings():
    assert "qwen2.5" in TOKENIZERS
    assert TOKENIZERS["qwen2.5"] == "Qwen/Qwen2.5-3B-Instruct"
    assert "llama3.2" in TOKENIZERS
    assert TOKENIZERS["llama3.2"] == "meta-llama/Llama-3.2-3B-Instruct"

def test_register_shortcut():
    registry = TokenizerRegistry()
    registry.register_shortcut("custom_model", "custom/path-v1")
    assert TOKENIZERS["custom_model"] == "custom/path-v1"

def test_registry_caching_mock(monkeypatch):
    registry = TokenizerRegistry()
    registry.clear_cache()

    class DummyTokenizer:
        def __init__(self, name):
            self.name = name

    def mock_from_pretrained(canonical_id, **kwargs):
        return DummyTokenizer(canonical_id)

    import app.tokenizer.registry as reg_module
    class MockAuto:
        from_pretrained = staticmethod(mock_from_pretrained)
    monkeypatch.setattr(reg_module, "AutoTokenizer", MockAuto)

    t1 = registry.get("qwen2.5")
    t2 = registry.get("qwen2.5")
    assert t1 is t2
    assert t1.name == "Qwen/Qwen2.5-3B-Instruct"
