import pytest
from app.tokenizer.collator import DataCollatorManager, DataCollatorForLanguageModeling

class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "[PAD]"
        self.eos_token_id = 2
        self.eos_token = "</s>"
        self.mask_token_id = 4
        self.mask_token = "[MASK]"
        self.padding_side = "right"

def test_data_collator_manager_causal_lm():
    if DataCollatorForLanguageModeling is None:
        pytest.skip("transformers not installed")
    manager = DataCollatorManager(DummyTokenizer())
    collator = manager.get_collator("causal_lm")
    assert isinstance(collator, DataCollatorForLanguageModeling)
    assert collator.mlm is False

def test_data_collator_manager_mlm():
    if DataCollatorForLanguageModeling is None:
        pytest.skip("transformers not installed")
    manager = DataCollatorManager(DummyTokenizer())
    collator = manager.get_collator("mlm", mlm_probability=0.2)
    assert isinstance(collator, DataCollatorForLanguageModeling)
    assert collator.mlm is True
    assert collator.mlm_probability == 0.2
