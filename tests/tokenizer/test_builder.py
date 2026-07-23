import pytest
from datasets import Dataset, DatasetDict
from app.tokenizer.dataset_builder import DatasetBuilder
from app.tokenizer.tokenizer_manager import TokenizerManager

class MockTokenizerForBuilder:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "</s>"
        self.mask_token = "[MASK]"
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.mask_token_id = 4
        self.padding_side = "right"

    def __call__(self, text, **kwargs):
        tokens = [ord(c) % 50 + 10 for c in text] + [self.eos_token_id]
        if kwargs.get("truncation") and kwargs.get("max_length"):
            max_l = kwargs["max_length"]
            if len(tokens) > max_l:
                tokens = tokens[:max_l]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens)
        }

    def decode(self, token_ids, **kwargs):
        return "mock decoded"

@pytest.fixture
def dummy_tokenizer_manager():
    return TokenizerManager(
        config={"max_length": 128, "padding_side": "right"},
        tokenizer_instance=MockTokenizerForBuilder()
    )

def test_dataset_builder_single_dataset_no_packing(dummy_tokenizer_manager):
    builder = DatasetBuilder(
        config={"max_length": 128, "packing": False},
        tokenizer_manager=dummy_tokenizer_manager
    )
    data = [
        {"en": "Hello world", "mni": "Chat"},
        {"en": "Good morning", "mni": "Ayuk"}
    ]
    ds = Dataset.from_list(data)
    out_ds = builder.build(ds, task_name="translation")

    assert len(out_ds) == 2
    assert "input_ids" in out_ds.column_names
    assert "attention_mask" in out_ds.column_names
    assert "labels" in out_ds.column_names
    # Old columns removed
    assert "en" not in out_ds.column_names

def test_dataset_builder_dataset_dict_with_packing(dummy_tokenizer_manager):
    builder = DatasetBuilder(
        config={"max_length": 30, "packing": True},
        tokenizer_manager=dummy_tokenizer_manager
    )
    data = [
        {"text": "Sentence one is here."},
        {"text": "Sentence two follows right along."},
        {"text": "And sentence three finishes."}
    ]
    ds_dict = DatasetDict({
        "train": Dataset.from_list(data)
    })
    out_dict = builder.build(ds_dict, task_name="pretraining")

    assert "train" in out_dict
    packed_train = out_dict["train"]
    # All blocks should be of size max_length=30 if any chunks generated
    for row in packed_train:
        assert len(row["input_ids"]) == 30
        assert len(row["attention_mask"]) == 30
        assert len(row["labels"]) == 30
