import pytest
from datasets import Dataset
from app.preprocessing.splitter import DatasetSplitter

def test_dataset_splitting():
    # Create a small dataset of 20 items
    data = {"text": [f"Sentence {i}" for i in range(20)]}
    dataset = Dataset.from_dict(data)
    
    # Split 80/10/10
    splitter = DatasetSplitter({"train": 0.8, "validation": 0.1, "test": 0.1})
    splits = splitter.split(dataset, seed=42)
    
    # 20 * 0.8 = 16 train, 2 val, 2 test
    assert len(splits["train"]) == 16
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2

def test_dataset_splitting_train_only():
    data = {"text": [f"Sentence {i}" for i in range(10)]}
    dataset = Dataset.from_dict(data)
    
    splitter = DatasetSplitter({"train": 1.0, "validation": 0.0, "test": 0.0})
    splits = splitter.split(dataset)
    
    assert "train" in splits
    assert "validation" not in splits
    assert "test" not in splits
    assert len(splits["train"]) == 10
