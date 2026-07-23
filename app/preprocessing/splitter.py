from typing import Dict, Any, Union
from datasets import Dataset, DatasetDict
from app.utils.logger import logger

class DatasetSplitter:
    """
    Component for splitting a single Dataset into train, validation, and test splits.
    Preserves paired columns (e.g. translation pairs) automatically.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.train_ratio = config.get("train", 0.90)
        self.val_ratio = config.get("validation", 0.05)
        self.test_ratio = config.get("test", 0.05)
        
        # Normalize ratios to sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total > 0:
            self.train_ratio /= total
            self.val_ratio /= total
            self.test_ratio /= total

    def split(self, dataset: Dataset, seed: int = 42) -> DatasetDict:
        """
        Splits a HuggingFace Dataset into train, validation, and test splits.
        """
        if not self.enabled:
            return DatasetDict({"train": dataset})

        logger.info(f"Splitter: Splitting dataset of size {len(dataset)} into Train({self.train_ratio:.2f}) / Val({self.val_ratio:.2f}) / Test({self.test_ratio:.2f})")

        # Handle splitting if validation and test are requested
        if self.val_ratio == 0 and self.test_ratio == 0:
            return DatasetDict({"train": dataset})

        n_samples = len(dataset)
        if n_samples < 3:
            logger.warning(f"Splitter: Dataset too small ({n_samples} samples) for full train/val/test split. Assigning all to train.")
            return DatasetDict({"train": dataset})

        # Calculate ratios for consecutive splits
        # 1. Split off the test set first
        test_size = self.test_ratio
        
        # We need to make sure test_size is not 0 or 1
        if test_size > 0 and test_size < 1:
            try:
                split1 = dataset.train_test_split(test_size=test_size, seed=seed)
                remaining_dataset = split1["train"]
                test_dataset = split1["test"]
            except ValueError as e:
                logger.warning(f"Splitter: test split failed ({e}). Assigning all to train.")
                return DatasetDict({"train": dataset})
        else:
            remaining_dataset = dataset
            test_dataset = None

        # 2. Split remaining into train and validation
        # Adjust validation ratio relative to the remaining part
        remaining_ratio = self.train_ratio + self.val_ratio
        if remaining_ratio > 0:
            val_adjusted = self.val_ratio / remaining_ratio
        else:
            val_adjusted = 0.0

        if val_adjusted > 0 and val_adjusted < 1 and len(remaining_dataset) >= 2:
            try:
                split2 = remaining_dataset.train_test_split(test_size=val_adjusted, seed=seed)
                train_dataset = split2["train"]
                val_dataset = split2["test"]
            except ValueError as e:
                logger.warning(f"Splitter: validation split failed ({e}). Keeping remaining in train.")
                train_dataset = remaining_dataset
                val_dataset = None
        else:
            train_dataset = remaining_dataset
            val_dataset = None

        # Construct final dict
        splits = {"train": train_dataset}
        if val_dataset is not None:
            splits["validation"] = val_dataset
        if test_dataset is not None:
            splits["test"] = test_dataset

        return DatasetDict(splits)
