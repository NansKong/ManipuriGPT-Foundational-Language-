"""
MultiDatasetBuilder module for assembling and partitioning multiple dataset types:
`Pretraining`, `Instruction Tuning`, `Chat`, `RAG`, `Translation`, `QA`, `Reasoning`, `Code`
with deterministic 90/5/5 train/validation/test splits.
"""

import random
from typing import Dict, Any, List, Optional, Union
from datasets import Dataset, DatasetDict
from app.utils.logger import logger


class MultiDatasetBuilder:
    """
    Orchestrates building datasets across 8 canonical tasks (`pretraining`, `instruction_tuning`,
    `chat`, `rag`, `translation`, `qa`, `reasoning`, `code`) and splitting them deterministically.
    """
    def __init__(
        self,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42
    ):
        if round(train_ratio + val_ratio + test_ratio, 4) != 1.0:
            raise ValueError("Split ratios (train_ratio, val_ratio, test_ratio) must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def build_splits(self, records: List[Dict[str, Any]], dataset_type: str = "pretraining") -> DatasetDict:
        """
        Takes a list of records (or loads them) and returns a deterministic DatasetDict
        containing `train`, `validation`, and `test` splits.
        """
        logger.info(f"MultiDatasetBuilder: Building '{dataset_type}' dataset with {len(records)} samples.")
        if not records:
            empty_ds = Dataset.from_dict({"text": [], "metadata": []})
            return DatasetDict({"train": empty_ds, "validation": empty_ds, "test": empty_ds})

        # Deterministic shuffle using seed
        rnd = random.Random(self.seed)
        shuffled = records.copy()
        rnd.shuffle(shuffled)

        num_samples = len(shuffled)
        # Handle small dataset edge cases safely
        if num_samples < 3:
            ds = Dataset.from_list(shuffled)
            return DatasetDict({"train": ds, "validation": ds, "test": ds})

        train_end = int(num_samples * self.train_ratio)
        val_end = train_end + max(1, int(num_samples * self.val_ratio))
        
        # Ensure at least 1 sample in each split if total >= 3
        if train_end == num_samples:
            train_end = num_samples - 2
            val_end = num_samples - 1
        elif val_end >= num_samples:
            val_end = num_samples - 1

        train_records = shuffled[:train_end]
        val_records = shuffled[train_end:val_end]
        test_records = shuffled[val_end:]

        # If test is empty due to rounding, borrow from train or val
        if not test_records and val_records:
            test_records = [val_records.pop()]

        ds_dict = DatasetDict({
            "train": Dataset.from_list(train_records),
            "validation": Dataset.from_list(val_records),
            "test": Dataset.from_list(test_records)
        })
        logger.info(
            f"MultiDatasetBuilder: Created splits -> "
            f"train: {len(ds_dict['train'])}, validation: {len(ds_dict['validation'])}, test: {len(ds_dict['test'])}"
        )
        return ds_dict

    def build_from_source(
        self,
        source: Union[str, Any],
        limit: int = 10000,
        mock_fallback: bool = False,
        dataset_type: str = "pretraining"
    ) -> DatasetDict:
        """
        Streams records from a real corpus source via CorpusAcquisitionManager and builds partitioned train/val/test splits.
        """
        from app.corpus.acquisition import CorpusAcquisitionManager
        logger.info(f"MultiDatasetBuilder: Building from source '{source}' (limit={limit}, mock_fallback={mock_fallback})...")
        mgr = CorpusAcquisitionManager()
        spec = mgr.get_source(source) if isinstance(source, str) else source
        if not spec:
            raise KeyError(f"Source '{source}' not found in registry.")

        stream = mgr.stream_source(spec, max_examples=limit, mock_fallback=mock_fallback)
        records = []
        for ex in stream:
            if isinstance(ex, dict) and "text" in ex:
                records.append(ex)
            elif isinstance(ex, dict):
                records.append({"text": str(ex.get(spec.default_text_column, "")), "metadata": ex})
            else:
                records.append({"text": str(ex), "metadata": {}})

        return self.build_splits(records, dataset_type=dataset_type)
