"""
DatasetProcessor module for transforming and formatting datasets for model training.
Handles tokenization, sequence packing, truncation, column removal, and train/val formatting.
"""

from typing import Any, Dict, List, Optional, Union
from app.training.config import TrainingConfig
from app.utils.logger import logger


class DatasetProcessor:
    """
    Stateless dataset processor performing tokenization, sequence packing,
    column filtering, and split formatting for pretraining and fine-tuning.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config

    def process(self, dataset: Any, tokenizer: Any) -> Any:
        """
        Processes a raw dataset (Dataset or DatasetDict) using the provided tokenizer.
        """
        logger.info("DatasetProcessor: Starting dataset processing...")
        if dataset is None:
            return None

        from datasets import DatasetDict, Dataset

        if isinstance(dataset, DatasetDict):
            processed_dict = {}
            for split_name, split_ds in dataset.items():
                logger.info(f"DatasetProcessor: Processing split '{split_name}' ({len(split_ds):,} samples)...")
                processed_dict[split_name] = self._process_single_split(split_ds, tokenizer)
            return DatasetDict(processed_dict)
        elif isinstance(dataset, Dataset):
            return self._process_single_split(dataset, tokenizer)
        else:
            logger.warning("DatasetProcessor: Unrecognized dataset type; returning as-is.")
            return dataset

    def _process_single_split(self, dataset: Any, tokenizer: Any) -> Any:
        # Check if already packed / tokenized (e.g. Phase 6 packed Parquet)
        col_names = getattr(dataset, "column_names", [])
        if "input_ids" in col_names:
            logger.info("DatasetProcessor: Dataset already contains 'input_ids'. Ensuring 'labels' exist and setting format.")
            if "labels" not in col_names and hasattr(dataset, "map"):
                dataset = dataset.map(
                    lambda ex: {"labels": ex["input_ids"].copy()},
                    batched=True,
                    desc="Setting labels from input_ids"
                )
            return dataset

        # Determine text column
        text_column = "text"
        if text_column not in col_names:
            possible = [c for c in col_names if "text" in c or "content" in c]
            if possible:
                text_column = possible[0]
            elif col_names:
                text_column = col_names[0]

        max_len = self.config.max_seq_length

        def tokenize_fn(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
            texts = examples[text_column]
            # Handle non-string entries safely
            texts = [str(t) if t is not None else "" for t in texts]
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_len,
                padding=False,
                return_attention_mask=True
            )
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
            return tokenized

        # Remove non-tensor columns to save memory
        cols_to_remove = [c for c in col_names if c not in ["input_ids", "attention_mask", "labels"]]
        
        logger.info(f"DatasetProcessor: Tokenizing split (max_seq_len={max_len}, removing columns: {cols_to_remove})...")
        processed = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=cols_to_remove,
            desc="Tokenizing dataset"
        )
        
        # Sequence packing for pretraining if requested
        if self.config.is_packed and self.config.mode in ["continued_pretraining", "full"]:
            processed = self._pack_sequences(processed, max_len)

        return processed

    def _pack_sequences(self, dataset: Any, block_size: int) -> Any:
        """
        Packs tokenized sequences into fixed length blocks for efficient causal LM pretraining.
        """
        logger.info(f"DatasetProcessor: Packing sequences into constant blocks of size {block_size}...")

        def group_texts(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        packed = dataset.map(
            group_texts,
            batched=True,
            desc=f"Packing sequences (block_size={block_size})"
        )
        return packed
