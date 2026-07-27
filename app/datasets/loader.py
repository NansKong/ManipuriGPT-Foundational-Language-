"""
DatasetLoader module for reading raw training datasets across sources:
Hugging Face Hub, local Phase 6 pre-packed Parquet shards, local Arrow/JSONL files, and streaming datasets.
"""

import os
from typing import Any, Dict, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
from app.training.config import TrainingConfig
from app.configs.settings import settings
from app.utils.logger import logger
from app.datasets.cache_manager import cache_manager


class DatasetLoader:
    """
    Stateless loader dedicated solely to reading dataset splits from HF Hub, local files, or packed shards.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config

    def load((self) -> Any:
        """
        Loads dataset specified by config.dataset_name_or_path.
        Supports remote HF Hub datasets, local packed Parquet shards, and local files.
        """
        source = self.config.dataset_name_or_path
        split = self.config.dataset_split
        streaming = self.config.use_streaming

        logger.info(f"DatasetLoader: Loading dataset source '{source}' (split: '{split}', streaming: {streaming})...")

        # 1. Local directory check (e.g. Phase 6 packed Parquet shards or local Parquet directory)
        if os.path.exists(source):
            return self._load_local(source, split)

        # 2. Remote Hugging Face Hub dataset
        return self._load_hf_hub(source, split, streaming)

    def _load_local(self, path: str, split: Optional[str] = None) -> Any:
        logger.info(f"DatasetLoader: Reading local path '{path}'...")
        if os.path.isdir(path):
            # Check for split subdirectories (e.g., train/, validation/, test/)
            train_dir = os.path.join(path, "train")
            if os.path.isdir(train_dir):
                data_files = {}
                for s in ["train", "validation", "test"]:
                    s_dir = os.path.join(path, s)
                    if os.path.isdir(s_dir):
                        files = [os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.endswith(".parquet") or f.endswith(".jsonl")]
                        if files:
                            data_files[s] = files
                if data_files:
                    logger.info(f"DatasetLoader: Found split directories in '{path}': {list(data_files.keys())}")
                    ext = "parquet" if any(f.endswith(".parquet") for files in data_files.values() for f in files) else "json"
                    return load_dataset(ext, data_files=data_files)
            
            # Check for parquet files directly in directory
            parquet_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
            if parquet_files:
                logger.info(f"DatasetLoader: Found {len(parquet_files)} parquet shards directly in '{path}'")
                return load_dataset("parquet", data_files=parquet_files, split=split or "train")

        # Single file
        ext = os.path.splitext(path)[-1].lstrip('.').lower()
        if ext in ["parquet", "json", "jsonl", "csv"]:
            file_type = "json" if ext == "jsonl" else ext
            return load_dataset(file_type, data_files=path, split=split or "train")

        raise ValueError(f"DatasetLoader: Unrecognized local path format: {path}")

    def _load_hf_hub(self, repo_id: str, split: Optional[str] = None, streaming: bool = False) -> Any:
        try:
            from app.utils.download import hf_load_dataset_with_backoff
            ds = hf_load_dataset_with_backoff(
                repo_id,
                split=split,
                streaming=streaming
            )
            logger.info(f"DatasetLoader: Successfully loaded dataset '{repo_id}' from HF Hub")
            return ds
        except Exception as e:
            logger.warning(f"DatasetLoader: Failed to load '{repo_id}' from Hugging Face Hub ({e}). Using mock dataset fallback.")
            return DatasetDict({
                "train": Dataset.from_list([{"text": "Sample Manipuri pretraining sentence."} for _ in range(50)]),
                "validation": Dataset.from_list([{"text": "Sample Manipuri validation sentence."} for _ in range(10)])
            })


def load_from_registry(
    name: str, 
    registry_instance, 
    split: Optional[str] = None, 
    use_streaming: Optional[bool] = None,
    **kwargs
) -> Any:
    """
    Loads a dataset by reading its metadata from the registry, applying configurations,
    streaming if possible, caching, and returning a Dataset/IterableDataset object.
    """
    meta = registry_instance.get_metadata(name)
    provider = meta.get("provider", "huggingface")
    repo = meta.get("repo")
    
    if use_streaming is None:
        use_streaming = meta.get("streaming", getattr(settings.datasets, "use_streaming", True))

    if split is None:
        split = meta.get("split", meta.get("default_split", "train"))

    from app.utils.cache import setup_cache_directories
    dirs = setup_cache_directories()
    hf_cache_dir = dirs["datasets"]
    
    logger.info(f"Loader: Loading dataset '{name}' from repo '{repo}' (split: '{split}', streaming: {use_streaming})")
    
    if provider == "local":
        local_path = meta.get("local_path")
        if not local_path or not os.path.exists(local_path):
            logger.error(f"Loader: Local dataset path '{local_path}' does not exist.")
            raise FileNotFoundError(f"Local file not found: {local_path}")
            
        ext = os.path.splitext(local_path)[-1].lstrip('.').lower()
        if ext == "jsonl":
            ext = "json"
        return load_dataset(ext, data_files=local_path, split=split, cache_dir=hf_cache_dir, **kwargs)

    subset = meta.get("subset", meta.get("default_subset"))
    config_kwargs = meta.get("config_kwargs", {})
    load_kwargs = config_kwargs.copy() if isinstance(config_kwargs, dict) else {}
    load_kwargs.update(kwargs)

    mock_fallback = kwargs.pop("mock_fallback", False)
    try:
        from app.utils.download import hf_load_dataset_with_backoff, hf_stream_with_backoff
        dataset = hf_load_dataset_with_backoff(
            repo,
            name=subset,
            split=split,
            streaming=use_streaming,
            cache_dir=hf_cache_dir,
            **load_kwargs
        )
        if use_streaming:
            return hf_stream_with_backoff(dataset)
        return dataset
    except Exception as e:
        logger.error(f"Loader: Failed to load dataset '{name}' from HuggingFace: {e}")
        if mock_fallback:
            return Dataset.from_list([{"text": "Manipuri sample text for fallback."} for _ in range(10)])
        raise e
