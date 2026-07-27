import os
from pathlib import Path
from typing import Any, Optional, Union
from datasets import load_from_disk, Dataset
from app.configs.settings import settings
from app.utils.logger import logger

class CacheManager:
    """Manages local directories and serialization for dataset caches."""
    
    def __init__(self, base_cache_dir: Optional[Union[str, Path]] = None):
        if base_cache_dir is None:
            base_cache_dir = getattr(settings.datasets, "cache_dir", "./cache")
        self.base_dir = Path(base_cache_dir).resolve()
        
        # Subdirectories
        self.hf_dir = self.base_dir / "hf"
        self.processed_dir = self.base_dir / "processed"
        self.tokenized_dir = self.base_dir / "tokenized"
        
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Creates the cached folders if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheManager: Initialized at {self.base_dir}")

    def get_processed_path(self, filename: str) -> Path:
        """Returns the path for a processed file."""
        return self.processed_dir / filename

    def get_tokenized_path(self, filename: str) -> Path:
        """Returns the path for a tokenized file."""
        return self.tokenized_dir / filename

    def is_processed_cached(self, filename: str) -> bool:
        """Checks if a processed dataset is cached."""
        path = self.get_processed_path(filename)
        # Hugging Face save_to_disk creates a directory with dataset_info.json or state.json
        return path.exists() and (path / "dataset_info.json").exists()

    def is_tokenized_cached(self, filename: str) -> bool:
        """Checks if a tokenized dataset is cached."""
        path = self.get_tokenized_path(filename)
        return path.exists() and (path / "dataset_info.json").exists()

    def save_processed(self, dataset: Any, filename: str) -> None:
        """Saves a processed HuggingFace dataset to disk."""
        path = self.get_processed_path(filename)
        logger.info(f"CacheManager: Saving processed dataset to {path}")
        dataset.save_to_disk(str(path))

    def load_processed(self, filename: str) -> Any:
        """Loads a processed HuggingFace dataset from disk."""
        path = self.get_processed_path(filename)
        logger.info(f"CacheManager: Loading processed dataset from {path}")
        return load_from_disk(str(path))

    def save_tokenized(self, dataset: Any, filename: str) -> None:
        """Saves a tokenized HuggingFace dataset to disk."""
        path = self.get_tokenized_path(filename)
        logger.info(f"CacheManager: Saving tokenized dataset to {path}")
        dataset.save_to_disk(str(path))

    def load_tokenized(self, filename: str) -> Any:
        """Loads a tokenized HuggingFace dataset from disk."""
        path = self.get_tokenized_path(filename)
        logger.info(f"CacheManager: Loading tokenized dataset from {path}")
        return load_from_disk(str(path))

# Global CacheManager instance
cache_manager = CacheManager()
