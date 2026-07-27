from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from app.utils.logger import logger

# Path to the YAML catalog
_DATASETS_YAML = Path(__file__).resolve().parent.parent / "configs" / "datasets.yaml"


def _load_sources_from_yaml() -> Dict[str, Dict[str, Any]]:
    """Load the dataset sources catalog from datasets.yaml."""
    if not _DATASETS_YAML.exists():
        logger.warning(f"Dataset config not found at {_DATASETS_YAML}. Starting with empty catalog.")
        return {}

    with open(_DATASETS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sources = data.get("datasets", {}).get("sources", {})
    logger.info(f"Registry: Loaded {len(sources)} dataset(s) from {_DATASETS_YAML.name}")
    return sources


class DatasetRegistry:
    """
    Configuration-driven dataset registry.

    Reads dataset metadata from configs/datasets.yaml on init.
    Also supports dynamic registration via .register() for
    runtime-defined or test-only datasets.
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = _load_sources_from_yaml()

    def register(self, name: str, metadata: Dict[str, Any]) -> None:
        """
        Registers a new dataset dynamically into the catalog.
        """
        if name in self._registry:
            logger.warning(f"Overwriting existing dataset entry in registry: {name}")
        self._registry[name] = metadata
        logger.info(f"Dataset '{name}' registered successfully.")

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Retrieves metadata for a registered dataset.
        """
        if name not in self._registry:
            logger.error(f"Dataset '{name}' is not registered in the catalog.")
            raise KeyError(f"Dataset '{name}' not found in registry.")
        return self._registry[name]

    def list_datasets(self) -> list:
        """
        Returns a list of all registered dataset names.
        """
        return list(self._registry.keys())

    def load(self, name: str, **kwargs) -> Any:
        """
        Loads a dataset by name using the dataset loader.
        """
        # Import inside method to avoid circular dependencies
        from app.datasets.loader import load_from_registry
        logger.info(f"Registry: Initiating load request for dataset '{name}'.")
        return load_from_registry(name, self, **kwargs)


# Global registry instance
registry = DatasetRegistry()
