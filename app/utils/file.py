import json
from pathlib import Path
from typing import Any, Union
from app.utils.logger import logger

def save_json(data: Any, file_path: Union[str, Path], indent: int = 4) -> None:
    """
    Saves a Python object/structure as JSON to the designated file path.
    Args:
        data: The serializable data structure to save.
        file_path: The target path to write the JSON file to.
        indent: JSON indentation spacing.
    """
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"JSON saved successfully to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise e

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Loads JSON from the specified file path.
    Args:
        file_path: The JSON file path to read.
    Returns:
        Any: The parsed JSON content.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"JSON file does not exist: {path}")
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"JSON loaded successfully from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise e

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensures that a directory exists, creating it and its parents if necessary.
    Args:
        dir_path: Directory path to create.
    Returns:
        Path: The confirmed directory path.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
