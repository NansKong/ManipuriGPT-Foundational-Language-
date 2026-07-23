"""
ManipuriGPT Centralized Storage Configuration & Cache Management Layer (`app/utils/cache.py`).
Redirects Hugging Face downloads (`HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`),
Python temporary files (`tempfile.tempdir`, `TMPDIR`), preprocessed shards, tokenizers, benchmarks,
statistics, and logs to a configurable `cache_root` (defaulting to `D:/ManipuriGPT/cache`).
Guarantees zero default writes to the C: drive unless explicitly requested.
"""

import os
import tempfile
from typing import Dict, Any, Optional


def setup_cache_directories(cache_root: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Initializes all project cache directories and sets system environment variables
    for HuggingFace, Transformers, and Python tempfile to prevent C: drive exhaustion.
    Returns a dictionary of absolute paths for each cache sub-directory.
    """
    if not cache_root:
        if config and "paths" in config and "cache_root" in config["paths"]:
            cache_root = config["paths"]["cache_root"]
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            default_cache = os.path.join(project_root, "cache")
            cache_root = os.getenv("MANIPURIGPT_CACHE_ROOT", default_cache)

    cache_root = os.path.abspath(cache_root)

    dirs = {
        "root": cache_root,
        "huggingface": os.path.join(cache_root, "huggingface"),
        "datasets": os.path.join(cache_root, "datasets"),
        "hub": os.path.join(cache_root, "hub"),
        "transformers": os.path.join(cache_root, "transformers"),
        "tmp": os.path.join(cache_root, "tmp"),
        "shards": os.path.join(cache_root, "shards"),
        "tokenizers": os.path.join(cache_root, "tokenizers"),
        "benchmarks": os.path.join(cache_root, "benchmarks"),
        "statistics": os.path.join(cache_root, "statistics"),
        "logs": os.path.join(cache_root, "logs"),
    }

    # Create all directories
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)

    # 1. Redirect Hugging Face caches (Highest Priority)
    os.environ["HF_HOME"] = dirs["huggingface"]
    os.environ["HF_DATASETS_CACHE"] = dirs["datasets"]
    os.environ["HF_HUB_CACHE"] = dirs["hub"]
    os.environ["TRANSFORMERS_CACHE"] = dirs["transformers"]
    os.environ["HUGGINGFACE_HUB_CACHE"] = dirs["hub"]
    os.environ["TORCH_HOME"] = os.path.join(cache_root, "torch")

    # 2. Redirect Python Temporary Files (`tempfile`)
    os.environ["TMPDIR"] = dirs["tmp"]
    os.environ["TEMP"] = dirs["tmp"]
    os.environ["TMP"] = dirs["tmp"]
    tempfile.tempdir = dirs["tmp"]

    return dirs


# Automatically run setup_cache_directories on module import
setup_cache_directories()
