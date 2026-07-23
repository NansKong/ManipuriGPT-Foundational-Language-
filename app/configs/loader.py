"""
Configuration Loader and Hashing Utility (`app/configs/loader.py`).
Loads centralized YAML configurations and computes exact SHA256 configuration hashes (`config_hash`)
to guarantee absolute provenance across pretraining shards and tokenizer models.
"""

import os
import yaml
import hashlib
import json
from typing import Dict, Any, Optional, List
from app.utils.logger import logger

CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_CONFIG_FILES = ["phase5.yaml", "sampling.yaml", "preprocessing.yaml", "tokenizer.yaml"]


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Loads a single YAML configuration file by base name from app/configs.
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    file_path = os.path.join(CONFIGS_DIR, config_name)
    if not os.path.exists(file_path):
        logger.warning(f"ConfigLoader: Config file '{file_path}' not found. Returning empty config.")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"ConfigLoader: Error loading config '{file_path}': {e}")
        return {}


def load_all_configs() -> Dict[str, Any]:
    """
    Loads all core Phase 5 configurations into a unified dictionary structure
    and initializes system cache/storage redirections.
    """
    configs = {
        "phase5": load_config("phase5.yaml"),
        "sampling": load_config("sampling.yaml"),
        "preprocessing": load_config("preprocessing.yaml"),
        "tokenizer": load_config("tokenizer.yaml")
    }
    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories(config=configs.get("phase5"))
    except Exception as e:
        logger.warning(f"ConfigLoader: Could not initialize cache directories: {e}")
    return configs


def compute_config_hash(extra_overrides: Optional[Dict[str, Any]] = None) -> str:
    """
    Computes a deterministic SHA256 configuration hash across all 4 core YAML configs
    plus any active runtime overrides.
    """
    combined_parts: List[str] = []
    for cfg_file in sorted(CORE_CONFIG_FILES):
        file_path = os.path.join(CONFIGS_DIR, cfg_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_parts.append(f"{cfg_file}:{f.read().strip()}")
            except Exception:
                combined_parts.append(f"{cfg_file}:empty")
        else:
            combined_parts.append(f"{cfg_file}:missing")

    if extra_overrides:
        try:
            combined_parts.append(f"overrides:{json.dumps(extra_overrides, sort_keys=True)}")
        except Exception:
            combined_parts.append(f"overrides:{str(extra_overrides)}")

    full_payload = "\n---\n".join(combined_parts)
    return hashlib.sha256(full_payload.encode("utf-8")).hexdigest()[:16]
