"""
TokenizerVersionManager module for systematic versioning of ManipuriGPT tokenizers.
Enforces corpus size guards (`v1-pretrain` requires >= 50 MB clean text) and records complete
reproducible metadata (git commit, dataset hash, sentence/token count, vocab size, evaluation summary).
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
from app.utils.logger import logger


class TokenizerVersionManager:
    """
    Manages tokenizer version tiers and ensures reproducible metadata saving across training experiments.
    
    Version tiers:
    - v0-experimental: For engineering, benchmarking, and pipeline verification (<50 MB corpus allowed)
    - v1-pretrain: For initial foundational pretraining (requires >= 50 MB clean text)
    - v2-expanded: For expanded scale (requires >= 250 MB clean text)
    - v3-final: For final production baseline (requires >= 500 MB clean text)
    """
    TIER_THRESHOLDS_BYTES = {
        "v0-experimental": 0,
        "v1-pretrain": 50 * 1024 * 1024,      # 50 MB
        "v2-expanded": 250 * 1024 * 1024,     # 250 MB
        "v3-final": 500 * 1024 * 1024,        # 500 MB
    }

    def __init__(self, base_output_dir: str = "cache/tokenizers"):
        self.base_output_dir = base_output_dir

    def get_tier_directory(self, tier: str) -> str:
        """Returns the absolute or relative directory path for the given version tier."""
        tier_clean = tier.lower().strip()
        if tier_clean not in self.TIER_THRESHOLDS_BYTES:
            valid = sorted(self.TIER_THRESHOLDS_BYTES.keys())
            raise ValueError(f"Unknown version tier '{tier}'. Valid tiers: {valid}")
        return os.path.join(self.base_output_dir, tier_clean)

    def validate_corpus_for_tier(
        self,
        total_bytes_or_chars: int,
        tier: str,
        dev_mode: bool = False,
        force: bool = False
    ) -> bool:
        """
        Validates that the buffered training corpus meets the size requirement for the target tier.
        Raises RuntimeError if insufficient, unless dev_mode or force is set.
        """
        tier_clean = tier.lower().strip()
        required_bytes = self.TIER_THRESHOLDS_BYTES.get(tier_clean, 0)

        if total_bytes_or_chars < required_bytes:
            current_mb = round(total_bytes_or_chars / (1024 * 1024), 2)
            required_mb = round(required_bytes / (1024 * 1024), 2)
            msg = (
                f"Corpus size ({current_mb} MB) is below the minimum threshold ({required_mb} MB) "
                f"required to promote/freeze tokenizer to tier '{tier_clean}'. "
                f"As per Phase 5.3 specifications, do not freeze Tokenizer v1 until the corpus expands to at least 50 MB. "
                f"Use tier='v0-experimental' for engineering and benchmarking."
            )
            if not dev_mode and not force:
                logger.error(f"TokenizerVersionManager: {msg}")
                raise RuntimeError(msg)
            else:
                logger.warning(f"TokenizerVersionManager: {msg} Proceeding due to dev_mode={dev_mode} / force={force}.")
                return False
        return True

    def _get_git_commit(self) -> str:
        """Retrieves the current git commit hash if available."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            ).strip()
            return commit
        except Exception:
            return os.environ.get("GIT_COMMIT", "unknown")

    def save_version_metadata(
        self,
        tier: str,
        algorithm: str,
        vocab_size: int,
        training_metadata: Dict[str, Any],
        evaluation_summary: Optional[Dict[str, Any]] = None,
        model_subdirectory: Optional[str] = None
    ) -> str:
        """
        Saves standardized reproducible metadata.json and training_config.json inside the tier/model directory.
        """
        tier_dir = self.get_tier_directory(tier)
        if model_subdirectory:
            target_dir = os.path.join(tier_dir, model_subdirectory)
        else:
            target_dir = os.path.join(tier_dir, f"{algorithm}_{vocab_size}")

        os.makedirs(target_dir, exist_ok=True)

        meta_path = os.path.join(target_dir, "metadata.json")
        config_path = os.path.join(target_dir, "training_config.json")

        git_commit = self._get_git_commit()

        # Build complete tier metadata
        complete_meta = {
            "version_tier": tier,
            "algorithm": algorithm,
            "vocab_size": vocab_size,
            "git_commit": git_commit,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "5.3",
            "dataset_hash": training_metadata.get("config_hash", "unknown"),
            "sentence_count": training_metadata.get("training_samples", 0),
            "token_count": training_metadata.get("total_characters_observed", 0),
            "training_duration_sec": training_metadata.get("training_duration_sec"),
            "training_metadata": training_metadata,
            "evaluation_summary": evaluation_summary or {},
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(complete_meta, f, indent=2, default=str)

        training_cfg = training_metadata.get("training_config", {})
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(training_cfg, f, indent=2, default=str)

        try:
            from app.tokenization.card_generator import generate_tokenizer_cards
            generate_tokenizer_cards(complete_meta, target_dir)
        except Exception as card_err:
            logger.warning(f"TokenizerVersionManager: Could not generate model cards: {card_err}")

        logger.info(f"TokenizerVersionManager: Saved versioned metadata to '{meta_path}'")
        return meta_path
