"""
CheckpointManager module for multi-model foundation training (Phase 5).
Handles periodic checkpoint storage, resume-from-checkpoint validation, best model tracking,
and automated retention pruning.
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional, Tuple
from app.utils.logger import logger


class CheckpointManager:
    """
    Manages model checkpointing lifecycle across training runs.
    Tracks best checkpoints based on validation metric thresholds (loss, perplexity, BLEU).
    """
    def __init__(
        self,
        base_output_dir: str = "artifacts/models/checkpoints",
        metric_name: str = "eval_loss",
        metric_minimize: bool = True,
        max_keep: int = 3
    ):
        self.base_output_dir = base_output_dir
        self.metric_name = metric_name
        self.metric_minimize = metric_minimize
        self.max_keep = max_keep
        os.makedirs(self.base_output_dir, exist_ok=True)

        self.checkpoints: List[Dict[str, Any]] = []
        self.best_checkpoint: Optional[Dict[str, Any]] = None
        self.state_file = os.path.join(self.base_output_dir, "checkpoint_registry.json")
        self._load_registry()

    def _load_registry(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.checkpoints = data.get("checkpoints", [])
                self.best_checkpoint = data.get("best_checkpoint", None)
            except Exception as e:
                logger.warning(f"CheckpointManager: Could not load registry ({e}). Starting fresh.")

    def _save_registry(self) -> None:
        data = {
            "metric_name": self.metric_name,
            "metric_minimize": self.metric_minimize,
            "best_checkpoint": self.best_checkpoint,
            "checkpoints": self.checkpoints
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"CheckpointManager: Failed to save registry: {e}")

    def save_checkpoint_metadata(
        self,
        step: int,
        checkpoint_path: str,
        metrics: Dict[str, float],
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Registers a new checkpoint directory and checks if it qualifies as the new best model.
        Prunes old checkpoints exceeding `max_keep`.
        """
        metric_val = metrics.get(self.metric_name, float("inf") if self.metric_minimize else float("-inf"))
        entry = {
            "step": step,
            "path": checkpoint_path,
            "metrics": metrics,
            "tag": tag or f"step-{step}"
        }
        self.checkpoints.append(entry)

        # Check best model
        is_new_best = False
        if self.best_checkpoint is None:
            is_new_best = True
        else:
            best_val = self.best_checkpoint["metrics"].get(self.metric_name, float("inf") if self.metric_minimize else float("-inf"))
            if self.metric_minimize and metric_val < best_val:
                is_new_best = True
            elif not self.metric_minimize and metric_val > best_val:
                is_new_best = True

        if is_new_best:
            logger.info(f"CheckpointManager: New best checkpoint achieved at step {step} ({self.metric_name}={metric_val})")
            self.best_checkpoint = entry

        # Prune old checkpoints
        if len(self.checkpoints) > self.max_keep:
            to_remove = self.checkpoints.pop(0)
            # Never prune the best checkpoint
            if self.best_checkpoint and to_remove["path"] == self.best_checkpoint["path"] and len(self.checkpoints) > 0:
                to_remove = self.checkpoints.pop(0)
            
            if os.path.exists(to_remove["path"]) and to_remove["path"] != getattr(self.best_checkpoint, "get", lambda k: "")("path"):
                try:
                    shutil.rmtree(to_remove["path"])
                    logger.debug(f"CheckpointManager: Pruned old checkpoint '{to_remove['path']}'")
                except Exception as e:
                    logger.warning(f"CheckpointManager: Could not prune directory '{to_remove['path']}' ({e})")

        self._save_registry()
        return entry

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Returns file path of the most recently recorded checkpoint for resume training.
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]["path"]

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Returns file path of the best performing checkpoint.
        """
        if not self.best_checkpoint:
            return self.get_latest_checkpoint()
        return self.best_checkpoint["path"]
