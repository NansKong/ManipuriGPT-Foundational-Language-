"""
Callbacks module for multi-model foundation training (Phase 5).
Handles periodic step logging, validation evaluation triggers, and checkpoint saving.
"""

import os
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class ManipuriTrainingCallbacks:
    """
    Manages training event hooks (on_step_end, on_epoch_end, on_train_end).
    Integrates with checkpoint managers and metrics trackers.
    """
    def __init__(self, log_steps: int = 10, eval_steps: int = 250, save_steps: int = 250):
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.step_history: List[Dict[str, Any]] = []

    def on_step_end(self, step: int, logs: Dict[str, Any], trainer_instance: Optional[Any] = None) -> None:
        """
        Invoked at the conclusion of each training gradient step.
        """
        if step % self.log_steps == 0:
            logger.info(f"Training Step {step}: {logs}")
            self.step_history.append({"step": step, **logs})

        if step % self.eval_steps == 0 and trainer_instance and hasattr(trainer_instance, "evaluate"):
            logger.info(f"Training Step {step}: Triggering periodic validation check...")
            try:
                eval_logs = trainer_instance.evaluate()
                logger.info(f"Validation step {step} results: {eval_logs}")
            except Exception as e:
                logger.warning(f"Validation check failed at step {step}: {e}")

        if step % self.save_steps == 0 and trainer_instance and hasattr(trainer_instance, "save_checkpoint"):
            logger.info(f"Training Step {step}: Saving periodic checkpoint...")
            trainer_instance.save_checkpoint(f"checkpoint-{step}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """
        Invoked at the conclusion of each training epoch.
        """
        logger.info(f"Epoch {epoch} completed. Summary -> {logs}")

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """
        Invoked when full training loop completes successfully.
        """
        logger.info(f"Training loop finalized successfully. Final state -> {logs}")
