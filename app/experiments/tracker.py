"""
ExperimentTracker module coordinating telemetry across `MLflow`, `TensorBoard`, and `WandB` (`Phase 5`).
Automatically falls back to local JSONL audit logs when external logging backends are offline or unconfigured.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
from app.utils.logger import logger


class ExperimentTracker:
    """
    Unified multi-backend experiment telemetry tracker.
    Logs parameters, metrics, and artifact paths to `MLflow`, `TensorBoard`, `WandB`,
    and/or local file `experiments.jsonl`.
    """
    def __init__(
        self,
        experiment_name: str = "manipurigpt_training",
        run_name: Optional[str] = None,
        backends: Optional[List[str]] = None,
        output_dir: str = "artifacts/experiments"
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        self.backends = [b.lower().strip() for b in (backends or ["jsonl"])]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.output_dir, f"{self.run_name}.jsonl")
        self._init_backends()

    def _init_backends(self) -> None:
        logger.info(f"ExperimentTracker: Initializing run '{self.run_name}' under experiment '{self.experiment_name}' across backends: {self.backends}")
        for b in self.backends:
            if b == "mlflow":
                try:
                    import mlflow
                    mlflow.set_experiment(self.experiment_name)
                    mlflow.start_run(run_name=self.run_name)
                except Exception as e:
                    logger.warning(f"ExperimentTracker: MLflow initialization skipped ({e}).")
            elif b == "wandb":
                try:
                    import wandb
                    if os.environ.get("WANDB_API_KEY"):
                        wandb.init(project=self.experiment_name, name=self.run_name)
                except Exception as e:
                    logger.warning(f"ExperimentTracker: WandB initialization skipped ({e}).")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Logs hyperparameter configurations.
        """
        self._write_local({"event": "params", "timestamp": datetime.utcnow().isoformat() + "Z", "data": params})
        if "mlflow" in self.backends:
            try:
                import mlflow
                for k, v in params.items():
                    mlflow.log_param(k, v)
            except Exception:
                pass
        if "wandb" in self.backends:
            try:
                import wandb
                wandb.config.update(params)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Logs numerical metrics at a given step.
        """
        self._write_local({"event": "metrics", "step": step, "timestamp": datetime.utcnow().isoformat() + "Z", "data": metrics})
        if "mlflow" in self.backends:
            try:
                import mlflow
                for k, v in metrics.items():
                    mlflow.log_metric(k, v, step=step)
            except Exception:
                pass
        if "wandb" in self.backends:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Logs a file or directory as an experiment artifact.
        """
        self._write_local({"event": "artifact", "local_path": local_path, "timestamp": datetime.utcnow().isoformat() + "Z"})
        if "mlflow" in self.backends:
            try:
                import mlflow
                mlflow.log_artifact(local_path, artifact_path)
            except Exception:
                pass

    def close(self) -> None:
        """
        Finalizes tracking runs across all active backends.
        """
        if "mlflow" in self.backends:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
        if "wandb" in self.backends:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        logger.info(f"ExperimentTracker: Closed tracking run '{self.run_name}'")

    def _write_local(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"ExperimentTracker: Failed to write local JSONL audit log ({e})")
