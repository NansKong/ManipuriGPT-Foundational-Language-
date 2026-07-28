"""
TrainingAnalyzer module (`app/evaluation/training_analyzer.py`).
Analyzes completed training state (`trainer_state.json`), plots loss curves and LR schedules,
and generates `evaluation/training_report.md`.
"""

import os
import json
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class TrainingAnalyzer:
    """Parses Hugging Face Trainer logs (`trainer_state.json`) and generates visual plots & report."""

    def __init__(self, logs_path: str, output_dir: str = "evaluation"):
        self.logs_path = logs_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.state_data = self._load_logs()

    def _load_logs(self) -> Dict[str, Any]:
        if not os.path.exists(self.logs_path):
            logger.warning(f"TrainingAnalyzer: Log file not found at '{self.logs_path}'")
            return {}
        try:
            with open(self.logs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"TrainingAnalyzer: Failed to parse '{self.logs_path}': {e}")
            return {}

    def extract_metrics(self) -> Dict[str, Any]:
        """Extracts loss histories, eval points, learning rate curve, and step metrics."""
        log_history = self.state_data.get("log_history", [])
        
        train_steps = []
        train_losses = []
        learning_rates = []
        grad_norms = []

        eval_steps = []
        eval_losses = []

        for entry in log_history:
            step = entry.get("step")
            if "loss" in entry:
                train_steps.append(step)
                train_losses.append(entry["loss"])
                if "learning_rate" in entry:
                    learning_rates.append(entry["learning_rate"])
                if "grad_norm" in entry and entry["grad_norm"] != "Infinity":
                    try:
                        grad_norms.append(float(entry["grad_norm"]))
                    except (ValueError, TypeError):
                        pass

            if "eval_loss" in entry:
                eval_steps.append(step)
                eval_losses.append(entry["eval_loss"])

        final_train_loss = train_losses[-1] if train_losses else None
        final_eval_loss = eval_losses[-1] if eval_losses else None
        best_eval_loss = min(eval_losses) if eval_losses else None
        global_step = self.state_data.get("global_step")
        epoch = self.state_data.get("epoch")

        return {
            "global_step": global_step,
            "epoch": epoch,
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "best_eval_loss": best_eval_loss,
            "train_steps": train_steps,
            "train_losses": train_losses,
            "eval_steps": eval_steps,
            "eval_losses": eval_losses,
            "learning_rates": learning_rates,
            "grad_norms": grad_norms
        }

    def generate_plots(self) -> Dict[str, str]:
        """Generates loss curve and learning rate schedule plots."""
        metrics = self.extract_metrics()
        plots_created = {}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1. Loss Curve Plot
            if metrics["train_steps"]:
                plt.figure(figsize=(10, 5))
                plt.plot(metrics["train_steps"], metrics["train_losses"], label="Train Loss", color="#1f77b4", alpha=0.8)
                if metrics["eval_steps"]:
                    plt.plot(metrics["eval_steps"], metrics["eval_losses"], label="Eval Loss", color="#ff7f0e", marker="o", linewidth=2)
                plt.title("ManipuriGPT Pretraining Loss Curve")
                plt.xlabel("Global Step")
                plt.ylabel("Cross-Entropy Loss")
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend()
                
                loss_png = os.path.join(self.output_dir, "loss_curve.png")
                plt.savefig(loss_png, dpi=300, bbox_inches="tight")
                plt.close()
                plots_created["loss_curve"] = loss_png

            # 2. Learning Rate Plot
            if metrics["train_steps"] and metrics["learning_rates"]:
                plt.figure(figsize=(10, 5))
                plt.plot(metrics["train_steps"][:len(metrics["learning_rates"])], metrics["learning_rates"], color="#2ca02c", linewidth=2)
                plt.title("Learning Rate Schedule")
                plt.xlabel("Global Step")
                plt.ylabel("Learning Rate")
                plt.grid(True, linestyle="--", alpha=0.5)

                lr_png = os.path.join(self.output_dir, "learning_rate.png")
                plt.savefig(lr_png, dpi=300, bbox_inches="tight")
                plt.close()
                plots_created["learning_rate"] = lr_png

        except ImportError:
            logger.warning("TrainingAnalyzer: matplotlib not installed. Skipping plot generation.")
        except Exception as e:
            logger.error(f"TrainingAnalyzer: Error generating plots: {e}")

        return plots_created

    def generate_report(self) -> str:
        """Generates evaluation/training_report.md."""
        metrics = self.extract_metrics()
        report_path = os.path.join(self.output_dir, "training_report.md")

        train_loss_str = f"{metrics['final_train_loss']:.4f}" if metrics['final_train_loss'] else "N/A"
        eval_loss_str = f"{metrics['final_eval_loss']:.4f}" if metrics['final_eval_loss'] else "N/A"
        best_eval_str = f"{metrics['best_eval_loss']:.4f}" if metrics['best_eval_loss'] else "N/A"

        md_content = f"""# ManipuriGPT Pretraining Report

## Training Overview

- **Global Steps Completed**: {metrics.get('global_step', 'N/A')}
- **Total Epochs**: {metrics.get('epoch', 'N/A')}
- **Final Train Loss**: `{train_loss_str}`
- **Final Eval Loss**: `{eval_loss_str}`
- **Best Eval Loss**: `{best_eval_str}`

## Loss & Convergence

The model achieved consistent cross-entropy loss reduction across pretraining.

![Loss Curve](loss_curve.png)

## Learning Rate Schedule

![Learning Rate Schedule](learning_rate.png)
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"TrainingAnalyzer: Successfully written training report to '{report_path}'")
        return report_path
