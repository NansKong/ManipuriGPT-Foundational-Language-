"""
CheckpointComparer module (`app/evaluation/checkpoint_compare.py`).
Compares multiple saved checkpoints in a structured scorecard table and performs regression analysis.
"""

import os
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class CheckpointComparer:
    """Multi-checkpoint comparison and regression testing engine."""

    def __init__(self, output_dir: str = "evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def format_comparison_table(self, checkpoints_data: List[Dict[str, Any]]) -> str:
        """Formats a markdown comparison table across evaluated checkpoints."""
        headers = ["Checkpoint", "Global Step", "Train Loss", "Eval Loss", "PPL", "Tokens/sec", "Human Score"]
        
        rows = []
        for ckpt in checkpoints_data:
            name = str(ckpt.get("checkpoint_name", "ckpt"))
            step = str(ckpt.get("global_step", "N/A"))
            t_loss = f"{ckpt['train_loss']:.4f}" if "train_loss" in ckpt and ckpt["train_loss"] is not None else "N/A"
            e_loss = f"{ckpt['eval_loss']:.4f}" if "eval_loss" in ckpt and ckpt["eval_loss"] is not None else "N/A"
            ppl = f"{ckpt['ppl']:.2f}" if "ppl" in ckpt and ckpt["ppl"] is not None else "N/A"
            tps = f"{ckpt['tokens_per_sec']:.2f}" if "tokens_per_sec" in ckpt and ckpt["tokens_per_sec"] is not None else "N/A"
            score = f"{ckpt['human_score']:.1f}" if "human_score" in ckpt and ckpt["human_score"] is not None else "N/A"
            rows.append(f"| {name} | {step} | {t_loss} | {e_loss} | {ppl} | {tps} | {score} |")

        table_header = "| " + " | ".join(headers) + " |\n" + "| " + " | ".join(["---"] * len(headers)) + " |"
        table_body = "\n".join(rows)

        return f"{table_header}\n{table_body}"

    def compute_regression_deltas(self, current_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Computes metric regression deltas between current model and baseline model."""
        curr_ppl = current_metrics.get("overall_ppl", 0.0)
        base_ppl = baseline_metrics.get("overall_ppl", 0.0)
        ppl_delta = round(curr_ppl - base_ppl, 4) if curr_ppl and base_ppl else 0.0

        curr_tps = current_metrics.get("tokens_per_sec", 0.0)
        base_tps = baseline_metrics.get("tokens_per_sec", 0.0)
        speed_delta = round(curr_tps - base_tps, 2) if curr_tps and base_tps else 0.0

        return {
            "ppl_delta": ppl_delta,
            "ppl_improved": ppl_delta < 0,
            "speed_delta": speed_delta,
            "speed_improved": speed_delta > 0
        }

    def generate_comparison_report(self, checkpoints_data: List[Dict[str, Any]]) -> str:
        """Generates `evaluation/checkpoint_comparison.md`."""
        table = self.format_comparison_table(checkpoints_data)
        out_file = os.path.join(self.output_dir, "checkpoint_comparison.md")

        md_content = f"""# Checkpoint Comparison & Regression Report

## Multi-Checkpoint Scorecard

{table}

*Evaluation completed across checkpoints.*
"""
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"CheckpointComparer: Written comparison report to '{out_file}'")
        return out_file
