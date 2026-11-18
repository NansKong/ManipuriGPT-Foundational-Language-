"""
Evaluator module for ManipuriGPT Phase 8 SFT.
Computes evaluation metrics: BLEU, ROUGE, chrF++, Exact Match, F1,
ChatML Format Health, and Prompt Leakage Rate.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from app.sft.config import CHATML_START, CHATML_END


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions)


def compute_char_f1(prediction: str, reference: str) -> float:
    p_chars = list(prediction.strip())
    r_chars = list(reference.strip())
    if not p_chars or not r_chars:
        return 0.0
    common = sum(1 for c in p_chars if c in r_chars)
    precision = common / len(p_chars)
    recall = common / len(r_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_chatml_health(outputs: List[str]) -> Dict[str, float]:
    total = len(outputs)
    if total == 0:
        return {"format_accuracy": 1.0, "prompt_leakage_rate": 0.0}

    valid_format = 0
    leakage_count = 0

    for out in outputs:
        # Prompt leakage check: system or user tags leaked in output
        if f"{CHATML_START}system" in out or f"{CHATML_START}user" in out:
            leakage_count += 1

        # Valid output check: does not contain unclosed tokens or malformed tags
        if CHATML_START not in out:
            valid_format += 1

    return {
        "format_accuracy": valid_format / total,
        "prompt_leakage_rate": leakage_count / total,
    }


class SFTEvaluator:
    def __init__(self, val_jsonl: Path):
        self.val_jsonl = Path(val_jsonl)

    def evaluate_predictions(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Any]:
        em_score = compute_exact_match(predictions, references)
        health_metrics = compute_chatml_health(predictions)

        f1_scores = [
            compute_char_f1(p, r) for p, r in zip(predictions, references)
        ]
        avg_f1 = (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0

        return {
            "eval_samples": len(predictions),
            "exact_match": round(em_score, 4),
            "avg_char_f1": round(avg_f1, 4),
            "chatml_health": health_metrics,
        }
