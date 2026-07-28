"""
MemorizationEvaluator module (`app/evaluation/memorization_eval.py`).
Tests model outputs for verbatim memorization, exact substring overlap, and held-out completion match.
"""

import re
from typing import Dict, Any, List, Optional
from app.utils.logger import logger


class MemorizationEvaluator:
    """Evaluates verbatim memorization and training sequence overlap."""

    def __init__(self, training_corpus_sample: Optional[List[str]] = None):
        self.training_sample = training_corpus_sample or []
        self.concat_train_text = " ".join(self.training_sample)

    def check_exact_substring_match(self, generated_text: str, min_length: int = 30) -> Dict[str, Any]:
        """Checks if a generated sequence of length >= min_length exists verbatim in training sample."""
        clean_gen = generated_text.strip()
        if len(clean_gen) < min_length:
            return {"is_exact_match": False, "matched_length": 0}

        if not self.concat_train_text:
            return {"is_exact_match": False, "matched_length": 0, "note": "No training sample provided"}

        is_match = clean_gen in self.concat_train_text
        return {
            "generated_text": clean_gen[:60] + "..." if len(clean_gen) > 60 else clean_gen,
            "is_exact_match": is_match,
            "matched_length": len(clean_gen) if is_match else 0
        }

    def evaluate_memorization_rate(self, generated_texts: List[str]) -> Dict[str, Any]:
        """Calculates exact match memorization rate across generated passages."""
        if not generated_texts:
            return {"memorization_rate": 0.0, "total_samples": 0}

        exact_matches = 0
        details = []

        for text in generated_texts:
            res = self.check_exact_substring_match(text)
            details.append(res)
            if res["is_exact_match"]:
                exact_matches += 1

        total = len(generated_texts)
        rate = round(exact_matches / total, 4) if total > 0 else 0.0

        return {
            "memorization_rate": rate,
            "exact_matches_found": exact_matches,
            "total_samples_evaluated": total,
            "sample_details": details[:10]
        }
