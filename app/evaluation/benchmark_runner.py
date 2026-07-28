"""
BenchmarkRunner module (`app/evaluation/benchmark_runner.py`).
Executes quantitative evaluations across Phase 6 held-out task benchmark splits.
"""

import os
import json
from typing import Dict, Any, List, Optional
from app.evaluation.evaluator import ManipuriEvaluator
from app.utils.logger import logger


class BenchmarkRunner:
    """Task-level benchmark suite evaluation engine."""

    def __init__(self, evaluator: Optional[ManipuriEvaluator] = None):
        self.evaluator = evaluator or ManipuriEvaluator()

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Utility to load benchmark JSONL records."""
        if not os.path.exists(file_path):
            logger.warning(f"BenchmarkRunner: Benchmark file '{file_path}' not found.")
            return []
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def run_translation_benchmark(self, file_path: str, model_generate_fn: Any) -> Dict[str, Any]:
        """Evaluates English ↔ Manipuri translation benchmark."""
        records = self.load_jsonl(file_path)
        if not records:
            return {"task": "translation", "error": "No records found"}

        predictions = []
        references = []

        for r in records[:100]:  # Evaluate sample subset
            eng = r.get("english", "")
            mni_ref = r.get("manipuri", "")
            if eng and mni_ref:
                pred = model_generate_fn(eng)
                predictions.append(pred)
                references.append(mni_ref)

        scorecard = self.evaluator.evaluate_task("translation", predictions, references)
        return scorecard

    def run_script_conversion_benchmark(self, file_path: str, model_generate_fn: Any) -> Dict[str, Any]:
        """Evaluates Meitei Mayek ↔ Bengali script conversion benchmark."""
        records = self.load_jsonl(file_path)
        if not records:
            return {"task": "script_conversion", "error": "No records found"}

        predictions = []
        references = []

        for r in records[:100]:
            meitei_src = r.get("meitei_mayek", "")
            bengali_ref = r.get("bengali_script", "")
            if meitei_src and bengali_ref:
                pred = model_generate_fn(meitei_src)
                predictions.append(pred)
                references.append(bengali_ref)

        scorecard = self.evaluator.evaluate_task("translation", predictions, references)
        scorecard["task"] = "script_conversion"
        return scorecard

    def run_ocr_spelling_benchmark(self, file_path: str, model_generate_fn: Any) -> Dict[str, Any]:
        """Evaluates OCR noise restoration benchmark."""
        records = self.load_jsonl(file_path)
        if not records:
            return {"task": "ocr_spelling", "error": "No records found"}

        predictions = []
        references = []

        for r in records[:100]:
            noisy = r.get("noisy_text", "")
            clean_ref = r.get("target_clean_text", "")
            if noisy and clean_ref:
                pred = model_generate_fn(noisy)
                predictions.append(pred)
                references.append(clean_ref)

        scorecard = self.evaluator.evaluate_task("qa", predictions, references)
        scorecard["task"] = "ocr_spelling"
        return scorecard
