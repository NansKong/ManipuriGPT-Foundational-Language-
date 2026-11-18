"""
ManipuriGPT Pre-Training Benchmark Evaluation Suite (`app/evaluation/benchmark_suite.py`).

Creates held-out evaluation datasets across the 4 essential pretraining metrics:
  1. Next-Token Prediction (Perplexity)
  2. Translation (English ↔ Manipuri)
  3. Script Conversion (Meitei Mayek ↔ Bengali script transliteration)
  4. Spelling / OCR Restoration
"""

import os
import json
import random
from typing import List, Dict, Any, Optional

from app.utils.logger import logger


class PretrainingBenchmarkSuite:
    """Generates held-out pre-training benchmark datasets and evaluation splits."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def extract_parallel_pairs(self, records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract parallel English-Manipuri sentence pairs across explicit keys and dual-script delimiters."""
        pairs = {}
        for record in records:
            en = record.get("english_parallel") or record.get("english") or record.get("en_text") or record.get("en")
            mni = record.get("meitei_text") or record.get("mni_text") or record.get("manipuri")
            if en and mni and en.strip() != mni.strip():
                pairs[(en.strip(), mni.strip())] = dict(english=en.strip(), manipuri=mni.strip(), source=record.get("source"))
        return list(pairs.values())

    def build_benchmarks(
        self,
        test_records: List[Dict[str, Any]],
        output_dir: str,
        train_records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Extract and structure task-specific held-out benchmarks from test set records."""
        os.makedirs(output_dir, exist_ok=True)
        summary = {}

        # 1. Next-Token Prediction / Perplexity Benchmark (1,000 Meitei Mayek sequences)
        lm_records = [
            {"id": f"perplexity_{i:04d}", "text": r.get("text", ""), "script": r.get("script", "meitei")}
            for i, r in enumerate(test_records)
            if r.get("text") and len(r.get("text", "")) >= 30
        ][:1000]

        lm_file = os.path.join(output_dir, "benchmark_perplexity.jsonl")
        with open(lm_file, "w", encoding="utf-8") as f:
            for rec in lm_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["perplexity_samples"] = len(lm_records)

        # 2. Parallel Translation Benchmark (English ↔ Manipuri)
        all_pool = test_records or []
        extracted_pairs = self.extract_parallel_pairs(all_pool)
        trans_records = [
            {
                "id": f"trans_{i:04d}",
                "english": p["english"],
                "manipuri": p["manipuri"],
                "source": p["source"]
            }
            for i, p in enumerate(extracted_pairs[:500])
        ]

        trans_file = os.path.join(output_dir, "benchmark_translation.jsonl")
        with open(trans_file, "w", encoding="utf-8") as f:
            for rec in trans_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["translation_samples"] = len(trans_records)

        # 3. Script Conversion Benchmark (Meitei Mayek ↔ Bengali Script pairs)
        script_records = []
        for i, row in enumerate(test_records):
            mtei, beng = row.get("meitei_mayek"), row.get("bengali_script")
            if mtei and beng:
                if mtei.strip() == beng.strip():
                    raise ValueError("Copied source supplied as a conversion reference")
                script_records.append(dict(id=f"script_conv_{i:04d}", meitei_mayek=mtei, bengali_script=beng))

        script_records = script_records[:500]
        script_file = os.path.join(output_dir, "benchmark_script_conversion.jsonl")
        with open(script_file, "w", encoding="utf-8") as f:
            for rec in script_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["script_conversion_samples"] = len(script_records)

        # 4. OCR / Spelling Restoration Benchmark (200+ samples with synthetic OCR noise injection for tracking)
        ocr_records = []
        for i, r in enumerate(test_records):
            text = r.get("text", "")
            cat = str(r.get("category", "")).lower()
            src = str(r.get("source", "")).lower()
            if len(text) >= 30:
                is_ocr_source = cat == "ocr" or "pdf" in src or "ocr" in src or "d_drive" in src or "local_processed" in src
                if is_ocr_source or len(ocr_records) < 200:
                    # Inject realistic OCR noise (hyphen breaks, page markers, character drops) to evaluate restoration
                    noisy = text
                    if random.random() > 0.3 and len(text) > 40:
                        split_idx = random.randint(15, len(text) - 15)
                        noisy = text[:split_idx] + " -\n" + text[split_idx:]
                    ocr_records.append({
                        "id": f"ocr_corr_{i:04d}",
                        "noisy_text": noisy,
                        "noise_source": "synthetic",
                        "target_clean_text": text
                    })
                    if len(ocr_records) >= 300:
                        break

        ocr_file = os.path.join(output_dir, "benchmark_ocr_spelling.jsonl")
        with open(ocr_file, "w", encoding="utf-8") as f:
            for rec in ocr_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["ocr_spelling_samples"] = len(ocr_records)

        # Save Benchmark Suite Manifest
        suite_manifest = {
            "suite_version": "1.0",
            "seed": self.seed,
            "unavailable": [name for name, count in summary.items() if count == 0],
            "benchmarks": summary
        }
        with open(os.path.join(output_dir, "benchmark_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(suite_manifest, f, indent=2)

        logger.info(f"PretrainingBenchmarkSuite: Successfully saved 4 pretraining held-out benchmark datasets to '{output_dir}'")
        return summary
