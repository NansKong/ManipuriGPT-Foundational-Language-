"""
Acceptance Gate module for ManipuriGPT Phase 8 Corpus Evolution (`app/corpus/acceptance_gate.py`).
Executes multi-dimensional quality audit over Candidate Corpus-v4 and generates `acceptance_report_v4.json`.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


class CorpusAcceptanceGate:
    """Evaluates Candidate Corpus-v4 against explicit quality metrics."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/datasets/Candidate-Corpus-v4")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_audit(self, candidate_sequences: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        total_seqs = len(candidate_sequences)
        if total_seqs == 0:
            return False, {"passed": False, "reason": "empty_candidate_corpus"}

        script_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        total_words = 0
        total_chars = 0

        high_quality_count = 0
        rejected_short_count = 0
        rejected_repetitive_count = 0

        for seq in candidate_sequences:
            text = seq.get("text", "")
            script = seq.get("script", "unknown")
            category = seq.get("category", "General")

            script_counts[script] = script_counts.get(script, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1

            w_cnt = len(text.split())
            c_cnt = len(text)
            total_words += w_cnt
            total_chars += c_cnt

            # Quality Audit Checks:
            # 1. Minimum sequence length check (w_cnt >= 3)
            # 2. OCR repetitive character noise check (unique chars < 3 in long sequence)
            is_too_short = w_cnt < 3
            is_repetitive_noise = (len(set(text)) < 3 and len(text) > 10)

            if is_too_short:
                rejected_short_count += 1
            elif is_repetitive_noise:
                rejected_repetitive_count += 1
            else:
                high_quality_count += 1

        quality_ratio = high_quality_count / total_seqs
        passed = (total_seqs > 0) and (quality_ratio >= 0.90)

        report = {
            "acceptance_gate": "Corpus-v4 Acceptance Gate",
            "passed": passed,
            "total_sequences_audited": total_seqs,
            "high_quality_sequences_passed": high_quality_count,
            "rejected_sequences_count": total_seqs - high_quality_count,
            "rejection_breakdown": {
                "sequence_too_short_words_lt_3": rejected_short_count,
                "repetitive_ocr_noise": rejected_repetitive_count,
            },
            "quality_ratio_pct": round(quality_ratio * 100, 2),
            "total_words": total_words,
            "total_characters": total_chars,
            "avg_words_per_sequence": round(total_words / total_seqs, 2) if total_seqs > 0 else 0.0,
            "avg_chars_per_sequence": round(total_chars / total_seqs, 2) if total_seqs > 0 else 0.0,
            "script_distribution": script_counts,
            "category_distribution": category_counts,
            "gate_criteria": {
                "min_quality_ratio_pct": 90.0,
                "passed_quality_check": quality_ratio >= 0.90,
            }
        }

        # Save report
        report_file = self.output_dir / "acceptance_report_v4.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        report["report_file"] = str(report_file)
        return passed, report
