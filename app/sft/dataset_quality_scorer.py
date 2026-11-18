"""
Quality Scorer Module for ManipuriGPT Phase 8 SFT (app/sft/dataset_quality_scorer.py).
Evaluates multi-metric composite quality score Q in [0.0, 1.0] for instruction-response records.
Filters records below quality threshold Q < 0.85 and logs rejections.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from app.sft.config import STATS_DATA_DIR


# Unicode Ranges
MEETEI_MAYEK_PATTERN = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAFF]')
BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')


class SFTQualityScorer:
    """Evaluates multi-metric quality scores for SFT records."""

    def __init__(self, min_threshold: float = 0.85):
        self.min_threshold = min_threshold
        self.rejections_log = STATS_DATA_DIR / "quality_rejections_v1.jsonl"
        self.rejected_records: List[Dict[str, Any]] = []

    @staticmethod
    def validate_unicode_purity(text: str) -> bool:
        """Validates that text contains clean Unicode without surrogate errors or broken byte sequences."""
        if not text:
            return False
        try:
            text.encode('utf-8')
            # Check for surrogate pairs or unprintable control characters
            return not bool(re.search(r'[\uD800-\uDFFF]', text))
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False

    @staticmethod
    def detect_script(text: str) -> str:
        """Detects primary script of a string."""
        meetei_count = len(MEETEI_MAYEK_PATTERN.findall(text))
        bengali_count = len(BENGALI_PATTERN.findall(text))
        latin_count = len(LATIN_PATTERN.findall(text))

        total = meetei_count + bengali_count + latin_count
        if total == 0:
            return "unknown"

        if meetei_count / total >= 0.5:
            return "MeeteiMayek"
        elif bengali_count / total >= 0.5:
            return "Bengali"
        elif latin_count / total >= 0.5:
            return "Latin"
        else:
            return "Mixed"

    def score_record(self, record: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Computes composite quality score Q in [0.0, 1.0].
        Sub-metrics:
          - completeness (0.20): Instruction non-empty, min 3 words.
          - response_quality (0.25): Non-empty, >= 6 chars, non-repetitive (n-gram repeat check).
          - script_validity (0.25): Recognized Unicode script range (script neutral quality).
          - formatting (0.15): Clean punctuation, zero broken HTML/control chars.
          - coherence (0.15): Input/output non-identical, subtask instruction-output alignment.
        """
        instruction = record.get("instruction", "").strip()
        input_text = record.get("input", "").strip()
        output = record.get("output", "").strip()
        task_type = record.get("task_type", "").lower()

        # Sub-metric 1: Instruction Completeness (0.20)
        words_inst = instruction.split()
        if len(words_inst) >= 3:
            s_completeness = 1.0
        elif len(words_inst) >= 1:
            s_completeness = 0.5
        else:
            s_completeness = 0.0

        # Sub-metric 2: Response Quality (0.25)
        len_out = len(output)
        if 6 <= len_out <= 2000:
            words_out = output.split()
            # 1. Word uniqueness ratio check
            unique_word_ratio = len(set(words_out)) / len(words_out) if words_out else 1.0

            # 2. Character 3-gram repetition check (detects OCR looping glitch like "lailailai")
            n_grams = [output[i:i+3] for i in range(len(output) - 2)]
            unique_ngram_ratio = len(set(n_grams)) / len(n_grams) if n_grams else 1.0

            if (len(words_out) > 6 and unique_word_ratio < 0.35) or (len(n_grams) > 10 and unique_ngram_ratio < 0.40):
                s_response = 0.2  # Severe penalty for repetitive looping text
            else:
                s_response = 1.0
        elif 1 <= len_out < 6:
            s_response = 0.2  # Too short/trivial single word
        else:
            s_response = 0.0

        # Sub-metric 3: Script Validity (0.25) - Objective Script Neutrality
        script_out = self.detect_script(output)
        if script_out in ["MeeteiMayek", "Bengali", "Latin", "Mixed"]:
            s_script = 1.0
        else:
            s_script = 0.2

        # Sub-metric 4: Formatting Correctness & Unicode Purity (0.15)
        has_html = bool(re.search(r'<[^>]+>', output)) or bool(re.search(r'<[^>]+>', instruction))
        has_control_chars = bool(re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', output))
        is_clean_unicode = self.validate_unicode_purity(instruction) and self.validate_unicode_purity(output)
        if not has_html and not has_control_chars and is_clean_unicode:
            s_format = 1.0
        elif has_html or not is_clean_unicode:
            s_format = 0.5
        else:
            s_format = 0.2

        # Sub-metric 5: Coherence & Instruction-Output Consistency (0.15)
        if input_text and input_text == output:
            s_coherence = 0.0  # Trivial identity copy
        else:
            # Check direction alignment for subtasks
            s_coherence = 1.0
            if task_type == "translation_mm_to_en" and script_out == "MeeteiMayek":
                s_coherence = 0.0  # Mismatch: asked for English output but got Meetei Mayek
            elif task_type in ["translation_en_to_mm", "script_conversion_bn_to_mm"] and script_out == "Latin":
                s_coherence = 0.0  # Mismatch: asked for Meetei Mayek output but got Latin

        sub_scores = {
            "completeness": s_completeness,
            "response_quality": s_response,
            "script_validity": s_script,
            "formatting": s_format,
            "coherence": s_coherence,
        }

        composite_score = round(
            0.20 * s_completeness
            + 0.25 * s_response
            + 0.25 * s_script
            + 0.15 * s_format
            + 0.15 * s_coherence,
            4
        )

        return composite_score, sub_scores

    def filter_records(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filters records based on min_threshold (default >= 0.85).
        Attaches quality_score, script (detected from output text), and quality breakdown to kept records.
        """
        kept = []
        rejected = []

        for rec in records:
            score, sub_scores = self.score_record(rec)
            rec["quality_score"] = score
            # Always base script attribute directly on assistant output script
            rec["script"] = self.detect_script(rec.get("output", ""))

            if score >= self.min_threshold:
                kept.append(rec)
            else:
                rec_rejected = dict(rec)
                rec_rejected["rejection_reason"] = sub_scores
                rejected.append(rec_rejected)

        if rejected:
            self.log_rejections(rejected)

        return kept, rejected

    def log_rejections(self, rejected: List[Dict[str, Any]]):
        """Logs rejected records to jsonl for auditing."""
        self.rejections_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rejections_log, "a", encoding="utf-8") as f:
            for rec in rejected:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
