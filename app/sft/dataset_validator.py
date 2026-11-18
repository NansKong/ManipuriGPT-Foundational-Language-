"""
Dataset Validator Module for ManipuriGPT Phase 8 SFT (app/sft/dataset_validator.py).
Executes automated pre-training validation:
- Task-specific guardrails (Translation, Definitions, QA, Script Conversion)
- MinHash Jaccard similarity near-duplicate detection (>0.85 similarity)
- Lexical diversity metrics (Type-Token Ratio, Distinct-1, Distinct-2, Shannon Entropy)
- Multi-token prefix checks (First 2, 3, 5 tokens)
- Automated audit report writer (sft_dataset_report.md)
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

from app.sft.config import STATS_DATA_DIR
from app.sft.dataset_quality_scorer import SFTQualityScorer


class MinHashDeduplicator:
    """Character n-gram MinHash for detecting near-duplicate outputs (>0.85 Jaccard similarity)."""

    def __init__(self, num_perm: int = 64, k: int = 4):
        self.num_perm = num_perm
        self.k = k
        # Fixed hash seed parameters
        self.a = [((i * 10007 + 3) % 2147483647) for i in range(num_perm)]
        self.b = [((i * 48611 + 17) % 2147483647) for i in range(num_perm)]
        self.prime = 2147483647

    def _get_shingles(self, text: str) -> Set[str]:
        cleaned = re.sub(r"\s+", "", text.lower())
        if len(cleaned) < self.k:
            return {cleaned}
        return {cleaned[i : i + self.k] for i in range(len(cleaned) - self.k + 1)}

    def compute_signature(self, text: str) -> List[int]:
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        sig = []
        shingle_hashes = [hash(s) & 0x7FFFFFFF for s in shingles]

        for i in range(self.num_perm):
            min_hash = min((self.a[i] * h + self.b[i]) % self.prime for h in shingle_hashes)
            sig.append(min_hash)
        return sig

    @staticmethod
    def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / float(len(sig1))


class SFTDatasetValidator:
    """Master Validator executing task-specific guardrails and diversity audit."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.report_dir = Path(STATS_DATA_DIR)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def validate_task_guardrails(self) -> Tuple[int, List[str]]:
        """Validates task-specific rules across all records."""
        violations = []
        valid_count = 0

        for idx, rec in enumerate(self.records):
            task_type = rec.get("task_type", "").lower()
            inp = rec.get("input", "").strip()
            out = rec.get("output", "").strip()
            rec_id = rec.get("id", f"rec_{idx}")

            if not out:
                violations.append(f"[{rec_id}] Empty output")
                continue

            inp_script = SFTQualityScorer.detect_script(inp) if inp else "unknown"
            out_script = SFTQualityScorer.detect_script(out)

            # 1. Translation Guardrails
            if task_type == "translation_en_to_mm":
                if inp and inp == out:
                    violations.append(f"[{rec_id}] Translation input equals output")
                    continue
                if out_script != "MeeteiMayek":
                    violations.append(f"[{rec_id}] EN->MM output is not Meetei Mayek script ({out_script})")
                    continue

            elif task_type == "translation_mm_to_en":
                if inp and inp == out:
                    violations.append(f"[{rec_id}] Translation input equals output")
                    continue
                if out_script != "Latin":
                    violations.append(f"[{rec_id}] MM->EN output is not Latin script ({out_script})")
                    continue

            # 2. Script Conversion Guardrail (BN -> MM)
            elif task_type == "script_conversion_bn_to_mm":
                bengali_chars = set("অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁ")
                if any(c in bengali_chars for c in out):
                    violations.append(f"[{rec_id}] Bengali characters detected in Meetei Mayek output")
                    continue
                if out_script != "MeeteiMayek":
                    violations.append(f"[{rec_id}] Script conversion output is not Meetei Mayek ({out_script})")
                    continue

            # 3. Definitions Guardrails
            elif task_type == "definitions":
                if len(out) < 15:
                    violations.append(f"[{rec_id}] Definition output too short (<15 chars)")
                    continue
                # Term check
                term_match = re.search(r"['\"`](.*?)['\"`]", rec.get("instruction", ""))
                term = term_match.group(1) if term_match else ""
                if term and term.strip().lower() == out.lower():
                    violations.append(f"[{rec_id}] Self-definition detected (term == output)")
                    continue

            # 4. QA Guardrails
            elif task_type == "qa":
                if len(out) < 5:
                    violations.append(f"[{rec_id}] QA output too short")
                    continue

            valid_count += 1

        return valid_count, violations

    def run_semantic_deduplication(self, threshold: float = 0.85) -> Tuple[int, List[Dict[str, Any]]]:
        """Detects and prunes near-duplicate outputs using MinHash Jaccard similarity."""
        minhash = MinHashDeduplicator()
        seen_signatures: List[List[int]] = []
        unique_records = []
        near_duplicate_count = 0

        for rec in self.records:
            out_text = rec.get("output", "")
            sig = minhash.compute_signature(out_text)

            is_duplicate = False
            for prev_sig in seen_signatures[-1500:]:  # Window of 1500 recent signatures
                if MinHashDeduplicator.jaccard_similarity(sig, prev_sig) >= threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                near_duplicate_count += 1
            else:
                seen_signatures.append(sig)
                unique_records.append(rec)

        return near_duplicate_count, unique_records

    def compute_lexical_diversity(self) -> Dict[str, float]:
        """Computes Type-Token Ratio, Distinct-1, Distinct-2, and Shannon Entropy."""
        all_tokens = []
        unigrams = []
        bigrams = []

        for rec in self.records:
            out_text = rec.get("output", "").strip()
            tokens = out_text.split()
            all_tokens.extend(tokens)
            unigrams.extend(tokens)

            for i in range(len(tokens) - 1):
                bigrams.append((tokens[i], tokens[i + 1]))

        total_tokens = len(all_tokens)
        if total_tokens == 0:
            return {"ttr": 0.0, "distinct_1": 0.0, "distinct_2": 0.0, "entropy": 0.0}

        unique_tokens = len(set(all_tokens))
        ttr = unique_tokens / float(total_tokens)
        distinct_1 = len(set(unigrams)) / float(len(unigrams)) if unigrams else 0.0
        distinct_2 = len(set(bigrams)) / float(len(bigrams)) if bigrams else 0.0

        # Shannon Entropy
        token_counts = Counter(all_tokens)
        entropy = -sum((c / total_tokens) * math.log2(c / total_tokens) for c in token_counts.values())

        return {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "ttr": round(ttr, 4),
            "distinct_1": round(distinct_1, 4),
            "distinct_2": round(distinct_2, 4),
            "entropy": round(entropy, 4),
        }

    def check_prefix_frequencies(self) -> Dict[str, Dict[str, float]]:
        """Checks top prefix frequency for first 2, 3, and 5 tokens."""
        prefixes_2 = Counter()
        prefixes_3 = Counter()
        prefixes_5 = Counter()
        total = len(self.records)

        if total == 0:
            return {}

        for rec in self.records:
            tokens = rec.get("output", "").strip().split()
            if len(tokens) >= 2:
                prefixes_2[" ".join(tokens[:2])] += 1
            if len(tokens) >= 3:
                prefixes_3[" ".join(tokens[:3])] += 1
            if len(tokens) >= 5:
                prefixes_5[" ".join(tokens[:5])] += 1

        top_2 = prefixes_2.most_common(1)
        top_3 = prefixes_3.most_common(1)
        top_5 = prefixes_5.most_common(1)

        return {
            "first_2_tokens": {
                "prefix": top_2[0][0] if top_2 else "N/A",
                "count": top_2[0][1] if top_2 else 0,
                "percentage": round((top_2[0][1] / total) * 100, 2) if top_2 else 0.0,
            },
            "first_3_tokens": {
                "prefix": top_3[0][0] if top_3 else "N/A",
                "count": top_3[0][1] if top_3 else 0,
                "percentage": round((top_3[0][1] / total) * 100, 2) if top_3 else 0.0,
            },
            "first_5_tokens": {
                "prefix": top_5[0][0] if top_5 else "N/A",
                "count": top_5[0][1] if top_5 else 0,
                "percentage": round((top_5[0][1] / total) * 100, 2) if top_5 else 0.0,
            },
        }

    def generate_audit_report(self) -> Tuple[bool, str]:
        """Generates comprehensive markdown report (sft_dataset_report.md) and returns (is_pass, report_text)."""
        total_records = len(self.records)
        outputs = [r.get("output", "").strip() for r in self.records]
        unique_outputs = len(set(outputs))
        unique_output_ratio = (unique_outputs / float(total_records)) * 100 if total_records > 0 else 0.0

        # Guardrails check
        valid_guardrails, violations = self.validate_task_guardrails()

        # Semantic Deduplication check
        near_dups, _ = self.run_semantic_deduplication()

        # Lexical Diversity
        lex_metrics = self.compute_lexical_diversity()

        # Prefixes check
        prefix_metrics = self.check_prefix_frequencies()

        # Task breakdown & Meetei Mayek script ratio
        task_counts = Counter(r.get("task_type", "unknown") for r in self.records)
        mm_count = sum(1 for r in self.records if SFTQualityScorer.detect_script(r.get("output", "")) == "MeeteiMayek")
        mm_ratio = (mm_count / float(total_records)) * 100 if total_records > 0 else 0.0

        # PASS / FAIL criteria
        max_prefix_pct = max(
            prefix_metrics.get("first_2_tokens", {}).get("percentage", 0.0),
            prefix_metrics.get("first_3_tokens", {}).get("percentage", 0.0),
            prefix_metrics.get("first_5_tokens", {}).get("percentage", 0.0),
        )

        is_pass = (
            unique_output_ratio >= 95.0
            and len(violations) == 0
            and max_prefix_pct <= 3.0
            and mm_ratio >= 40.0
        )
        status_str = "PASS [OK]" if is_pass else "FAIL [FAILED]"

        # Construct Report Content
        report_lines = [
            "=========================================================",
            "             MANIPURIGPT SFT DATASET AUDIT REPORT        ",
            "=========================================================",
            f"Dataset Version: v1.1-grounded-30k",
            f"Audit Status: {status_str}",
            f"Total Records: {total_records}",
            f"Meetei Mayek Script Ratio: {mm_ratio:.2f}%",
            "",
            "--- Task Distribution Breakdown ---",
        ]
        for task, cnt in task_counts.items():
            report_lines.append(f"  - {task}: {cnt} ({(cnt/total_records)*100:.1f}%)")

        report_lines.extend([
            "",
            "--- Diversity & Uniqueness Metrics ---",
            f"  - Unique Target Outputs: {unique_outputs} ({unique_output_ratio:.2f}%)",
            f"  - Exact Duplicate Outputs: {total_records - unique_outputs}",
            f"  - Near-Duplicate Outputs (MinHash >0.85): {near_dups}",
            f"  - Type-Token Ratio (TTR): {lex_metrics.get('ttr', 0.0)}",
            f"  - Distinct-1 Score: {lex_metrics.get('distinct_1', 0.0)}",
            f"  - Distinct-2 Score: {lex_metrics.get('distinct_2', 0.0)}",
            f"  - Shannon Entropy: {lex_metrics.get('entropy', 0.0)}",
            "",
            "--- Multi-Token Prefix Frequency ---",
            f"  - First 2 Tokens Top Prefix: '{prefix_metrics.get('first_2_tokens', {}).get('prefix')}' ({prefix_metrics.get('first_2_tokens', {}).get('percentage')}%)",
            f"  - First 3 Tokens Top Prefix: '{prefix_metrics.get('first_3_tokens', {}).get('prefix')}' ({prefix_metrics.get('first_3_tokens', {}).get('percentage')}%)",
            f"  - First 5 Tokens Top Prefix: '{prefix_metrics.get('first_5_tokens', {}).get('prefix')}' ({prefix_metrics.get('first_5_tokens', {}).get('percentage')}%)",
            "",
            "--- Task Guardrail Validation ---",
            f"  - Valid Guardrail Records: {valid_guardrails} / {total_records}",
            f"  - Guardrail Violations: {len(violations)}",
        ])

        if violations:
            report_lines.append("  Sample Violations:")
            for v in violations[:10]:
                report_lines.append(f"    * {v}")

        report_lines.append("=========================================================")
        report_text = "\n".join(report_lines)

        # Write to sft_dataset_report.md
        report_file = self.report_dir / "sft_dataset_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        return is_pass, report_text
