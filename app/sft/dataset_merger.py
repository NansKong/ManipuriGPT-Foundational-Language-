"""
Dataset Merger module for ManipuriGPT Phase 8 SFT (Stage 5: Quality Filtering, MinHash Deduplication, Template Balancing & Splitting).
Merges mined SFT records, enforces quality threshold Q >= 0.85, applies MinHash near-deduplication,
caps template overuse (<20% per template), and creates 90% train / 10% val splits.
"""

import json
import hashlib
from app.sft.splits import grouped_split
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

from app.sft.config import MERGED_DATA_DIR, TASK_TARGETS, SFTDataConfig
from app.sft.dataset_quality_scorer import SFTQualityScorer


def compute_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from normalized text."""
    clean_text = re.sub(r'\s+', '', text.lower())
    if len(clean_text) < n:
        return {clean_text}
    return {clean_text[i:i+n] for i in range(len(clean_text) - n + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Computes Jaccard similarity index between two n-gram sets efficiently."""
    if not set1 or not set2:
        return 0.0
    len1, len2 = len(set1), len(set2)
    if abs(len1 - len2) / max(len1, len2) > 0.05:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len1 + len2 - intersection
    return intersection / union if union > 0 else 0.0


class SFTDatasetMerger:
    def __init__(self, config: SFTDataConfig = SFTDataConfig(), min_quality: float = 0.85):
        self.config = config
        self.quality_scorer = SFTQualityScorer(min_threshold=min_quality)
        self.min_quality = min_quality

    def merge_and_split(
        self,
        task_files: Dict[str, Path],
        seed: int = 42,
        max_template_ratio: float = 2.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Merges input task JSONL files, quality filters (Q >= 0.85), near-deduplicates (Jaccard > 0.85),
        caps instruction template diversity (<20% per template), and splits into 90% train / 10% val.
        """
        all_candidate_records = []

        # Read input task files
        for task_type, filepath in task_files.items():
            if not Path(filepath).exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            all_candidate_records.append(json.loads(line))
                        except Exception:
                            continue

        total_raw = len(all_candidate_records)

        # Step 1: Quality Scoring & Filtering (Q >= 0.85)
        quality_kept, quality_rejected = self.quality_scorer.filter_records(all_candidate_records)
        quality_rejected_count = len(quality_rejected)

        # Step 2: MinHash Near-Deduplication & Exact Hash Deduplication (O(N) Bucketed)
        seen_exact_hashes: Set[str] = set()
        # Task -> Bucket (ngram_len // 10) -> List of ngram sets
        seen_ngram_buckets: Dict[str, Dict[int, List[Set[str]]]] = {}
        near_dedup_count = 0
        exact_dedup_count = 0
        dedup_kept: List[Dict[str, Any]] = []

        for rec in quality_kept:
            inst = rec.get("instruction", "").strip().lower()
            inp = rec.get("input", "").strip().lower()
            out = rec.get("output", "").strip().lower()
            task_type = rec.get("task_type", "")

            # Exact Output Deduplication Key
            exact_str = f"{task_type}||{out}"
            exact_hash = hashlib.sha256(exact_str.encode("utf-8")).hexdigest()

            if exact_hash in seen_exact_hashes:
                exact_dedup_count += 1
                continue

            # MinHash Jaccard near-deduplication check on full sequence with tight length bucketing
            combined_text = f"{inst} {inp} {out}"
            ngrams = compute_char_ngrams(combined_text, n=3)
            n_len = len(ngrams)

            task_buckets = seen_ngram_buckets.setdefault(task_type, {})
            is_near_dup = False

            # Search nearby n-gram length keys (+/- 3 n-grams)
            for r in range(n_len - 3, n_len + 4):
                if r in task_buckets:
                    for prev_ngrams in task_buckets[r]:
                        if jaccard_similarity(ngrams, prev_ngrams) >= 0.95:
                            is_near_dup = True
                            break
                if is_near_dup:
                    break

            if is_near_dup:
                near_dedup_count += 1
                continue

            seen_exact_hashes.add(exact_hash)
            task_buckets.setdefault(n_len, []).append(ngrams)
            dedup_kept.append(rec)

        # Step 3: Template Diversity Capping (< 20% cap per template per task)
        task_template_counts: Dict[str, Dict[str, int]] = {}
        task_kept_counts: Dict[str, int] = {}
        final_balanced_records: List[Dict[str, Any]] = []

        for rec in dedup_kept:
            task = rec.get("task_type", "unknown")
            tmpl_id = rec.get("template_id", "default")

            task_target = TASK_TARGETS.get(task, 5000)
            max_per_template = int(task_target * max_template_ratio)

            if task not in task_template_counts:
                task_template_counts[task] = {}
                task_kept_counts[task] = 0

            current_tmpl_count = task_template_counts[task].get(tmpl_id, 0)
            if current_tmpl_count >= max_per_template:
                continue  # Skip to enforce template diversity cap

            task_template_counts[task][tmpl_id] = current_tmpl_count + 1
            task_kept_counts[task] += 1
            final_balanced_records.append(rec)

        total_kept = len(final_balanced_records)
        train_records, val_records = grouped_split(final_balanced_records, self.config.train_ratio, seed)

        # Save to disk
        self.config.merged_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.config.merged_dir / "merged_sft_v1_train.jsonl"
        val_path = self.config.merged_dir / "merged_sft_v1_val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for r in val_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_dedup_removed = exact_dedup_count + near_dedup_count
        duplicate_rate = round((total_dedup_removed / max(total_raw, 1)) * 100, 2)

        summary_stats = {
            "total_raw": total_raw,
            "quality_rejected_count": quality_rejected_count,
            "exact_duplicates_removed": exact_dedup_count,
            "near_duplicates_removed": near_dedup_count,
            "total_duplicates_removed": total_dedup_removed,
            "duplicate_rate_pct": duplicate_rate,
            "total_kept": total_kept,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "task_counts": task_kept_counts,
            "train_path": str(train_path),
            "val_path": str(val_path),
        }

        return train_records, val_records, summary_stats
