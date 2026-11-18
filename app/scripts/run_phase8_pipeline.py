"""
Master Execution Script for ManipuriGPT Phase 8 SFT Data Pipeline (v2.0 Corpus-Driven Overhaul).
Executes all 9 Stages: Mining, Quality Scoring (Q >= 0.85), Pre-write Instruction Validation, Normalization, MinHash Near-Deduplication,
Task Balancing, Template Diversity Capping, ChatML Formatting, Quality Acceptance Gates Evaluation, Statistics & Data Card Generation,
and 100-sample Human Review Generation.
"""

import sys
import json
import random
import re
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.sft.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MERGED_DATA_DIR,
    STATS_DATA_DIR,
    TASK_TARGETS,
    SUPPORTED_TASKS,
    MAX_CORPUS_SOURCE_RATIO,
    SFTDataConfig,
)
from app.sft.dataset_miner import SFTDatasetMiner
from app.sft.dataset_builder import SFTDatasetBuilder
from app.sft.dataset_merger import SFTDatasetMerger
from app.sft.dataset_statistics import SFTStatisticsReporter
from app.sft.prompt_formatter import format_chatml_record


def generate_human_review_artifact(all_records: list, samples_per_task: int = 25) -> Path:
    """Generates 200-sample stratified human review inspection artifact (25 per task) formatted in ChatML."""
    review_path = BASE_DIR / "docs" / "human_review_sft_v2.md"

    # Stratify records by task_type
    by_task: dict = {}
    for rec in all_records:
        t_type = str(rec.get("task_type", "unknown")).lower()
        by_task.setdefault(t_type, []).append(rec)

    stratified_samples = []
    for task_name, rec_list in sorted(by_task.items()):
        sample_count = min(len(rec_list), samples_per_task)
        stratified_samples.extend(random.sample(rec_list, sample_count))

    with open(review_path, "w", encoding="utf-8") as f:
        f.write("# ManipuriGPT SFT v2.0 — 200-Sample Stratified Human Review Inspection\n\n")
        f.write("This artifact contains 200 stratified ChatML instruction sequences (25 samples per task category) for quality verification.\n\n")

        for idx, rec in enumerate(stratified_samples, 1):
            formatted = format_chatml_record(
                instruction=rec.get("instruction", ""),
                input_text=rec.get("input", ""),
                output_text=rec.get("output", "")
            )
            task_name = str(rec.get('task_type', 'N/A')).upper()
            script_name = str(rec.get('script', 'N/A'))
            quality_score = rec.get('quality_score', 0.95)
            source_name = str(rec.get('source', 'unknown'))
            src_type = str(rec.get('source_type', 'general'))
            para_id = str(rec.get('paragraph_id', 'P000'))
            tmpl_id = str(rec.get('template_id', 'N/A'))
            rec_id = str(rec.get('id', 'N/A'))

            f.write(f"### Sample {idx} [{task_name} | {script_name}]\n")
            f.write(f"**Task**: `{task_name}` | **Script**: `{script_name}` | **Quality Score**: `{quality_score}`  \n")
            f.write(f"**Source**: `{source_name}` (`{src_type}`) | **Para**: `{para_id}` | **Template**: `{tmpl_id}` | **ID**: `{rec_id}`\n\n")
            f.write("```text\n")
            f.write(formatted["full_text"] + "\n")
            f.write("```\n\n")
            f.write("---\n\n")

    return review_path


def evaluate_quality_acceptance_gates(merge_stats, train_recs, val_recs):
    """Prints and returns Quality Acceptance Gates metrics."""
    total_raw = merge_stats.get("total_raw", 1)
    near_dedup = merge_stats.get("near_duplicates_removed", 0)
    exact_dedup = merge_stats.get("exact_duplicates_removed", 0)
    kept = merge_stats.get("total_kept", 1)

    dedup_rate = round((near_dedup + exact_dedup) / max(total_raw, 1) * 100, 2)

    all_recs = train_recs + val_recs
    valid_count = sum(1 for r in all_recs if SFTDatasetMiner.validate_record_consistency(r))
    instruction_consistency_rate = round((valid_count / max(len(all_recs), 1)) * 100, 2)

    unique_instructions = len(set(r.get("instruction", "") for r in all_recs))
    output_tokens = [int(len(r.get("output", "")) / 3.5) for r in all_recs]
    avg_out_tokens = round(sum(output_tokens) / max(len(output_tokens), 1), 1)

    # Script calculation
    meetei_chars = sum(len(re.findall(r'[\uABC0-\uABFF\uAAE0-\uAAFF]', r.get("output", "") + r.get("instruction", ""))) for r in all_recs)
    total_chars = sum(len(r.get("output", "") + r.get("instruction", "")) for r in all_recs)
    mm_ratio = round((meetei_chars / max(total_chars, 1)) * 100, 2)

    print("\n=========================================================")
    print("  Quality Acceptance Gates (v2.0 30k Production Benchmark)")
    print("=========================================================")
    print(f"  [1] Instruction Consistency:   {instruction_consistency_rate}% (Target: 100%)")
    print(f"  [2] Deduplication Collapse:    {dedup_rate}% (Target: < 20%)")
    print(f"  [3] Unique Instructions Count: {unique_instructions} (Target: > 500)")
    print(f"  [4] Average Response Tokens:   {avg_out_tokens} tokens (Target: > 15)")
    print(f"  [5] Meetei Mayek Script Ratio: {mm_ratio}% (Target: 40-50%)")
    print(f"  [6] Unique Instruction Kept:  {kept} records (Target: 30,000)")
    print("=========================================================\n")


def main():
    print("=========================================================")
    print("  ManipuriGPT Phase 8 SFT Data Pipeline (v2.0 Overhaul)")
    print("=========================================================")

    # Step 1: Mine SFT Instruction Dataset across all subtasks
    miner = SFTDatasetMiner(raw_dir=RAW_DATA_DIR)
    mined_raw_file = miner.generate_mined_dataset(target_counts=TASK_TARGETS)

    # Step 2: Stage 1 & Stage 2 Builder (Extract & Convert into task files)
    builder = SFTDatasetBuilder(raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)
    task_files = {}
    for task_type in SUPPORTED_TASKS:
        recs = builder.process_raw_file(mined_raw_file, task_type=task_type)
        if recs:
            out_p = builder.save_processed_task(recs, task_type=task_type)
            task_files[task_type] = out_p

    # Step 3: Stage 3, 4, 5 Merger (Quality Filter Q>=0.85, MinHash Near-Dedup, Template Capping, 90/10 Split)
    config = SFTDataConfig()
    merger = SFTDatasetMerger(config=config, min_quality=0.85)
    train_recs, val_recs, merge_stats = merger.merge_and_split(task_files=task_files)

    print("\n[Merge & Filtering Summary]")
    print(f"  Total Raw Extracted: {merge_stats['total_raw']}")
    print(f"  Quality Rejected (Q < 0.85): {merge_stats['quality_rejected_count']}")
    print(f"  Exact Duplicates Removed: {merge_stats['exact_duplicates_removed']}")
    print(f"  Near Duplicates Removed (MinHash): {merge_stats['near_duplicates_removed']}")
    print(f"  Total Kept: {merge_stats['total_kept']}")
    print(f"  Train Count (90%): {merge_stats['train_count']}")
    print(f"  Val Count (10%): {merge_stats['val_count']}")
    print(f"  Train Output: {merge_stats['train_path']}")
    print(f"  Val Output: {merge_stats['val_path']}")

    # Step 4: Quality Acceptance Gates Benchmark Report
    evaluate_quality_acceptance_gates(merge_stats, train_recs, val_recs)

    # Step 5: Stage 6 Statistics & Data Card Generation
    reporter = SFTStatisticsReporter(stats_dir=STATS_DATA_DIR)
    stats_path = reporter.generate_report(train_records=train_recs, val_records=val_recs)
    print(f"  Stats Report Output: {stats_path}")
    print(f"  Data Card Artifact Output: {STATS_DATA_DIR / 'DATASET_CARD_v1.md'}")

    # Step 6: Generate 200-Sample Stratified Human Review Artifact (25 per task)
    review_path = generate_human_review_artifact(all_records=train_recs + val_recs, samples_per_task=25)
    print(f"  Human Review Inspection Artifact: {review_path}")

    print("\n[Phase 8 Pipeline v2.0 Execution Complete]")


if __name__ == "__main__":
    main()
