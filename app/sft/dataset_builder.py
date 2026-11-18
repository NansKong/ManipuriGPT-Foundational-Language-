"""
Dataset Builder module for ManipuriGPT Phase 8 SFT (app/sft/dataset_builder.py).
Orchestrates Task Grounding v1.1 Dataset Mining, Merging, Validation, and Reporting.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.sft.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MERGED_DATA_DIR
from app.sft.dataset_miner import SFTDatasetMiner
from app.sft.dataset_merger import SFTDatasetMerger
from app.sft.dataset_validator import SFTDatasetValidator


def build_grounded_sft_dataset() -> Dict[str, Any]:
    """
    Master entry point for building Task Grounding v1.1 SFT Dataset.
    Generates 30k grounded records, merges & splits into train/val JSONL,
    runs automated validation, and generates sft_dataset_report.md.
    """
    print("=========================================================")
    print("  ManipuriGPT Phase 8 Task Grounding SFT Dataset Builder")
    print("  Dataset Version: v1.1-grounded-30k")
    print("=========================================================\n")

    # Step 1: Mine Grounded Records
    miner = SFTDatasetMiner()
    mined_records = miner.generate_mined_dataset()

    # Save mined records to processed dir
    processed_file = PROCESSED_DATA_DIR / "mined_grounded_sft_records.jsonl"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(processed_file, "w", encoding="utf-8") as f:
        for rec in mined_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Step 2: Merge, Deduplicate, Quality Score, and Split (90/10)
    merger = SFTDatasetMerger()
    task_files = {"grounded": processed_file}
    train_recs, val_recs, merge_stats = merger.merge_and_split(task_files)

    print(f"\n[Merge Complete] Total Kept: {merge_stats['total_kept']} | Train: {merge_stats['train_count']} | Val: {merge_stats['val_count']}")

    # Step 3: Run Automated Dataset Validator & Generate Audit Report
    validator = SFTDatasetValidator(train_recs + val_recs)
    is_pass, report_text = validator.generate_audit_report()

    print("\n" + report_text + "\n")

    if is_pass:
        print("[SUCCESS] SFT Dataset v1.1 successfully generated and passed all audit checks!")
    else:
        print("[WARNING] SFT Dataset built but failed one or more validation criteria. Review audit report.")

    return merge_stats


if __name__ == "__main__":
    build_grounded_sft_dataset()
