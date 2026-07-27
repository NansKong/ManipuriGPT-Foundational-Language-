"""
ManipuriGPT Phase 6 Foundation Pretraining Preparation CLI (`app/scripts/run_phase6_prep.py`).

Packs the frozen corpus into configurable context window lengths (512, 1024, 2048, 4096)
and constructs the held-out pre-training benchmark suite (Perplexity, Translation,
Script Conversion, and OCR Correction).

Usage:
  python -m app.scripts.run_phase6_prep --corpus-dir artifacts/datasets/ManipuriGPT-Corpus-v1.0
"""

import os
import sys
import argparse
from typing import List, Optional

from app.utils.logger import logger
from app.training.packer import SequencePacker
from app.evaluation.benchmark_suite import PretrainingBenchmarkSuite


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ManipuriGPT Phase 6 Foundation Pretraining Preparation CLI"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1.0",
        help="Directory containing frozen ManipuriGPT-Corpus-v1.0 snapshot",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer model (auto-discovers inside corpus-dir/tokenizer if unset)",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="List of sequence context window lengths to pack",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/phase6_pretraining",
        help="Output directory for packed datasets and benchmark suite",
    )
    return parser.parse_args(args_list)


def load_split_records(dataset_dir: str, split: str = "train") -> List[dict]:
    """Load records from a dataset split."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow is required to load dataset records.")
        return []

    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        return []

    records = []
    for f in sorted(os.listdir(split_dir)):
        if f.endswith(".parquet"):
            fp = os.path.join(split_dir, f)
            try:
                table = pq.read_table(fp)
                df = table.to_pandas()
                records.extend(df.to_dict(orient="records"))
            except Exception as e:
                logger.warning(f"Error loading {fp}: {e}")
    return records


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 6 — FOUNDATION PRETRAINING PREPARATION ENGINE")
    logger.info("=" * 80)
    logger.info(f"Frozen Corpus Dir : {args.corpus_dir}")
    logger.info(f"Output Directory  : {args.output_dir}")
    logger.info(f"Sequence Lengths  : {args.seq_lens}")

    if not os.path.exists(args.corpus_dir):
        logger.error(f"Frozen corpus directory not found at '{args.corpus_dir}'. Aborting.")
        return 1

    tok_path = args.tokenizer_path
    if not tok_path:
        tok_path = os.path.join(args.corpus_dir, "tokenizer", "tokenizer.model")
    if not os.path.exists(tok_path):
        tok_path = "artifacts/tokenizer_v1.0/tokenizer.model"

    if not os.path.exists(tok_path):
        logger.error(f"No tokenizer model found at '{tok_path}'. Aborting.")
        return 1

    logger.info(f"Using tokenizer: {tok_path}")

    # Load Train & Test records
    train_records = load_split_records(args.corpus_dir, split="train")
    test_records = load_split_records(args.corpus_dir, split="test")

    logger.info(f"Loaded {len(train_records):,} train records and {len(test_records):,} test records")

    if not train_records:
        logger.error("0 train records loaded! Aborting.")
        return 1

    # 1. Configurable Sequence Packing
    train_texts = [r.get("text", "") for r in train_records if r.get("text")]
    packer = SequencePacker(tokenizer_path=tok_path)
    pack_summary = packer.pack_and_save_shards(
        texts=train_texts,
        output_dir=os.path.join(args.output_dir, "packed_datasets"),
        seq_lens=args.seq_lens
    )

    # 2. Pre-Training Benchmark Suite Setup
    benchmark_dir = os.path.join(args.output_dir, "benchmarks")
    suite = PretrainingBenchmarkSuite()
    bench_summary = suite.build_benchmarks(test_records=test_records, output_dir=benchmark_dir, train_records=train_records)

    logger.info("\n" + "=" * 80)
    logger.info(" PHASE 6 PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info("Packed Dataset Summary:")
    for s_len, block_count in pack_summary.items():
        logger.info(f"  max_seq_len={s_len:4d} : {block_count:,} packed blocks")

    logger.info("\nHeld-Out Pre-Training Benchmark Suite Summary:")
    for b_name, count in bench_summary.items():
        logger.info(f"  {b_name:30s} : {count:,} samples")

    logger.info(f"\nPhase 6 artifacts ready in: {args.output_dir}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
