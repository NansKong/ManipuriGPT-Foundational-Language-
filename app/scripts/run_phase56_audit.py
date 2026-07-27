"""
ManipuriGPT Phase 5.6 Corpus Quality Audit CLI (`app/scripts/run_phase56_audit.py`).

Performs a comprehensive quality audit of the dataset snapshots and produces
detailed metrics for source novelty %, domain balance, document length outliers,
script distribution, token entropy, and vocabulary growth.

Usage:
  python -m app.scripts.run_phase56_audit --v3-dir artifacts/datasets/ManipuriGPT-Corpus-v3
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional, List

from app.utils.logger import logger
from app.evaluation.corpus_auditor import CorpusAuditor


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ManipuriGPT Phase 5.6 Corpus Quality Audit CLI"
    )
    parser.add_argument(
        "--v3-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v3",
        help="Directory containing Phase 5.5 / v3 corpus shards",
    )
    parser.add_argument(
        "--v1-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1",
        help="Directory containing v1 corpus shards",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="artifacts/tokenizer_v1/tokenizer.model",
        help="Path to SentencePiece tokenizer model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Custom output JSON path for the audit report",
    )
    return parser.parse_args(args_list)


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.6 — CORPUS QUALITY AUDIT ENGINE")
    logger.info("=" * 80)
    logger.info(f"Target Dataset Dir : {args.v3_dir}")
    logger.info(f"Tokenizer Path     : {args.tokenizer_path}")

    target_dir = args.v3_dir
    if not os.path.exists(target_dir):
        logger.warning(f"Target dataset directory '{target_dir}' does not exist. Falling back to '{args.v1_dir}'")
        target_dir = args.v1_dir

    if not os.path.exists(target_dir):
        logger.error(f"No corpus directory found at '{target_dir}' or '{args.v1_dir}'. Aborting.")
        return 1

    # Load prior source stats from metadata/corpus_report.json if available
    prior_stats = None
    report_file = os.path.join(target_dir, "metadata", "corpus_report.json")
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
                prior_stats = rep_data.get("per_source_stats")
        except Exception as e:
            logger.warning(f"Could not read per-source stats from {report_file}: {e}")

    auditor = CorpusAuditor(tokenizer_path=args.tokenizer_path)
    audit_results = auditor.audit_dataset_directory(
        dataset_dir=target_dir,
        prior_source_stats=prior_stats
    )

    audit_results["audit_metadata"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "5.6",
        "audited_directory": target_dir,
        "tokenizer_used": args.tokenizer_path if os.path.exists(args.tokenizer_path) else "none"
    }

    # Display audit highlights
    summary = audit_results.get("summary", {})
    logger.info("\n" + "=" * 60)
    logger.info(" CORPUS AUDIT HIGHLIGHTS")
    logger.info("=" * 60)
    logger.info(f"Total Sequences    : {summary.get('total_sequences', 0):,}")
    logger.info(f"Total Characters   : {summary.get('total_characters', 0):,}")
    logger.info(f"Total Words        : {summary.get('total_words', 0):,}")
    logger.info(f"Total Tokens       : {summary.get('total_tokens', 0):,}")
    logger.info(f"Unknown Rate (`<unk>%`): {summary.get('unknown_token_rate_pct', 0.0):.4f}%")
    logger.info(f"Avg Doc Length (chars): {summary.get('avg_doc_length_chars', 0.0)}")
    logger.info(f"Avg Doc Length (words): {summary.get('avg_doc_length_words', 0.0)}")
    logger.info(f"Observed Unique Vocab: {summary.get('unique_vocab_size_observed', 0):,}")

    logger.info("\n--- Script Distribution ---")
    for scr, data in audit_results.get("script_distribution", {}).items():
        logger.info(f"  {scr:12s}: {data['count']:,} seqs ({data['pct']}%)")

    logger.info("\n--- Source Novelty Breakdown ---")
    for row in audit_results.get("source_novelty", []):
        src_name = row.get("source", "unknown")
        raw = row.get("raw_scanned", "-")
        kept = row.get("unique_kept", 0)
        novelty = row.get("novelty_pct", "-")
        logger.info(f"  {src_name:35s} | Kept: {kept:,} | Raw: {raw} | Novelty: {novelty}%")

    outliers = audit_results.get("outliers_and_long_docs", {})
    logger.info("\n--- Outlier Summary ---")
    logger.info(f"  OCR Noise Outliers (>35% Latin) : {outliers.get('ocr_noise_outliers_count', 0)}")
    logger.info(f"  Repeated Token Outliers (>60%)  : {outliers.get('repeated_token_outliers_count', 0)}")
    if outliers.get("longest_document"):
        ld = outliers["longest_document"]
        logger.info(f"  Longest Doc  : {ld['char_length']:,} chars, {ld['token_length']:,} tokens ({ld['source']})")
    if outliers.get("shortest_document"):
        sd = outliers["shortest_document"]
        logger.info(f"  Shortest Doc : {sd['char_length']:,} chars, {sd['token_length']:,} tokens ({sd['source']})")

    # Output file
    output_path = args.output_file
    if not output_path:
        meta_dir = os.path.join(target_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        output_path = os.path.join(meta_dir, "corpus_audit_report.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved complete audit report to: {output_path}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
