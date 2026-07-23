"""
ManipuriGPT Phase 5.2 Top-Level Orchestration Suite (`app/scripts/run_phase52.py`).
Executes the complete production lifecycle:
    Balanced Sampling -> Benchmarking -> Preprocessing & Sharding -> Tokenizer Training -> Evaluation -> Reporting
Supports `--resume`, `--skip-existing`, and `--dry-run`.
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.configs.loader import load_all_configs, compute_config_hash
from app.scripts import benchmark_scale, preprocess_shards, train_tokenizers
from app.tokenization.benchmark import TokenizerBenchmarker
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Phase 5.2 Top-Level Orchestration Suite")
    parser.add_argument(
        "--config",
        type=str,
        default="phase5.yaml",
        help="Primary Phase 5 YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline and sharding from previous checkpoints"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages where target output manifests/reports already exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan, configurations, and paths without executing data processing"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum raw examples for sharding stage"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of samples for tokenizer training stage"
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        help="Allow mock fallback across all pipeline stages if live HF connection fails"
    )
    return parser.parse_args(args)


def main(args_list: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(args_list)
    start_total_t = time.time()
    
    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories()
    except Exception as e:
        logger.warning(f"RunPhase52: Could not initialize cache directories: {e}")

    configs = load_all_configs()
    phase5_cfg = configs.get("phase5", {})
    config_hash = compute_config_hash()

    limit = args.limit if args.limit is not None else phase5_cfg.get("sharding", {}).get("max_examples", 2000)
    train_samples = args.train_samples if args.train_samples is not None else 5000
    mock_fallback = args.mock_fallback or phase5_cfg.get("execution", {}).get("mock_fallback", True)

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.2 TOP-LEVEL ORCHESTRATION SUITE")
    logger.info("=" * 80)
    logger.info(f"Config Hash    : {config_hash}")
    logger.info(f"Resume Mode    : {args.resume}")
    logger.info(f"Skip Existing  : {args.skip_existing}")
    logger.info(f"Dry Run        : {args.dry_run}")
    logger.info(f"Sharding Limit : {limit} examples")
    logger.info(f"Train Samples  : {train_samples} sequences")

    if args.dry_run:
        logger.info("\n[DRY-RUN ACTIVE] Execution Plan Summary:")
        logger.info("  1. Stage 1: Load YAML Configs & Compute Hash (COMPLETE)")
        logger.info("  2. Stage 2: Scale & Throughput Benchmark -> cache/benchmarks/throughput.json")
        logger.info("  3. Stage 3: Preprocessing & Sharding -> cache/shards/manifest.json")
        logger.info("  4. Stage 4: 6-Tokenizer Training -> cache/tokenizers/training_summary.json")
        logger.info("  5. Stage 5: Tokenizer Evaluation -> cache/benchmarks/tokenizer_examples.md")
        logger.info("\n[DRY-RUN] No files modified. Exiting successfully.")
        return {"status": "dry_run_complete", "config_hash": config_hash}

    summary: Dict[str, Any] = {
        "pipeline_version": "5.2",
        "config_hash": config_hash,
        "created": datetime.utcnow().isoformat() + "Z",
        "stages": {}
    }

    # Stage 2: Benchmarking
    benchmark_report_path = "cache/benchmarks/throughput.json"
    if args.skip_existing and os.path.exists(benchmark_report_path):
        logger.info(f"\n[Stage 1/4] Skipping Throughput Benchmarking (Report exists at {benchmark_report_path})")
        summary["stages"]["benchmarking"] = "skipped_existing"
    else:
        logger.info("\n[Stage 1/4] Running Scale & Throughput Benchmarking...")
        bench_args = ["--limit", str(min(limit, 500))]
        if mock_fallback:
            bench_args.append("--mock-fallback")
        bench_res = benchmark_scale.main(bench_args)
        summary["stages"]["benchmarking"] = "completed"

    # Stage 3: Sharding & Preprocessing
    manifest_path = "cache/shards/manifest.json"
    if args.skip_existing and os.path.exists(manifest_path) and not args.resume:
        logger.info(f"\n[Stage 2/4] Skipping Large-Scale Sharding (Manifest exists at {manifest_path})")
        summary["stages"]["sharding"] = "skipped_existing"
    else:
        logger.info("\n[Stage 2/4] Executing Large-Scale Preprocessing & Sharding...")
        shard_args = ["--limit", str(limit)]
        if args.resume:
            shard_args.append("--resume")
        if mock_fallback:
            shard_args.append("--mock-fallback")
        shard_res = preprocess_shards.main(shard_args)
        summary["stages"]["sharding"] = "completed"

    # Stage 4: Tokenizer Training
    training_summary_path = "cache/tokenizers/training_summary.json"
    if args.skip_existing and os.path.exists(training_summary_path):
        logger.info(f"\n[Stage 3/4] Skipping Tokenizer Training (Summary exists at {training_summary_path})")
        summary["stages"]["tokenizer_training"] = "skipped_existing"
    else:
        logger.info("\n[Stage 3/4] Executing 6-Candidate Tokenizer Training Suite...")
        tok_args = ["--train-samples", str(train_samples)]
        if mock_fallback:
            tok_args.append("--mock-fallback")
        tok_res = train_tokenizers.main(tok_args)
        summary["stages"]["tokenizer_training"] = "completed"

    # Stage 5: Evaluation & Report Generation
    eval_report_path = "cache/benchmarks/tokenizer_examples.md"
    logger.info("\n[Stage 4/4] Generating Qualitative & Quantitative Human Evaluation Report...")
    try:
        if os.path.exists(training_summary_path):
            with open(training_summary_path, "r", encoding="utf-8") as f:
                models_dict = json.load(f).get("model_metadata", {})
            report_path = TokenizerBenchmarker.generate_human_evaluation_report(models_dict, output_path=eval_report_path)
            summary["stages"]["evaluation"] = f"completed -> {report_path}"
        else:
            summary["stages"]["evaluation"] = "skipped_missing_training_summary"
    except Exception as e:
        logger.error(f"Error generating human evaluation report: {e}")
        summary["stages"]["evaluation"] = f"error -> {e}"

    duration_sec = round(time.time() - start_total_t, 2)
    summary["duration_sec"] = duration_sec

    summary_file = "cache/phase52_orchestration_summary.json"
    os.makedirs("cache", exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(" PHASE 5.2 ORCHESTRATION COMPLETED SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Duration : {duration_sec}s")
    logger.info(f"Stage Summary  : {summary['stages']}")
    logger.info(f"Final Manifest : {summary_file}")
    logger.info("=" * 80)

    return summary


if __name__ == "__main__":
    main()
