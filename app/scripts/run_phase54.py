"""
ManipuriGPT Phase 5.4 Dataset Builder & Corpus Validation CLI (`app/scripts/run_phase54.py`).
Unified entry point bridging tokenization and model pretraining:
1. Freezes Tokenizer v1 (`tokenizer.model`, `tokenizer.json`, `vocab.json`, `coverage_report.json`)
2. Builds clean, deduplicated, quality-scored dataset across deterministic splits (98/1/1)
3. Exports chunked Parquet training shards (`train/`, `validation/`, `test/`)
4. Generates `corpus_report.json` and Hugging Face `README.md` Dataset Card
"""

import os
import sys
import argparse
from typing import List, Optional
from app.dataset_builder.tokenizer_freezer import TokenizerFreezer
from app.dataset_builder.dataset_assembler import DatasetAssembler
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Phase 5.4 Dataset Builder & Corpus Validation CLI")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "dayananda_meitei_mayek_sample",
            "dayananda_english_to_meitei",
            "joyson_bible",
            "joyson_pib_pmi"
        ],
        help="Corpus sources for balanced dataset construction"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum raw examples to stream and process for this dataset run"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Explicit path to source tokenizer.model (if not provided, auto-discovers Phase 5.3 winner)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1",
        help="Directory where training dataset shards and cards will be saved"
    )
    parser.add_argument(
        "--freeze-dir",
        type=str,
        default="artifacts/tokenizer_v1",
        help="Directory where frozen Tokenizer v1 artifacts will be saved"
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.98, 0.01, 0.01],
        help="Split ratios for Train, Validation, and Test (default: 0.98 0.01 0.01)"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of sequences per exported Parquet shard file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for splitting and sampling"
    )
    parser.add_argument(
        "--skip-freeze",
        action="store_true",
        help="Skip freezing Tokenizer v1 if already present in --freeze-dir"
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        default=True,
        help="Allow mock fallback if live Hugging Face dataset streaming encounters network issues"
    )
    return parser.parse_args(args)


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    # Ensure centralized storage redirection (protect C: drive)
    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories()
    except Exception as e:
        logger.warning(f"RunPhase54: Could not initialize cache directories: {e}")

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.4 ORCHESTRATION ENGINE")
    logger.info("=" * 80)

    # 1. Freeze Tokenizer v1
    frozen_model_path: Optional[str] = None
    if not args.skip_freeze:
        logger.info("\n[Step 1/2] Freezing Tokenizer v1...")
        freezer = TokenizerFreezer(default_output_dir=args.freeze_dir)
        try:
            frozen_files = freezer.freeze(source_model_path=args.tokenizer_path, output_dir=args.freeze_dir)
            frozen_model_path = frozen_files.get("tokenizer.model")

            # Also copy frozen tokenizer directly inside the Hugging Face dataset output folder
            ds_tok_dir = os.path.join(args.output_dir, "tokenizer")
            freezer.freeze(source_model_path=frozen_model_path, output_dir=ds_tok_dir)
        except Exception as e:
            logger.error(f"RunPhase54: Freezing Tokenizer v1 failed: {e}")
            if not args.tokenizer_path or not os.path.exists(args.tokenizer_path):
                logger.error("Please provide a valid --tokenizer-path or ensure Phase 5.3 candidates are available.")
                return 1
            frozen_model_path = args.tokenizer_path
    else:
        logger.info("\n[Step 1/2] Skipping Tokenizer v1 freeze (--skip-freeze set). Checking available model...")
        candidate = os.path.join(args.freeze_dir, "tokenizer.model")
        if os.path.exists(candidate):
            frozen_model_path = candidate
        elif args.tokenizer_path and os.path.exists(args.tokenizer_path):
            frozen_model_path = args.tokenizer_path
        else:
            logger.warning("No frozen tokenizer found. Proceeding with estimated subword lengths.")

    # 2. Assemble Sharded Dataset across Train/Validation/Test
    logger.info("\n[Step 2/2] Assembling Sharded Dataset & Generating Corpus Report...")
    assembler = DatasetAssembler(
        output_dir=args.output_dir,
        tokenizer_path=frozen_model_path,
        split_ratios=tuple(args.ratios),
        shard_size=args.shard_size,
        seed=args.seed
    )

    try:
        manifest = assembler.assemble(
            sources=args.sources,
            max_examples=args.limit,
            mock_fallback=args.mock_fallback
        )
        logger.info("\n🎉 Phase 5.4 Dataset Builder & Corpus Validation successfully finished!")
        return 0
    except Exception as e:
        logger.error(f"RunPhase54: Dataset assembly failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
