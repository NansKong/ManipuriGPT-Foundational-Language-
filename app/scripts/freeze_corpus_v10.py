"""
ManipuriGPT Corpus Snapshot Freezing CLI (`app/scripts/freeze_corpus_v10.py`).

Consolidates all v1 and v3 clean sequences into the final immutable snapshot
`artifacts/datasets/ManipuriGPT-Corpus-v1.0/` with explicit SHA256 checksums,
tokenizer freezing, release manifest, and dataset identity fingerprint.

Usage:
  python -m app.scripts.freeze_corpus_v10 --output-dir artifacts/datasets/ManipuriGPT-Corpus-v1.0
"""

import os
import sys
import argparse
from typing import List, Optional

from app.utils.logger import logger
from app.dataset_builder.corpus_freezer import CorpusFreezer


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ManipuriGPT Corpus Snapshot Freezing CLI"
    )
    parser.add_argument(
        "--v1-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1",
        help="Directory containing v1 corpus shards",
    )
    parser.add_argument(
        "--v3-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v3",
        help="Directory containing v3 corpus shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1.0",
        help="Directory where the final frozen dataset snapshot will be saved",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="artifacts/tokenizer_v1.0/tokenizer.model",
        help="Path to tokenizer model to freeze alongside dataset",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of sequences per Parquet shard file",
    )
    return parser.parse_args(args_list)


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT CORPUS SNAPSHOT FREEZING ENGINE")
    logger.info("=" * 80)
    logger.info(f"V1 Directory   : {args.v1_dir}")
    logger.info(f"V3 Directory   : {args.v3_dir}")
    logger.info(f"Frozen Output  : {args.output_dir}")
    logger.info(f"Tokenizer Path : {args.tokenizer_path}")

    freezer = CorpusFreezer()
    try:
        manifest = freezer.freeze(
            v1_dir=args.v1_dir,
            v3_dir=args.v3_dir,
            output_dir=args.output_dir,
            tokenizer_model_path=args.tokenizer_path,
            shard_size=args.shard_size
        )
    except Exception as e:
        logger.error(f"Corpus freezing failed: {e}")
        return 1

    logger.info("\n" + "=" * 80)
    logger.info(" FROZEN SNAPSHOT CREATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Name         : {manifest['name']}")
    logger.info(f"Version      : {manifest['version']}")
    logger.info(f"Status       : {manifest['status']}")
    logger.info(f"Total Seqs   : {manifest['records']:,}")
    logger.info(f"Total Chars  : {manifest['characters']:,}")
    logger.info(f"Total Tokens : {manifest['tokens']:,}")
    logger.info(f"Fingerprint  : {manifest['fingerprint'][:16]}...")
    logger.info(f"Output Path  : {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
