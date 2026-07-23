"""
Production CLI script for streaming corpus acquisition and pipeline cleaning (`Phase 5`).
Usage: `python -m app.scripts.ingest --source ai4bharat --limit 5000`
"""

import argparse
import sys
from typing import Optional, List, Dict, Any, Union, Tuple
from app.corpus.acquisition import CorpusAcquisitionManager
from app.preprocessing.pipeline import PreprocessingPipeline
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Streaming Corpus Ingestion CLI")
    parser.add_argument("--source", type=str, default="wikipedia", help="Target dataset source to stream")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of examples to stream and process")
    parser.add_argument("--no-mock", action="store_true", help="Disable mock fallback and enforce real live corpus streaming")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    mock_fallback = not parsed.no_mock
    logger.info(f"CLI: Initiating streaming ingestion from source '{parsed.source}' (limit={parsed.limit}, mock_fallback={mock_fallback})...")

    mgr = CorpusAcquisitionManager()
    source_spec = mgr.get_source(parsed.source)
    if not source_spec:
        logger.error(f"CLI: Source '{parsed.source}' not found in registry.")
        return 1

    stream = mgr.stream_source(source_spec, max_examples=parsed.limit, mock_fallback=mock_fallback)
    pipeline = PreprocessingPipeline(config={})

    count = 0
    progress_step = max(1, parsed.limit // 5)
    for idx, ex in enumerate(stream):
        if idx >= parsed.limit:
            break
        processed = pipeline.process_example(ex, chunk=True)
        if processed:
            count += len(processed)
        if (idx + 1) % progress_step == 0:
            logger.info(f"CLI: Processed {idx + 1}/{parsed.limit} raw items -> {count} cleaned chunks so far.")

    logger.info(f"CLI: Streaming ingestion and cleaning completed -> yielded {count} cleaned chunks across {idx if 'idx' in locals() else 0} raw examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
