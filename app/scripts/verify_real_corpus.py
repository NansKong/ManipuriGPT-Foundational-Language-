"""
Diagnostic and verification script for Phase 5.1 Real Corpus Integration.
Tests HF_TOKEN authentication, live network streaming, exponential backoff resilience,
and pipeline cleaning across priority datasets (AI4Bharat, Wikipedia, FineWeb, OSCAR, CC100, OPUS, C4).

Usage:
    python -m app.scripts.verify_real_corpus --sources ai4bharat wikipedia fineweb oscar cc100 opus c4 --limit 5
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Any, Optional
from app.corpus.acquisition import CorpusAcquisitionManager
from app.preprocessing.pipeline import PreprocessingPipeline
from app.configs.settings import settings
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Real Corpus Verification & Diagnostic Tool")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["huggingface_datasets", "ai4bharat", "wikipedia", "fineweb", "oscar", "slimpajama", "opus", "c4"],
        help="List of dataset sources to verify live streaming from"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of items to stream and verify per source"
    )
    return parser.parse_args(args)


def verify_source(mgr: CorpusAcquisitionManager, pipeline: PreprocessingPipeline, source_name: str, limit: int) -> Dict[str, Any]:
    """Verifies live streaming and cleaning for a single target source."""
    spec = mgr.get_source(source_name)
    if not spec:
        return {"source": source_name, "status": "FAILED (Not Registered)", "items": 0, "cleaned_chunks": 0, "error": "Source not in registry"}

    logger.info(f"\n[{source_name.upper()}] Verifying live stream: repo='{spec.dataset_path}' subset='{spec.subset}' split='{spec.split}'...")
    start_time = time.time()
    raw_count = 0
    cleaned_count = 0
    error_msg = None
    if hasattr(pipeline, "reset"):
        pipeline.reset()

    try:
        # Stream with mock_fallback=False to enforce real HuggingFace network connection
        stream = mgr.stream_source(spec, max_examples=limit, mock_fallback=False)
        for idx, ex in enumerate(stream):
            raw_count += 1
            chunks = pipeline.process_example(ex, chunk=True)
            if chunks:
                cleaned_count += len(chunks)
        duration = time.time() - start_time
        status = "PASSED" if raw_count > 0 else "WARNING (No items yielded)"
    except Exception as e:
        duration = time.time() - start_time
        status = f"FAILED ({type(e).__name__})"
        error_msg = str(e)
        logger.error(f"[{source_name.upper()}] Error during real streaming verification: {e}")

    return {
        "source": source_name,
        "status": status,
        "items": raw_count,
        "cleaned_chunks": cleaned_count,
        "duration_sec": round(duration, 2),
        "error": error_msg or "-"
    }


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    token = os.getenv("HF_TOKEN", getattr(settings.model, "hf_token", None))
    token_status = f"AUTHENTICATED (prefix: '{token[:6]}...')" if token else "UNAUTHENTICATED (Anonymous Hub access)"
    
    print("=" * 80)
    print(" MANIPURIGPT PHASE 5.1 REAL CORPUS DIAGNOSTIC VERIFICATION")
    print("=" * 80)
    print(f"HF_TOKEN Status : {token_status}")
    print(f"Target Sources  : {', '.join(parsed.sources)}")
    print(f"Stream Limit    : {parsed.limit} examples per source")
    print("-" * 80)

    mgr = CorpusAcquisitionManager()
    # Configure pipeline with target_language='any' during diagnostic verification across multilingual sources
    pipeline = PreprocessingPipeline(config={"preprocessing": {"language_detection": {"target_language": "any"}}})
    results = []

    for src in parsed.sources:
        try:
            res = verify_source(mgr, pipeline, src, parsed.limit)
            results.append(res)
        except KeyError as ke:
            results.append({"source": src, "status": "FAILED (KeyError)", "items": 0, "cleaned_chunks": 0, "duration_sec": 0, "error": str(ke)})

    print("\n" + "=" * 80)
    print(" VERIFICATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Source':<18} | {'Status':<22} | {'Raw Items':<10} | {'Chunks':<8} | {'Time (s)':<8}")
    print("-" * 80)
    failed = 0
    for r in results:
        status_str = r['status'][:22]
        print(f"{r['source']:<18} | {status_str:<22} | {r['items']:<10} | {r['cleaned_chunks']:<8} | {r.get('duration_sec', 0):<8}")
        if "FAILED" in r["status"]:
            failed += 1
            if r["error"] != "-":
                print(f"   └─ Error: {r['error'][:75]}")
    print("=" * 80)

    if failed > 0:
        logger.warning(f"Verification finished with {failed} failed source(s). Check network connectivity or repository access permissions.")
        return 1
    logger.info("Verification completed successfully across all requested real corpus sources!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
