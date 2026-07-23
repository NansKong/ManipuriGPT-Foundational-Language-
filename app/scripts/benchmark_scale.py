"""
ManipuriGPT Phase 5.2 Scale & Throughput Benchmarking Tool.
Evaluates streaming speed, preprocessing throughput, tokenization speed, system memory/CPU, and disk write metrics.
"""

import os
import time
import json
import psutil
import random
import tempfile
import argparse
from typing import List, Dict, Any, Optional
from app.corpus.acquisition import CorpusAcquisitionManager
from app.preprocessing.pipeline import PreprocessingPipeline
from app.tokenizer.tokenizer_manager import TokenizerManager
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Scale & Throughput Benchmarker")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["huggingface_datasets", "ai4bharat", "wikipedia", "fineweb", "oscar", "slimpajama", "opus", "c4"],
        help="Sources to include in balanced sampling stream"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of items to stream and benchmark"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced sampling"
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        help="Allow mock fallback if live HF connection fails"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="cache/benchmarks/throughput.json",
        help="Path to save benchmark JSON report"
    )
    return parser.parse_args(args)


def measure_disk_write_speed(sample_texts: List[str], temp_path: Optional[str] = None) -> float:
    """Measures disk write speed in MB/sec."""
    payload = "\n".join(sample_texts).encode("utf-8")
    payload_mb = len(payload) / (1024.0 * 1024.0)
    
    start_t = time.time()
    if temp_path:
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        target_path = temp_path
    else:
        fd, target_path = tempfile.mkstemp(prefix="manipurigpt_bench_", suffix=".tmp")
        os.close(fd)

    try:
        with open(target_path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        duration = max(time.time() - start_t, 0.0001)
    finally:
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass

    return round(payload_mb / duration, 2)


def main(args_list: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(args_list)
    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories()
    except Exception as e:
        logger.warning(f"BenchmarkScale: Could not initialize cache directories: {e}")
    random.seed(args.seed)
    process = psutil.Process()
    
    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.2 SCALE & THROUGHPUT BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Target Sources : {args.sources}")
    logger.info(f"Stream Limit   : {args.limit}")
    logger.info(f"Random Seed    : {args.seed}")

    # Baseline memory and CPU
    process.cpu_percent()
    start_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
    peak_rss_mb = start_rss_mb
    cpu_samples: List[float] = []

    # 1. Streaming Benchmark
    logger.info("\n[1/4] Benchmarking Balanced Streaming Acquisition...")
    mgr = CorpusAcquisitionManager(sources=args.sources)
    start_stream_t = time.time()
    
    raw_examples: List[Dict[str, Any]] = []
    stream = mgr.stream_balanced(seed=args.seed, max_examples=args.limit, mock_fallback=args.mock_fallback)
    
    for ex in stream:
        raw_examples.append(ex)
        current_rss = process.memory_info().rss / (1024.0 * 1024.0)
        peak_rss_mb = max(peak_rss_mb, current_rss)
        cpu_samples.append(process.cpu_percent())
        if len(raw_examples) >= args.limit:
            break

    stream_duration = max(time.time() - start_stream_t, 0.001)
    total_raw_chars = sum(len(ex.get("text", "")) for ex in raw_examples)
    total_raw_bytes = sum(len(ex.get("text", "").encode("utf-8")) for ex in raw_examples)
    total_raw_mb = total_raw_bytes / (1024.0 * 1024.0)

    stream_samples_sec = round(len(raw_examples) / stream_duration, 2)
    stream_chars_sec = round(total_raw_chars / stream_duration, 2)
    stream_mb_sec = round(total_raw_mb / stream_duration, 3)

    # 2. Preprocessing Benchmark
    logger.info("\n[2/4] Benchmarking Preprocessing Pipeline Throughput...")
    pipeline = PreprocessingPipeline()
    start_prep_t = time.time()
    
    processed_chunks: List[Dict[str, Any]] = []
    for ex in raw_examples:
        chunks = pipeline.process_example(ex, chunk=True)
        processed_chunks.extend(chunks)
        current_rss = process.memory_info().rss / (1024.0 * 1024.0)
        peak_rss_mb = max(peak_rss_mb, current_rss)
        cpu_samples.append(process.cpu_percent())

    prep_duration = max(time.time() - start_prep_t, 0.001)
    prep_samples_sec = round(len(raw_examples) / prep_duration, 2)
    prep_chunks_sec = round(len(processed_chunks) / prep_duration, 2)
    total_prep_chars = sum(len(c.get("text", "")) for c in processed_chunks)
    prep_chars_sec = round(total_prep_chars / prep_duration, 2)

    # 3. Tokenizer Throughput Benchmark
    logger.info("\n[3/4] Benchmarking Tokenization Throughput...")
    local_tok_path = None
    if os.path.exists("cache/tokenizers"):
        for root, dirs, files in os.walk("cache/tokenizers"):
            if "tokenizer.json" in files or "tokenizer.model" in files:
                local_tok_path = root
                break

    if local_tok_path and not args.mock_fallback:
        logger.info(f"Loading local tokenizer from '{local_tok_path}' for benchmark...")
        tok_mgr = TokenizerManager(config={"model_name": local_tok_path})
        tok_wrapper = tok_mgr.get_tokenizer("indic")
    elif args.mock_fallback:
        logger.info("Using fast local whitespace tokenization for benchmark (avoiding remote HF download)...")
        class WhitespaceTokenizerWrapper:
            def count_tokens(self, text: str) -> int:
                return len(text.split())
            def encode(self, text: str) -> Dict[str, List[int]]:
                words = text.split()
                return {"input_ids": list(range(len(words)))}
        tok_wrapper = WhitespaceTokenizerWrapper()
    else:
        logger.info("Loading default remote tokenizer for benchmark...")
        tok_mgr = TokenizerManager()
        tok_wrapper = tok_mgr.get_tokenizer("indic")
    
    chunk_texts = [c.get("text", "") for c in processed_chunks if c.get("text")]
    start_tok_t = time.time()
    
    total_tokens = 0
    for text in chunk_texts:
        if hasattr(tok_wrapper, "count_tokens"):
            total_tokens += tok_wrapper.count_tokens(text)
        elif hasattr(tok_wrapper, "encode"):
            enc = tok_wrapper.encode(text)
            total_tokens += len(enc["input_ids"]) if isinstance(enc, dict) and "input_ids" in enc else len(enc)
        current_rss = process.memory_info().rss / (1024.0 * 1024.0)
        peak_rss_mb = max(peak_rss_mb, current_rss)
        cpu_samples.append(process.cpu_percent())

    tok_duration = max(time.time() - start_tok_t, 0.001)
    tok_tokens_sec = round(total_tokens / tok_duration, 2)
    tok_chunks_sec = round(len(chunk_texts) / tok_duration, 2)

    # 4. Disk Write Speed Benchmark
    logger.info("\n[4/4] Benchmarking Disk Write Speed...")
    disk_write_mb_sec = measure_disk_write_speed(chunk_texts if chunk_texts else ["sample text"])

    # Calculate system metrics
    avg_rss_mb = round((start_rss_mb + peak_rss_mb) / 2.0, 2)
    peak_rss_mb = round(peak_rss_mb, 2)
    avg_cpu = round(sum(cpu_samples) / max(len(cpu_samples), 1), 1)
    peak_cpu = round(max(cpu_samples) if cpu_samples else 0.0, 1)

    report = {
        "pipeline_version": "5.2",
        "seed": args.seed,
        "streaming": {
            "total_samples": len(raw_examples),
            "samples_per_sec": stream_samples_sec,
            "chars_per_sec": stream_chars_sec,
            "mb_per_sec": stream_mb_sec,
            "duration_sec": round(stream_duration, 3)
        },
        "preprocessing": {
            "raw_samples_processed": len(raw_examples),
            "output_chunks": len(processed_chunks),
            "samples_per_sec": prep_samples_sec,
            "chunks_per_sec": prep_chunks_sec,
            "chars_per_sec": prep_chars_sec,
            "duration_sec": round(prep_duration, 3)
        },
        "tokenization": {
            "chunks_tokenized": len(chunk_texts),
            "total_tokens": total_tokens,
            "chunks_per_sec": tok_chunks_sec,
            "tokens_per_sec": tok_tokens_sec,
            "duration_sec": round(tok_duration, 3)
        },
        "system_resources": {
            "peak_ram_mb": peak_rss_mb,
            "avg_ram_mb": avg_rss_mb,
            "peak_cpu_percent": peak_cpu,
            "avg_cpu_percent": avg_cpu,
            "disk_write_mb_sec": disk_write_mb_sec
        }
    }

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    runtime_stats_path = "cache/statistics/runtime_stats.json"
    os.makedirs("cache/statistics", exist_ok=True)
    with open(runtime_stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(" BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Streaming Speed       : {stream_samples_sec} samples/sec ({stream_mb_sec} MB/sec)")
    logger.info(f"Preprocessing Speed   : {prep_samples_sec} samples/sec -> {prep_chunks_sec} chunks/sec")
    logger.info(f"Tokenization Speed    : {tok_tokens_sec} tokens/sec ({tok_chunks_sec} chunks/sec)")
    logger.info(f"System RAM Consumed   : Peak={peak_rss_mb} MB | Avg={avg_rss_mb} MB")
    logger.info(f"System CPU Utilized   : Peak={peak_cpu}% | Avg={avg_cpu}%")
    logger.info(f"Disk Write Throughput : {disk_write_mb_sec} MB/sec")
    logger.info(f"Report saved to       : {args.output_report}")
    logger.info("=" * 80)

    return report


if __name__ == "__main__":
    main()
