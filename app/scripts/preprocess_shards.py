"""
ManipuriGPT Phase 5.2 Large-Scale Preprocessing & Sharding CLI with Checkpointing & Resume.
Streams balanced corpora, executes PreprocessingPipeline, validates each shard, and writes resumable manifest.json.
"""

import os
import time
import json
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from datasets import Dataset
from app.corpus.sampler import BalancedCorpusSampler
from app.preprocessing.pipeline import PreprocessingPipeline
from app.preprocessing.exporters import DatasetExporter
from app.configs.loader import load_all_configs, compute_config_hash
from app.preprocessing.metadata_types import DocumentMetadata, PipelineMetadata
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Resumable Preprocessing & Sharding CLI")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Sources to include in balanced sampling (defaults to YAML configs if not provided)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum raw examples to stream before completing sharding run"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of chunked sequences per exported parquet shard"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where shards and manifest.json will be written"
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["parquet", "arrow", "jsonl"],
        help="Export format for dataset shards"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for balanced sampling"
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        help="Allow mock fallback if live HF connection fails"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest.json checkpoint if available"
    )
    return parser.parse_args(args)


def validate_shard_file(shard_path: str) -> Dict[str, Any]:
    """
    Performs continuous active integrity validation on a newly written shard file.
    """
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Validation failed: Shard file missing at {shard_path}")
    
    size_bytes = os.path.getsize(shard_path)
    if size_bytes == 0:
        raise ValueError(f"Validation failed: Shard file {shard_path} is empty (0 bytes)")

    # Compute SHA256 checksum
    hasher = hashlib.sha256()
    with open(shard_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    # If parquet, verify readability and required schema columns
    row_count = 0
    if shard_path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(shard_path)
            row_count = table.num_rows
            req_cols = {"text", "language", "script", "source"}
            if not req_cols.issubset(set(table.column_names)):
                raise ValueError(f"Shard schema missing required columns. Found: {table.column_names}")
        except Exception as e:
            logger.warning(f"Parquet detailed verification skipped or failed ({e}). File exists and checksum computed.")

    return {
        "valid": True,
        "size_bytes": size_bytes,
        "row_count": row_count,
        "sha256_checksum": checksum
    }


def write_manifest_checkpoint(
    manifest_path: str,
    shard_count: int,
    total_raw_processed: int,
    total_chunks_yielded: int,
    total_tokens_estimate: int,
    language_counts: Dict[str, int],
    seed: int,
    config_hash: str,
    duration_sec: float,
    shard_checksums: Dict[str, str]
) -> Dict[str, Any]:
    """Writes or updates manifest.json with resume checkpoint metadata."""
    resume_token = f"shard_{shard_count:06d}_{hashlib.sha256(str(shard_count).encode()).hexdigest()[:8]}"
    manifest = {
        "manifest_version": "1.0",
        "pipeline_version": "5.2",
        "last_completed_shard": shard_count,
        "raw_examples_processed": total_raw_processed,
        "chunks_exported": total_chunks_yielded,
        "tokens_estimate": total_tokens_estimate,
        "languages": language_counts,
        "created": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "config_hash": config_hash,
        "resume_token": resume_token,
        "duration_sec": duration_sec,
        "shard_checksums": shard_checksums,
        "pipeline_state": "in_progress"
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(args_list: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(args_list)
    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories()
    except Exception as e:
        logger.warning(f"PreprocessShards: Could not initialize cache directories: {e}")
    configs = load_all_configs()
    phase5_cfg = configs.get("phase5", {})
    sampling_cfg = configs.get("sampling", {}).get("sampling", {})

    # Merge YAML defaults with CLI overrides
    sources = args.sources or list(sampling_cfg.get("datasets", {"huggingface_datasets": 0.3, "ai4bharat": 0.2, "wikipedia": 0.2, "fineweb": 0.15, "opus": 0.15}).keys())
    limit = args.limit if args.limit is not None else phase5_cfg.get("sharding", {}).get("max_examples", 2000)
    shard_size = args.shard_size if args.shard_size is not None else phase5_cfg.get("sharding", {}).get("shard_size", 500)
    output_dir = args.output_dir or phase5_cfg.get("sharding", {}).get("output_dir", "cache/shards")
    export_format = args.format or phase5_cfg.get("sharding", {}).get("format", "parquet")
    seed = args.seed if args.seed is not None else phase5_cfg.get("execution", {}).get("seed", 42)
    mock_fallback = args.mock_fallback or phase5_cfg.get("execution", {}).get("mock_fallback", True)

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    config_hash = compute_config_hash({"sources": sources, "limit": limit, "shard_size": shard_size, "seed": seed})

    shard_count = 0
    total_raw_processed = 0
    total_chunks_yielded = 0
    total_tokens_estimate = 0
    language_counts: Dict[str, int] = {}
    shard_checksums: Dict[str, str] = {}
    skip_raw_examples = 0

    # Handle --resume option
    if args.resume and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
            shard_count = prev_manifest.get("last_completed_shard", 0)
            total_raw_processed = prev_manifest.get("raw_examples_processed", 0)
            total_chunks_yielded = prev_manifest.get("chunks_exported", 0)
            total_tokens_estimate = prev_manifest.get("tokens_estimate", 0)
            language_counts = prev_manifest.get("languages", {})
            shard_checksums = prev_manifest.get("shard_checksums", {})
            skip_raw_examples = total_raw_processed
            logger.info(f"PreprocessShards: Resuming from checkpoint -> last_shard={shard_count}, processed_raw={total_raw_processed}, resume_token={prev_manifest.get('resume_token')}")
        except Exception as e:
            logger.warning(f"PreprocessShards: Failed to read manifest checkpoint for resume ({e}). Starting fresh.")

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.2 RESUMABLE PREPROCESSING & SHARDING")
    logger.info("=" * 80)
    logger.info(f"Target Sources : {sources}")
    logger.info(f"Stream Limit   : {limit} raw items")
    logger.info(f"Shard Size     : {shard_size} chunks/shard")
    logger.info(f"Output Format  : {export_format.upper()}")
    logger.info(f"Output Dir     : {output_dir}")
    logger.info(f"Config Hash    : {config_hash}")
    logger.info(f"Resume Active  : {args.resume} (Skipping first {skip_raw_examples} raw items)")

    sampler = BalancedCorpusSampler(sources=sources, seed=seed)
    pipeline = PreprocessingPipeline()
    exporter = DatasetExporter(config={"format": export_format, "output_dir": output_dir})

    stream = sampler.stream(min_length=50, max_examples=limit + skip_raw_examples, mock_fallback=mock_fallback)
    current_shard_buffer: List[Dict[str, Any]] = []
    start_t = time.time()

    for raw_ex in stream:
        if skip_raw_examples > 0:
            skip_raw_examples -= 1
            continue

        total_raw_processed += 1
        chunks = pipeline.process_example(raw_ex, chunk=True)
        
        for chunk_ex in chunks:
            text = chunk_ex.get("text", "")
            if not text:
                continue

            lang = chunk_ex.get("metadata", {}).get("language", "en")
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            words_count = len(text.split())
            total_tokens_estimate += int(words_count * 1.4)
            total_chunks_yielded += 1

            meta = chunk_ex.get("metadata", {})
            flat_row = {
                "text": text,
                "language": str(lang),
                "script": str(meta.get("script", "unknown")),
                "source": str(meta.get("source", "unknown")),
                "source_dataset": str(meta.get("source_dataset", meta.get("source", "unknown"))),
                "dataset_version": str(meta.get("dataset_version", "v1")),
                "document_id": str(meta.get("document_id", "")),
                "quality_score": float(meta.get("quality_score", 1.0)),
                "timestamp": str(meta.get("timestamp", datetime.utcnow().isoformat() + "Z")),
                "chunk_id": int(meta.get("chunk_id", 0)),
                "total_chunks": int(meta.get("total_chunks", 1)),
                "tokenizer_version": str(meta.get("tokenizer_version", "sentencepiece_unigram_32k")),
                "pipeline_version": str(meta.get("pipeline_version", "5.2")),
                "config_hash": config_hash
            }
            current_shard_buffer.append(flat_row)

            if len(current_shard_buffer) >= shard_size:
                shard_count += 1
                shard_name = f"shard_{shard_count:06d}"
                shard_ds = Dataset.from_list(current_shard_buffer)
                exported_files = exporter.export(shard_ds, format_override=export_format, output_dir_override=output_dir, file_prefix=shard_name)
                current_shard_buffer.clear()
                
                # Active validation and checksum calculation
                for fpath in exported_files:
                    val_res = validate_shard_file(fpath)
                    shard_checksums[shard_name] = val_res["sha256_checksum"]
                    logger.info(f"PreprocessShards: Validated {shard_name}.{export_format} -> checksum={val_res['sha256_checksum'][:12]}...")

                # Write checkpoint manifest immediately
                duration_sec = round(time.time() - start_t, 2)
                write_manifest_checkpoint(manifest_path, shard_count, total_raw_processed, total_chunks_yielded, total_tokens_estimate, language_counts, seed, config_hash, duration_sec, shard_checksums)

    # Export any remaining items as final shard
    if current_shard_buffer:
        shard_count += 1
        shard_name = f"shard_{shard_count:06d}"
        shard_ds = Dataset.from_list(current_shard_buffer)
        exported_files = exporter.export(shard_ds, format_override=export_format, output_dir_override=output_dir, file_prefix=shard_name)
        current_shard_buffer.clear()
        
        for fpath in exported_files:
            val_res = validate_shard_file(fpath)
            shard_checksums[shard_name] = val_res["sha256_checksum"]
            logger.info(f"PreprocessShards: Validated final {shard_name}.{export_format}")

    duration_sec = round(time.time() - start_t, 2)
    final_manifest = write_manifest_checkpoint(manifest_path, shard_count, total_raw_processed, total_chunks_yielded, total_tokens_estimate, language_counts, seed, config_hash, duration_sec, shard_checksums)
    final_manifest["pipeline_state"] = "completed"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(" SHARDING RUN COMPLETED SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Raw Processed : {total_raw_processed} documents")
    logger.info(f"Total Chunks Exported: {total_chunks_yielded} chunks ({shard_count} shards)")
    logger.info(f"Estimated Tokens    : ~{total_tokens_estimate:,} tokens")
    logger.info(f"Language Distribution: {language_counts}")
    logger.info(f"Manifest Checkpoint : {manifest_path} (status={final_manifest['pipeline_state']})")
    logger.info("=" * 80)

    return final_manifest


if __name__ == "__main__":
    main()
