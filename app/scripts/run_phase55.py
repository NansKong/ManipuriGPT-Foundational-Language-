"""
ManipuriGPT Phase 5.5 Master Corpus Scaling CLI (`app/scripts/run_phase55.py`).

Ingests ALL available local and cached corpus sources into a unified v2 master
corpus with strict provenance metadata, SHA256 deduplication (including cross-dedup
against existing v1 shards), and deterministic train/val/test splitting.

Sources ingested:
  1. Processed OCR PDFs (cache/processed/*.jsonl)
  2. EMA Lon monolingual (TXT)
  3. EMA Lon parallel - Manipuri side (TSV)
  4. AI4Bharat Sangraha (Arrow cache)
  5. Dayananda Meitei Mayek sample (Parquet cache)
  6. Dayananda English-to-Meitei Mayek (Parquet cache)
  7. Joyson English-Manipuri parallel (Arrow cache)
  8. Existing v1 corpus shards (cross-dedup seed)

Usage:
  python -m app.scripts.run_phase55 --output-dir artifacts/datasets/ManipuriGPT-Corpus-v2 --skip-freeze
"""

import os
import sys
import time
import json
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from app.utils.logger import logger


# All local sources to ingest (Phase 5.5)
PHASE55_LOCAL_SOURCES = [
    "local_processed_pdfs",
    "local_ema_lon_mono",
    "local_ema_lon_parallel",
    "local_sangraha_cached",
    "local_dayananda_meitei",
    "local_dayananda_eng_to_meitei",
    "local_joyson_parallel",
]

# HF sources from Phase 5.4 to re-ingest (already cached locally)
PHASE55_HF_SOURCES = [
    "dayananda_meitei_mayek_sample",
    "dayananda_english_to_meitei",
    "joyson_bible",
    "joyson_pib_pmi",
]


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ManipuriGPT Phase 5.5 Master Corpus Scaling CLI"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v2",
        help="Directory where v2 training dataset shards will be saved",
    )
    parser.add_argument(
        "--v1-dir",
        type=str,
        default="artifacts/datasets/ManipuriGPT-Corpus-v1",
        help="Existing v1 corpus directory (for cross-dedup seed)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Explicit path to tokenizer.model (auto-discovers if not set)",
    )
    parser.add_argument(
        "--freeze-dir",
        type=str,
        default="artifacts/tokenizer_v1",
        help="Directory containing frozen Tokenizer v1 artifacts",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.98, 0.01, 0.01],
        help="Split ratios for Train / Val / Test (default: 0.98 0.01 0.01)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of sequences per Parquet shard file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--skip-freeze",
        action="store_true",
        help="Skip tokenizer freezing (reuse existing frozen artifacts)",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HF streaming sources (use only local sources)",
    )
    parser.add_argument(
        "--skip-v1-dedup",
        action="store_true",
        help="Skip cross-dedup against v1 corpus",
    )
    return parser.parse_args(args)


def _load_v1_hashes(v1_dir: str) -> set:
    """Load SHA256 hashes of all text sequences from v1 Parquet shards for cross-dedup."""
    hashes = set()
    if not os.path.isdir(v1_dir):
        logger.warning(f"Phase55: v1 directory not found: {v1_dir}. Skipping cross-dedup.")
        return hashes

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("Phase55: pyarrow not installed. Cannot cross-dedup against v1.")
        return hashes

    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(v1_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for f in sorted(os.listdir(split_dir)):
            if not f.endswith(".parquet"):
                continue
            fp = os.path.join(split_dir, f)
            try:
                table = pq.read_table(fp, columns=["text"])
                for i in range(table.num_rows):
                    text = table.column("text")[i].as_py()
                    if text:
                        h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
                        hashes.add(h)
            except Exception as e:
                logger.warning(f"Phase55: Error reading v1 shard {fp}: {e}")

    logger.info(f"Phase55: Loaded {len(hashes):,} v1 hashes for cross-deduplication.")
    return hashes


def _validate_shard_file(shard_path: str) -> Dict[str, Any]:
    """SHA256 checksum and integrity check on exported Parquet shard."""
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file missing: {shard_path}")
    size_bytes = os.path.getsize(shard_path)
    if size_bytes == 0:
        raise ValueError(f"Shard file empty: {shard_path}")

    hasher = hashlib.sha256()
    with open(shard_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return {"size_bytes": size_bytes, "sha256_checksum": hasher.hexdigest()}


def main(args_list: Optional[List[str]] = None) -> int:
    args = parse_args(args_list)

    # Cache setup
    try:
        from app.utils.cache import setup_cache_directories
        setup_cache_directories()
    except Exception as e:
        logger.warning(f"Phase55: Could not initialize cache directories: {e}")

    logger.info("=" * 80)
    logger.info(" MANIPURIGPT PHASE 5.5 — MASTER CORPUS SCALING ENGINE")
    logger.info("=" * 80)
    logger.info(f"Output Dir       : {args.output_dir}")
    logger.info(f"V1 Corpus Dir    : {args.v1_dir}")
    logger.info(f"Split Ratios     : {args.ratios}")
    logger.info(f"Shard Size       : {args.shard_size}")
    logger.info(f"Seed             : {args.seed}")

    start_t = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 0: Tokenizer
    # ---------------------------------------------------------------
    frozen_model_path: Optional[str] = None
    if not args.skip_freeze:
        logger.info("\n[Step 0] Freezing Tokenizer v1...")
        from app.dataset_builder.tokenizer_freezer import TokenizerFreezer
        freezer = TokenizerFreezer(default_output_dir=args.freeze_dir)
        try:
            frozen_files = freezer.freeze(
                source_model_path=args.tokenizer_path, output_dir=args.freeze_dir
            )
            frozen_model_path = frozen_files.get("tokenizer.model")
        except Exception as e:
            logger.warning(f"Phase55: Tokenizer freeze failed: {e}")
    else:
        candidate = os.path.join(args.freeze_dir, "tokenizer.model")
        if os.path.exists(candidate):
            frozen_model_path = candidate
        elif args.tokenizer_path and os.path.exists(args.tokenizer_path):
            frozen_model_path = args.tokenizer_path
        logger.info(f"[Step 0] Using existing tokenizer: {frozen_model_path}")

    # ---------------------------------------------------------------
    # Step 1: Load v1 hashes for cross-dedup
    # ---------------------------------------------------------------
    if not args.skip_v1_dedup:
        logger.info("\n[Step 1] Loading v1 corpus hashes for cross-deduplication...")
        v1_hashes = _load_v1_hashes(args.v1_dir)
    else:
        v1_hashes = set()
        logger.info("\n[Step 1] Skipping v1 cross-dedup (--skip-v1-dedup set).")

    # ---------------------------------------------------------------
    # Step 2: Stream all sources through the pipeline
    # ---------------------------------------------------------------
    logger.info("\n[Step 2] Streaming & processing ALL sources through the pipeline...")

    from app.preprocessing.normalizer import UnicodeNormalizer
    from app.preprocessing.cleaner import TextCleaner
    from app.preprocessing.validator import SentenceValidator
    from app.preprocessing.script_detector import ScriptDetector

    normalizer = UnicodeNormalizer({})
    cleaner = TextCleaner({})
    validator = SentenceValidator({})
    script_detector = ScriptDetector({})

    clean_rows: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, int]] = {}
    total_raw = 0
    total_dups = 0
    total_quality_filtered = 0

    def _process_source(source_name: str):
        nonlocal total_raw, total_dups, total_quality_filtered

        try:
            spec = get_source_spec(source_name)
        except KeyError as e:
            logger.warning(f"Phase55: Source '{source_name}' not in registry: {e}")
            return

        logger.info(f"\n--- Processing source: {source_name} ---")

        streamer = CorpusStreamer(
            source=spec,
            min_length=10,
            max_examples=None,
            mock_fallback=False,
        )

        src_raw = 0
        src_kept = 0
        src_dups = 0
        src_filtered = 0

        for raw_ex in streamer:
            src_raw += 1
            total_raw += 1

            if total_raw % 10000 == 0:
                logger.info(
                    f"  ... Processed {total_raw:,} raw records total, "
                    f"{len(clean_rows):,} kept so far"
                )

            # --- Lightweight inline processing (no heavy langdetect/MinHash) ---
            text = raw_ex.get("text", "")
            if not text or not isinstance(text, str):
                src_filtered += 1
                continue

            # 1. Unicode normalize + clean
            text = normalizer.normalize(text)
            text = cleaner.clean(text)
            text = text.strip()

            # 2. Validate
            if not text or len(text) < 10 or not validator.validate(text):
                src_filtered += 1
                continue

            # 3. SHA256 exact dedup (including cross-dedup against v1)
            txt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if txt_hash in seen_hashes:
                src_dups += 1
                total_dups += 1
                continue
            seen_hashes.add(txt_hash)

            # 4. Fast script detection (regex-based, no ML)
            script_info = script_detector.detect(text)

            # 5. Extract metadata from raw example (already tagged by LocalCorpusLoader)
            meta = raw_ex.get("metadata", {})
            lang = meta.get("language", "mni")
            script = script_info.get("script", meta.get("script", "unknown"))

            row = {
                "text": text,
                "language": str(lang),
                "script": str(script),
                "source": str(meta.get("source", source_name)),
                "source_dataset": str(
                    meta.get("source_dataset", meta.get("source", source_name))
                ),
                "category": str(meta.get("category", "unknown")),
                "license": str(meta.get("license", "Various")),
                "year": int(meta.get("year", 2024)),
                "quality_score": float(meta.get("quality_score", 1.0)),
                "document_id": str(meta.get("document_id", "")),
                "chunk_id": int(meta.get("chunk_id", 0)),
                "timestamp": str(
                    meta.get("timestamp", datetime.utcnow().isoformat() + "Z")
                ),
                "tokenizer_version": "v1" if frozen_model_path else "estimated",
            }

            # Preserve English parallel if available
            eng_parallel = meta.get("english_parallel", "")
            if eng_parallel:
                row["english_parallel"] = str(eng_parallel)

            clean_rows.append(row)
            src_kept += 1

        source_stats[source_name] = {
            "raw": src_raw,
            "kept": src_kept,
            "duplicates": src_dups,
            "filtered": src_raw - src_kept - src_dups,
        }
        logger.info(
            f"  → {source_name}: raw={src_raw:,}, kept={src_kept:,}, "
            f"dups={src_dups:,}, filtered={src_raw - src_kept - src_dups:,}"
        )

    # Process all local sources
    for source_name in PHASE55_LOCAL_SOURCES:
        _process_source(source_name)

    # Process HF sources (if not skipped)
    if not args.skip_hf:
        for source_name in PHASE55_HF_SOURCES:
            _process_source(source_name)
    else:
        logger.info("\n[Skipping HF sources per --skip-hf flag]")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Total raw processed : {total_raw:,}")
    logger.info(f"Total kept (unique) : {len(clean_rows):,}")
    logger.info(f"Total duplicates    : {total_dups:,}")
    logger.info(f"V1 cross-dedup seed : {len(v1_hashes):,}")
    logger.info(f"{'=' * 60}")

    if not clean_rows:
        logger.error("Phase55: 0 sequences remaining after processing! Aborting.")
        return 1

    # ---------------------------------------------------------------
    # Step 3: Deterministic splitting
    # ---------------------------------------------------------------
    logger.info(f"\n[Step 3] Splitting {len(clean_rows):,} sequences into Train/Val/Test...")

    from datasets import Dataset, DatasetDict
    from app.preprocessing.splitter import DatasetSplitter

    splitter = DatasetSplitter({
        "enabled": True,
        "train": args.ratios[0],
        "validation": args.ratios[1],
        "test": args.ratios[2],
    })

    full_ds = Dataset.from_list(clean_rows)
    split_dict: DatasetDict = splitter.split(full_ds, seed=args.seed)

    for split_name, ds in split_dict.items():
        logger.info(f"  {split_name}: {len(ds):,} sequences")

    # ---------------------------------------------------------------
    # Step 4: Export Parquet shards
    # ---------------------------------------------------------------
    logger.info(f"\n[Step 4] Exporting Parquet shards to '{args.output_dir}'...")

    shard_checksums: Dict[str, Dict[str, Any]] = {}
    total_shards = 0

    for split_name, ds in split_dict.items():
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        rows_in_split = len(ds)
        shard_idx = 0
        for start_i in range(0, rows_in_split, args.shard_size):
            end_i = min(start_i + args.shard_size, rows_in_split)
            shard_rows = [ds[i] for i in range(start_i, end_i)]
            shard_ds = Dataset.from_list(shard_rows)

            shard_filename = f"shard-{shard_idx:05d}.parquet"
            shard_path = os.path.join(split_dir, shard_filename)
            shard_ds.to_parquet(shard_path)

            val_meta = _validate_shard_file(shard_path)
            shard_key = f"{split_name}/{shard_filename}"
            shard_checksums[shard_key] = {
                "split": split_name,
                "rows": len(shard_rows),
                "size_bytes": val_meta["size_bytes"],
                "sha256": val_meta["sha256_checksum"],
            }
            shard_idx += 1
            total_shards += 1
            logger.info(
                f"  → Saved {shard_key} ({len(shard_rows):,} seqs, "
                f"sha256={val_meta['sha256_checksum'][:12]}...)"
            )

    # ---------------------------------------------------------------
    # Step 5: Corpus report & dataset card
    # ---------------------------------------------------------------
    logger.info("\n[Step 5] Generating corpus report & dataset card...")

    # Compute provenance breakdown
    source_dist = {}
    category_dist = {}
    script_dist = {}
    language_dist = {}
    total_chars = 0
    total_words = 0

    for row in clean_rows:
        src = row.get("source", "unknown")
        cat = row.get("category", "unknown")
        scr = row.get("script", "unknown")
        lang = row.get("language", "unknown")
        text = row.get("text", "")

        source_dist[src] = source_dist.get(src, 0) + 1
        category_dist[cat] = category_dist.get(cat, 0) + 1
        script_dist[scr] = script_dist.get(scr, 0) + 1
        language_dist[lang] = language_dist.get(lang, 0) + 1
        total_chars += len(text)
        total_words += len(text.split())

    # Tokenizer evaluation (if available)
    total_tokens = 0
    total_unk = 0
    if frozen_model_path and os.path.exists(frozen_model_path):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(frozen_model_path)
            unk_id = sp.unk_id()

            for row in clean_rows:
                ids = sp.Encode(row["text"])
                total_tokens += len(ids)
                total_unk += sum(1 for t in ids if t == unk_id)
        except Exception as e:
            logger.warning(f"Phase55: Tokenizer evaluation failed: {e}")

    unk_rate = (total_unk / max(total_tokens, 1)) * 100.0

    report = {
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "5.5",
        "tokenizer_used": frozen_model_path or "estimated",
        "overall_statistics": {
            "total_sequences": len(clean_rows),
            "total_characters": total_chars,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "total_unk_tokens": total_unk,
            "unknown_token_rate_pct": round(unk_rate, 4),
            "avg_sequence_length_chars": round(total_chars / max(len(clean_rows), 1), 2),
            "avg_sequence_length_words": round(total_words / max(len(clean_rows), 1), 2),
        },
        "deduplication_metrics": {
            "raw_examples_before_deduplication": total_raw,
            "v1_cross_dedup_seed_hashes": len(v1_hashes),
            "clean_sequences_retained": len(clean_rows),
            "duplicate_count": total_dups,
            "duplicate_percentage": round(total_dups / max(total_raw, 1) * 100, 2),
            "retention_percentage": round(len(clean_rows) / max(total_raw, 1) * 100, 2),
        },
        "source_distribution": dict(sorted(source_dist.items(), key=lambda x: -x[1])),
        "category_distribution": dict(sorted(category_dist.items(), key=lambda x: -x[1])),
        "script_distribution": dict(sorted(script_dist.items(), key=lambda x: -x[1])),
        "language_distribution": dict(sorted(language_dist.items(), key=lambda x: -x[1])),
        "per_source_stats": source_stats,
        "split_statistics": {
            split_name: len(ds) for split_name, ds in split_dict.items()
        },
        "shard_checksums": shard_checksums,
    }

    # Save report
    meta_dir = os.path.join(args.output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    report_path = os.path.join(meta_dir, "corpus_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"  → Saved corpus report: {report_path}")

    # Manifest
    manifest_path = os.path.join(meta_dir, "manifest.json")
    manifest = {
        "dataset_name": "ManipuriGPT-Corpus-v2",
        "pipeline_version": "5.5",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "total_sequences": len(clean_rows),
        "split_ratios": args.ratios,
        "total_shards": total_shards,
        "shard_size_limit": args.shard_size,
        "tokenizer_path": frozen_model_path or "estimated",
        "v1_corpus_dir": args.v1_dir,
        "duration_sec": round(time.time() - start_t, 2),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  → Saved manifest: {manifest_path}")

    # Dataset Card (README.md)
    card_path = os.path.join(args.output_dir, "README.md")
    _generate_dataset_card(card_path, report, manifest)
    logger.info(f"  → Saved dataset card: {card_path}")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    duration = round(time.time() - start_t, 2)
    logger.info("\n" + "=" * 80)
    logger.info(" PHASE 5.5 MASTER CORPUS SCALING — COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Sequences     : {len(clean_rows):,}")
    logger.info(f"Total Parquet Shards: {total_shards}")
    logger.info(f"Total Characters    : {total_chars:,}")
    logger.info(f"Total Tokens        : {total_tokens:,}")
    logger.info(f"Unknown Token Rate  : {unk_rate:.4f}%")
    logger.info(f"Sources Ingested    : {len(source_stats)}")
    logger.info(f"Duration            : {duration}s")
    logger.info(f"Output Directory    : {args.output_dir}")
    logger.info("=" * 80)

    return 0


def _generate_dataset_card(card_path: str, report: Dict, manifest: Dict):
    """Generate a Hugging Face-compatible README.md dataset card."""
    total_seqs = report["overall_statistics"]["total_sequences"]
    total_chars = report["overall_statistics"]["total_characters"]
    total_tokens = report["overall_statistics"]["total_tokens"]
    unk_rate = report["overall_statistics"]["unknown_token_rate_pct"]

    source_lines = "\n".join(
        f"| {src} | {count:,} | {count / max(total_seqs, 1) * 100:.1f}% |"
        for src, count in report["source_distribution"].items()
    )
    script_lines = "\n".join(
        f"| {scr} | {count:,} | {count / max(total_seqs, 1) * 100:.1f}% |"
        for scr, count in report["script_distribution"].items()
    )
    category_lines = "\n".join(
        f"| {cat} | {count:,} | {count / max(total_seqs, 1) * 100:.1f}% |"
        for cat, count in report["category_distribution"].items()
    )

    card = f"""---
language:
  - mni
  - en
  - bn
license: cc-by-nc-4.0
task_categories:
  - text-generation
  - fill-mask
tags:
  - manipuri
  - meitei-mayek
  - bengali-script
  - indic
  - low-resource
  - foundation-model
pretty_name: ManipuriGPT Corpus v2
size_categories:
  - 100K<n<1M
---

# ManipuriGPT Corpus v2

A comprehensive, deduplicated, quality-scored Manipuri language corpus for
foundation model pretraining. Built by the ManipuriGPT Phase 5.5 pipeline.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Sequences | {total_seqs:,} |
| Total Characters | {total_chars:,} |
| Total Tokens (SentencePiece) | {total_tokens:,} |
| Unknown Token Rate | {unk_rate:.4f}% |
| Pipeline Version | 5.5 |
| Created | {manifest['created_at']} |

## Source Distribution

| Source | Count | % |
|--------|-------|---|
{source_lines}

## Script Distribution

| Script | Count | % |
|--------|-------|---|
{script_lines}

## Category Distribution

| Category | Count | % |
|----------|-------|---|
{category_lines}

## Data Provenance

Every record includes full provenance metadata:
- `source`: Origin dataset identifier
- `category`: Content type (monolingual, parallel, textbook, etc.)
- `script`: Writing system (Mtei, Beng, Latn, Deva)
- `license`: License information
- `year`: Publication year
- `quality_score`: Quality score from the pipeline

## Processing Pipeline

1. Raw data ingestion from 7+ local and cached sources
2. Unicode normalization (NFC)
3. Text cleaning and validation
4. SHA256 exact deduplication (including cross-dedup against v1)
5. Script detection
6. Language detection
7. Quality scoring and toxicity filtering
8. PII removal
9. MinHash near-duplicate detection
10. Canonicalization
11. Sequence chunking
12. Deterministic train/val/test splitting ({manifest['split_ratios']})
13. Parquet sharding with SHA256 integrity checksums

## License

Mixed licenses. See individual record `license` fields for details.
Primary corpus sources are CC BY-NC 4.0 (EMA Lon) and CC BY 4.0 (AI4Bharat Sangraha).
"""

    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)


if __name__ == "__main__":
    sys.exit(main())
