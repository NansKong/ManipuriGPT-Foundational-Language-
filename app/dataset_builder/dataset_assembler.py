"""
DatasetAssembler module (`app/dataset_builder/dataset_assembler.py`).
Core engine for Phase 5.4: transforms raw/streamed corpus data into deterministic,
sharded Parquet datasets (`train/`, `validation/`, `test/`) enriched with Tokenizer v1 metadata,
quality metrics (`corpus_report.json`), and SHA256 integrity checks.
"""

import os
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import Dataset, DatasetDict
from app.corpus.sampler import BalancedCorpusSampler
from app.preprocessing.pipeline import PreprocessingPipeline
from app.preprocessing.quality_scorer import QualityScorer, ToxicityFilter
from app.preprocessing.splitter import DatasetSplitter
from app.dataset_builder.corpus_validator import CorpusValidator
from app.utils.logger import logger


class DatasetAssembler:
    """
    Orchestrates Phase 5.4 dataset building:
    1. Streaming & Normalization
    2. Exact & Fuzzy Deduplication
    3. Quality & Toxicity Screening
    4. Deterministic Splitting (default: 98% train, 1% val, 1% test)
    5. Chunked Parquet Sharding with active SHA256 verification
    6. Corpus Quality Evaluation Report & Dataset Card generation
    """
    def __init__(
        self,
        output_dir: str = "artifacts/datasets/ManipuriGPT-Corpus-v1",
        tokenizer_path: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.98, 0.01, 0.01),
        shard_size: int = 10000,
        seed: int = 42
    ):
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path
        self.split_ratios = split_ratios
        self.shard_size = shard_size
        self.seed = seed

        self.pipeline = PreprocessingPipeline()
        self.scorer = QualityScorer(min_score=0.40)
        self.toxicity_filter = ToxicityFilter()
        self.validator = CorpusValidator(tokenizer_path=tokenizer_path)

        # Configure splitter with custom ratios
        self.splitter = DatasetSplitter({
            "enabled": True,
            "train": split_ratios[0],
            "validation": split_ratios[1],
            "test": split_ratios[2]
        })

    def _validate_shard_file(self, shard_path: str) -> Dict[str, Any]:
        """Performs SHA256 checksum calculation and integrity check on exported Parquet."""
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file missing: {shard_path}")
        size_bytes = os.path.getsize(shard_path)
        if size_bytes == 0:
            raise ValueError(f"Shard file empty (0 bytes): {shard_path}")

        hasher = hashlib.sha256()
        with open(shard_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return {
            "size_bytes": size_bytes,
            "sha256_checksum": hasher.hexdigest()
        }

    def assemble(
        self,
        sources: Optional[List[str]] = None,
        max_examples: int = 10000,
        mock_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end dataset assembly from balanced sources or raw stream.
        """
        start_t = time.time()
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info(" MANIPURIGPT PHASE 5.4 DATASET ASSEMBLER")
        logger.info("=" * 80)
        logger.info(f"Output Base Dir  : {self.output_dir}")
        logger.info(f"Target Sources   : {sources or 'Default Balanced Sampler'}")
        logger.info(f"Max Stream Limit : {max_examples} raw items")
        logger.info(f"Split Ratios     : Train={self.split_ratios[0]}, Val={self.split_ratios[1]}, Test={self.split_ratios[2]}")
        logger.info(f"Shard Size       : {self.shard_size} sequences / shard")
        logger.info(f"Random Seed      : {self.seed}")

        # 1. Stream, Normalize & Deduplicate
        logger.info("\n[1/4] Streaming, Normalizing & Deduplicating corpus sequences...")
        sampler = BalancedCorpusSampler(sources=sources, seed=self.seed)
        stream = sampler.stream(min_length=30, max_examples=max_examples, mock_fallback=mock_fallback)

        clean_rows: List[Dict[str, Any]] = []
        seen_exact_hashes = set()
        raw_count = 0
        toxic_count = 0
        quality_filtered_count = 0
        exact_dups = 0

        for raw_ex in stream:
            raw_count += 1
            chunks = self.pipeline.process_example(raw_ex, chunk=True)
            for chunk_ex in chunks:
                text = chunk_ex.get("text", "").strip()
                if not text:
                    continue

                # Exact deduplication via SHA256 of normalized text
                txt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if txt_hash in seen_exact_hashes:
                    exact_dups += 1
                    continue
                seen_exact_hashes.add(txt_hash)

                # Toxicity and Quality check
                filtered_ex = self.toxicity_filter.filter_example(chunk_ex, scorer=self.scorer)
                if filtered_ex is None:
                    toxic, _ = self.toxicity_filter.is_toxic(text)
                    if toxic:
                        toxic_count += 1
                    else:
                        quality_filtered_count += 1
                    continue

                meta = filtered_ex.get("metadata", {})
                row = {
                    "text": text,
                    "language": str(meta.get("language", "en")),
                    "script": str(meta.get("script", "unknown")),
                    "source": str(meta.get("source", "unknown")),
                    "source_dataset": str(meta.get("source_dataset", meta.get("source", "unknown"))),
                    "quality_score": float(meta.get("quality_score", 1.0)),
                    "document_id": str(meta.get("document_id", "")),
                    "chunk_id": int(meta.get("chunk_id", 0)),
                    "timestamp": str(meta.get("timestamp", datetime.utcnow().isoformat() + "Z")),
                    "tokenizer_version": "v1" if self.tokenizer_path else "estimated"
                }
                clean_rows.append(row)

        logger.info(f"Buffered {len(clean_rows):,} clean unique sequences (Raw={raw_count:,}, Dups={exact_dups:,}, Toxic/QualityFiltered={toxic_count+quality_filtered_count:,}).")

        if not clean_rows:
            raise RuntimeError("DatasetAssembler: 0 sequences remaining after cleaning and deduplication.")

        # 2. Deterministic Splitting
        logger.info(f"\n[2/4] Splitting dataset ({len(clean_rows):,} sequences) into Train / Val / Test ({self.split_ratios})...")
        full_ds = Dataset.from_list(clean_rows)
        split_dict: DatasetDict = self.splitter.split(full_ds, seed=self.seed)

        # 3. Export Chunked Parquet Shards
        logger.info(f"\n[3/4] Exporting chunked Parquet shards to '{self.output_dir}'...")
        shard_checksums: Dict[str, Dict[str, Any]] = {}
        total_shards = 0

        for split_name, ds in split_dict.items():
            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            rows_in_split = len(ds)
            shard_idx = 0
            for start_i in range(0, rows_in_split, self.shard_size):
                end_i = min(start_i + self.shard_size, rows_in_split)
                shard_rows = [ds[i] for i in range(start_i, end_i)]
                shard_ds = Dataset.from_list(shard_rows)

                shard_filename = f"shard-{shard_idx:05d}.parquet"
                shard_path = os.path.join(split_dir, shard_filename)
                shard_ds.to_parquet(shard_path)

                val_meta = self._validate_shard_file(shard_path)
                shard_key = f"{split_name}/{shard_filename}"
                shard_checksums[shard_key] = {
                    "split": split_name,
                    "rows": len(shard_rows),
                    "size_bytes": val_meta["size_bytes"],
                    "sha256": val_meta["sha256_checksum"]
                }
                shard_idx += 1
                total_shards += 1
                logger.info(f"  -> Saved {shard_key} ({len(shard_rows):,} sequences, checksum={val_meta['sha256_checksum'][:12]}...)")

        # 4. Corpus Quality Evaluation & Report Generation
        logger.info("\n[4/4] Running comprehensive corpus validation & generating dataset cards...")
        report = self.validator.evaluate(split_dict, raw_count_before_dedup=raw_count)
        report["shard_checksums"] = shard_checksums

        meta_dir = os.path.join(self.output_dir, "metadata")
        card_path = os.path.join(self.output_dir, "README.md")
        saved_reports = self.validator.save_report(report, output_dir=meta_dir, dataset_card_path=card_path)

        # Write top-level manifest.json
        manifest_path = os.path.join(meta_dir, "manifest.json")
        manifest_data = {
            "dataset_name": "ManipuriGPT-Corpus-v1",
            "pipeline_version": "5.4",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "seed": self.seed,
            "total_sequences": len(clean_rows),
            "split_ratios": list(self.split_ratios),
            "total_shards": total_shards,
            "shard_size_limit": self.shard_size,
            "tokenizer_path": self.tokenizer_path or "estimated",
            "duration_sec": round(time.time() - start_t, 2),
            "reports": saved_reports
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info(" DATASET ASSEMBLY COMPLETE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Sequences Retained : {len(clean_rows):,}")
        logger.info(f"Total Parquet Shards     : {total_shards} across train/validation/test")
        logger.info(f"Manifest Checkpoint      : {manifest_path}")
        logger.info(f"Quality Report           : {saved_reports['corpus_report.json']}")
        logger.info(f"Dataset Card             : {saved_reports['README.md']}")
        logger.info("=" * 80)

        return manifest_data
