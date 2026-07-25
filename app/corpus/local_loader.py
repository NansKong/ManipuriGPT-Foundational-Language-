"""
LocalCorpusLoader — Reads local files (JSONL, TXT, Parquet, Arrow/HF-cached)
and yields records in the same {"text": ..., "metadata": {...}} format used by
the rest of the ManipuriGPT ingestion pipeline.

Supports:
  - JSONL  (cache/processed/*.jsonl)
  - TXT    (monolingual line-by-line)
  - TSV    (bilingual tab-separated, extracts Manipuri side)
  - Parquet (Dayananda cached datasets)
  - Arrow  (AI4Bharat / Joyson HF-cached datasets via load_from_disk)

Phase 5.5 — Incremental Master Corpus Scaling.
"""

import os
import json
import hashlib
from typing import Iterator, Dict, Any, Optional, List
from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Per-source provenance configuration
# ---------------------------------------------------------------------------
# Standard canonical category mapping helper
CANONICAL_CATEGORIES = {
    "grammar": ["grammar", "grammar_book", "linguistics"],
    "textbook": ["textbook", "school_textbook", "curriculum", "education"],
    "dictionary": ["dictionary", "dict", "glossary", "lexicon"],
    "story": ["story", "stories", "children_story", "folktale"],
    "government": ["government", "official", "gazette", "pib", "pmi"],
    "news": ["news", "newspaper", "journalism", "article"],
    "literature": ["literature", "ema_lon", "classical", "poetry"],
    "books": ["book", "books", "novel", "prose"],
}


def normalize_category(raw_cat: str, default: str = "literature") -> str:
    """Normalizes raw category strings to canonical domain tags."""
    if not raw_cat or raw_cat == "unknown":
        return default
    cat_lower = raw_cat.lower().strip()
    for canon, keywords in CANONICAL_CATEGORIES.items():
        if any(kw in cat_lower for kw in keywords):
            return canon
    return default


LOCAL_SOURCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "local_processed_pdfs": {
        "description": "OCR'd Manipuri PDFs (Books, Textbooks, Dictionaries, Literature)",
        "license": "To be determined",
        "year": 2024,
        "default_category": "books",
        "ocr_confidence_min": 50.0,
        "quality_min": 65.0,
    },
    "local_ema_lon_mono": {
        "description": "EMA Lon monolingual Manipuri corpus",
        "license": "CC BY-NC 4.0",
        "year": 2024,
        "default_category": "literature",
    },
    "local_ema_lon_parallel": {
        "description": "EMA Lon bilingual Manipuri-English corpus",
        "license": "CC BY-NC 4.0",
        "year": 2024,
        "default_category": "literature",
    },
    "local_sangraha_cached": {
        "description": "AI4Bharat Sangraha Manipuri (local HF cache)",
        "license": "CC BY 4.0",
        "year": 2024,
        "default_category": "news",
    },
    "local_dayananda_meitei": {
        "description": "Dayananda Thokchom Meitei Mayek sample (local cache)",
        "license": "Various",
        "year": 2024,
        "default_category": "literature",
    },
    "local_dayananda_eng_to_meitei": {
        "description": "Dayananda Thokchom English-to-Meitei Mayek (local cache)",
        "license": "Various",
        "year": 2024,
        "default_category": "textbook",
    },
    "local_joyson_parallel": {
        "description": "Joyson English-Manipuri parallel corpus (local cache)",
        "license": "Various",
        "year": 2024,
        "default_category": "government",
    },
    "d_drive_manipuri_corpus_processed": {
        "description": "Manipuri Corpus (Books, Dictionaries, Literature, Textbooks from D:/manipuri corpus)",
        "license": "To be determined",
        "year": 2024,
        "default_category": "literature",
        "ocr_confidence_min": 40.0,
        "quality_min": 50.0,
        "metadata_dir": "D:/manipuri corpus/metadata",
    },
}


class LocalCorpusLoader:
    """
    Bridges local file data into the standard pipeline record format:
        {"text": "...", "metadata": {"source": ..., "category": ..., ...}}
    """

    def __init__(self, spec, min_text_length: int = 10):
        """
        Args:
            spec: A CorpusSourceSpec with source_type="local" and extra_configs
                  containing 'format' and 'text_column' overrides.
            min_text_length: Minimum character length to accept a record.
        """
        self.spec = spec
        self.min_text_length = min_text_length
        self.source_cfg = LOCAL_SOURCE_CONFIGS.get(spec.name, {})

        # Resolve format and path from spec
        self.file_format = spec.extra_configs.get("format", "jsonl")
        self.text_column = spec.default_text_column
        self.dataset_path = spec.dataset_path
        self.yielded = 0
        self._doc_metadata_cache: Dict[str, Dict[str, Any]] = {}

    def _load_sidecar_metadata(self, doc_key: str) -> Dict[str, Any]:
        """Loads companion document metadata from sidecar JSON files if present."""
        if doc_key in self._doc_metadata_cache:
            return self._doc_metadata_cache[doc_key]

        meta_dir = self.source_cfg.get("metadata_dir")
        if not meta_dir or not os.path.isdir(meta_dir):
            self._doc_metadata_cache[doc_key] = {}
            return {}

        meta_path = os.path.join(meta_dir, f"{doc_key}.json")
        if not os.path.isfile(meta_path):
            self._doc_metadata_cache[doc_key] = {}
            return {}

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._doc_metadata_cache[doc_key] = data
                return data
        except Exception as e:
            logger.debug(f"LocalCorpusLoader: Error reading sidecar metadata {meta_path}: {e}")
            self._doc_metadata_cache[doc_key] = {}
            return {}

    def stream(self) -> Iterator[Dict[str, Any]]:
        """Main entry point — dispatches to format-specific reader."""
        fmt = self.file_format.lower()
        logger.info(
            f"LocalCorpusLoader: Streaming '{self.spec.name}' "
            f"(format={fmt}, path={self.dataset_path})"
        )

        if fmt == "jsonl":
            yield from self._stream_jsonl()
        elif fmt == "txt":
            yield from self._stream_txt()
        elif fmt == "tsv":
            yield from self._stream_tsv()
        elif fmt == "parquet":
            yield from self._stream_parquet()
        elif fmt in ("arrow", "hf_cache"):
            yield from self._stream_arrow()
        else:
            logger.warning(f"LocalCorpusLoader: Unknown format '{fmt}' for '{self.spec.name}'")
            return

        logger.info(
            f"LocalCorpusLoader: Finished '{self.spec.name}'. "
            f"Yielded {self.yielded:,} records."
        )

    # ------------------------------------------------------------------
    # JSONL reader  (cache/processed/*.jsonl)
    # ------------------------------------------------------------------
    def _stream_jsonl(self) -> Iterator[Dict[str, Any]]:
        path = self.dataset_path
        if os.path.isdir(path):
            files = sorted(
                f for f in os.listdir(path)
                if f.endswith(".jsonl")
            )
            filepaths = [os.path.join(path, f) for f in files]
        elif os.path.isfile(path):
            filepaths = [path]
        else:
            logger.warning(f"LocalCorpusLoader: JSONL path not found: {path}")
            return

        ocr_conf_min = self.source_cfg.get("ocr_confidence_min", 0.0)
        quality_min = self.source_cfg.get("quality_min", 0.0)
        skipped_quality = 0

        for fp in filepaths:
            fname = os.path.basename(fp)
            doc_key = fname.split('.')[0]
            sidecar_meta = self._load_sidecar_metadata(doc_key)

            with open(fp, "r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = rec.get("text", "").strip()
                    if len(text) < self.min_text_length:
                        continue

                    # OCR quality gate
                    ocr_conf = rec.get("ocr_confidence", sidecar_meta.get("avg_ocr_confidence"))
                    quality = rec.get("quality", sidecar_meta.get("avg_quality_score"))
                    if ocr_conf is not None and ocr_conf < ocr_conf_min:
                        skipped_quality += 1
                        continue
                    if quality is not None and quality < quality_min:
                        skipped_quality += 1
                        continue

                    category_raw = rec.get("category") or sidecar_meta.get("category")
                    script_raw = rec.get("script") or sidecar_meta.get("actual_script") or sidecar_meta.get("script") or "unknown"
                    license_raw = rec.get("license") or sidecar_meta.get("license") or self.source_cfg.get("license", "Various")

                    metadata = {
                        "source": self.spec.name,
                        "source_dataset": rec.get("source") or sidecar_meta.get("source") or self.source_cfg.get("description", self.spec.name),
                        "category": normalize_category(category_raw, self.source_cfg.get("default_category", "literature")),
                        "script": script_raw,
                        "license": license_raw,
                        "year": sidecar_meta.get("year", self.source_cfg.get("year", 2024)),
                        "language": rec.get("language") or sidecar_meta.get("actual_language") or "mni",
                        "document_id": rec.get("id", f"{fname}_{line_no}"),
                        "quality_score": float(quality) / 100.0 if quality else 1.0,
                        "ocr_engine": rec.get("ocr_engine") or sidecar_meta.get("processing_method"),
                        "original_source": rec.get("source") or sidecar_meta.get("source") or "unknown",
                        "title": sidecar_meta.get("title"),
                        "author": sidecar_meta.get("author"),
                    }

                    self.yielded += 1
                    yield {"text": text, "metadata": metadata}

        if skipped_quality:

            logger.info(
                f"LocalCorpusLoader: Skipped {skipped_quality} JSONL records "
                f"below quality/OCR threshold (ocr_conf>={ocr_conf_min}, quality>={quality_min})"
            )

    # ------------------------------------------------------------------
    # TXT reader  (monolingual line-by-line)
    # ------------------------------------------------------------------
    def _stream_txt(self) -> Iterator[Dict[str, Any]]:
        path = self.dataset_path
        if not os.path.isfile(path):
            logger.warning(f"LocalCorpusLoader: TXT file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                text = line.strip()
                if len(text) < self.min_text_length:
                    continue

                metadata = {
                    "source": self.spec.name,
                    "source_dataset": self.source_cfg.get("description", self.spec.name),
                    "category": self.source_cfg.get("default_category", "monolingual"),
                    "script": "Beng",  # Will be re-detected by pipeline
                    "license": self.source_cfg.get("license", "Various"),
                    "year": self.source_cfg.get("year", 2024),
                    "language": "mni",
                    "document_id": f"ema_mono_{line_no:06d}",
                    "quality_score": 1.0,
                }

                self.yielded += 1
                yield {"text": text, "metadata": metadata}

    # ------------------------------------------------------------------
    # TSV reader  (bilingual tab-separated — extracts Manipuri side)
    # ------------------------------------------------------------------
    def _stream_tsv(self) -> Iterator[Dict[str, Any]]:
        path = self.dataset_path
        if not os.path.isfile(path):
            logger.warning(f"LocalCorpusLoader: TSV file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue

                # Try tab-separated first, then fall back to line as-is
                parts = line.split("\t")
                if len(parts) >= 2:
                    manipuri_text = parts[0].strip()
                    english_text = parts[1].strip()
                else:
                    manipuri_text = line
                    english_text = ""

                if len(manipuri_text) < self.min_text_length:
                    continue

                metadata = {
                    "source": self.spec.name,
                    "source_dataset": self.source_cfg.get("description", self.spec.name),
                    "category": self.source_cfg.get("default_category", "parallel_corpus"),
                    "script": "Beng",  # Will be re-detected by pipeline
                    "license": self.source_cfg.get("license", "Various"),
                    "year": self.source_cfg.get("year", 2024),
                    "language": "mni",
                    "document_id": f"ema_parallel_{line_no:06d}",
                    "quality_score": 1.0,
                }

                # Preserve English side in metadata for future fine-tuning
                if english_text:
                    metadata["english_parallel"] = english_text

                self.yielded += 1
                yield {"text": manipuri_text, "metadata": metadata}

    # ------------------------------------------------------------------
    # Parquet reader  (Dayananda cached datasets)
    # ------------------------------------------------------------------
    def _stream_parquet(self) -> Iterator[Dict[str, Any]]:
        path = self.dataset_path
        parquet_files: List[str] = []

        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for f in files:
                    if f.endswith(".parquet"):
                        parquet_files.append(os.path.join(root, f))
        elif os.path.isfile(path) and path.endswith(".parquet"):
            parquet_files = [path]
        else:
            logger.warning(f"LocalCorpusLoader: Parquet path not found: {path}")
            return

        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("LocalCorpusLoader: pyarrow not installed. Cannot read Parquet.")
            return

        text_col = self.text_column

        for pf in sorted(parquet_files):
            table = pq.read_table(pf)
            columns = table.column_names

            # Determine text column
            actual_text_col = text_col if text_col in columns else None
            if not actual_text_col:
                # Try common alternatives
                for candidate in ["meitei_mayek", "text", "Manipuri", "mni"]:
                    if candidate in columns:
                        actual_text_col = candidate
                        break

            if not actual_text_col:
                logger.warning(
                    f"LocalCorpusLoader: No text column found in {pf}. "
                    f"Available: {columns}"
                )
                continue

            for row_idx in range(table.num_rows):
                text_val = table.column(actual_text_col)[row_idx].as_py()
                if not text_val or not isinstance(text_val, str):
                    continue
                text = text_val.strip()
                if len(text) < self.min_text_length:
                    continue

                metadata = {
                    "source": self.spec.name,
                    "source_dataset": self.source_cfg.get("description", self.spec.name),
                    "category": self.source_cfg.get("default_category", "unknown"),
                    "script": "Mtei",  # Will be re-detected by pipeline
                    "license": self.source_cfg.get("license", "Various"),
                    "year": self.source_cfg.get("year", 2024),
                    "language": "mni",
                    "document_id": f"{self.spec.name}_{row_idx:08d}",
                    "quality_score": 1.0,
                }

                # Preserve parallel English if present
                for eng_col in ["english", "English", "eng", "en"]:
                    if eng_col in columns:
                        eng_val = table.column(eng_col)[row_idx].as_py()
                        if eng_val and isinstance(eng_val, str):
                            metadata["english_parallel"] = eng_val.strip()
                        break

                self.yielded += 1
                yield {"text": text, "metadata": metadata}

    # ------------------------------------------------------------------
    # Arrow / HF cache reader  (load_from_disk or iterate arrow files)
    # ------------------------------------------------------------------
    def _stream_arrow(self) -> Iterator[Dict[str, Any]]:
        path = self.dataset_path

        if not os.path.isdir(path):
            logger.warning(f"LocalCorpusLoader: Arrow/HF cache path not found: {path}")
            return

        dataset = None
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(path)
            logger.info(f"LocalCorpusLoader: Loaded HF dataset from disk: {path}")
        except Exception:
            pass

        if dataset is None:
            # Try loading arrow files directly
            try:
                from datasets import Dataset as HFDataset
                import pyarrow as pa

                arrow_files = []
                for root, _dirs, files in os.walk(path):
                    for f in files:
                        if f.endswith(".arrow"):
                            arrow_files.append(os.path.join(root, f))

                if not arrow_files:
                    logger.warning(f"LocalCorpusLoader: No .arrow files found in {path}")
                    return

                # Concatenate arrow tables
                tables = []
                for af in sorted(arrow_files):
                    reader = pa.ipc.open_file(af)
                    tables.append(reader.read_all())

                combined = pa.concat_tables(tables)
                dataset = HFDataset(combined)
                logger.info(
                    f"LocalCorpusLoader: Loaded {len(dataset)} records from "
                    f"{len(arrow_files)} arrow files in {path}"
                )
            except Exception as e:
                logger.error(f"LocalCorpusLoader: Failed to load arrow data from {path}: {e}")
                return

        text_col = self.text_column
        columns = dataset.column_names if hasattr(dataset, "column_names") else []

        # Auto-detect text column
        actual_text_col = text_col if text_col in columns else None
        if not actual_text_col:
            for candidate in ["text", "Manipuri", "meitei_mayek", "mni", "sentence"]:
                if candidate in columns:
                    actual_text_col = candidate
                    break

        if not actual_text_col:
            logger.warning(
                f"LocalCorpusLoader: No text column found in arrow dataset. "
                f"Available: {columns}"
            )
            return

        for row_idx, row in enumerate(dataset):
            text_val = row.get(actual_text_col, "")
            if isinstance(text_val, dict):
                # Handle translation dicts
                text_val = text_val.get("mni", text_val.get("bn", text_val.get("en", "")))
            if not text_val or not isinstance(text_val, str):
                continue
            text = text_val.strip()
            if len(text) < self.min_text_length:
                continue

            metadata = {
                "source": self.spec.name,
                "source_dataset": self.source_cfg.get("description", self.spec.name),
                "category": self.source_cfg.get("default_category", "unknown"),
                "script": "unknown",  # Will be re-detected by pipeline
                "license": self.source_cfg.get("license", "Various"),
                "year": self.source_cfg.get("year", 2024),
                "language": row.get("language", "mni"),
                "document_id": f"{self.spec.name}_{row_idx:08d}",
                "quality_score": 1.0,
            }

            # Preserve parallel English
            for eng_col in ["english", "English", "eng", "en"]:
                if eng_col in row and row[eng_col]:
                    eng_val = row[eng_col]
                    if isinstance(eng_val, str):
                        metadata["english_parallel"] = eng_val.strip()
                    break

            # Preserve any source-specific metadata
            for meta_col in ["source", "script", "url", "title"]:
                if meta_col in row and row[meta_col] and meta_col != actual_text_col:
                    if meta_col == "source":
                        metadata["original_source"] = str(row[meta_col])
                    else:
                        metadata[meta_col] = str(row[meta_col])

            self.yielded += 1
            yield {"text": text, "metadata": metadata}
