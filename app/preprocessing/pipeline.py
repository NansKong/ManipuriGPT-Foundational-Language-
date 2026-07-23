import os
import yaml
import json
from datetime import datetime
from typing import List, Dict, Any, Union, Iterator, Optional, Tuple
from datasets import Dataset, DatasetDict, IterableDataset

from app.preprocessing.normalizer import UnicodeNormalizer
from app.preprocessing.cleaner import TextCleaner
from app.preprocessing.script_detector import ScriptDetector
from app.preprocessing.transliteration import ScriptCanonicalizer
from app.preprocessing.language_detector import LanguageDetector
from app.preprocessing.validator import SentenceValidator
from app.preprocessing.sentence_filter import SentenceFilter
from app.preprocessing.deduplicator import Deduplicator
from app.preprocessing.statistics import StatisticsTracker
from app.preprocessing.splitter import DatasetSplitter
from app.preprocessing.exporters import DatasetExporter
from app.preprocessing.pii_remover import PIIRemover
from app.preprocessing.quality_scorer import QualityScorer, ToxicityFilter
from app.preprocessing.minhash_deduplicator import MinHashDeduplicator
from app.preprocessing.chunker import SequenceChunker
from app.preprocessing.metadata_types import DocumentMetadata
from app.utils.logger import logger


class PreprocessingPipeline:
    """
    Orchestrator for the entire preprocessing pipeline.
    Loads configs, runs normalization/cleaning, language/script verification,
    filtering, deduplication, collects stats, splits and exports datasets.
    """
    def __init__(self, config_path: Union[str, Dict[str, Any], None] = None, config: Optional[Dict[str, Any]] = None):
        if config is not None:
            self.config = config
        elif isinstance(config_path, dict):
            self.config = config_path
        else:
            if config_path is None:
                config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "..", "configs", "preprocessing.yaml"
                )
            self.config = self._load_config(config_path)
        p_cfg = self.config.get("preprocessing", {})

        # Instantiate independent blocks
        self.normalizer = UnicodeNormalizer(p_cfg.get("unicode", {}))
        self.cleaner = TextCleaner(p_cfg.get("cleaning", {}))
        self.script_detector = ScriptDetector(p_cfg.get("script_detection", {}))
        self.canonicalizer = ScriptCanonicalizer(p_cfg.get("canonicalization", {}))
        self.language_detector = LanguageDetector(p_cfg.get("language_detection", {}))
        self.validator = SentenceValidator(p_cfg.get("validation", {}))
        self.sentence_filter = SentenceFilter(p_cfg.get("filtering", {}))
        self.deduplicator = Deduplicator(p_cfg.get("deduplication", {}))
        self.splitter = DatasetSplitter(p_cfg.get("splitting", {}))
        self.exporter = DatasetExporter(p_cfg.get("export", {}))
        
        # Phase 5 components
        self.pii_remover = PIIRemover(
            mask_replacement=p_cfg.get("pii", {}).get("mask_replacement", "<PII>"),
            remove_pii=p_cfg.get("pii", {}).get("remove_pii", False)
        )
        self.quality_scorer = QualityScorer(
            min_score=p_cfg.get("quality", {}).get("min_score", 0.45)
        )
        self.toxicity_filter = ToxicityFilter()
        self.minhash_dedup = MinHashDeduplicator(
            similarity_threshold=p_cfg.get("minhash_dedup", {}).get("similarity_threshold", 0.85)
        )
        self.chunker = SequenceChunker(
            max_chunk_size=p_cfg.get("chunking", {}).get("max_chunk_size", 2048),
            chunk_overlap=p_cfg.get("chunking", {}).get("chunk_overlap", 128)
        )
        
        self.stats = StatisticsTracker()
        try:
            from app.configs.loader import compute_config_hash
            self._config_hash = compute_config_hash()
        except Exception:
            self._config_hash = "5.3_default"

    def reset(self) -> None:
        """Resets the deduplication and statistics state."""
        self.deduplicator.reset()
        if hasattr(self.minhash_dedup, "reset"):
            self.minhash_dedup.reset()
        self.stats = StatisticsTracker()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Loads configuration from YAML file."""
        if not os.path.exists(path):
            logger.warning(f"Pipeline config not found at {path}. Using default configuration.")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def run(self, dataset: Dataset, text_keys: List[str]) -> Dataset:
        """
        Runs the cleaning and validation pipeline on a HuggingFace Dataset.
        """
        logger.info(f"Pipeline: Starting processing on dataset with {len(dataset)} samples. Text columns: {text_keys}")
        
        # Reset tracking/stats state
        self.deduplicator.reset()
        self.stats = StatisticsTracker()

        # Define batched mapping function for HF Dataset map
        def process_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            # Initialize empty lists for all columns in the batch
            processed_batch = {k: [] for k in batch.keys()}
            num_samples = len(next(iter(batch.values())))

            for i in range(num_samples):
                sample_is_valid = True
                cleaned_texts = {}

                # 1. First extract and clean text for all specified keys
                for key in text_keys:
                    val = batch[key][i]
                    if val is None or not isinstance(val, str):
                        # Empty/None values
                        self.stats.empty_removed += 1
                        sample_is_valid = False
                        break

                    # Unicode Normalization
                    norm_text = self.normalizer.normalize(val)
                    self.stats.record_unicode_fix(val, norm_text)

                    # Cleaner
                    clean_text = self.cleaner.clean(norm_text)
                    self.stats.record_cleaner_fix(norm_text, clean_text)

                    # Sentence Quality Validator
                    if not self.validator.validate(clean_text):
                        if not clean_text.strip():
                            self.stats.empty_removed += 1
                        elif '\uFFFD' in clean_text:
                            self.stats.invalid_unicode_removed += 1
                        elif self.validator.punctuation_only_regex.match(clean_text):
                            self.stats.only_punctuation_removed += 1
                        elif self.validator.digits_only_regex.match(clean_text):
                            self.stats.only_numbers_removed += 1
                        else:
                            self.stats.repeated_chars_removed += 1
                        sample_is_valid = False
                        break

                    # Length Filtering
                    if not self.sentence_filter.filter(clean_text):
                        self.stats.length_filtered_removed += 1
                        sample_is_valid = False
                        break

                    # Script Detection
                    script_info = self.script_detector.detect(clean_text)
                    self.stats.record_script(script_info["script"])
                    
                    target_script = self.script_detector.target_script
                    if target_script != "any" and script_info["script"] != target_script:
                        # Script mismatch
                        sample_is_valid = False
                        break

                    # Language Detection
                    lang_info = self.language_detector.detect(clean_text)
                    self.stats.record_language(lang_info["language"])

                    target_lang = self.language_detector.config.get("target_language", "any")
                    min_conf = self.language_detector.min_confidence
                    if target_lang != "any" and lang_info["language"] != target_lang:
                        # Language mismatch
                        sample_is_valid = False
                        break
                    if target_lang != "any" and lang_info["confidence"] < min_conf:
                        # Low confidence language
                        sample_is_valid = False
                        break

                    # Deduplication
                    if self.deduplicator.is_duplicate(clean_text):
                        self.stats.duplicates_removed += 1
                        sample_is_valid = False
                        break

                    # Canonicalization (after cleaning, validation, filtering, script/lang detection, and deduplication)
                    clean_text, _ = self.canonicalizer.process_text(clean_text)
                    cleaned_texts[key] = clean_text

                if sample_is_valid:
                    # Update row values
                    for k in batch.keys():
                        if k in text_keys:
                            processed_batch[k].append(cleaned_texts[k])
                        else:
                            processed_batch[k].append(batch[k][i])
                    
                    # Track statistics for accepted sample
                    # Use the first text key for final lengths tracking
                    if text_keys:
                        self.stats.record_final_sample(cleaned_texts[text_keys[0]])

            return processed_batch

        # Run mapping in batched mode (which allows filtering out bad samples)
        processed_ds = dataset.map(
            process_batch, 
            batched=True, 
            remove_columns=None,  # Keep columns
            desc="Running preprocessing pipeline"
        )
        
        logger.info(f"Pipeline: Completed. Cleaned dataset has {len(processed_ds)} samples.")
        return processed_ds

    def _generate_preprocessing_metadata(self) -> Dict[str, Any]:
        p_cfg = self.config.get("preprocessing", {})
        
        # Unicode
        uni_cfg = p_cfg.get("unicode", {})
        if uni_cfg.get("enabled", True):
            unicode_meta = uni_cfg.get("form", "NFC")
        else:
            unicode_meta = "disabled"
            
        # Deduplication
        dedup_cfg = p_cfg.get("deduplication", {})
        if dedup_cfg.get("enabled", True):
            active = []
            if dedup_cfg.get("exact", True):
                active.append("exact")
            if dedup_cfg.get("normalized", True):
                active.append("normalized")
            if dedup_cfg.get("fuzzy", True):
                active.append("fuzzy")
            dedup_meta = "+".join(active) if active else "none"
        else:
            dedup_meta = "disabled"
            
        # Language Detector
        lang_cfg = p_cfg.get("language_detection", {})
        if lang_cfg.get("enabled", True):
            lang_meta = lang_cfg.get("detector_type", "langdetect")
        else:
            lang_meta = "disabled"
            
        return {
            "unicode": unicode_meta,
            "deduplication": dedup_meta,
            "language_detector": lang_meta
        }

    def process_split_export(
        self, 
        dataset: Dataset, 
        text_keys: List[str], 
        dataset_name: str = "dataset",
        report_path: str = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Runs pipeline, splits the processed dataset, generates stats reports, and exports.
        """
        # 1. Preprocess
        processed_ds = self.run(dataset, text_keys)

        # 2. Split
        final_dataset = self.splitter.split(processed_ds)

        # 3. Save reports automatically
        # Save auto reports in reports/ folder at the root
        os.makedirs("reports", exist_ok=True)
        self.stats.save_markdown_report(os.path.join("reports", "corpus_report.md"))
        self.stats.save_json_report(os.path.join("reports", "statistics.json"))
        logger.info("Pipeline: Automatically generated reports/corpus_report.md and reports/statistics.json")

        if report_path:
            self.stats.save_markdown_report(report_path)
            logger.info(f"Pipeline: Saved execution report to {report_path}")

        # 4. Export
        self.exporter.export(final_dataset)

        # 5. Manifest Generation
        resolved_out_dir = self.exporter.get_resolved_output_dir()
        manifest_path = os.path.join(resolved_out_dir, "manifest.json")
        manifest = {
            "dataset": dataset_name,
            "version": self.exporter.version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "preprocessing": self._generate_preprocessing_metadata(),
            "samples": self.stats.final_count
        }
        try:
            os.makedirs(resolved_out_dir, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Pipeline: Saved dataset manifest to {manifest_path}")
        except Exception as e:
            logger.error(f"Pipeline: Failed to save dataset manifest: {e}")
        return final_dataset

    def process_example(self, example: Dict[str, Any], chunk: bool = True) -> List[Dict[str, Any]]:
        """
        Processes a single dataset dictionary example through the Phase 5 pipeline order:
        Raw Data -> Cleaning -> Exact Dedup -> Language Detection -> Quality/Toxicity -> MinHash Dedup -> PII -> Canonicalization -> Chunking.
        Returns a list of chunked/processed examples (or empty list if filtered out).
        
        Optimization notes:
        - Exact dedup is done BEFORE language detection (cheap O(1) check short-circuits expensive detection)
        - Script detection is done once and reused by language detector
        - target_lang is cached on self to avoid per-example dict lookup
        """
        text = example.get("text", "")
        if not text and "translation" in example and isinstance(example["translation"], dict):
            trans_dict = example["translation"]
            text = trans_dict.get("mni", trans_dict.get("bn", trans_dict.get("en", "")))
        if not text or not isinstance(text, str):
            return []

        # 1. Unicode & Cleaning (canonicalization moved after deduplication)
        norm_text = self.normalizer.normalize(text)
        clean_text = self.cleaner.clean(norm_text)
        if not clean_text.strip() or not self.validator.validate(clean_text):
            return []

        # 2. EARLY exact dedup check (O(1) set lookup — before expensive lang/quality detection)
        if self.deduplicator.is_duplicate(clean_text):
            return []

        # 3. Script Detection (done once, reused by language detector)
        script_info = self.script_detector.detect(clean_text)

        # 4. Language Detection (uses script_info internally for Meitei heuristic)
        lang_info = self.language_detector.detect(clean_text)
        # Cache target_lang to avoid repeated config dict lookups
        if not hasattr(self, '_target_lang'):
            self._target_lang = self.language_detector.config.get("target_language", "any")
        target_lang = self._target_lang
        if target_lang != "any":
            if isinstance(target_lang, list) and lang_info["language"] not in target_lang:
                return []
            elif isinstance(target_lang, str) and target_lang != "any" and lang_info["language"] != target_lang:
                return []
        if lang_info["confidence"] < self.language_detector.min_confidence:
            return []

        # 5. Quality Scoring & Toxicity Filtering
        # Build dict in-place instead of example.copy() to reduce allocation
        metadata = {"language": lang_info["language"], "script": script_info["script"]}
        ex_dict = {"text": clean_text, "metadata": metadata}

        tox_res = self.toxicity_filter.filter_example(ex_dict, scorer=self.quality_scorer)
        if tox_res is None:
            return []

        # 6. PII Removal
        pii_res = self.pii_remover.process(tox_res)

        # 7. MinHash Deduplication (near-duplicate, LSH-based O(1) avg lookup)
        minhash_res = self.minhash_dedup.process_example(pii_res)
        if minhash_res is None:
            return []

        # 8. Canonicalization & Non-Destructive Multi-Script Tracking
        canon_text, canon_meta = self.canonicalizer.process_text(minhash_res.get("text", clean_text))
        minhash_res["text"] = canon_text

        # Enrich metadata before chunking using strongly typed DocumentMetadata
        raw_meta = minhash_res.get("metadata", {})
        doc_meta = DocumentMetadata.from_dict(raw_meta)
        doc_meta.language = raw_meta.get("language", lang_info["language"])
        doc_meta.script = raw_meta.get("script", script_info["script"])
        doc_meta.source = raw_meta.get("source", "unknown")
        doc_meta.source_dataset = raw_meta.get("source_dataset", doc_meta.source)
        doc_meta.document_id = raw_meta.get("document_id", f"doc_{hash(clean_text) & 0xffffffff:08x}")
        doc_meta.quality_score = float(raw_meta.get("quality_score", 1.0))
        doc_meta.timestamp = datetime.utcnow().isoformat() + "Z"
        doc_meta.tokenizer_version = "sentencepiece_unigram_32k"
        doc_meta.pipeline_version = "5.3"
        doc_meta.extra["original_text"] = canon_meta["original_text"]
        doc_meta.extra["canonical_text"] = canon_meta["canonical_text"]
        doc_meta.extra["canonicalization_mode"] = canon_meta["canonicalization_mode"]
        doc_meta.config_hash = getattr(self, "_config_hash", "5.3_default")

        minhash_res["metadata"] = doc_meta.to_dict()

        # 9. Chunking
        if chunk:
            chunked = self.chunker.process_example(minhash_res)
            total = len(chunked)
            for idx, c in enumerate(chunked):
                cmeta = DocumentMetadata.from_dict(c["metadata"])
                cmeta.chunk_id = idx
                cmeta.total_chunks = total
                c["metadata"] = cmeta.to_dict()
            return chunked

        doc_meta.chunk_id = 0
        doc_meta.total_chunks = 1
        minhash_res["metadata"] = doc_meta.to_dict()
        return [minhash_res]

    def process_stream(self, example_stream: Iterator[Dict[str, Any]], chunk: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Processes an iterable/generator stream of dataset examples on the fly.
        """
        for example in example_stream:
            processed_chunks = self.process_example(example, chunk=chunk)
            for chunk_ex in processed_chunks:
                yield chunk_ex


class PreprocessingWorker:
    """
    Abstract base interface for parallel or distributed preprocessing workers.
    Allows swapping in ThreadPool, ProcessPool, or Ray backends without changing pipeline interfaces.
    """
    def __init__(self, pipeline: Optional[PreprocessingPipeline] = None):
        self.pipeline = pipeline or PreprocessingPipeline()

    def process_batch(self, batch: List[Dict[str, Any]], chunk: bool = True) -> List[Dict[str, Any]]:
        raise NotImplementedError


class SequentialPreprocessingWorker(PreprocessingWorker):
    """
    Standard sequential in-process worker implementation of PreprocessingWorker.
    """
    def process_batch(self, batch: List[Dict[str, Any]], chunk: bool = True) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for example in batch:
            results.extend(self.pipeline.process_example(example, chunk=chunk))
        return results


