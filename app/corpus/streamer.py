"""
CorpusStreamer module for streaming dataset yields without downloading full archives to disk.
Supports Hugging Face streaming datasets (`streaming=True`) and local generator fallbacks.
"""

from typing import Iterator, Dict, Any, Optional, List, Union
from app.corpus.sources import CorpusSourceSpec, get_source_spec
from app.utils.logger import logger


class CorpusStreamer:
    """
    Streams dataset examples one by one using iterable streams.
    Filters examples on the fly by language, minimum text length, or script requirements.
    """
    def __init__(
        self,
        source: Union[str, CorpusSourceSpec],
        min_length: int = 10,
        max_length: Optional[int] = None,
        allowed_languages: Optional[List[str]] = None,
        max_examples: Optional[int] = None,
        mock_fallback: bool = False
    ):
        if isinstance(source, str):
            self.spec = get_source_spec(source)
        else:
            self.spec = source

        self.min_length = min_length
        self.max_length = max_length
        self.allowed_languages = allowed_languages or self.spec.supported_languages
        self.max_examples = max_examples
        self.mock_fallback = mock_fallback
        self.streamed_count = 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields filtered examples dictionary containing text and metadata.
        """
        self.streamed_count = 0
        raw_stream = self._get_raw_stream()

        try:
            for example in raw_stream:
                if self.max_examples is not None and self.streamed_count >= self.max_examples:
                    break

                processed = self._extract_and_filter(example)
                if processed is not None:
                    self.streamed_count += 1
                    yield processed
        finally:
            if hasattr(raw_stream, "close") and callable(raw_stream.close):
                try:
                    raw_stream.close()
                except Exception:
                    pass

    def _get_raw_stream(self) -> Iterator[Dict[str, Any]]:
        """
        Retrieves raw dataset stream from HuggingFace datasets or mock fallback.
        """
        import os
        if self.mock_fallback and (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("MANIPURIGPT_OFFLINE") == "1"):
            return self._generate_mock_stream()

        if self.spec.source_type == "local":
            from app.corpus.local_loader import LocalCorpusLoader
            loader = LocalCorpusLoader(self.spec, min_text_length=self.min_length)
            return loader.stream()

        if self.spec.source_type == "hf":
            try:
                from app.utils.download import hf_load_dataset_with_backoff, hf_stream_with_backoff
                from app.utils.cache import setup_cache_directories
                dirs = setup_cache_directories()
                strategy = getattr(self.spec, "caching_strategy", "stream")

                if strategy == "local_cache":
                    logger.info(f"CorpusStreamer: Using local_cache (non-streaming) for '{self.spec.dataset_path}' -> {dirs['datasets']}")
                    dataset = hf_load_dataset_with_backoff(
                        self.spec.dataset_path,
                        self.spec.subset,
                        split=self.spec.split,
                        streaming=False,
                        cache_dir=dirs["datasets"]
                    )
                    return iter(dataset)
                elif strategy == "shard_prefetch":
                    logger.info(f"CorpusStreamer: Using shard_prefetch for '{self.spec.dataset_path}' -> {dirs['shards']}")
                    try:
                        dataset = hf_load_dataset_with_backoff(
                            self.spec.dataset_path,
                            self.spec.subset,
                            split=f"{self.spec.split}[:2%]",
                            streaming=False,
                            cache_dir=dirs["shards"]
                        )
                        return iter(dataset)
                    except Exception as prefetch_err:
                        logger.warning(f"CorpusStreamer: Shard prefetch failed ({prefetch_err}). Falling back to live stream.")

                # Default network stream
                dataset = hf_load_dataset_with_backoff(
                    self.spec.dataset_path,
                    self.spec.subset,
                    split=self.spec.split,
                    streaming=True
                )
                return hf_stream_with_backoff(dataset)
            except Exception as e:
                logger.warning(
                    f"CorpusStreamer: Failed to connect to HuggingFace stream '{self.spec.dataset_path}' "
                    f"({e}). Using mock fallback={self.mock_fallback}."
                )
                if not self.mock_fallback:
                    raise

        return self._generate_mock_stream()

    def _generate_mock_stream(self) -> Iterator[Dict[str, Any]]:
        """
        Generates realistic mock dataset examples for offline/test environments.
        """
        samples = [
            {
                self.spec.default_text_column: "Manipuri is a Sino-Tibetan language spoken predominantly in Manipur, India.",
                "language": "en",
                "url": "https://example.org/manipuri_overview"
            },
            {
                self.spec.default_text_column: "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯃꯅꯤꯄꯨꯔꯒꯤ ꯃꯔꯨꯑꯣꯏꯕ ꯂꯣꯟꯅꯤ꯫ ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ",
                "language": "mni",
                "script": "meitei",
                "url": "https://example.org/meitei_article"
            },
            {
                self.spec.default_text_column: "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯃꯅꯤꯄꯨꯔꯒꯤ ꯃꯔꯨꯑꯣꯏꯕ ꯂꯣꯟꯅꯤ꯫ ꯑꯅꯤꯁꯨꯕ ꯄꯔꯦꯡ꯫",
                "language": "mni",
                "script": "meitei",
                "url": "https://example.org/meitei_article_2"
            },
            {
                self.spec.default_text_column: "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯃꯅꯤꯄꯨꯔꯒꯤ ꯃꯔꯨꯑꯣꯏꯕ ꯂꯣꯟꯅꯤ꯫ ꯑꯍꯨꯝꯁꯨꯕ ꯄꯔꯦꯡ꯫",
                "language": "mni",
                "script": "meitei",
                "url": "https://example.org/meitei_article_3"
            },
            {
                self.spec.default_text_column: "মণিপুরী ভাষা ভারতের একটি অন্যতম প্রধান ভাষা এবং সংবিধানের অষ্টম তফসিলে অন্তর্ভুক্ত।",
                "language": "bn",
                "script": "bengali",
                "url": "https://example.org/bengali_manipuri"
            },
            {
                self.spec.default_text_column: "Short",
                "language": "en",
                "url": "https://example.org/short"
            }
        ]
        # Yield samples that pass language constraints up to max_examples
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            for sample in samples:
                attempts += 1
                yield sample
                if self.max_examples is not None and self.streamed_count >= self.max_examples:
                    return
            if self.max_examples is None:
                break

    def _extract_and_filter(self, raw_example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extracts text column and validates against length and language filters.
        """
        text = raw_example.get(self.spec.default_text_column, "")
        metadata = {col: raw_example[col] for col in self.spec.metadata_columns if col in raw_example}
        metadata["source"] = self.spec.name

        if isinstance(text, dict):
            metadata["translation"] = text
            text = text.get("mni", text.get("bn", text.get("en", next(iter(text.values()), ""))))
        elif not isinstance(text, str):
            return None

        text_clean = text.strip()
        if len(text_clean) < self.min_length:
            return None
        if self.max_length is not None and len(text_clean) > self.max_length:
            return None

        # Check language if explicitly tagged in example
        ex_lang = raw_example.get("language")
        if ex_lang and self.allowed_languages and ex_lang not in self.allowed_languages:
            return None

        if ex_lang:
            metadata["language"] = ex_lang

        return {
            "text": text_clean,
            "metadata": metadata
        }
