"""
CorpusAcquisitionManager module orchestrating multi-source dataset streams.
Enables parallel/sequential acquisition across multiple corpus sources.
"""

from typing import Iterator, Dict, Any, List, Optional, Union
from app.corpus.sources import CorpusSourceSpec, SOURCE_REGISTRY, get_source_spec
from app.corpus.streamer import CorpusStreamer
from app.utils.logger import logger


class CorpusAcquisitionManager:
    """
    Orchestrates streaming acquisition across dozens of corpus sources.
    Aggregates metrics including total yielded examples, total bytes processed, and source counts.
    """
    def __init__(self, sources: Optional[List[Union[str, CorpusSourceSpec]]] = None):
        if sources is None:
            self.sources = list(SOURCE_REGISTRY.keys())
        else:
            self.sources = sources
        self.stats = {
            "total_examples_yielded": 0,
            "total_bytes_yielded": 0,
            "source_counts": {}
        }

    def stream_all(
        self,
        min_length: int = 10,
        max_examples_per_source: Optional[int] = None,
        mock_fallback: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams examples sequentially across all registered or configured sources.
        """
        for source_item in self.sources:
            spec = get_source_spec(source_item) if isinstance(source_item, str) else source_item
            logger.info(f"CorpusAcquisitionManager: Starting stream for source '{spec.name}'")

            streamer = CorpusStreamer(
                source=spec,
                min_length=min_length,
                max_examples=max_examples_per_source,
                mock_fallback=mock_fallback
            )

            source_count = 0
            for example in streamer:
                source_count += 1
                self.stats["total_examples_yielded"] += 1
                self.stats["total_bytes_yielded"] += len(example["text"].encode("utf-8"))
                self.stats["source_counts"][spec.name] = source_count
                yield example

            logger.debug(f"CorpusAcquisitionManager: Streamed {source_count} examples from '{spec.name}'")

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns acquisition metrics summary.
        """
        return dict(self.stats)

    def list_available_sources(self) -> List[str]:
        """
        Returns sorted list of all available source canonical names.
        """
        return sorted(SOURCE_REGISTRY.keys())

    def get_source(self, name: str) -> CorpusSourceSpec:
        """
        Retrieves source specification by canonical name or alias.
        """
        return get_source_spec(name)

    def stream_source(
        self,
        source: Union[str, CorpusSourceSpec],
        min_length: int = 10,
        max_examples: Optional[int] = None,
        mock_fallback: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams examples sequentially from a single specified source.
        """
        spec = get_source_spec(source) if isinstance(source, str) else source
        logger.info(f"CorpusAcquisitionManager: Starting stream for source '{spec.name}'")

        streamer = CorpusStreamer(
            source=spec,
            min_length=min_length,
            max_examples=max_examples,
            mock_fallback=mock_fallback
        )

        source_count = 0
        for example in streamer:
            source_count += 1
            self.stats["total_examples_yielded"] += 1
            self.stats["total_bytes_yielded"] += len(example["text"].encode("utf-8"))
            self.stats["source_counts"][spec.name] = source_count
            yield example

        logger.debug(f"CorpusAcquisitionManager: Streamed {source_count} examples from '{spec.name}'")

    def stream_balanced(
        self,
        dataset_weights: Optional[Dict[str, float]] = None,
        language_weights: Optional[Dict[str, float]] = None,
        mode: str = "probabilistic",
        temperature: float = 1.0,
        seed: int = 42,
        min_length: int = 50,
        max_examples: Optional[int] = None,
        mock_fallback: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Delegates balanced multi-source acquisition to BalancedCorpusSampler.
        """
        from app.corpus.sampler import BalancedCorpusSampler
        sampler = BalancedCorpusSampler(
            sources=self.sources,
            dataset_weights=dataset_weights,
            language_weights=language_weights,
            mode=mode,
            temperature=temperature,
            seed=seed
        )
        return sampler.stream(
            min_length=min_length,
            max_examples=max_examples,
            mock_fallback=mock_fallback
        )
