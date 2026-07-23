"""
BalancedCorpusSampler module for weighted, language-aware, and reproducible multi-source
streaming acquisition across Phase 5 pretraining corpora.
"""

import os
import json
import random
from typing import Iterator, Dict, Any, List, Optional, Union, Tuple
from app.corpus.sources import CorpusSourceSpec, SOURCE_REGISTRY, get_source_spec
from app.corpus.streamer import CorpusStreamer
from app.utils.logger import logger


class BalancedCorpusSampler:
    """
    Samples examples from multiple corpus streams using combined dataset and language weighting:
        P(sample) = (dataset_weight ^ (1/T)) * language_weight
    Supports deterministic seeds, probabilistic or round-robin modes, and continuous statistics logging.
    """
    def __init__(
        self,
        sources: Optional[List[Union[str, CorpusSourceSpec]]] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        language_weights: Optional[Dict[str, float]] = None,
        mode: str = "probabilistic",
        temperature: float = 1.0,
        seed: int = 42,
        stats_dir: str = "cache/statistics",
        stats_interval: int = 1000
    ):
        self.sources = sources or list(SOURCE_REGISTRY.keys())
        self.dataset_weights = dataset_weights or {
            "ai4bharat": 0.25,
            "wikipedia": 0.20,
            "opus": 0.15,
            "c4": 0.15,
            "fineweb": 0.15,
            "slimpajama": 0.10,
            "oscar": 0.10,
            "arxiv": 0.05,
            "pubmed": 0.05
        }
        self.language_weights = language_weights or {
            "mni": 1.5,
            "bn": 1.2,
            "hi": 1.0,
            "en": 0.8
        }
        self.mode = mode.lower()
        self.temperature = max(temperature, 0.01)
        self.seed = seed
        self.stats_dir = stats_dir
        self.stats_interval = stats_interval

        self.stats = {
            "total_yielded": 0,
            "total_bytes": 0,
            "total_chars": 0,
            "total_words": 0,
            "dataset_counts": {},
            "language_counts": {},
            "avg_chars": 0.0,
            "avg_words": 0.0
        }

    def _compute_source_probabilities(self, active_specs: List[CorpusSourceSpec]) -> List[float]:
        """
        Computes sampling probabilities based on dataset_weight * language_weight scaled by temperature.
        """
        probs = []
        for spec in active_specs:
            ds_weight = self.dataset_weights.get(spec.name, 0.1)
            # Estimate primary language weight from supported languages or default
            primary_lang = spec.supported_languages[0] if spec.supported_languages else "en"
            lang_weight = self.language_weights.get(primary_lang, 1.0)

            # Apply temperature scaling to combined probability
            raw_p = ds_weight * lang_weight
            probs.append(raw_p ** (1.0 / self.temperature))

        total_p = sum(probs)
        if total_p <= 0:
            return [1.0 / max(len(active_specs), 1)] * len(active_specs)
        return [p / total_p for p in probs]

    def save_stats(self) -> str:
        """Saves current acquisition statistics to corpus_stats.json and sampling_stats.json."""
        os.makedirs(self.stats_dir, exist_ok=True)
        corpus_path = os.path.join(self.stats_dir, "corpus_stats.json")
        sampling_path = os.path.join(self.stats_dir, "sampling_stats.json")

        total = max(self.stats["total_yielded"], 1)
        actual_ds_dist = {ds: round(count / total, 4) for ds, count in self.stats["dataset_counts"].items()}
        actual_lang_dist = {lang: round(count / total, 4) for lang, count in self.stats["language_counts"].items()}

        sampling_meta = {
            "seed": self.seed,
            "mode": self.mode,
            "temperature": self.temperature,
            "target_dataset_weights": self.dataset_weights,
            "target_language_weights": self.language_weights,
            "actual_dataset_distribution": actual_ds_dist,
            "actual_language_distribution": actual_lang_dist
        }

        try:
            with open(corpus_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
            with open(sampling_path, "w", encoding="utf-8") as f:
                json.dump(sampling_meta, f, indent=2)
        except Exception as e:
            logger.debug(f"BalancedCorpusSampler: Failed to save separated stats: {e}")
        return corpus_path

    def stream(
        self,
        min_length: int = 50,
        max_examples: Optional[int] = None,
        mock_fallback: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams balanced examples across all configured sources until max_examples is reached.
        """
        random.seed(self.seed)
        logger.info(f"BalancedCorpusSampler: Starting balanced stream (mode='{self.mode}', seed={self.seed}) across {len(self.sources)} sources.")

        # Initialize iterators for all sources
        active_sources: List[Tuple[CorpusSourceSpec, Iterator[Dict[str, Any]]]] = []
        for source_item in self.sources:
            spec = get_source_spec(source_item) if isinstance(source_item, str) else source_item
            if not spec:
                continue
            streamer = CorpusStreamer(
                source=spec,
                min_length=min_length,
                max_examples=max_examples,
                mock_fallback=mock_fallback
            )
            active_sources.append((spec, iter(streamer)))

        if not active_sources:
            logger.warning("BalancedCorpusSampler: No active source streams available.")
            return

        round_robin_idx = 0
        while active_sources:
            if max_examples is not None and self.stats["total_yielded"] >= max_examples:
                break

            if self.mode == "round_robin":
                selected_idx = round_robin_idx % len(active_sources)
                round_robin_idx += 1
            else:
                active_specs = [pair[0] for pair in active_sources]
                probs = self._compute_source_probabilities(active_specs)
                selected_idx = random.choices(range(len(active_sources)), weights=probs, k=1)[0]

            spec, iterator = active_sources[selected_idx]
            try:
                example = next(iterator)
            except StopIteration:
                # Remove exhausted source iterator
                active_sources.pop(selected_idx)
                if selected_idx < round_robin_idx and round_robin_idx > 0:
                    round_robin_idx -= 1
                continue
            except Exception as e:
                logger.warning(f"BalancedCorpusSampler: Error pulling from '{spec.name}' ({e}). Dropping stream.")
                active_sources.pop(selected_idx)
                continue

            # Update metrics
            text = example.get("text", "")
            if not text:
                continue

            char_len = len(text)
            word_len = len(text.split())
            byte_len = len(text.encode("utf-8"))
            lang = example.get("metadata", {}).get("language", spec.supported_languages[0] if spec.supported_languages else "en")

            self.stats["total_yielded"] += 1
            self.stats["total_bytes"] += byte_len
            self.stats["total_chars"] += char_len
            self.stats["total_words"] += word_len
            self.stats["dataset_counts"][spec.name] = self.stats["dataset_counts"].get(spec.name, 0) + 1
            self.stats["language_counts"][lang] = self.stats["language_counts"].get(lang, 0) + 1

            self.stats["avg_chars"] = round(self.stats["total_chars"] / max(self.stats["total_yielded"], 1), 1)
            self.stats["avg_words"] = round(self.stats["total_words"] / max(self.stats["total_yielded"], 1), 1)

            if self.stats["total_yielded"] % self.stats_interval == 0:
                self.save_stats()

            yield example

        self.save_stats()
        logger.info(f"BalancedCorpusSampler: Completed stream. Yielded {self.stats['total_yielded']} examples across {len(self.stats['dataset_counts'])} sources.")
