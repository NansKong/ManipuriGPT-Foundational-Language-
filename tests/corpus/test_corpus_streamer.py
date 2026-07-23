"""
Unit tests for Corpus acquisition and streaming (Phase 5).
Ensures source registry, streamer generator yields, and acquisition manager function correctly.
"""

import pytest
from app.corpus.sources import SOURCE_REGISTRY, get_source_spec
from app.corpus.streamer import CorpusStreamer
from app.corpus.acquisition import CorpusAcquisitionManager


def test_source_registry_has_required_datasets():
    expected_sources = [
        "huggingface_datasets", "common_crawl", "wikipedia", "oscar", "cc100",
        "fineweb", "c4", "arxiv", "pubmed", "stackexchange", "github_code",
        "opensubtitles", "wiktionary", "opus", "ai4bharat", "manipuri_specific"
    ]
    for src in expected_sources:
        assert src in SOURCE_REGISTRY, f"Source '{src}' missing from SOURCE_REGISTRY"
        spec = get_source_spec(src)
        assert spec.name == src


def test_corpus_streamer_mock_fallback_yields():
    streamer = CorpusStreamer("manipuri_specific", min_length=10, max_examples=3, mock_fallback=True)
    examples = list(streamer)
    assert len(examples) == 3
    for ex in examples:
        assert "text" in ex
        assert "metadata" in ex
        assert len(ex["text"]) >= 10
        assert ex["metadata"]["source"] == "manipuri_specific"


def test_corpus_streamer_length_filtering():
    # min_length 50 will filter out short samples
    streamer = CorpusStreamer("common_crawl", min_length=50, max_examples=5, mock_fallback=True)
    examples = list(streamer)
    for ex in examples:
        assert len(ex["text"]) >= 50


def test_corpus_acquisition_manager_stream_all():
    manager = CorpusAcquisitionManager(sources=["wikipedia", "c4"])
    examples = list(manager.stream_all(max_examples_per_source=2, mock_fallback=True))
    assert len(examples) == 4
    stats = manager.get_stats()
    assert stats["total_examples_yielded"] == 4
    assert stats["source_counts"]["wikipedia"] == 2
    assert stats["source_counts"]["c4"] == 2
