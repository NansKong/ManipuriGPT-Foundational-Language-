"""
ManipuriGPT Corpus Acquisition & Streaming Module (Phase 5).
Provides streaming ingestion across multi-source corpora without full disk downloads.
"""

from app.corpus.sources import CorpusSourceSpec, SOURCE_REGISTRY, get_source_spec
from app.corpus.streamer import CorpusStreamer
from app.corpus.acquisition import CorpusAcquisitionManager

__all__ = [
    "CorpusSourceSpec",
    "SOURCE_REGISTRY",
    "get_source_spec",
    "CorpusStreamer",
    "CorpusAcquisitionManager",
]
