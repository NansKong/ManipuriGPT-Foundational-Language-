"""
Strongly typed metadata objects (`app/preprocessing/metadata_types.py`).
Provides DocumentMetadata and PipelineMetadata dataclasses for IDE completion,
type safety, validation, and structured serialization across Phase 5 pretraining pipelines.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class DocumentMetadata:
    """
    Metadata associated with every preprocessed document and chunk shard.
    """
    language: str = "en"
    script: str = "unknown"
    source: str = "unknown"
    source_dataset: str = "unknown"
    dataset_version: str = "v1"
    document_id: str = ""
    quality_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    chunk_id: int = 0
    total_chunks: int = 1
    tokenizer_version: str = "sentencepiece_unigram_32k"
    pipeline_version: str = "5.2"
    config_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return data

    @classmethod
    def from_dict(cls, raw_data: Optional[Dict[str, Any]] = None) -> "DocumentMetadata":
        if not raw_data:
            return cls()
        
        known_fields = {
            "language", "script", "source", "source_dataset", "dataset_version",
            "document_id", "quality_score", "timestamp", "chunk_id", "total_chunks",
            "tokenizer_version", "pipeline_version", "config_hash"
        }
        
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in raw_data.items():
            if k in known_fields:
                kwargs[k] = v
            else:
                extra[k] = v
                
        kwargs["extra"] = extra
        return cls(**kwargs)


@dataclass
class PipelineMetadata:
    """
    Top-level manifest metadata for sharding runs and execution pipelines.
    """
    pipeline_version: str = "5.2"
    config_hash: str = ""
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    total_documents_processed: int = 0
    total_chunks_yielded: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw_data: Optional[Dict[str, Any]] = None) -> "PipelineMetadata":
        if not raw_data:
            return cls()
        return cls(**{k: v for k, v in raw_data.items() if hasattr(cls, k)})
