"""
ManipuriGPT Phase 5.4 Dataset Builder Engine (`app/dataset_builder`).
Provides utilities for freezing Tokenizer v1, evaluating corpus quality (`CorpusValidator`),
and assembling training-ready Parquet shards across deterministic splits (`DatasetAssembler`).
"""

from app.dataset_builder.tokenizer_freezer import TokenizerFreezer
from app.dataset_builder.corpus_validator import CorpusValidator
from app.dataset_builder.dataset_assembler import DatasetAssembler

__all__ = [
    "TokenizerFreezer",
    "CorpusValidator",
    "DatasetAssembler",
]
