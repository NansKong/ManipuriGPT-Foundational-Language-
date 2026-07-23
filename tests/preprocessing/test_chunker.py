"""
Unit test for SequenceChunker module (Phase 5).
Verifies sliding window chunking with overlap across long document texts.
"""

import pytest
from app.preprocessing.chunker import SequenceChunker


def test_sequence_chunker_word_level():
    chunker = SequenceChunker(max_chunk_size=10, chunk_overlap=2, mode="word")
    words = [f"word_{i}" for i in range(25)]
    text = " ".join(words)

    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    # Check that chunks overlap correctly
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert len(first_chunk_words) <= 10
    # Overlap of 2 means last 2 words of chunk 0 equal first 2 words of chunk 1
    assert first_chunk_words[-2:] == second_chunk_words[:2]


def test_sequence_chunker_process_example():
    chunker = SequenceChunker(max_chunk_size=5, chunk_overlap=1, mode="word")
    example = {
        "text": "One two three four five six seven eight nine ten.",
        "metadata": {"source": "test"}
    }
    chunk_list = chunker.process_example(example)
    assert len(chunk_list) >= 2
    for idx, ex in enumerate(chunk_list):
        assert ex["metadata"]["chunk_index"] == idx
        assert ex["metadata"]["total_chunks"] == len(chunk_list)
