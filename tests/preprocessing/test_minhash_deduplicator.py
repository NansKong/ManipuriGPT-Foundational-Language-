"""
Unit test for MinHashDeduplicator module (Phase 5).
Verifies Locality-Sensitive Hashing (LSH) near-duplicate detection and removal.
"""

import pytest
from app.preprocessing.minhash_deduplicator import MinHashDeduplicator


def test_minhash_deduplicator_exact_and_near_duplicates():
    dedup = MinHashDeduplicator(num_perm=128, num_bands=16, similarity_threshold=0.80)
    
    doc1 = "Manipuri language is one of the official languages of the Indian state of Manipur."
    doc2 = "Manipuri language is one of the official languages of the Indian state of Manipur and surrounding region."
    doc3 = "Quantum computing harnesses phenomena from quantum mechanics to solve complex computation problems."

    is_dup1, id1 = dedup.is_duplicate_or_add(doc1)
    assert not is_dup1
    assert id1 is not None

    # doc2 is a near-duplicate of doc1
    is_dup2, id2 = dedup.is_duplicate_or_add(doc2)
    assert is_dup2
    assert id2 == id1

    # doc3 is completely different
    is_dup3, id3 = dedup.is_duplicate_or_add(doc3)
    assert not is_dup3
    assert id3 != id1


def test_minhash_process_example():
    dedup = MinHashDeduplicator(similarity_threshold=0.75)
    ex1 = {"text": "Manipuri script alignment and normalization pipeline across three scripts."}
    ex2 = {"text": "Manipuri script alignment and normalization pipeline across three scripts today."}
    
    res1 = dedup.process_example(ex1)
    assert res1 is not None
    assert res1["metadata"]["minhash_deduplicated"] is True

    res2 = dedup.process_example(ex2)
    assert res2 is None
