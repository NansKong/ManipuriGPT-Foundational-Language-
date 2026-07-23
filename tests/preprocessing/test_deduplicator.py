import pytest
from app.preprocessing.deduplicator import Deduplicator

def test_exact_deduplication():
    dedup = Deduplicator({"exact": True, "normalized": False, "fuzzy": False})
    
    assert dedup.is_duplicate("Hello World") is False
    assert dedup.is_duplicate("Hello World") is True
    assert dedup.is_duplicate("hello world") is False

def test_normalized_deduplication():
    dedup = Deduplicator({"exact": False, "normalized": True, "fuzzy": False})
    
    assert dedup.is_duplicate("Hello   World") is False
    assert dedup.is_duplicate("hello world") is True
    assert dedup.is_duplicate("HELLO WORLD  ") is True

def test_fuzzy_deduplication():
    # Fuzzy threshold at 90
    dedup = Deduplicator({"exact": False, "normalized": False, "fuzzy": True, "fuzzy_threshold": 90.0})
    
    assert dedup.is_duplicate("How are you?") is False
    # "How are you!" matches > 90%
    assert dedup.is_duplicate("How are you!") is True
    # "Where are you?" matches < 90%
    assert dedup.is_duplicate("Where are you?") is False
