"""
Unit test for PIIRemover module (Phase 5).
Verifies email, phone, IP, URL, and ID detection and masking.
"""

import pytest
from app.preprocessing.pii_remover import PIIRemover


def test_pii_remover_masking():
    remover = PIIRemover(remove_pii=False)
    text = "Contact me at test.user@example.com or call +91 9876543210 from IP 192.168.1.100."
    cleaned, counts = remover.clean_text(text)
    
    assert "<EMAIL>" in cleaned
    assert "<PHONE>" in cleaned
    assert "<IP_ADDRESS>" in cleaned
    assert counts["email"] == 1
    assert counts["phone"] == 1
    assert counts["ip_address"] == 1


def test_pii_remover_process_dict():
    remover = PIIRemover(remove_pii=False)
    example = {
        "text": "Check https://manipuri.org for details and email support@manipuri.org.",
        "metadata": {"source": "test"}
    }
    res = remover.process(example)
    assert "<URL>" in res["text"]
    assert "<EMAIL>" in res["text"]
    assert res["metadata"]["pii_cleaned"] is True
    assert res["metadata"]["pii_detected"]["url"] == 1
