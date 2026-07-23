import pytest
from app.preprocessing.transliteration import ScriptCanonicalizer


def test_script_canonicalizer_bengali_to_mayek():
    canonicalizer = ScriptCanonicalizer({"enabled": True})
    
    # Test standard vowels and consonants mapping
    bengali_text = "মণিপুরী ভাষা ভারতের অন্যতম প্রধান ভাষা।"
    canonical_text = canonicalizer.canonicalize_to_mayek(bengali_text)
    
    # Check that Bengali characters are replaced by Meitei Mayek characters
    assert not canonicalizer.bengali_range.search(canonical_text), f"Bengali characters should be transliterated, got: {canonical_text}"
    assert canonicalizer.mayek_range.search(canonical_text), f"Result should contain Meitei Mayek characters, got: {canonical_text}"


def test_script_canonicalizer_preserve_mayek_and_english():
    canonicalizer = ScriptCanonicalizer({"enabled": True})
    
    mixed_text = "ꯑꯩꯍꯥꯛ Manipuri ꯂꯣꯟ ꯇꯝꯂꯤ।"
    canonical_text = canonicalizer.canonicalize_to_mayek(mixed_text)
    
    # Should remain exactly the same since there are no Bengali characters
    assert canonical_text == mixed_text


def test_script_canonicalizer_process_text():
    # Test default/canonical mode
    canonicalizer = ScriptCanonicalizer({"enabled": True, "mode": "canonical"})
    
    bengali_input = "মণিপুরী ভাষা"
    processed, meta = canonicalizer.process_text(bengali_input)
    
    assert meta["canonicalized"] is True
    assert meta["original_script_dist"]["bengali"] > 0.0
    assert meta["final_script_dist"]["meitei"] > 0.0
    assert meta["original_text"] == bengali_input
    assert meta["canonical_text"] == processed


def test_script_canonicalizer_preserve_mode():
    # Test preserve mode (non-destructive)
    canonicalizer = ScriptCanonicalizer({"enabled": True, "mode": "preserve"})
    
    bengali_input = "মণিপুরী ভাষা"
    processed, meta = canonicalizer.process_text(bengali_input)
    
    assert processed == bengali_input
    assert meta["canonicalization_mode"] == "preserve"
    assert meta["original_text"] == bengali_input
    assert meta["canonical_text"] != bengali_input  # Canonical form still computed and recorded in metadata


def test_script_canonicalizer_hybrid_mode():
    canonicalizer = ScriptCanonicalizer({"enabled": True, "mode": "hybrid"})
    
    bengali_input = "মণিপুরী ভাষা"
    processed, meta = canonicalizer.process_text(bengali_input)
    
    assert processed == meta["canonical_text"]
    assert meta["canonicalization_mode"] == "hybrid"
    assert meta["original_text"] == bengali_input
