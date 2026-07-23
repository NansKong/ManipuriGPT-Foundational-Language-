import pytest
from app.preprocessing.normalizer import UnicodeNormalizer

def test_unicode_normalization_nfc():
    # Character combining: 'o' + Combining Diaeresis (U+0308) -> 'ö' (U+00F6)
    normalizer = UnicodeNormalizer({"form": "NFC"})
    text_nfd = "o\u0308"
    normalized = normalizer.normalize(text_nfd)
    assert normalized == "\u00F6"

def test_remove_zero_width_spaces():
    normalizer = UnicodeNormalizer({"remove_zero_width": True})
    text = "Hello\u200bWorld\ufeff!"
    normalized = normalizer.normalize(text)
    assert normalized == "HelloWorld!"

def test_remove_control_characters():
    normalizer = UnicodeNormalizer({"remove_control_chars": True})
    text = "Hello\x00\x07World!"
    normalized = normalizer.normalize(text)
    assert normalized == "HelloWorld!"

def test_normalize_quotes():
    normalizer = UnicodeNormalizer({"normalize_quotes": True})
    text = "“Hello” ‘World’"
    normalized = normalizer.normalize(text)
    assert normalized == '"Hello" \'World\''

def test_normalize_punctuation():
    normalizer = UnicodeNormalizer({"normalize_punctuation": True})
    text = "Hello , World  ...  How are you ?"
    normalized = normalizer.normalize(text)
    assert normalized == "Hello, World... How are you?"
