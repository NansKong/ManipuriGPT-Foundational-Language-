import pytest
from app.datasets.validator import DatasetValidator

def test_empty_filtering():
    validator = DatasetValidator(target_script="any")
    samples = [
        {"en": "Hello", "mni": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},
        {"en": "  ", "mni": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},  # Empty English
        {"en": "Hi", "mni": ""},              # Empty Manipuri
    ]
    cleaned, report = validator.validate_samples(samples, text_keys=["en", "mni"])
    
    assert len(cleaned) == 1
    assert report["Loaded"] == 3
    assert report["Empty"] == 2
    assert report["Final"] == 1

def test_duplicate_filtering():
    validator = DatasetValidator(target_script="any")
    samples = [
        {"text": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},
        {"text": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},  # Duplicate
        {"text": "ꯀꯔꯤ ꯇꯧꯔꯤꯕꯅꯣ"},
    ]
    cleaned, report = validator.validate_samples(samples, text_keys=["text"])
    
    assert len(cleaned) == 2
    assert report["Removed duplicates"] == 1
    assert report["Final"] == 2

def test_unicode_filtering():
    validator = DatasetValidator(target_script="any")
    samples = [
        {"text": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},
        {"text": "Sample \uFFFD text"},  # Invalid Unicode replacement character
        {"text": "Sample \x00 text"},     # Control character
    ]
    cleaned, report = validator.validate_samples(samples, text_keys=["text"])
    
    assert len(cleaned) == 1
    assert report["Invalid Unicode"] == 2
    assert report["Final"] == 1

def test_script_filtering_meitei():
    validator = DatasetValidator(target_script="meitei_mayek")
    samples = [
        {"text": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},     # Meitei Mayek
        {"text": "নূঙঙাইরিব্রা"},         # Bengali script
        {"text": "nungngairibra"},       # Romanized/Latin
    ]
    cleaned, report = validator.validate_samples(samples, text_keys=["text"])
    
    assert len(cleaned) == 1
    assert report["Script mismatch"] == 2
    assert report["Final"] == 1

def test_script_filtering_bengali():
    validator = DatasetValidator(target_script="bengali")
    samples = [
        {"text": "নূঙঙাইরিব্রা"},         # Bengali script
        {"text": "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ"},     # Meitei Mayek
        {"text": "nungngairibra"},       # Romanized/Latin
    ]
    cleaned, report = validator.validate_samples(samples, text_keys=["text"])
    
    assert len(cleaned) == 1
    assert report["Script mismatch"] == 2
    assert report["Final"] == 1
