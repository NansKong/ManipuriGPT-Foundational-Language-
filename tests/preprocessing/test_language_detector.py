import pytest
from app.preprocessing.language_detector import LanguageDetector

def test_language_detection_english():
    detector = LanguageDetector({"detector_type": "langdetect"})
    result = detector.detect("This is a simple English sentence.")
    assert result["language"] == "en"
    assert result["confidence"] > 0.8

def test_language_detection_meitei_heuristic():
    detector = LanguageDetector({"detector_type": "langdetect"})
    # Even if langdetect doesn't know Manipuri, the Meitei Mayek script heuristic
    # should identify it as 'mni'
    result = detector.detect("ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ")
    assert result["language"] == "mni"
    assert result["confidence"] > 0.8
