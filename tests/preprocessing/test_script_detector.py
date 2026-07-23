import pytest
from app.preprocessing.script_detector import ScriptDetector

def test_script_detection_meitei():
    detector = ScriptDetector()
    # "ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ" is Meitei Mayek
    result = detector.detect("ꯅꯨꯡꯉꯥꯏꯔꯤꯕ꯭ꯔꯥ")
    assert result["script"] == "meitei"
    assert result["confidence"] > 0.9

def test_script_detection_bengali():
    detector = ScriptDetector()
    # "নূঙঙাইরিব্রা" is Bengali
    result = detector.detect("নূঙঙাইরিব্রা")
    assert result["script"] == "bengali"
    assert result["confidence"] > 0.9

def test_script_detection_latin():
    detector = ScriptDetector()
    result = detector.detect("Hello World")
    assert result["script"] == "latin"
    assert result["confidence"] > 0.9

def test_script_detection_devanagari():
    detector = ScriptDetector()
    result = detector.detect("नमस्ते")
    assert result["script"] == "devanagari"
    assert result["confidence"] > 0.9

def test_script_detection_mixed():
    detector = ScriptDetector()
    result = detector.detect("Hello নূঙঙাইরিব্রা")
    assert result["script"] == "mixed"
