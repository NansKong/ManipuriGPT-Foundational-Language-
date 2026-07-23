"""
Unit test for InferenceEngine and InferenceValidator (`Phase 5`).
Verifies generation outputs, streaming token yields, and 7-dimension validation verification.
"""

import pytest
from app.inference.engine import InferenceEngine
from app.inference.validator import InferenceValidator


def test_inference_engine_generate_and_stream():
    engine = InferenceEngine(model_name="smollm_135m", backend="transformers")
    
    # Test generation
    res = engine.generate("Translate to Manipuri: Hello, how are you?")
    assert res["status"] == "success"
    assert "ꯈꯨꯔꯨꯝꯖꯔꯤ" in res["output"] or "Khurumjari" in res["output"]
    assert res["tokens_generated"] > 0

    # Test streaming
    tokens = list(engine.stream_generate("Summarize Manipur history", max_new_tokens=16))
    assert len(tokens) > 1
    full_stream = "".join(tokens)
    assert "Summary:" in full_stream


def test_inference_validator_seven_dimensions():
    engine = InferenceEngine(model_name="tinyllama_1_1b")
    validator = InferenceValidator(engine)

    report = validator.validate_all_dimensions()
    assert report["dimensions_tested"] == 7
    assert report["dimensions_passed"] == 7
    assert report["all_passed"] is True
    assert "chat" in report["details"]
    assert "translation" in report["details"]
    assert "summarization" in report["details"]
    assert "reasoning" in report["details"]
    assert "rag" in report["details"]
    assert "long_context" in report["details"]
    assert "safety" in report["details"]
