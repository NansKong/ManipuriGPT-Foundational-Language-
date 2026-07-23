"""
Unit tests for TokenizerEvaluator (Phase 5.3).

Tests verify:
- Single candidate evaluation computes correct metrics
- Multi-candidate comparison discovers models correctly
- Winner selection scoring function works as expected
- Threshold compliance checking
- Report generation
"""

import os
import json
import pytest
from app.tokenization.evaluator import (
    TokenizerEvaluator,
    EVALUATION_SENTENCES,
    _flatten_samples,
)


class MockSPProcessor:
    """Mock SentencePieceProcessor for evaluator unit tests."""
    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size

    def GetPieceSize(self):
        return self._vocab_size

    def unk_id(self):
        return 0

    def Encode(self, text, out_type=int):
        words = text.split()
        if out_type == int:
            # 2 token IDs per word (simulating moderate fertility)
            ids = []
            for i, w in enumerate(words):
                ids.extend([10 + i * 2, 11 + i * 2])
            return ids
        else:
            # Subword pieces
            pieces = []
            for w in words:
                mid = len(w) // 2
                if mid > 0:
                    pieces.extend([f"▁{w[:mid]}", w[mid:]])
                else:
                    pieces.append(f"▁{w}")
            return pieces

    def Decode(self, ids):
        # Simple mock decode that reconstructs approximately
        return "mock decoded text"


# ============================================================
# Evaluator Unit Tests
# ============================================================

def test_evaluation_sentences_structure():
    """Verifies EVALUATION_SENTENCES has all required script categories."""
    assert "meitei_mayek" in EVALUATION_SENTENCES
    assert "bengali" in EVALUATION_SENTENCES
    assert "latin" in EVALUATION_SENTENCES
    assert "mixed" in EVALUATION_SENTENCES

    for script, sentences in EVALUATION_SENTENCES.items():
        assert len(sentences) >= 2, f"Script '{script}' should have at least 2 test sentences"
        for s in sentences:
            assert isinstance(s, str) and len(s) > 0


def test_flatten_samples():
    """Verifies _flatten_samples correctly flattens the dict structure."""
    samples = {
        "a": ["sentence 1", "sentence 2"],
        "b": ["sentence 3"],
    }
    flat = _flatten_samples(samples)
    assert len(flat) == 3
    assert "sentence 1" in flat
    assert "sentence 3" in flat


def test_evaluator_thresholds_from_config():
    """Verifies evaluator loads thresholds."""
    evaluator = TokenizerEvaluator()
    assert "max_acceptable_fertility" in evaluator.thresholds
    assert "max_acceptable_unknown_rate" in evaluator.thresholds
    assert "min_compression_ratio" in evaluator.thresholds
    assert "min_round_trip_accuracy" in evaluator.thresholds


def test_evaluator_custom_thresholds():
    """Verifies evaluator accepts custom thresholds."""
    custom = {
        "max_acceptable_fertility": 2.0,
        "max_acceptable_unknown_rate": 0.01,
        "min_compression_ratio": 4.0,
        "min_round_trip_accuracy": 0.99,
    }
    evaluator = TokenizerEvaluator(thresholds=custom)
    assert evaluator.thresholds["max_acceptable_fertility"] == 2.0
    assert evaluator.thresholds["min_round_trip_accuracy"] == 0.99


def test_evaluator_missing_model():
    """Verifies evaluator handles missing model path gracefully."""
    evaluator = TokenizerEvaluator()
    result = evaluator.evaluate_candidate("/nonexistent/path/tokenizer.model")
    assert "error" in result


def test_evaluator_compare_empty_dir(tmp_path):
    """Verifies compare_candidates handles empty directory."""
    evaluator = TokenizerEvaluator()
    results = evaluator.compare_candidates(str(tmp_path))
    assert len(results) == 0


def test_evaluator_select_winner_empty():
    """Verifies select_winner raises on empty results."""
    evaluator = TokenizerEvaluator()
    with pytest.raises(ValueError, match="No candidates"):
        evaluator.select_winner({})


def test_evaluator_select_winner_scoring():
    """Verifies winner selection scoring prefers lower fertility and higher compression."""
    evaluator = TokenizerEvaluator()

    candidates = {
        "sentencepiece_unigram/16384": {
            "fertility": 2.0,
            "unknown_rate": 0.001,
            "round_trip_accuracy": 0.98,
            "compression_ratio": 4.5,
            "vocab_size": 16384,
        },
        "sentencepiece_bpe/16384": {
            "fertility": 3.0,
            "unknown_rate": 0.002,
            "round_trip_accuracy": 0.95,
            "compression_ratio": 3.5,
            "vocab_size": 16384,
        },
    }

    winner_name, winner_metrics = evaluator.select_winner(candidates)

    # Unigram/16384 should win: lower fertility, lower unknown rate, higher compression
    assert winner_name == "sentencepiece_unigram/16384", (
        f"Expected unigram/16384 to win, got: {winner_name}"
    )
    assert "selection_score" in winner_metrics


def test_evaluator_select_winner_skips_errors():
    """Verifies winner selection skips candidates with errors."""
    evaluator = TokenizerEvaluator()

    candidates = {
        "bad_model": {"error": "model not found"},
        "sentencepiece_unigram/16384": {
            "fertility": 2.5,
            "unknown_rate": 0.001,
            "round_trip_accuracy": 0.97,
            "compression_ratio": 4.0,
        },
    }

    winner_name, _ = evaluator.select_winner(candidates)
    assert winner_name == "sentencepiece_unigram/16384"


def test_evaluator_threshold_check():
    """Verifies threshold compliance checking works correctly."""
    evaluator = TokenizerEvaluator(thresholds={
        "max_acceptable_fertility": 3.0,
        "max_acceptable_unknown_rate": 0.005,
        "min_compression_ratio": 3.0,
        "min_round_trip_accuracy": 0.95,
    })

    # Passing metrics
    good = {
        "fertility": 2.0,
        "unknown_rate": 0.001,
        "compression_ratio": 4.5,
        "round_trip_accuracy": 0.98,
    }
    passes = evaluator._check_thresholds(good)
    assert all(passes.values()), f"Good metrics should pass all thresholds: {passes}"

    # Failing metrics
    bad = {
        "fertility": 5.0,
        "unknown_rate": 0.1,
        "compression_ratio": 1.0,
        "round_trip_accuracy": 0.5,
    }
    fails = evaluator._check_thresholds(bad)
    assert not any(fails.values()), f"Bad metrics should fail all thresholds: {fails}"


def test_evaluator_report_generation(tmp_path):
    """Verifies comparison report is generated as valid markdown."""
    evaluator = TokenizerEvaluator()

    results = {
        "sentencepiece_unigram/16384": {
            "vocab_size": 16384,
            "fertility": 2.0,
            "compression_ratio": 4.5,
            "unknown_rate": 0.001,
            "round_trip_accuracy": 0.98,
            "selection_score": 0.75,
            "passes_thresholds": {
                "fertility_ok": True,
                "unknown_rate_ok": True,
                "compression_ok": True,
                "round_trip_ok": True,
            },
            "per_script": {
                "meitei_mayek": {"fertility": 2.5, "compression_ratio": 3.5, "unknown_rate": 0.002, "round_trip_accuracy": 0.95, "avg_tokens_per_sentence": 8.0},
                "latin": {"fertility": 1.5, "compression_ratio": 5.0, "unknown_rate": 0.0, "round_trip_accuracy": 1.0, "avg_tokens_per_sentence": 6.0},
            },
        }
    }

    report_path = str(tmp_path / "test_report.md")
    output = evaluator.generate_comparison_report(results, "sentencepiece_unigram/16384", report_path)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        content = f.read()
    assert "ManipuriGPT Tokenizer Candidate Evaluation Report" in content
    assert "sentencepiece_unigram/16384" in content
    assert "Per-Script Breakdown" in content
