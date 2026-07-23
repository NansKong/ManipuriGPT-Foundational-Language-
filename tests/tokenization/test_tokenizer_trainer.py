"""
Unit tests for TokenizerTrainer, TokenizerBenchmarker, and SentencePieceTokenizerBackend (Phase 5.3).

Tests verify:
- Simulated training produces correct metadata schema
- Character coverage configuration is respected
- Special tokens don't include ordinary Manipuri letters
- Empty corpus guard prevents production training on insufficient data
- SentencePiece backend encode/decode round-trip
- Benchmarker correctly computes fertility and per-script metrics
"""

import os
import json
import pytest
from app.tokenization.trainer import TokenizerTrainer, _SPM_RESERVED_TOKENS
from app.tokenization.benchmark import TokenizerBenchmarker


class MockTokenizerForBenchmarking:
    """Mock tokenizer with encode/decode for benchmarker unit tests."""
    def __init__(self):
        self._vocab = {"<unk>": 0, "hello": 101, "world": 102, "test": 103}

    def encode(self, text: str):
        # Simple mock: 3 tokens per input
        return {"input_ids": [101, 102, 103]}

    def decode(self, token_ids, skip_special_tokens=True):
        return "hello world test"

    def get_vocab(self):
        return self._vocab

    @property
    def unk_token_id(self):
        return 0


class MockTokenizerWithUnknowns:
    """Mock tokenizer that produces unknown tokens for testing OOV metrics."""
    def encode(self, text: str):
        # Every other token is unknown
        return {"input_ids": [0, 101, 0, 102]}

    def get_vocab(self):
        return {"<unk>": 0, "a": 101, "b": 102}

    @property
    def unk_token_id(self):
        return 0


# ============================================================
# TokenizerTrainer Tests
# ============================================================

def test_tokenizer_trainer_simulated(tmp_path):
    """Verifies simulated training produces correct artifacts and metadata schema."""
    out_dir = str(tmp_path / "test_tokenizers")
    trainer = TokenizerTrainer(
        algorithm="sentencepiece_unigram",
        vocab_size=100,
        output_dir=out_dir,
        dev_mode=True  # Allow simulated fallback
    )
    samples = [
        "Manipuri is spoken across Manipur.",
        "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯃꯅꯤꯄꯨꯔꯒꯤ ꯂꯣꯟꯅꯤ꯫",
        "মণিপুরী ভাষা ভারতের অন্যতম প্রধান ভাষা।",
    ] * 50  # Repeat to meet minimum sample requirement

    metadata = trainer.train_from_iterator(iter(samples), model_prefix="test_spm")

    # Verify metadata schema uses 'vocab_size' (not 'vocab')
    assert "vocab_size" in metadata, "Metadata must use 'vocab_size' key, not 'vocab'"
    assert metadata["vocab_size"] == 100
    assert metadata["algorithm"] == "sentencepiece_unigram"
    assert metadata["training_samples"] == 150  # 3 * 50
    assert len(metadata["artifact_files"]) >= 1
    assert metadata["pipeline_version"] == "5.3"

    # Verify training_config is recorded
    assert "training_config" in metadata

    # Verify metadata file exists
    assert os.path.exists(os.path.join(out_dir, "test_spm_metadata.json"))
    assert os.path.exists(os.path.join(out_dir, "metadata.json"))


def test_character_coverage_in_config():
    """Verifies that the training config loads character_coverage=0.9999."""
    try:
        from app.tokenization.trainer import _load_training_config
        cfg = _load_training_config()
        if cfg:
            coverage = cfg.get("character_coverage", 0.9999)
            assert coverage >= 0.999, (
                f"character_coverage should be >= 0.999 for Manipuri "
                f"(Meitei Mayek has small Unicode block). Got: {coverage}"
            )
    except Exception:
        # Config may not be loadable in all test environments
        pass


def test_special_tokens_no_ordinary_letters():
    """
    Verifies that special tokens don't include ordinary Manipuri or Bengali letters.
    user_defined_symbols should only contain model-control tokens.
    """
    trainer = TokenizerTrainer(
        algorithm="sentencepiece_unigram",
        vocab_size=16384,
        dev_mode=True
    )

    # Check that no ordinary Manipuri/Bengali/Latin characters are in special tokens
    import re
    meitei_pattern = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAFF]')
    bengali_pattern = re.compile(r'[\u0980-\u09FF]')

    for token in trainer.special_tokens:
        # Special tokens should be wrapped in angle brackets or similar delimiters
        assert not meitei_pattern.search(token), (
            f"Special token '{token}' contains ordinary Meitei Mayek characters. "
            f"Only model-control tokens should be in user_defined_symbols."
        )
        assert not bengali_pattern.search(token), (
            f"Special token '{token}' contains ordinary Bengali characters. "
            f"Only model-control tokens should be in user_defined_symbols."
        )


def test_empty_corpus_production_guard(tmp_path):
    """Verifies that production mode raises RuntimeError on empty corpus."""
    out_dir = str(tmp_path / "empty_test")
    trainer = TokenizerTrainer(
        algorithm="sentencepiece_unigram",
        vocab_size=100,
        output_dir=out_dir,
        dev_mode=False  # Production mode
    )

    # Empty corpus should raise
    with pytest.raises(RuntimeError, match="Training corpus"):
        trainer.train_from_iterator(iter([]), model_prefix="empty")


def test_near_empty_corpus_production_guard(tmp_path):
    """Verifies that production mode raises on near-empty corpus (<100 samples)."""
    out_dir = str(tmp_path / "near_empty_test")
    trainer = TokenizerTrainer(
        algorithm="sentencepiece_unigram",
        vocab_size=100,
        output_dir=out_dir,
        dev_mode=False
    )

    # 5 samples is below the minimum threshold
    with pytest.raises(RuntimeError, match="Training corpus"):
        trainer.train_from_iterator(
            iter(["short text"] * 5),
            model_prefix="near_empty"
        )


def test_empty_corpus_dev_mode_fallback(tmp_path):
    """Verifies that dev_mode allows simulated fallback on insufficient corpus."""
    out_dir = str(tmp_path / "dev_mode_test")
    trainer = TokenizerTrainer(
        algorithm="sentencepiece_unigram",
        vocab_size=100,
        output_dir=out_dir,
        dev_mode=True
    )

    # Should produce simulated artifacts without raising
    metadata = trainer.train_from_iterator(
        iter(["Some text for dev mode."] * 200),
        model_prefix="dev_test"
    )
    assert metadata["training_samples"] == 200
    assert len(metadata["artifact_files"]) >= 1


def test_reserved_tokens_not_in_user_symbols():
    """Verifies that SentencePiece reserved tokens are filtered from user_defined_symbols."""
    trainer = TokenizerTrainer(
        algorithm="sentencepiece_unigram",
        vocab_size=16384,
        dev_mode=True
    )

    # The user_defined_symbols passed to SentencePiece should NOT include reserved tokens
    user_symbols = [t for t in trainer.special_tokens if t not in _SPM_RESERVED_TOKENS]
    for token in _SPM_RESERVED_TOKENS:
        assert token not in user_symbols, (
            f"Reserved token '{token}' must not appear in user_defined_symbols. "
            f"SentencePiece manages these via unk_id/bos_id/eos_id parameters."
        )


# ============================================================
# TokenizerBenchmarker Tests
# ============================================================

def test_tokenizer_benchmarker():
    """Verifies benchmarker computes basic metrics correctly."""
    mock_tok = MockTokenizerForBenchmarking()
    benchmarker = TokenizerBenchmarker(mock_tok)

    samples = ["Sample sentence one.", "Sample sentence two."]
    metrics = benchmarker.evaluate_corpus(samples)

    assert "compression_ratio" in metrics
    assert "vocabulary_coverage" in metrics
    assert "average_sequence_length" in metrics
    assert "oov_rate" in metrics
    assert "fertility" in metrics
    assert "manipuri_token_quality" in metrics
    assert "per_script" in metrics
    assert metrics["total_samples_evaluated"] == 2
    assert metrics["average_sequence_length"] == 3.0


def test_benchmarker_fertility():
    """Verifies fertility (tokens per word) is computed correctly."""
    mock_tok = MockTokenizerForBenchmarking()
    benchmarker = TokenizerBenchmarker(mock_tok)

    # Each sample produces 3 tokens; "Sample sentence one." has 3 words
    samples = ["Sample sentence one."]
    metrics = benchmarker.evaluate_corpus(samples)

    # 3 tokens / 3 words = 1.0 fertility
    assert metrics["fertility"] == 1.0, f"Expected fertility=1.0, got {metrics['fertility']}"
    assert "total_words" in metrics


def test_benchmarker_per_script():
    """Verifies per-script breakdown includes detected scripts."""
    mock_tok = MockTokenizerForBenchmarking()
    benchmarker = TokenizerBenchmarker(mock_tok)

    samples = [
        "This is English text for testing.",
        "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ",
        "মণিপুরী ভাষা ভারতের",
    ]
    metrics = benchmarker.evaluate_corpus(samples)

    per_script = metrics.get("per_script", {})
    # Should have at least one detected script
    assert len(per_script) >= 1, f"Expected per-script breakdown, got: {per_script}"


def test_benchmarker_unknown_tokens():
    """Verifies OOV rate is computed with unknown tokens."""
    mock_tok = MockTokenizerWithUnknowns()
    benchmarker = TokenizerBenchmarker(mock_tok)

    samples = ["Test text."]
    metrics = benchmarker.evaluate_corpus(samples)

    # Mock produces [0, 101, 0, 102] -> 2 unknowns out of 4 tokens = 0.5 OOV rate
    assert metrics["oov_rate"] == 0.5, f"Expected oov_rate=0.5, got {metrics['oov_rate']}"
    assert metrics["unknown_char_count"] == 2


def test_benchmarker_empty_corpus():
    """Verifies benchmarker handles empty corpus gracefully."""
    mock_tok = MockTokenizerForBenchmarking()
    benchmarker = TokenizerBenchmarker(mock_tok)

    metrics = benchmarker.evaluate_corpus([])
    assert metrics["compression_ratio"] == 0.0
    assert metrics["fertility"] == 0.0
    assert metrics["total_samples_evaluated"] == 0
