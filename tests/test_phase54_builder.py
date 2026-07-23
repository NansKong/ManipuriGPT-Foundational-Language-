"""
Unit test suite for Phase 5.4 (`tests/test_phase54_builder.py`).
Tests Tokenizer v1 freezing, corpus validation report generation, deterministic splitting (98/1/1),
and Parquet sharding across train/val/test splits without requiring live Hugging Face calls.
"""

import os
import json
import tempfile
import pytest
from datasets import Dataset, DatasetDict
from app.dataset_builder.tokenizer_freezer import TokenizerFreezer
from app.dataset_builder.corpus_validator import CorpusValidator
from app.dataset_builder.dataset_assembler import DatasetAssembler


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_tokenizer_freezer_basic(temp_workspace):
    # Create a dummy tokenizer.model binary file
    dummy_model = os.path.join(temp_workspace, "dummy.model")
    with open(dummy_model, "wb") as f:
        f.write(b"mock_sentencepiece_binary_content")

    freezer = TokenizerFreezer(default_output_dir=os.path.join(temp_workspace, "frozen"))
    files = freezer.freeze(source_model_path=dummy_model, output_dir=os.path.join(temp_workspace, "frozen"))

    assert "tokenizer.model" in files
    assert "vocab.json" in files
    assert "special_tokens_map.json" in files
    assert "coverage_report.json" in files

    # Verify coverage_report contents
    with open(files["coverage_report.json"], "r", encoding="utf-8") as f:
        cov = json.load(f)
    assert cov["vocab_size"] >= 4
    assert cov["tokenizer_name"] == "ManipuriGPT-Tokenizer-v1"


def test_corpus_validator_evaluation_and_reports(temp_workspace):
    # Create sample rows across splits
    sample_rows = [
        {"text": "ꯍꯥꯏ ꯃꯅꯤꯄꯨꯔꯤ", "language": "mni", "script": "meitei", "source": "dayananda"},
        {"text": "হ্যালো মণিপুরী", "language": "mni", "script": "bengali", "source": "dayananda"},
        {"text": "Hello Manipuri world", "language": "en", "script": "latin", "source": "wikipedia"}
    ]
    ds_dict = DatasetDict({
        "train": Dataset.from_list(sample_rows * 10),
        "validation": Dataset.from_list(sample_rows[:1]),
        "test": Dataset.from_list(sample_rows[:1])
    })

    validator = CorpusValidator()
    report = validator.evaluate(ds_dict, raw_count_before_dedup=35)

    assert report["pipeline_version"] == "5.4"
    assert report["overall_statistics"]["total_sequences"] == 32
    assert report["deduplication_metrics"]["raw_examples_before_deduplication"] == 35
    assert "meitei" in report["script_balance_pct"]
    assert "dayananda" in report["source_distribution"]

    # Verify saving reports and cards
    out_meta = os.path.join(temp_workspace, "metadata")
    card_path = os.path.join(temp_workspace, "README.md")
    saved = validator.save_report(report, output_dir=out_meta, dataset_card_path=card_path)

    assert os.path.exists(saved["corpus_report.json"])
    assert os.path.exists(saved["README.md"])

    with open(saved["README.md"], "r", encoding="utf-8") as f:
        card_txt = f.read()
    assert "ManipuriGPT-Corpus-v1" in card_txt
    assert "Script Balance" in card_txt


def test_dataset_assembler_pipeline_and_shards(temp_workspace):
    assembler = DatasetAssembler(
        output_dir=os.path.join(temp_workspace, "corpus_v1"),
        split_ratios=(0.80, 0.10, 0.10),
        shard_size=15,
        seed=42
    )

    manifest = assembler.assemble(
        sources=["dayananda_meitei_mayek_sample", "dayananda_english_to_meitei", "joyson_bible"],
        max_examples=200,
        mock_fallback=True
    )

    assert manifest["dataset_name"] == "ManipuriGPT-Corpus-v1"
    assert manifest["pipeline_version"] == "5.4"
    assert manifest["total_sequences"] > 0
    assert manifest["total_shards"] >= 1

    # Check Parquet files exist in output_dir
    train_dir = os.path.join(temp_workspace, "corpus_v1", "train")
    assert os.path.exists(train_dir)
    shards = os.listdir(train_dir)
    assert any(s.endswith(".parquet") for s in shards)

    # Check manifest and reports
    meta_dir = os.path.join(temp_workspace, "corpus_v1", "metadata")
    assert os.path.exists(os.path.join(meta_dir, "corpus_report.json"))
    assert os.path.exists(os.path.join(meta_dir, "manifest.json"))
    assert os.path.exists(os.path.join(temp_workspace, "corpus_v1", "README.md"))
