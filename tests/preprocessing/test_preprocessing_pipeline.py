import pytest
import os
import shutil
import json
from datasets import Dataset
from app.preprocessing.pipeline import PreprocessingPipeline

def test_pipeline_run():
    data = {
        "text": [
            "  Hello World!  ",
            "Hello World!",
            "<html><body>Clean Me</body></html>",
            "!",
            "12345",
            "aaaaa",
            "This is a very nice English sentence.",
        ]
    }
    dataset = Dataset.from_dict(data)

    pipeline = PreprocessingPipeline()
    pipeline.language_detector.config["target_language"] = "any"
    pipeline.script_detector.target_script = "any"

    processed = pipeline.run(dataset, text_keys=["text"])

    assert len(processed) == 3
    texts = processed["text"]
    assert "Hello World!" in texts
    assert "Clean Me" in texts
    assert "This is a very nice English sentence." in texts

    stats = pipeline.stats.generate_report()
    assert stats["total_processed"] == 7
    assert stats["final_accepted"] == 3
    assert stats["skipped"]["duplicates_removed"] == 1
    assert stats["skipped"]["only_punctuation_removed"] == 1
    assert stats["skipped"]["only_numbers_removed"] == 1
    assert stats["skipped"]["repeated_chars_removed"] == 1


def test_pipeline_process_split_export(tmp_path):
    if os.path.exists("reports"):
        shutil.rmtree("reports")

    data = {"text": [f"Sentence number {i}" for i in range(10)]}
    dataset = Dataset.from_dict(data)

    pipeline = PreprocessingPipeline()
    pipeline.language_detector.config["target_language"] = "any"
    pipeline.script_detector.target_script = "any"
    pipeline.deduplicator.fuzzy_enabled = False
    pipeline.splitter.train_ratio = 0.8
    pipeline.splitter.val_ratio = 0.2
    pipeline.splitter.test_ratio = 0.0

    out_dir = os.path.join(tmp_path, "processed")
    pipeline.exporter.output_dir = out_dir
    pipeline.exporter.format = "jsonl"
    pipeline.exporter.version = "v1"
    
    report_file = os.path.join(tmp_path, "report.md")

    result = pipeline.process_split_export(
        dataset, 
        text_keys=["text"], 
        dataset_name="test-corpus",
        report_path=report_file
    )

    assert "train" in result
    assert "validation" in result
    assert len(result["train"]) == 8
    assert len(result["validation"]) == 2

    versioned_dir = os.path.join(out_dir, "v1")
    assert os.path.exists(os.path.join(versioned_dir, "dataset_train.jsonl"))
    assert os.path.exists(os.path.join(versioned_dir, "dataset_validation.jsonl"))
    
    manifest_file = os.path.join(versioned_dir, "manifest.json")
    assert os.path.exists(manifest_file)
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
    assert manifest["dataset"] == "test-corpus"
    assert manifest["version"] == "v1"
    assert manifest["samples"] == 10
    assert "preprocessing" in manifest
    assert manifest["preprocessing"]["unicode"] == "NFC"
    
    assert os.path.exists(report_file)
    assert os.path.exists(os.path.join("reports", "corpus_report.md"))
    assert os.path.exists(os.path.join("reports", "statistics.json"))

    if os.path.exists("reports"):
        shutil.rmtree("reports")
