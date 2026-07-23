"""
Unit test for production CLI scripts (`train.py`, `evaluate.py`, `export.py`, `ingest.py`).
Verifies argparse parsing and dry-run execution without blocking.
"""

import pytest
from app.scripts.train import main as train_main
from app.scripts.evaluate import main as eval_main
from app.scripts.export import main as export_main
from app.scripts.ingest import main as ingest_main


def test_cli_train_dry_run(tmp_path):
    out_dir = str(tmp_path / "cli_checkpoints")
    ret = train_main(["--model", "smollm_135m", "--mode", "lora", "--dry-run", "--output-dir", out_dir])
    assert ret == 0


def test_cli_evaluate():
    ret = eval_main(["--model", "tinyllama_1_1b", "--task", "translation", "--run-validation"])
    assert ret == 0


def test_cli_export(tmp_path):
    ckpt_dir = str(tmp_path / "dummy_ckpt")
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    ret = export_main(["--checkpoint", ckpt_dir, "--model", "test_model", "--targets", "hf", "gguf"])
    assert ret == 0


def test_cli_ingest():
    ret = ingest_main(["--source", "wikipedia", "--limit", "5"])
    assert ret == 0
