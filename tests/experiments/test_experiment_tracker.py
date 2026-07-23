"""
Unit test for ExperimentTracker module (Phase 5).
Verifies hyperparameter recording, step metric logging, and offline JSONL audit trails.
"""

import os
import json
import pytest
from app.experiments.tracker import ExperimentTracker


def test_experiment_tracker_local_jsonl(tmp_path):
    out_dir = str(tmp_path / "experiments")
    tracker = ExperimentTracker(
        experiment_name="test_experiment",
        run_name="run_001",
        backends=["jsonl"],
        output_dir=out_dir
    )

    tracker.log_params({"learning_rate": 2e-4, "batch_size": 4})
    tracker.log_metrics({"loss": 1.45, "bleu": 25.4}, step=10)
    tracker.close()

    log_path = os.path.join(out_dir, "run_001.jsonl")
    assert os.path.exists(log_path)

    lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))

    assert len(lines) == 2
    assert lines[0]["event"] == "params"
    assert lines[0]["data"]["learning_rate"] == 2e-4
    assert lines[1]["event"] == "metrics"
    assert lines[1]["step"] == 10
    assert lines[1]["data"]["bleu"] == 25.4
