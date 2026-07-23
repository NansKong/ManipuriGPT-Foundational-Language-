"""
Unit test for model checkpointing (`CheckpointManager`) and multi-target exports (`UnifiedExporter`).
Verifies best-checkpoint tracking, pruning, and HF/GGUF/ONNX export generation.
"""

import os
import pytest
from app.models.checkpointing import CheckpointManager
from app.exports.exporter import UnifiedExporter


def test_checkpoint_manager_best_tracking_and_pruning(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    mgr = CheckpointManager(base_output_dir=ckpt_dir, metric_name="eval_loss", metric_minimize=True, max_keep=2)

    # Step 100: loss 1.5
    dir1 = os.path.join(ckpt_dir, "step-100")
    os.makedirs(dir1, exist_ok=True)
    mgr.save_checkpoint_metadata(step=100, checkpoint_path=dir1, metrics={"eval_loss": 1.5})
    assert mgr.get_best_checkpoint() == dir1

    # Step 200: loss 1.2 (better)
    dir2 = os.path.join(ckpt_dir, "step-200")
    os.makedirs(dir2, exist_ok=True)
    mgr.save_checkpoint_metadata(step=200, checkpoint_path=dir2, metrics={"eval_loss": 1.2})
    assert mgr.get_best_checkpoint() == dir2

    # Step 300: loss 1.4 (worse), causes step-100 to be pruned since max_keep=2
    dir3 = os.path.join(ckpt_dir, "step-300")
    os.makedirs(dir3, exist_ok=True)
    mgr.save_checkpoint_metadata(step=300, checkpoint_path=dir3, metrics={"eval_loss": 1.4})
    assert mgr.get_best_checkpoint() == dir2  # Best remains dir2
    assert not os.path.exists(dir1)  # Pruned


def test_unified_exporter_all_targets(tmp_path):
    ckpt_dir = str(tmp_path / "dummy_checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    exporter = UnifiedExporter()
    results = exporter.export_all(checkpoint_dir=ckpt_dir, model_name="test_manipuri_135m", simulate=True)
    
    assert "hf" in results
    assert "gguf" in results
    assert "onnx" in results
    assert os.path.exists(results["hf"]["readme_path"])
    assert os.path.exists(results["gguf"]["output_path"])
    assert os.path.exists(results["onnx"]["output_path"])
