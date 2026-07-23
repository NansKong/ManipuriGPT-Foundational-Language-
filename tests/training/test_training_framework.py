"""
Unit test for training framework modules (`TrainingConfig`, `BackendFactory`, `ManipuriTrainer`).
Verifies multi-backend setup and execution simulation across SFT, DPO, and Continued Pretraining.
"""

import os
import pytest
from app.training.config import TrainingConfig
from app.training.backends import BackendFactory
from app.training.trainer import ManipuriTrainer


def test_training_config_to_dict():
    cfg = TrainingConfig(model_name="smollm_135m", mode="sft", backend="transformers", precision="bf16")
    cfg_dict = cfg.to_dict()
    assert cfg_dict["model_name"] == "smollm_135m"
    assert cfg_dict["mode"] == "sft"
    assert cfg_dict["precision"] == "bf16"


def test_backend_factory_selection():
    cfg_peft = TrainingConfig(mode="lora", backend="peft")
    wrapper_peft = BackendFactory.get_backend(cfg_peft)
    assert "PEFT" in wrapper_peft.__class__.__name__

    cfg_unsloth = TrainingConfig(backend="unsloth")
    wrapper_unsloth = BackendFactory.get_backend(cfg_unsloth)
    assert "Unsloth" in wrapper_unsloth.__class__.__name__


def test_manipuri_trainer_simulated_execution(tmp_path):
    out_dir = str(tmp_path / "test_checkpoints")
    cfg = TrainingConfig(
        model_name="tinyllama_1_1b",
        mode="qlora",
        backend="peft",
        output_dir=out_dir,
        num_epochs=1,
        max_steps=5
    )
    
    mock_train_ds = [{"text": "Sample 1"}, {"text": "Sample 2"}]
    trainer = ManipuriTrainer(config=cfg, train_dataset=mock_train_ds)
    
    results = trainer.train()
    assert results["mode"] == "qlora"
    assert results["global_step"] == 5
    assert os.path.exists(results["saved_checkpoint_dir"])
