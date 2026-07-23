"""
Production CLI script for launching model training runs (`Phase 5`).
Usage: `python -m app.scripts.train --model smollm_135m --mode qlora --backend peft --epochs 3`
"""

import argparse
import sys
from typing import Optional, List, Dict, Any, Union, Tuple
from app.training.config import TrainingConfig
from app.training.trainer import ManipuriTrainer
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Training CLI")
    parser.add_argument("--model", type=str, default="smollm_135m", help="Short name of target model from registry")
    parser.add_argument("--mode", type=str, default="sft", choices=["full", "lora", "qlora", "dpo", "orpo", "continued_pretraining", "sft"], help="Training mode")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "trl", "peft", "unsloth", "deepspeed"], help="Training execution backend")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Compute precision")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/checkpoints", help="Output directory for saved checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without running lengthy training loops")
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    logger.info(f"CLI: Launching training run with parameters: {parsed}")

    cfg = TrainingConfig(
        model_name=parsed.model,
        mode=parsed.mode,
        backend=parsed.backend,
        precision=parsed.precision,
        num_epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        output_dir=parsed.output_dir,
        max_steps=2 if parsed.dry_run else None
    )

    dummy_ds = [{"text": "Sample training data text for Manipuri language."}]
    trainer = ManipuriTrainer(config=cfg, train_dataset=dummy_ds)
    results = trainer.train()
    
    logger.info(f"CLI: Training execution finished -> {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
