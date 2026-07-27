"""
Production CLI script for launching model training runs (`Phase 5/6`).
Supports foundation pretraining, continued pretraining (DAPT), SFT, QLoRA, and DPO.

Usage:
  # Continued Pretraining on Google Colab (Tesla T4)
  python -m app.scripts.train --model smollm_135m --mode continued_pretraining --dataset-source nanskong/ManipuriGPT-Corpus-v1.0

  # 20-Step Smoke Test Validation
  python -m app.scripts.train --model smollm_135m --mode continued_pretraining --max-steps 20

  # Dry-Run Initialization Check
  python -m app.scripts.train --dry-run
"""

import argparse
import sys
from typing import Optional, List
from app.training.config import TrainingConfig
from app.training.trainer import ManipuriTrainer
from app.utils.logger import logger


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ManipuriGPT Foundation Pretraining & Fine-Tuning CLI")
    
    # Model & Tokenizer
    parser.add_argument("--model", type=str, default="smollm_135m", help="Registered model short name (e.g. smollm_135m, qwen_2_5_0_5b)")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="Hugging Face hub model ID or local directory override")
    parser.add_argument("--tokenizer-path", "--tokenizer-name-or-path", dest="tokenizer_path", type=str, default="nanskong/ManipuriGPT-Tokenizer-v1", help="Tokenizer repo ID or path")
    
    # Mode & Backend
    parser.add_argument("--mode", type=str, default="continued_pretraining", choices=["continued_pretraining", "sft", "lora", "qlora", "dpo", "orpo", "full"], help="Training execution mode")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "trl", "peft", "unsloth", "deepspeed"], help="Training execution backend")
    
    # Dataset
    parser.add_argument("--dataset-source", "--dataset-name-or-path", dest="dataset_source", type=str, default="nanskong/ManipuriGPT-Corpus-v1.0", help="HF dataset ID or local path to parquet shards")
    parser.add_argument("--dataset-split", type=str, default=None, help="Target split to load")
    parser.add_argument("--is-packed", action="store_true", help="Set if dataset is already tokenized and packed into sequence blocks")
    parser.add_argument("--use-streaming", action="store_true", help="Enable streaming mode for large dataset processing")

    # Precision & Hyperparameters
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="Compute precision (auto selects fp16 on Tesla T4)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence context window length")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (set to e.g. 20 for smoke test validation)")
    
    # Checkpointing & Strategy
    parser.add_argument("--output-dir", type=str, default="artifacts/models/checkpoints", help="Output directory for saved checkpoints")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Checkpoint directory path to resume training from")
    
    # Hugging Face Hub Integration
    parser.add_argument("--push-to-hub", action="store_true", help="Automatically push trained model to Hugging Face Hub")
    parser.add_argument("--hub-model-id", type=str, default=None, help="Hugging Face model repository ID for hub push")
    
    # Validation & Dry Run
    parser.add_argument("--dry-run", action="store_true", help="Simulate initialization and dry-run execution without running training loops")

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    logger.info("=" * 70)
    logger.info(" MANIPURIGPT PRETRAINING & FINE-TUNING EXECUTION ENGINE")
    logger.info("=" * 70)
    logger.info(f"Launch Parameters: {parsed}")

    precision_setting = "fp16" if parsed.precision == "auto" else parsed.precision

    cfg = TrainingConfig(
        model_name=parsed.model,
        model_name_or_path=parsed.model_name_or_path,
        tokenizer_name_or_path=parsed.tokenizer_path,
        mode=parsed.mode,
        backend=parsed.backend,
        dataset_name_or_path=parsed.dataset_source,
        dataset_split=parsed.dataset_split,
        is_packed=parsed.is_packed,
        use_streaming=parsed.use_streaming,
        precision=precision_setting,
        num_epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        gradient_accumulation_steps=parsed.grad_accum_steps,
        max_seq_length=parsed.max_seq_len,
        max_steps=parsed.max_steps,
        output_dir=parsed.output_dir,
        resume_from_checkpoint=parsed.resume_from_checkpoint,
        push_to_hub=parsed.push_to_hub,
        hub_model_id=parsed.hub_model_id,
        dry_run=parsed.dry_run
    )

    trainer = ManipuriTrainer(config=cfg)
    results = trainer.train()
    
    logger.info("=" * 70)
    logger.info(f"CLI: Execution completed successfully -> {results}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
