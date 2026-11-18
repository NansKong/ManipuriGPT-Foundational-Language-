"""
Master Production Training Script for ManipuriGPT Phase 8 LoRA SFT Training.
Executes full LoRA SFT training across 3 epochs (~26,969 training records) on RTX 2060 (6GB VRAM)
with automatic checkpoint detection, fault tolerance recovery, and best model evaluation selection.
"""

import sys
import argparse
import torch
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.sft.config import MERGED_DATA_DIR, SFTTrainingConfig, QLoRAConfig
from app.sft.trainer import ManipuriGPTSFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="ManipuriGPT Phase 8 SFT Production Training CLI")
    parser.add_argument("--fresh", action="store_true", help="Force fresh training from scratch, ignoring existing checkpoints.")
    parser.add_argument("--resume", type=str, default=None, nargs="?", const="auto", help="Explicitly resume training. Pass 'auto' or leave blank for automatic latest checkpoint recovery, or provide path.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--max-steps", type=int, help="Optional optimizer-step limit for a pilot")
    parser.add_argument("--base-model", type=Path, help="Local base or continued-pretraining final_model directory")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "models" / "Meitei-lon-SFT-v1.2")
    parser.add_argument("--save-steps", type=int, default=500, help="Save a checkpoint every N steps (default: 500)")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate model every N steps (default: 500)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1.")

    from app.training.meitei_lon import SFT_DATA
    train_jsonl = SFT_DATA / "train.jsonl"
    val_jsonl = SFT_DATA / "validation.jsonl"

    if not train_jsonl.exists() or not val_jsonl.exists():
        print(f"[ERROR] Split datasets not found at {SFT_DATA}")
        print("Prepare the reviewed data with python -m app.scripts.prepare_meitei_lon.")
        sys.exit(1)

    train_config = SFTTrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    lora_config = QLoRAConfig()
    if args.base_model:
        train_config.base_model_path = str(args.base_model.resolve())

    output_dir = Path(train_config.output_dir)

    # Checkpoint Recovery Detection
    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    resume_target = None

    if args.fresh:
        resume_target = None
    elif args.resume and args.resume != "auto":
        resume_target = args.resume
    elif args.resume == "auto" and last_checkpoint:
        resume_target = last_checkpoint
    elif args.resume == "auto":
        raise ValueError("No checkpoint exists in the selected output directory")
    elif last_checkpoint and not args.fresh:
        raise ValueError("Output contains a checkpoint. Use --resume auto or --fresh explicitly.")

    print("=========================================================")
    print("  ManipuriGPT Phase 8 SFT — Full Production Training")
    print("=========================================================")

    # Hardware Diagnostics
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Hardware] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[Hardware] Running on CPU (CUDA not available)")

    print(f"[Config] Output Directory: {output_dir}")
    print(f"[Config] Base Model: {train_config.base_model_path}")
    print(f"[Config] Tokenizer: {train_config.tokenizer_path}")
    print(f"[Config] Max Seq Length: {train_config.max_seq_length}")
    print(f"[Config] Batch Size: {train_config.per_device_train_batch_size} per device")
    print(f"[Config] Gradient Accumulation Steps: {train_config.gradient_accumulation_steps}")
    print(f"[Config] Effective Batch Size: {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps}")
    print(f"[Config] Epochs: {train_config.num_train_epochs}")
    print(f"[Config] Learning Rate: {train_config.learning_rate} ({train_config.lr_scheduler_type} scheduler, {train_config.warmup_ratio} warmup ratio)")
    print(f"[Config] Save & Eval Frequency: Every {train_config.save_steps} steps (Keeping max {train_config.save_total_limit} checkpoints)")
    print(f"[Config] Gradient Checkpointing: {train_config.gradient_checkpointing}")
    print(f"[Config] Best Model Selection: {train_config.load_best_model_at_end} (metric: {train_config.metric_for_best_model})")
    print(f"[Config] LoRA Rank r={lora_config.r}, Alpha={lora_config.lora_alpha}")

    if resume_target:
        print(f"\n[Checkpoint Recovery] Resuming: YES -> {resume_target}")
    else:
        print("\n[Checkpoint Recovery] Resuming: NO (Starting fresh training)")

    from dataclasses import asdict
    from app.training.meitei_lon import record_run_inputs
    from app.sft.config import SYSTEM_PROMPT
    record_run_inputs(output_dir, resume_target, train_config.base_model_path, train_config.tokenizer_path,
                      [train_jsonl, val_jsonl], {"lora": asdict(lora_config), "system_prompt": SYSTEM_PROMPT,
                      "training": {k: v for k, v in asdict(train_config).items() if k not in
                                   {"output_dir", "num_train_epochs", "base_model_path", "tokenizer_path"}}})
    trainer = ManipuriGPTSFTTrainer(train_config=train_config, lora_config=lora_config)

    print("\n--- Launching Fault-Tolerant LoRA SFT Training Engine ---")
    result = trainer.train(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        max_steps=args.max_steps,
        resume_from_checkpoint=resume_target,
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"\n[VRAM Usage] Peak Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    print("\n=========================================================")
    print("  [SUCCESS] Phase 8 SFT Production Training Complete!")
    print(f"  Best fine-tuned model saved to: {train_config.output_dir}")
    print("=========================================================")


if __name__ == "__main__":
    main()
