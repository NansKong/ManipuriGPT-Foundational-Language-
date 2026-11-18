"""
Master Dry-Run Script for ManipuriGPT Phase 8 LoRA SFT Training.
Executes a 10-step trial training run on RTX 2060 (6GB VRAM) to verify:
1. Model loading & PEFT LoRA adapter initialization
2. PyTorch CUDA memory allocation & VRAM headroom
3. ChatML DataCollator & loss masking (-100 index)
4. Training loss reduction & adapter saving
"""

import sys
import torch
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.sft.config import MERGED_DATA_DIR, SFTTrainingConfig, QLoRAConfig
from app.training.meitei_lon import SFT_DATA
from app.sft.trainer import ManipuriGPTSFTTrainer


def main():
    print("=========================================================")
    print("  ManipuriGPT Phase 8 SFT — QLoRA Dry-Run (10 Steps)")
    print("=========================================================")

    train_jsonl = SFT_DATA / "train.jsonl"
    val_jsonl = SFT_DATA / "validation.jsonl"

    if not train_jsonl.exists() or not val_jsonl.exists():
        print(f"[ERROR] Split datasets not found at {SFT_DATA}")
        print("Please run 'python -m app.scripts.prepare_meitei_lon' first.")
        sys.exit(1)

    # CUDA / GPU Hardware Diagnostics
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Hardware] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[Hardware] Running on CPU (CUDA not available)")

    # Configure dry-run hyperparameters
    train_config = SFTTrainingConfig(
        output_dir=BASE_DIR / "models" / "ManipuriGPT-SFT-dry-run",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        fp16=torch.cuda.is_available(),
    )
    lora_config = QLoRAConfig()

    print(f"[Config] Base Model: {train_config.base_model_path}")
    print(f"[Config] Tokenizer: {train_config.tokenizer_path}")
    print(f"[Config] Max Seq Length: {train_config.max_seq_length}")
    print(f"[Config] LoRA rank r={lora_config.r}, alpha={lora_config.lora_alpha}")

    trainer = ManipuriGPTSFTTrainer(train_config=train_config, lora_config=lora_config)

    print("\n--- Starting 10-Step LoRA SFT Dry-Run ---")
    result = trainer.train(train_jsonl=train_jsonl, val_jsonl=val_jsonl, max_steps=10)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"\n[VRAM Usage] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    print("\n=========================================================")
    print("  [SUCCESS] Phase 8 SFT Dry-Run Complete!")
    print(f"  Fine-tuned adapter saved to: {train_config.output_dir}")
    print("=========================================================")


if __name__ == "__main__":
    main()
