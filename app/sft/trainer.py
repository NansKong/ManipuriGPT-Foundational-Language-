"""
Trainer module for ManipuriGPT Phase 8 LoRA SFT.
Configured for RTX 2060 (6GB VRAM) memory optimization with PEFT LoRA adapters,
gradient accumulation, FP16 precision, and ChatML loss masking.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from app.sft.config import SFTTrainingConfig, QLoRAConfig
from app.sft.prompt_formatter import ChatMLCollator


class JSONLDataset(torch.utils.data.Dataset):
    """Simple PyTorch Dataset loading JSONL SFT records."""

    def __init__(self, jsonl_path: Path):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class ManipuriGPTSFTTrainer:
    def __init__(
        self,
        train_config: SFTTrainingConfig = SFTTrainingConfig(),
        lora_config: QLoRAConfig = QLoRAConfig(),
    ):
        self.train_config = train_config
        self.lora_config = lora_config

    def setup_model_and_tokenizer(self):
        print(f"[SFT Trainer] Loading tokenizer: {self.train_config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.train_config.tokenizer_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|im_end|>"

        print(f"[SFT Trainer] Loading base model: {self.train_config.base_model_path}")
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if (torch.cuda.is_available() and self.train_config.fp16) else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.train_config.base_model_path,
            device_map=device_map,
            dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Configure LoRA PEFT
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, peft_config)

        if getattr(self.train_config, "gradient_checkpointing", True):
            print("[SFT Trainer] Enabling gradient checkpointing for VRAM efficiency...")
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model.print_trainable_parameters()

        return model, tokenizer

    def train(
        self,
        train_jsonl: Path,
        val_jsonl: Path,
        max_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Any] = None,
    ):
        model, tokenizer = self.setup_model_and_tokenizer()

        train_dataset = JSONLDataset(train_jsonl)
        val_dataset = JSONLDataset(val_jsonl)

        collator = ChatMLCollator(tokenizer, max_seq_length=self.train_config.max_seq_length)

        out_dir = Path(self.train_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Calculate dynamic warmup_steps from warmup_ratio to avoid deprecation warnings
        eff_batch_size = self.train_config.per_device_train_batch_size * self.train_config.gradient_accumulation_steps
        approx_total_steps = (len(train_dataset) // max(1, eff_batch_size)) * self.train_config.num_train_epochs
        calc_warmup_steps = int((max_steps or approx_total_steps) * self.train_config.warmup_ratio)

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            learning_rate=self.train_config.learning_rate,
            num_train_epochs=self.train_config.num_train_epochs,
            max_steps=max_steps if max_steps is not None else -1,
            warmup_steps=calc_warmup_steps,
            lr_scheduler_type=self.train_config.lr_scheduler_type,
            logging_steps=self.train_config.logging_steps,
            save_strategy=self.train_config.save_strategy,
            save_steps=self.train_config.save_steps,
            save_total_limit=getattr(self.train_config, "save_total_limit", 3),
            eval_strategy=self.train_config.eval_strategy,
            eval_steps=self.train_config.eval_steps,
            fp16=torch.cuda.is_available() and self.train_config.fp16,
            gradient_checkpointing=getattr(self.train_config, "gradient_checkpointing", True),
            load_best_model_at_end=getattr(self.train_config, "load_best_model_at_end", True),
            metric_for_best_model=getattr(self.train_config, "metric_for_best_model", "eval_loss"),
            greater_is_better=getattr(self.train_config, "greater_is_better", False),
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        if resume_from_checkpoint:
            print(f"[SFT Trainer] Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print("[SFT Trainer] Starting fresh LoRA SFT Training...")

        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model weights & tokenizer
        print(f"[SFT Trainer] Saving fine-tuned adapter to {out_dir}")
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        return train_result
