"""
TrainingConfig module defining unified configurations for multi-model foundation training,
continued pretraining (DAPT), instruction tuning (SFT), and preference optimization (DPO/ORPO).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class TrainingConfig:
    """
    Declarative configuration for ManipuriGPT model training across backends.
    Supports modes: `full`, `lora`, `qlora`, `dpo`, `orpo`, `continued_pretraining`, `sft`.
    """
    model_name: str = "qwen_2_5_3b"
    mode: str = "sft"  # "full", "lora", "qlora", "dpo", "orpo", "continued_pretraining", "sft"
    backend: str = "transformers"  # "transformers", "trl", "peft", "unsloth", "deepspeed"
    
    # Precision & Optimization
    precision: str = "bf16"  # "bf16", "fp16", "fp32"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Batching & Context
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    num_epochs: int = 3
    max_steps: Optional[int] = None
    
    # LoRA / QLoRA specific settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    use_qlora_4bit: bool = True
    
    # DPO / ORPO specific settings
    dpo_beta: float = 0.1
    orpo_beta: float = 0.1
    
    # Distributed & Hardware
    deepspeed_config_path: Optional[str] = None
    gradient_checkpointing: bool = True
    output_dir: str = "artifacts/models/checkpoints"
    save_steps: int = 250
    eval_steps: int = 250
    logging_steps: int = 10
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }
