"""
TrainingConfig module defining unified configurations for multi-model foundation training,
continued pretraining (DAPT), instruction tuning (SFT), and preference optimization (DPO/ORPO).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class TrainingConfig:
    """
    Declarative configuration for ManipuriGPT model training across backends.
    Supports modes: `full`, `lora`, `qlora`, `dpo`, `orpo`, `continued_pretraining`, `sft`.
    """
    model_name: str = "smollm_135m"
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    mode: str = "continued_pretraining"  # "full", "lora", "qlora", "dpo", "orpo", "continued_pretraining", "sft"
    backend: str = "transformers"  # "transformers", "trl", "peft", "unsloth", "deepspeed"
    
    # Dataset configuration
    dataset_name_or_path: str = "nanskong/ManipuriGPT-Corpus-v1.0"
    dataset_split: Optional[str] = None
    use_streaming: bool = False
    is_packed: bool = False
    
    # Precision & Optimization
    precision: str = "fp16"  # "fp16", "bf16", "fp32" (Auto-selects fp16 for Tesla T4)
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
    
    # Checkpointing & Strategy
    output_dir: str = "artifacts/models/checkpoints"
    resume_from_checkpoint: Optional[Union[str, bool]] = None
    save_total_limit: int = 3
    save_steps: int = 500
    save_strategy: str = "steps"
    eval_steps: int = 500
    eval_strategy: str = "steps"
    logging_steps: int = 10
    logging_strategy: str = "steps"
    
    # Hugging Face Hub Integration
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    hub_strategy: str = "every_save"
    
    # Distributed & Hardware
    deepspeed_config_path: Optional[str] = None
    gradient_checkpointing: bool = True
    seed: int = 42
    dry_run: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }
