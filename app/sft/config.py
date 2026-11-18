"""
Configuration module for ManipuriGPT Phase 8 SFT (Supervised Fine-Tuning).
Defines paths, prompt templates, ChatML parameters, task distribution targets,
and QLoRA hyperparameter presets.
"""

from pathlib import Path
from app.training.meitei_lon import TOKENIZER
from dataclasses import dataclass, field
from typing import Dict, List, Any

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SFT_DATA_DIR = BASE_DIR / "datasets" / "sft"
RAW_DATA_DIR = SFT_DATA_DIR / "raw"
PROCESSED_DATA_DIR = SFT_DATA_DIR / "processed"
MERGED_DATA_DIR = SFT_DATA_DIR / "merged" / "v1"
STATS_DATA_DIR = SFT_DATA_DIR / "stats" / "v1"
DEFAULT_OUTPUT_MODEL_DIR = BASE_DIR / "models" / "Meitei-lon-SFT-v1.2"

# Fixed System Prompt & Special Tokens
SYSTEM_PROMPT = (
    "You are ManipuriGPT, a helpful assistant whose primary language is Meitei-lon (Manipuri). "
    "Default to Meitei-lon in Meitei Mayek. Match Bengali-script or Romanized Meitei-lon when the user writes that way. Follow explicit language and writing-system requests."
)

CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"
IGNORE_INDEX = -100

import json
import yaml

SFT_CONFIG_YAML = Path(__file__).resolve().parent / "sft_config.yaml"

def load_sft_config() -> Dict[str, Any]:
    """Loads configuration dynamically from sft_config.yaml."""
    if SFT_CONFIG_YAML.exists():
        try:
            with open(SFT_CONFIG_YAML, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}

_yaml_cfg = load_sft_config()

# Task Distribution Targets (Default Target Sum: 30,000 Grounded)
TASK_TARGETS: Dict[str, int] = _yaml_cfg.get(
    "task_targets",
    {
        "translation_en_to_mm": 8000,
        "translation_mm_to_en": 5500,
        "definitions": 4500,
        "conversation": 4500,
        "qa": 4500,
        "script_conversion_bn_to_mm": 3000,
    }
)

MAX_CORPUS_SOURCE_RATIO = float(_yaml_cfg.get("max_corpus_source_ratio", 0.30))
TOTAL_TARGET_PAIRS = sum(TASK_TARGETS.values())
SUPPORTED_TASKS = list(TASK_TARGETS.keys())

@dataclass
class SFTDataConfig:
    raw_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    merged_dir: Path = MERGED_DATA_DIR
    stats_dir: Path = STATS_DATA_DIR
    train_ratio: float = 0.90
    val_ratio: float = 0.10
    seed: int = 42

@dataclass
class QLoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

LOCAL_MODEL_PATH = BASE_DIR / "models" / "ManipuriGPT-135M-Base-v1.0"

def resolve_model_path(hf_repo: str = "nanskong/ManipuriGPT-135M-Base-v1.0") -> str:
    """Checks if model exists locally for faster loading; falls back to HF Hub if missing."""
    if (LOCAL_MODEL_PATH / "config.json").exists() and (
        (LOCAL_MODEL_PATH / "model.safetensors").exists() or (LOCAL_MODEL_PATH / "pytorch_model.bin").exists()
    ):
        return str(LOCAL_MODEL_PATH)
    return hf_repo

def resolve_tokenizer_path(hf_repo: str = "nanskong/ManipuriGPT-Tokenizer-v1") -> str:
    """Checks if tokenizer exists locally for faster loading; falls back to HF Hub if missing."""
    if (LOCAL_MODEL_PATH / "tokenizer.json").exists() or (LOCAL_MODEL_PATH / "tokenizer_config.json").exists():
        return str(LOCAL_MODEL_PATH)
    return hf_repo

@dataclass
class SFTTrainingConfig:
    base_model_path: str = field(default_factory=resolve_model_path)
    tokenizer_path: str = str(TOKENIZER)
    output_dir: Path = DEFAULT_OUTPUT_MODEL_DIR
    max_seq_length: int = 512
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 500
    fp16: bool = True
    gradient_checkpointing: bool = True
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    seed: int = 42
