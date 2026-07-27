"""
ModelLoader module for instantiating base causal models and tokenizers.
Handles model registry resolution, automatic precision selection (T4 fp16 optimization),
vocab embedding resizing, 4-bit / 8-bit QLoRA configs, and FlashAttention / SDPA setup.
"""

from typing import Tuple, Any, Optional, Dict
from app.training.config import TrainingConfig
from app.models.registry import model_registry
from app.utils.logger import logger


class ModelLoader:
    """
    Stateless loader instantiating Hugging Face models and tokenizers
    configured for foundation training, DAPT, SFT, and QLoRA.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config

    def select_torch_dtype(self) -> Tuple[Any, str]:
        """
        Automatically selects the optimal torch dtype and precision string based on GPU hardware.
        Defaults Tesla T4 (Colab free tier) to fp16, Ampere/Ada/Hopper to bf16, and CPU to fp32.
        """
        import torch

        if not torch.cuda.is_available():
            logger.info("ModelLoader: CUDA unavailable. Selected precision: fp32 (CPU)")
            return torch.float32, "fp32"

        device_name = torch.cuda.get_device_name(0)
        logger.info(f"ModelLoader: Detected GPU hardware -> '{device_name}'")

        # Explicit precision override from config if user explicitly set non-default
        if self.config.precision == "fp16":
            return torch.float16, "fp16"
        elif self.config.precision == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"

        # Hardware-aware auto selection:
        # Tesla T4, GTX 1080/2080, V100 do not support fast native BF16
        if "T4" in device_name or "V100" in device_name or "1080" in device_name or "2080" in device_name:
            logger.info("ModelLoader: Hardware lacks fast native BF16 support (Tesla T4/V100). Auto-selected precision: fp16")
            return torch.float16, "fp16"
        elif torch.cuda.is_bf16_supported():
            logger.info("ModelLoader: Ampere/Ada/Hopper GPU detected with BF16 support. Auto-selected precision: bf16")
            return torch.bfloat16, "bf16"
        else:
            logger.info("ModelLoader: Defaulting GPU precision to fp16")
            return torch.float16, "fp16"

    def load(self) -> Tuple[Any, Any]:
        """
        Loads and prepares target model and tokenizer.
        Returns (model, tokenizer).
        """
        # Resolve target model spec & paths
        model_name_or_path = self.config.model_name_or_path
        try:
            spec = model_registry.get(self.config.model_name)
            if not model_name_or_path:
                model_name_or_path = spec.name
        except KeyError:
            if not model_name_or_path:
                model_name_or_path = self.config.model_name

        tokenizer_path = self.config.tokenizer_name_or_path or model_name_or_path

        logger.info(f"ModelLoader: Loading model '{model_name_or_path}' and tokenizer '{tokenizer_path}'...")

        torch_dtype, effective_precision = self.select_torch_dtype()

        tokenizer = self._load_tokenizer(tokenizer_path)
        model = self._load_model(model_name_or_path, torch_dtype, len(tokenizer))

        return model, tokenizer

    def _load_tokenizer(self, tokenizer_path: str) -> Any:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"ModelLoader: Tokenizer loaded successfully (vocab_size={len(tokenizer):,})")
            return tokenizer
        except Exception as e:
            logger.warning(f"ModelLoader: Could not load real tokenizer '{tokenizer_path}' ({e}). Using mock tokenizer fallback.")
            return MockTokenizerWrapper()

    def _load_model(self, model_path: str, torch_dtype: Any, vocab_size: int) -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            quantization_config = None
            if self.config.mode == "qlora" and self.config.use_qlora_4bit and torch.cuda.is_available():
                logger.info("ModelLoader: Configuring BitsAndBytes 4-bit NF4 quantization for QLoRA...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Resize token embeddings if tokenizer vocab size differs
            if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
                current_vocab = model.get_input_embeddings().weight.shape[0]
                if current_vocab != vocab_size:
                    logger.info(f"ModelLoader: Resizing model token embeddings ({current_vocab} -> {vocab_size})...")
                    model.resize_token_embeddings(vocab_size)

            if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                logger.info("ModelLoader: Enabling gradient checkpointing...")
                model.gradient_checkpointing_enable()

            logger.info("ModelLoader: Model loaded successfully.")
            return model
        except Exception as e:
            logger.warning(f"ModelLoader: Could not load real model '{model_path}' ({e}). Using simulated model object.")
            return {
                "name": model_path,
                "simulated": True,
                "vocab_size": vocab_size,
                "dtype": str(torch_dtype)
            }


class MockTokenizerWrapper:
    """Mock tokenizer fallback for offline dry-run testing without external dependencies."""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.eos_token_id = 3

    def __len__(self):
        return 32000

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {
                "input_ids": [[101, 200, 300, 102] for _ in text],
                "attention_mask": [[1, 1, 1, 1] for _ in text]
            }
        return {
            "input_ids": [101, 200, 300, 102],
            "attention_mask": [1, 1, 1, 1]
        }
