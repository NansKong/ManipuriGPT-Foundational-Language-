from typing import Dict, Any, List, Optional
from app.tokenizer.tokenizer_manager import TokenizerManager
from app.utils.logger import logger

class ModelSpecification:
    """
    Specification for a model family/architecture. Encapsulates tokenization, context limits,
    chat template requirements, LoRA targets, and default quantization configs.
    """
    def __init__(
        self,
        short_name: str,
        name: str,
        max_context_length: int,
        chat_template: str,
        lora_target_modules: List[str],
        supported_tasks: List[str],
        eos_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        memory_estimates: Optional[Dict[str, Any]] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None
    ):
        self.short_name = short_name.lower()
        self.name = name
        self.aliases = [a.lower() for a in (aliases or [])]
        self.max_context_length = max_context_length
        self.chat_template = chat_template
        self.lora_target_modules = lora_target_modules
        self.supported_tasks = [t.lower() for t in supported_tasks]
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.quantization_config = quantization_config or {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        }
        self.memory_estimates = memory_estimates or {
            "4bit_vram_gb": 4.0,
            "8bit_vram_gb": 8.0,
            "16bit_vram_gb": 16.0
        }
        self.rope_scaling = rope_scaling or {"type": "default", "factor": 1.0}
        self._cached_tokenizer_manager: Optional[TokenizerManager] = None

    def tokenizer(
        self,
        config_override: Optional[Dict[str, Any]] = None,
        tokenizer_instance: Optional[Any] = None,
        backend: str = "hf"
    ) -> TokenizerManager:
        """
        Returns a configured TokenizerManager for this model specification.
        Caches the instance unless overrides or custom instances are supplied.
        """
        if self._cached_tokenizer_manager is not None and config_override is None and tokenizer_instance is None:
            return self._cached_tokenizer_manager

        config = {
            "model_name": self.name,
            "max_length": self.max_context_length,
            "padding_side": "right"
        }
        if config_override:
            config.update(config_override)

        manager = TokenizerManager(
            config=config,
            tokenizer_instance=tokenizer_instance,
            backend=backend
        )
        if config_override is None and tokenizer_instance is None:
            self._cached_tokenizer_manager = manager
        return manager


class ModelRegistry:
    """
    Registry for managing and selecting ModelSpecifications by short name or full model path.
    """
    def __init__(self):
        self._specs: Dict[str, ModelSpecification] = {}

    def register(self, spec: ModelSpecification) -> None:
        """Registers a ModelSpecification under its short name, full path, and any aliases."""
        self._specs[spec.short_name.lower()] = spec
        self._specs[spec.name.lower()] = spec
        for alias in getattr(spec, "aliases", []):
            self._specs[alias.lower()] = spec
        logger.debug(f"ModelRegistry: Registered model '{spec.short_name}' -> '{spec.name}'")

    def get(self, name: str) -> ModelSpecification:
        """Retrieves a ModelSpecification by short name (e.g. 'qwen2.5') or canonical path."""
        key = name.lower()
        if key not in self._specs:
            available = list(set(s.short_name for s in self._specs.values()))
            raise KeyError(f"Model specification '{name}' not found in ModelRegistry. Available models: {sorted(available)}")
        return self._specs[key]

    def list_models(self) -> List[str]:
        """Returns a sorted list of unique short model names registered."""
        return sorted(list(set(s.short_name for s in self._specs.values())))


# Global singleton instance
model_registry = ModelRegistry()
