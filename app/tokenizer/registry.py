from typing import Dict, Any, Optional
from app.utils.logger import logger

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# Standard model shortcuts mapped to canoncial HF model IDs
TOKENIZERS: Dict[str, str] = {
    "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma3": "google/gemma-3-4b-it",
    "gemma2": "google/gemma-2-2b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

class TokenizerRegistry:
    """
    Registry resolving model shortcut keys or direct HF model paths to tokenizer instances with caching.
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, model_name_or_id: str, trust_remote_code: bool = True, use_fast: bool = True, **kwargs) -> Any:
        """
        Resolves model_name_or_id using TOKENIZERS shortcut dictionary or loads directly via AutoTokenizer.
        Caches loaded instances in memory.
        """
        canonical_id = TOKENIZERS.get(model_name_or_id.lower(), model_name_or_id)
        cache_key = f"{canonical_id}_{trust_remote_code}_{use_fast}"

        if cache_key in self._cache:
            logger.debug(f"TokenizerRegistry: Returning cached tokenizer for '{canonical_id}'")
            return self._cache[cache_key]

        if AutoTokenizer is None:
            raise RuntimeError("transformers package is not installed. Cannot load AutoTokenizer.")

        logger.info(f"TokenizerRegistry: Loading tokenizer from '{canonical_id}' (trust_remote_code={trust_remote_code})")
        tokenizer = AutoTokenizer.from_pretrained(
            canonical_id,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            **kwargs
        )
        self._cache[cache_key] = tokenizer
        return tokenizer

    def register_shortcut(self, shortcut: str, canonical_id: str) -> None:
        """Registers a custom shortcut mapping."""
        TOKENIZERS[shortcut.lower()] = canonical_id
        logger.debug(f"TokenizerRegistry: Registered shortcut '{shortcut}' -> '{canonical_id}'")

    def clear_cache(self) -> None:
        self._cache.clear()

# Global singleton instance
tokenizer_registry = TokenizerRegistry()
