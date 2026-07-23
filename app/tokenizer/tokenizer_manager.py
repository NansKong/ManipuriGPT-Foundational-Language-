from typing import Dict, Any, Optional, Union, List
from app.configs.settings import settings
from app.tokenizer.backends import (
    BaseTokenizerBackend,
    HFTokenizerBackend,
    SentencePieceTokenizerBackend,
    TikTokenTokenizerBackend,
    CustomTokenizerBackend
)
from app.utils.logger import logger

class TokenizerManager:
    """
    Centralized manager handling tokenizer loading (with primary focus on HuggingFace backend),
    caching, padding tokens, EOS/BOS setup, max sequence length, and token counting utilities.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tokenizer_instance: Optional[Any] = None,
        backend: Optional[str] = None
    ):
        config = config or {}
        settings_config = {}
        if hasattr(settings, "tokenizer") and hasattr(settings.tokenizer, "to_dict"):
            settings_config = settings.tokenizer.to_dict()
        elif isinstance(settings, dict):
            settings_config = settings
        
        self.model_name = config.get("model_name", config.get("model", settings_config.get("model", "qwen2.5")))
        self.max_length = config.get("max_length", settings_config.get("max_length", 2048))
        self.padding_side = config.get("padding_side", settings_config.get("padding_side", "right"))
        self.truncation = config.get("truncation", settings_config.get("truncation", True))
        self.trust_remote_code = config.get("trust_remote_code", settings_config.get("trust_remote_code", True))
        self.add_special_tokens = config.get("add_special_tokens", settings_config.get("add_special_tokens", True))
        self.backend_name = (backend or config.get("backend", "hf")).lower()

        self.config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "padding_side": self.padding_side,
            "truncation": self.truncation,
            "trust_remote_code": self.trust_remote_code,
            "add_special_tokens": self.add_special_tokens,
            "backend": self.backend_name
        }
        config.update(self.config)

        if tokenizer_instance is not None:
            # Wrap instance in HFTokenizerBackend to keep active interface unified
            self.backend = HFTokenizerBackend(tokenizer_instance=tokenizer_instance)
            self.tokenizer = tokenizer_instance
            self._ensure_special_tokens()
        else:
            self.backend = self._initialize_backend(self.backend_name)
            self.tokenizer = self.backend.load_tokenizer(self.model_name, trust_remote_code=self.trust_remote_code)
            self._ensure_special_tokens()

    def _initialize_backend(self, backend_name: str) -> BaseTokenizerBackend:
        if backend_name in ["hf", "huggingface", "auto", "custom"]:
            return HFTokenizerBackend()
        elif backend_name in ["spm", "sentencepiece"]:
            return SentencePieceTokenizerBackend()
        elif backend_name in ["tiktoken"]:
            return TikTokenTokenizerBackend()
        else:
            logger.warning(f"TokenizerManager: Backend '{backend_name}' not active, defaulting to HuggingFace backend.")
            return HFTokenizerBackend()

    def _ensure_special_tokens(self, tokenizer: Optional[Any] = None) -> None:
        tok = tokenizer or self.tokenizer
        if tok is None:
            return
        if not hasattr(tok, "pad_token") or tok.pad_token is None:
            if hasattr(tok, "eos_token") and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif hasattr(tok, "unk_token") and tok.unk_token is not None:
                tok.pad_token = tok.unk_token
            else:
                tok.pad_token = "[PAD]"
                if hasattr(tok, "add_special_tokens"):
                    try:
                        tok.add_special_tokens({"pad_token": "[PAD]"})
                    except Exception:
                        pass

        if hasattr(tok, "padding_side"):
            tok.padding_side = self.padding_side

    def encode(self, text: Union[str, List[str]], max_length: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        max_len = max_length or self.max_length
        return self.backend.encode(
            text,
            max_length=max_len,
            truncation=self.truncation,
            padding=kwargs.get("padding", False),
            add_special_tokens=kwargs.get("add_special_tokens", self.add_special_tokens),
            return_tensors=kwargs.get("return_tensors", None)
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def get_pad_token_id(self) -> int:
        return self.backend.get_pad_token_id()

    def get_eos_token_id(self) -> int:
        return self.backend.get_eos_token_id()

    def get_bos_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "bos_token_id", None)

    def count_tokens(self, text: str) -> int:
        return self.backend.count_tokens(text)

    def get_tokenizer(self, name: str = "default") -> "TokenizerManager":
        """
        Standardized API returning a unified tokenizer wrapper/instance exposing
        encode, decode, count_tokens, get_pad_token_id, get_eos_token_id, and vocab_size.
        """
        if name in ["default", "indic", self.model_name, self.backend_name]:
            return self
        try:
            return TokenizerManager(config={"model_name": name, "backend": self.backend_name})
        except Exception as e:
            logger.warning(f"TokenizerManager: Could not load requested tokenizer '{name}' ({e}), returning self.")
            return self

    @property
    def vocab_size(self) -> int:
        if hasattr(self.backend, "get_vocab_size"):
            try:
                return self.backend.get_vocab_size()
            except Exception:
                pass
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        if hasattr(self.tokenizer, "get_vocab_size"):
            return self.tokenizer.get_vocab_size()
        if hasattr(self.tokenizer, "__len__"):
            return len(self.tokenizer)
        return 32768

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        return self.encode(text, **kwargs)
