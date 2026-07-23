from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from app.utils.logger import logger

class BaseTokenizerBackend(ABC):
    """
    Abstract base class for all tokenizer backends (HuggingFace, SentencePiece, TikToken, Custom).
    Ensures unified interface across diverse tokenization implementations.
    """
    def __init__(self, tokenizer_instance: Optional[Any] = None):
        self.tokenizer = tokenizer_instance

    @abstractmethod
    def load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def encode(self, text: Union[str, List[str]], max_length: Optional[int] = None, truncation: bool = True, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def get_pad_token_id(self) -> int:
        pass

    @abstractmethod
    def get_eos_token_id(self) -> int:
        pass


class HFTokenizerBackend(BaseTokenizerBackend):
    """
    Standard HuggingFace AutoTokenizer backend. Active primary backend for Phase 4.
    """
    def load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        from app.tokenizer.registry import tokenizer_registry
        self.tokenizer = tokenizer_registry.get(model_name_or_path, **kwargs)
        return self.tokenizer

    def encode(self, text: Union[str, List[str]], max_length: Optional[int] = None, truncation: bool = True, **kwargs) -> Dict[str, Any]:
        if callable(self.tokenizer):
            return self.tokenizer(
                text,
                max_length=max_length,
                truncation=truncation,
                padding=kwargs.get("padding", False),
                add_special_tokens=kwargs.get("add_special_tokens", True),
                return_tensors=kwargs.get("return_tensors", None)
            )
        # Fallback if self.tokenizer is a mock without __call__ or None
        return {"input_ids": [], "attention_mask": []}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return ""

    def count_tokens(self, text: str) -> int:
        if callable(self.tokenizer):
            tokens = self.tokenizer(text, add_special_tokens=False)
            if isinstance(tokens, dict) and "input_ids" in tokens:
                return len(tokens["input_ids"])
            return len(tokens)
        return len(text)

    def get_pad_token_id(self) -> int:
        return getattr(self.tokenizer, "pad_token_id", 0) or 0

    def get_eos_token_id(self) -> int:
        return getattr(self.tokenizer, "eos_token_id", 2) or 2


class SentencePieceTokenizerBackend(BaseTokenizerBackend):
    """
    SentencePiece backend for loading and using trained .model files.
    Provides unified encode/decode/tokenize interface compatible with the
    TokenizerManager and TokenizerBenchmarker.
    """

    def __init__(self, tokenizer_instance: Optional[Any] = None):
        super().__init__(tokenizer_instance)
        self._sp = None
        self._vocab: Optional[Dict[str, int]] = None

    def load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Loads a SentencePiece .model file.

        Args:
            model_name_or_path: Path to a .model file

        Returns:
            The loaded SentencePieceProcessor instance
        """
        import sentencepiece as spm

        if not model_name_or_path.endswith(".model"):
            # Try appending .model extension
            candidate = model_name_or_path + ".model"
            if not __import__("os").path.exists(candidate):
                # Try looking in standard cache directory
                candidate = __import__("os").path.join("cache", "tokenizers", model_name_or_path, "tokenizer.model")
            model_name_or_path = candidate

        sp = spm.SentencePieceProcessor()
        sp.Load(model_name_or_path)

        self._sp = sp
        self.tokenizer = sp
        self._vocab = None  # Lazy-build vocab dict

        logger.info(
            f"SentencePieceTokenizerBackend: Loaded model from '{model_name_or_path}' "
            f"(vocab_size={sp.GetPieceSize()})"
        )
        return sp

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        truncation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encodes text to token IDs using the loaded SentencePiece model.

        Returns:
            Dict with 'input_ids' and 'attention_mask' keys for HF compatibility.
        """
        if self._sp is None:
            raise RuntimeError("SentencePiece model not loaded. Call load_tokenizer() first.")

        if isinstance(text, str):
            ids = self._sp.Encode(text, out_type=int)
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids)
            }
        else:
            # Batch encoding
            batch_ids = []
            batch_masks = []
            for t in text:
                ids = self._sp.Encode(t, out_type=int)
                if truncation and max_length and len(ids) > max_length:
                    ids = ids[:max_length]
                batch_ids.append(ids)
                batch_masks.append([1] * len(ids))
            return {
                "input_ids": batch_ids,
                "attention_mask": batch_masks
            }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """Decodes token IDs back to text."""
        if self._sp is None:
            raise RuntimeError("SentencePiece model not loaded. Call load_tokenizer() first.")

        if skip_special_tokens:
            # Filter out special token IDs (typically 0=unk, 1=bos, 2=eos, 3=pad)
            filtered = [tid for tid in token_ids if tid >= 4]
            return self._sp.Decode(filtered)

        return self._sp.Decode(token_ids)

    def tokenize(self, text: str) -> List[str]:
        """
        Returns human-readable subword pieces for visualization.

        Example:
            "ꯃꯅꯤꯄꯨꯔꯤ" → ["▁ꯃꯅꯤ", "ꯄꯨꯔꯤ"]
        """
        if self._sp is None:
            raise RuntimeError("SentencePiece model not loaded. Call load_tokenizer() first.")
        return self._sp.Encode(text, out_type=str)

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens for the given text."""
        if self._sp is None:
            raise RuntimeError("SentencePiece model not loaded. Call load_tokenizer() first.")
        return len(self._sp.Encode(text, out_type=int))

    def get_pad_token_id(self) -> int:
        """Returns the pad token ID (default: 3 for SentencePiece convention)."""
        if self._sp is not None:
            pad_id = self._sp.pad_id()
            return pad_id if pad_id >= 0 else 3
        return 3

    def get_eos_token_id(self) -> int:
        """Returns the EOS token ID (default: 2 for SentencePiece convention)."""
        if self._sp is not None:
            eos_id = self._sp.eos_id()
            return eos_id if eos_id >= 0 else 2
        return 2

    def get_bos_token_id(self) -> int:
        """Returns the BOS token ID (default: 1 for SentencePiece convention)."""
        if self._sp is not None:
            bos_id = self._sp.bos_id()
            return bos_id if bos_id >= 0 else 1
        return 1

    def get_unk_token_id(self) -> int:
        """Returns the UNK token ID (default: 0 for SentencePiece convention)."""
        if self._sp is not None:
            unk_id = self._sp.unk_id()
            return unk_id if unk_id >= 0 else 0
        return 0

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the full vocabulary as a {piece: id} dictionary.
        Required by TokenizerBenchmarker for vocabulary coverage analysis.
        """
        if self._sp is None:
            raise RuntimeError("SentencePiece model not loaded. Call load_tokenizer() first.")

        if self._vocab is None:
            self._vocab = {}
            for i in range(self._sp.GetPieceSize()):
                piece = self._sp.IdToPiece(i)
                self._vocab[piece] = i

        return self._vocab

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size."""
        if self._sp is not None:
            return self._sp.GetPieceSize()
        return 0


class TikTokenTokenizerBackend(BaseTokenizerBackend):
    """
    TikToken backend stub.
    TODO: Implement when TikToken-exclusive models are introduced.
    """
    def load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        raise NotImplementedError("TODO: TikTokenTokenizerBackend will be implemented in future training phases.")

    def encode(self, text: Union[str, List[str]], max_length: Optional[int] = None, truncation: bool = True, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("TODO: TikTokenTokenizerBackend")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        raise NotImplementedError("TODO: TikTokenTokenizerBackend")

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError("TODO: TikTokenTokenizerBackend")

    def get_pad_token_id(self) -> int:
        raise NotImplementedError("TODO: TikTokenTokenizerBackend")

    def get_eos_token_id(self) -> int:
        raise NotImplementedError("TODO: TikTokenTokenizerBackend")


class CustomTokenizerBackend(BaseTokenizerBackend):
    """
    Custom tokenizer backend stub.
    TODO: Implement when non-HuggingFace custom backends are required.
    """
    def load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        raise NotImplementedError("TODO: CustomTokenizerBackend will be implemented in future training phases.")

    def encode(self, text: Union[str, List[str]], max_length: Optional[int] = None, truncation: bool = True, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("TODO: CustomTokenizerBackend")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        raise NotImplementedError("TODO: CustomTokenizerBackend")

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError("TODO: CustomTokenizerBackend")

    def get_pad_token_id(self) -> int:
        raise NotImplementedError("TODO: CustomTokenizerBackend")

    def get_eos_token_id(self) -> int:
        raise NotImplementedError("TODO: CustomTokenizerBackend")
