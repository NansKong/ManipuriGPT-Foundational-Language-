"""
TokenizerTrainer module for training custom SentencePiece tokenizers on Manipuri corpora.

Supports SentencePiece Unigram (recommended for agglutinative Manipuri) and BPE algorithms,
with configurable character_coverage, byte_fallback, split_digits, and input_sentence_size.

Includes HuggingFace PreTrainedTokenizerFast export for Transformers integration.
"""

import os
import json
from datetime import datetime
from typing import Iterator, Dict, Any, List, Optional, Union, Callable
from app.utils.logger import logger


class TokenizerTrainerRegistry:
    """
    Registry for tokenizer training algorithm backends.
    Allows registering custom training functions or classes for any subword/BPE/Unigram algorithm.
    """
    _backends: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, backend_func: Callable) -> None:
        cls._backends[name.lower()] = backend_func

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        return cls._backends.get(name.lower())

    @classmethod
    def list_algorithms(cls) -> List[str]:
        return sorted(cls._backends.keys())


# SentencePiece reserved control token IDs (must NOT appear in user_defined_symbols)
_SPM_RESERVED_TOKENS = {"<unk>", "<s>", "</s>"}


def _load_training_config() -> Dict[str, Any]:
    """Loads training configuration from tokenizer.yaml."""
    try:
        from app.configs.loader import load_config
        tok_cfg = load_config("tokenizer.yaml")
        return tok_cfg.get("tokenizer", {}).get("training", {})
    except Exception:
        return {}


def _validate_training_corpus(
    buffered_texts: List[str],
    dev_mode: bool,
    min_samples: int = 100
) -> None:
    """
    Validates that the training corpus meets minimum quality requirements.
    Raises RuntimeError in production mode if corpus is insufficient.
    """
    if len(buffered_texts) < min_samples:
        msg = (
            f"Training corpus has only {len(buffered_texts)} samples "
            f"(minimum required: {min_samples}). "
            f"This will produce a low-quality tokenizer."
        )
        if not dev_mode:
            raise RuntimeError(
                f"{msg} Set dev_mode=True to allow simulated fallback, "
                f"or provide more training data."
            )
        logger.warning(f"TokenizerTrainer: {msg} Proceeding in dev_mode.")

    total_chars = sum(len(t) for t in buffered_texts)
    if total_chars < 10000 and not dev_mode:
        raise RuntimeError(
            f"Training corpus has only {total_chars} total characters "
            f"(minimum recommended: 10,000). Provide more training data."
        )


def _train_sentencepiece(
    algorithm: str,
    buffered_texts: List[str],
    save_path_base: str,
    vocab_size: int,
    special_tokens: List[str],
    output_dir: str,
    dev_mode: bool = False,
    training_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Trains a SentencePiece model (Unigram or BPE) with production-grade parameters.

    Key parameters sourced from tokenizer.yaml:
    - character_coverage: 0.9999 (Meitei Mayek has small Unicode block)
    - input_sentence_size: prevents OOM on large corpora
    - split_digits: separates digit characters
    - byte_fallback: enables byte-level fallback for unknown characters
    - max_sentencepiece_length: limits subword piece length
    """
    cfg = training_config or _load_training_config()

    spm_type = "unigram" if "unigram" in algorithm else ("char" if "char" in algorithm else "bpe")

    try:
        import sentencepiece as spm

        temp_input = os.path.join(output_dir, f"{os.path.basename(save_path_base)}_spm_train.txt")
        with open(temp_input, "w", encoding="utf-8") as f:
            for line in buffered_texts:
                stripped = line.strip()
                if stripped:
                    f.write(stripped + "\n")

        # Split tokens: SentencePiece owns <unk>/<s>/</s> via unk_id/bos_id/eos_id.
        # Only non-reserved tokens may appear in user_defined_symbols.
        # IMPORTANT: Do NOT put ordinary Manipuri/Bengali letters here.
        # These are for model-control tokens only (e.g., <system>, <tool>, <code>).
        user_symbols = [t for t in special_tokens if t not in _SPM_RESERVED_TOKENS]

        # Determine pad_id: use index of <pad> if present, else -1 (disabled)
        pad_id = special_tokens.index("<pad>") if "<pad>" in special_tokens else -1

        # Load configurable training parameters from tokenizer.yaml
        character_coverage = cfg.get("character_coverage", 0.9999)
        input_sentence_size = cfg.get("input_sentence_size", 1000000)
        shuffle_input_sentence = cfg.get("shuffle_input_sentence", True)
        num_threads = cfg.get("num_threads", 4)
        max_sentencepiece_length = cfg.get("max_sentencepiece_length", 32)
        split_digits = cfg.get("split_digits", True)
        byte_fallback = cfg.get("byte_fallback", True)

        logger.info(
            f"TokenizerTrainer: SentencePiece training config -> "
            f"model_type={spm_type}, vocab_size={vocab_size}, "
            f"character_coverage={character_coverage}, "
            f"input_sentence_size={input_sentence_size}, "
            f"split_digits={split_digits}, byte_fallback={byte_fallback}, "
            f"user_defined_symbols={user_symbols}, "
            f"unk_id=0, bos_id=1, eos_id=2, pad_id={pad_id}"
        )

        spm.SentencePieceTrainer.train(
            input=temp_input,
            model_prefix=save_path_base,
            vocab_size=vocab_size,
            model_type=spm_type,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=pad_id,
            user_defined_symbols=user_symbols,
            character_coverage=character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            num_threads=num_threads,
            max_sentencepiece_length=max_sentencepiece_length,
            split_digits=split_digits,
            byte_fallback=byte_fallback,
        )

        # Clean up temporary training file
        if os.path.exists(temp_input):
            os.remove(temp_input)

        model_path = f"{save_path_base}.model"
        vocab_path = f"{save_path_base}.vocab"

        # Verify model was actually produced
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece training completed but model file not found at {model_path}")

        model_size = os.path.getsize(model_path)
        logger.info(f"TokenizerTrainer: SentencePiece model saved -> {model_path} ({model_size:,} bytes)")

        return [model_path, vocab_path]

    except ImportError as e:
        logger.warning(f"TokenizerTrainer: SentencePiece library not installed ({e}).")
        if not dev_mode:
            raise RuntimeError(
                f"SentencePiece library is required for training. "
                f"Install with: pip install sentencepiece. Error: {e}"
            ) from e
        logger.warning("TokenizerTrainer: dev_mode=True — generating simulated model (sentencepiece not installed).")
        return _generate_simulated_artifacts(algorithm, save_path_base, buffered_texts, vocab_size, special_tokens)

    except Exception as e:
        logger.warning(f"TokenizerTrainer: SentencePiece training failed ({e}).")
        if not dev_mode:
            raise RuntimeError(
                f"SentencePiece training failed in production mode. "
                f"Set dev_mode=True to allow simulated fallback. Error: {e}"
            ) from e
        logger.warning("TokenizerTrainer: dev_mode=True — generating simulated model.")
        return _generate_simulated_artifacts(algorithm, save_path_base, buffered_texts, vocab_size, special_tokens)


def _train_tokenizers_lib(
    algorithm: str,
    buffered_texts: List[str],
    save_path_base: str,
    vocab_size: int,
    special_tokens: List[str],
    output_dir: str,
    dev_mode: bool = False,
    training_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Trains using the HuggingFace tokenizers library (BPE, Unigram, WordPiece)."""
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        if algorithm == "bpe":
            model = models.BPE()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        elif algorithm == "unigram":
            model = models.Unigram()
            trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        elif algorithm in ["wordpiece", "bytelevel_bpe"]:
            model = models.WordPiece(unk_token="<unk>")
            trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        else:
            model = models.BPE()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.train_from_iterator(buffered_texts, trainer=trainer)

        json_path = f"{save_path_base}.json"
        tokenizer.save(json_path)
        return [json_path]
    except Exception as e:
        logger.warning(f"TokenizerTrainer: Tokenizers library training failed ({e}).")
        if not dev_mode:
            raise RuntimeError(
                f"Tokenizers library training failed in production mode. "
                f"Set dev_mode=True to allow simulated fallback. Error: {e}"
            ) from e
        logger.warning("TokenizerTrainer: dev_mode=True — generating simulated artifact.")
        return _generate_simulated_artifacts(algorithm, save_path_base, buffered_texts, vocab_size, special_tokens)


def _generate_simulated_artifacts(
    algorithm: str,
    save_path_base: str,
    buffered_texts: List[str],
    vocab_size: int,
    special_tokens: List[str]
) -> List[str]:
    """Generates simulated vocabulary files for offline and testing validation."""
    char_freqs: Dict[str, int] = {}
    for text in buffered_texts[:5000]:
        for c in text:
            char_freqs[c] = char_freqs.get(c, 0) + 1

    vocab_dict: Dict[str, int] = {}
    for idx, token in enumerate(special_tokens):
        vocab_dict[token] = idx

    sorted_chars = sorted(char_freqs.keys(), key=lambda c: char_freqs[c], reverse=True)
    current_id = len(vocab_dict)
    for char in sorted_chars:
        if current_id >= vocab_size:
            break
        vocab_dict[char] = current_id
        current_id += 1

    synthetic_subwords = ["ꯃꯅꯤ", "ꯄꯨꯔꯤ", "ꯂꯣꯟ", "মণি", "পুরী", "lang", "manipuri", "meitei"]
    for sw in synthetic_subwords:
        if current_id < vocab_size and sw not in vocab_dict:
            vocab_dict[sw] = current_id
            current_id += 1

    json_path = f"{save_path_base}_simulated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": {"type": algorithm, "vocab": vocab_dict},
            "simulated": True
        }, f, indent=2)

    return [json_path]


# Register standard algorithms
TokenizerTrainerRegistry.register("sentencepiece_unigram", _train_sentencepiece)
TokenizerTrainerRegistry.register("sentencepiece_bpe", _train_sentencepiece)
TokenizerTrainerRegistry.register("sentencepiece_character", _train_sentencepiece)
TokenizerTrainerRegistry.register("bpe", _train_tokenizers_lib)
TokenizerTrainerRegistry.register("unigram", _train_tokenizers_lib)
TokenizerTrainerRegistry.register("wordpiece", _train_tokenizers_lib)
TokenizerTrainerRegistry.register("tiktoken", _train_tokenizers_lib)
TokenizerTrainerRegistry.register("bytelevel_bpe", _train_tokenizers_lib)


class TokenizerTrainer:
    """
    Trains custom SentencePiece tokenizers on text corpora and exports production artifacts.

    Production workflow:
        SentencePiece training → .model/.vocab → convert_to_hf_fast() → PreTrainedTokenizerFast → Transformers

    Key design decisions (based on expert review for ManipuriGPT):
    - character_coverage=0.9999: Meitei Mayek has a small Unicode block, don't drop chars
    - user_defined_symbols: model-control tokens only (<system>, <tool>, etc.), NOT ordinary letters
    - byte_fallback=True: robustness for unseen characters
    - split_digits=True: cleaner number handling
    """
    def __init__(
        self,
        algorithm: str = "sentencepiece_unigram",
        vocab_size: int = 32768,
        special_tokens: Optional[List[str]] = None,
        output_dir: str = "cache/tokenizers",
        dev_mode: bool = False,
        training_config: Optional[Dict[str, Any]] = None
    ):
        self.algorithm = algorithm.lower()
        self.vocab_size = vocab_size
        self.output_dir = output_dir
        self.dev_mode = dev_mode

        # Load training config from YAML if not provided
        self._training_config = training_config or _load_training_config()

        # Use config-specified special tokens, falling back to production defaults
        if special_tokens is not None:
            self.special_tokens = special_tokens
        else:
            cfg_tokens = self._training_config.get("special_tokens")
            if cfg_tokens:
                self.special_tokens = cfg_tokens
            else:
                self.special_tokens = [
                    "<unk>", "<s>", "</s>", "<pad>",
                    "<|im_start|>", "<|im_end|>",
                    "<system>", "<user>", "<assistant>",
                    "<tool>", "<code>", "<math>", "<sep>"
                ]

        os.makedirs(self.output_dir, exist_ok=True)

    def train_from_iterator(
        self,
        text_iterator: Iterator[str],
        model_prefix: str = "tokenizer",
        languages: Optional[List[str]] = None,
        seed: int = 42,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trains the tokenizer using an iterable stream of strings via registered backends.
        Returns training metadata and saved model paths.

        Raises RuntimeError if corpus is empty/insufficient in production mode.
        """
        logger.info(f"TokenizerTrainer: Starting '{self.algorithm}' training with target vocab={self.vocab_size} in '{self.output_dir}'")

        buffered_texts: List[str] = []
        total_chars = 0

        for i, text in enumerate(text_iterator):
            if not text or not isinstance(text, str):
                continue
            stripped = text.strip()
            if not stripped:
                continue
            if i < 100000:  # Buffer up to 100k samples for training
                buffered_texts.append(stripped)
            total_chars += len(stripped)

        # Validate corpus quality before training
        _validate_training_corpus(buffered_texts, self.dev_mode)

        backend_func = TokenizerTrainerRegistry.get(self.algorithm)
        if not backend_func:
            if self.algorithm.startswith("sentencepiece"):
                backend_func = _train_sentencepiece
            else:
                backend_func = _train_tokenizers_lib

        save_path_base = os.path.join(self.output_dir, model_prefix)
        artifact_files = backend_func(
            self.algorithm,
            buffered_texts,
            save_path_base,
            self.vocab_size,
            self.special_tokens,
            self.output_dir,
            self.dev_mode,
            self._training_config
        )

        try:
            from app.configs.loader import compute_config_hash
            config_hash = compute_config_hash()
        except Exception:
            config_hash = "5.2_default"

        meta_path = os.path.join(self.output_dir, "metadata.json")
        metadata = {
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "training_samples": len(buffered_texts),
            "languages": languages or ["en", "mni", "bn"],
            "seed": seed,
            "experiment_id": experiment_id or f"SPM_{self.algorithm[:3].upper()}_{self.vocab_size}_{datetime.utcnow().strftime('%Y%m%d')}",
            "config_hash": config_hash,
            "date": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "5.3",
            "total_characters_observed": total_chars,
            "special_tokens": self.special_tokens,
            "artifact_files": artifact_files,
            "training_config": {
                "character_coverage": self._training_config.get("character_coverage", 0.9999),
                "input_sentence_size": self._training_config.get("input_sentence_size", 1000000),
                "split_digits": self._training_config.get("split_digits", True),
                "byte_fallback": self._training_config.get("byte_fallback", True),
                "max_sentencepiece_length": self._training_config.get("max_sentencepiece_length", 32),
            }
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Also write legacy model_prefix_metadata.json for compatibility with existing tests
        legacy_meta_path = f"{save_path_base}_metadata.json"
        with open(legacy_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"TokenizerTrainer: Completed training. Artifacts saved in '{self.output_dir}'")
        return metadata

    def convert_to_hf_fast(
        self,
        spm_model_path: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Converts a trained SentencePiece .model to HuggingFace PreTrainedTokenizerFast.

        Production workflow:
            SentencePiece .model → PreTrainedTokenizerFast → tokenizer.json + tokenizer_config.json

        This enables:
        - Fast Rust-backed encoding via tokenizers library
        - Parallel tokenization for streaming datasets
        - Native HuggingFace Transformers Trainer integration
        - Seamless model uploading to HuggingFace Hub

        Args:
            spm_model_path: Path to the trained .model file
            output_dir: Directory for HF tokenizer output (default: cache/tokenizers/hf_fast)

        Returns:
            Path to the saved HF tokenizer directory
        """
        export_dir = output_dir or self._training_config.get("hf_export_dir", "cache/tokenizers/hf_fast")
        os.makedirs(export_dir, exist_ok=True)

        if not os.path.exists(spm_model_path):
            raise FileNotFoundError(f"SentencePiece model not found at {spm_model_path}")

        try:
            from transformers import LlamaTokenizerFast, PreTrainedTokenizerFast

            # Try loading via LlamaTokenizerFast which natively supports SentencePiece
            try:
                tokenizer = LlamaTokenizerFast(
                    vocab_file=spm_model_path,
                    legacy=False
                )
            except Exception:
                # Fallback: convert via sentencepiece_model_pb2
                from tokenizers import SentencePieceBPETokenizerFast
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=spm_model_path
                )

            # Set special tokens
            special_tokens_map = {}
            for token in self.special_tokens:
                if token == "<unk>":
                    special_tokens_map["unk_token"] = token
                elif token == "<s>":
                    special_tokens_map["bos_token"] = token
                elif token == "</s>":
                    special_tokens_map["eos_token"] = token
                elif token == "<pad>":
                    special_tokens_map["pad_token"] = token

            if special_tokens_map:
                tokenizer.add_special_tokens(special_tokens_map)

            # Add additional special tokens that aren't standard roles
            additional_special = [
                t for t in self.special_tokens
                if t not in _SPM_RESERVED_TOKENS and t != "<pad>"
            ]
            if additional_special:
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special})

            tokenizer.save_pretrained(export_dir)
            logger.info(f"TokenizerTrainer: HF Fast tokenizer exported to '{export_dir}'")
            return export_dir

        except ImportError as e:
            raise RuntimeError(
                f"transformers library required for HF conversion. "
                f"Install with: pip install transformers. Error: {e}"
            ) from e

    def train_on_source(
        self,
        source: Union[str, Any],
        limit: int = 50000,
        mock_fallback: bool = False,
        model_prefix: str = "manipuri_tokenizer"
    ) -> Dict[str, Any]:
        """
        Trains the tokenizer directly from a streamed corpus source.
        """
        from app.corpus.acquisition import CorpusAcquisitionManager
        logger.info(f"TokenizerTrainer: Training on real/streamed corpus source '{source}' (limit={limit}, mock_fallback={mock_fallback})...")
        mgr = CorpusAcquisitionManager()
        spec = mgr.get_source(source) if isinstance(source, str) else source
        if not spec:
            raise KeyError(f"Source '{source}' not found in registry.")

        stream = mgr.stream_source(spec, max_examples=limit, mock_fallback=mock_fallback)
        def _text_generator(ex_stream):
            for ex in ex_stream:
                if isinstance(ex, dict):
                    text = ex.get(spec.default_text_column, ex.get("text", ""))
                else:
                    text = str(ex)
                if text and isinstance(text, str):
                    yield text

        return self.train_from_iterator(_text_generator(stream), model_prefix=model_prefix)
