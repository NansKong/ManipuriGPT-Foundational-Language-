"""
TokenizerFreezer module (`app/dataset_builder/tokenizer_freezer.py`).
Locks down Tokenizer v1 and exports all essential artifacts required for pretraining
and Hugging Face dataset publishing:
- `tokenizer.model`
- `tokenizer.json`
- `vocab.json`
- `special_tokens_map.json`
- `coverage_report.json`
"""

import os
import shutil
import json
import glob
from datetime import datetime
from typing import Dict, Any, Optional, List
from app.utils.logger import logger


class TokenizerFreezer:
    """
    Freezes candidate subword tokenizers into canonical Tokenizer v1 artifacts.
    Supports auto-discovering the evaluation winner from Phase 5.3 candidate directories
    or freezing an explicitly provided model path.
    """
    def __init__(self, default_output_dir: str = "artifacts/tokenizer_v1"):
        self.default_output_dir = default_output_dir

    def find_best_candidate(self, base_search_dir: str = "cache/tokenizers") -> Optional[str]:
        """
        Auto-discovers the winning tokenizer model from evaluation_results.json or
        by scanning candidate directories for the highest vocab size model.
        """
        if not os.path.exists(base_search_dir):
            return None

        # Check for evaluation_results.json across version tiers
        for tier in ["v3-final", "v2-expanded", "v1-pretrain", "v0-experimental"]:
            eval_path = os.path.join(base_search_dir, tier, "evaluation_results.json")
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    winner = data.get("winner")
                    if winner:
                        if "/" in winner:
                            alg, vocab = winner.split("/")
                            model_path = os.path.join(base_search_dir, tier, alg, vocab, "tokenizer.model")
                        else:
                            model_path = os.path.join(base_search_dir, tier, winner, "tokenizer.model")
                        if os.path.exists(model_path):
                            logger.info(f"TokenizerFreezer: Discovered evaluation winner '{winner}' at '{model_path}'")
                            return model_path
                except Exception as e:
                    logger.warning(f"TokenizerFreezer: Could not parse {eval_path}: {e}")

        # If no evaluation results file, search for any valid tokenizer.model across tiers
        candidates: List[str] = []
        for tier in ["v3-final", "v2-expanded", "v1-pretrain", "v0-experimental"]:
            tier_dir = os.path.join(base_search_dir, tier)
            if os.path.exists(tier_dir):
                models = glob.glob(os.path.join(tier_dir, "*", "*", "tokenizer.model"))
                candidates.extend(models)

        if not candidates:
            # Check base dir itself
            candidates = glob.glob(os.path.join(base_search_dir, "**", "tokenizer.model"), recursive=True)

        if candidates:
            # Sort by vocab size (extract integer from parent folder name if possible)
            def _vocab_key(p: str) -> int:
                try:
                    parent = os.path.basename(os.path.dirname(p))
                    return int(parent)
                except ValueError:
                    return 0
            sorted_candidates = sorted(candidates, key=_vocab_key, reverse=True)
            best_model = sorted_candidates[0]
            logger.info(f"TokenizerFreezer: Auto-selected candidate '{best_model}'")
            return best_model

        return None

    def freeze(
        self,
        source_model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        tokenizer_name: str = "ManipuriGPT-Tokenizer-v1"
    ) -> Dict[str, str]:
        """
        Freezes the tokenizer and exports `tokenizer.model`, `tokenizer.json`, `vocab.json`,
        `special_tokens_map.json`, and `coverage_report.json`.
        Returns a dictionary of generated file paths.
        """
        out_dir = output_dir or self.default_output_dir
        os.makedirs(out_dir, exist_ok=True)

        if not source_model_path:
            source_model_path = self.find_best_candidate()

        if not source_model_path or not os.path.exists(source_model_path):
            raise FileNotFoundError(
                f"TokenizerFreezer: Could not locate source model file at '{source_model_path}'. "
                "Please provide a valid path or run Phase 5.3 tokenizer training first."
            )

        logger.info(f"TokenizerFreezer: Freezing Tokenizer v1 from '{source_model_path}' into '{out_dir}'")
        exported_files: Dict[str, str] = {}

        # 1. Export tokenizer.model binary
        dest_model = os.path.join(out_dir, "tokenizer.model")
        shutil.copy2(source_model_path, dest_model)
        exported_files["tokenizer.model"] = dest_model
        logger.info(f"  -> Exported raw binary: {dest_model}")

        # 2. Inspect SentencePiece vocabulary & statistics
        vocab_dict: Dict[str, int] = {}
        unk_id = 0
        bos_id = 1
        eos_id = 2
        pad_id = 3
        piece_count = 0
        piece_lengths: List[int] = []

        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(dest_model)
            piece_count = sp.get_piece_size()
            unk_id = sp.unk_id()
            bos_id = sp.bos_id()
            eos_id = sp.eos_id()
            pad_id = sp.pad_id()

            for i in range(piece_count):
                piece = sp.id_to_piece(i)
                vocab_dict[piece] = i
                piece_lengths.append(len(piece))
        except Exception as e:
            logger.warning(f"TokenizerFreezer: SentencePiece direct parsing skipped/failed ({e}). Attempting fallback.")
            if not vocab_dict:
                vocab_dict = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
                piece_count = 4

        # 3. Export vocab.json
        vocab_path = os.path.join(out_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        exported_files["vocab.json"] = vocab_path
        logger.info(f"  -> Exported vocabulary table ({piece_count} items): {vocab_path}")

        # 4. Export special_tokens_map.json
        special_tokens = {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "additional_special_tokens": []
        }
        special_path = os.path.join(out_dir, "special_tokens_map.json")
        with open(special_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, indent=2)
        exported_files["special_tokens_map.json"] = special_path
        logger.info(f"  -> Exported special tokens map: {special_path}")

        # 5. Export tokenizer.json & fast tokenizer via Transformers (if available)
        tokenizer_json_path = os.path.join(out_dir, "tokenizer.json")
        try:
            from transformers import PreTrainedTokenizerFast
            # Try loading via PreTrainedTokenizerFast directly from spm model or vocab
            fast_tok = PreTrainedTokenizerFast(
                tokenizer_file=dest_model if dest_model.endswith(".json") else None,
                vocab_file=vocab_path if not dest_model.endswith(".json") else None,
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>",
                pad_token="<pad>"
            )
            # If loaded from spm model directly via sentencepiece backend in transformers:
            try:
                from transformers import AutoTokenizer
                auto_tok = AutoTokenizer.from_pretrained(os.path.dirname(dest_model), use_fast=True)
                auto_tok.save_pretrained(out_dir)
                if os.path.exists(tokenizer_json_path):
                    exported_files["tokenizer.json"] = tokenizer_json_path
                    logger.info(f"  -> Exported HuggingFace fast tokenizer.json: {tokenizer_json_path}")
            except Exception:
                fast_tok.save_pretrained(out_dir)
                if os.path.exists(tokenizer_json_path):
                    exported_files["tokenizer.json"] = tokenizer_json_path
                    logger.info(f"  -> Exported fast tokenizer via PreTrainedTokenizerFast: {tokenizer_json_path}")
        except Exception as e:
            logger.warning(f"TokenizerFreezer: Transformers fast tokenizer export fallback skipped ({e}).")

        # Ensure tokenizer_config.json exists
        config_path = os.path.join(out_dir, "tokenizer_config.json")
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 2048,
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "clean_up_tokenization_spaces": True,
            "name_or_path": tokenizer_name,
            "frozen_at": datetime.utcnow().isoformat() + "Z"
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2)
        exported_files["tokenizer_config.json"] = config_path

        # 6. Export coverage_report.json
        avg_piece_len = sum(piece_lengths) / max(len(piece_lengths), 1) if piece_lengths else 0.0
        max_piece_len = max(piece_lengths) if piece_lengths else 0
        coverage_report = {
            "tokenizer_name": tokenizer_name,
            "source_model_path": source_model_path,
            "frozen_at": datetime.utcnow().isoformat() + "Z",
            "vocab_size": piece_count,
            "special_tokens": {
                "unk_id": unk_id,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "pad_id": pad_id
            },
            "piece_statistics": {
                "avg_piece_length_chars": round(avg_piece_len, 2),
                "max_piece_length_chars": max_piece_len,
            },
            "exported_artifacts": list(exported_files.keys())
        }
        coverage_path = os.path.join(out_dir, "coverage_report.json")
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(coverage_report, f, indent=2)
        exported_files["coverage_report.json"] = coverage_path
        logger.info(f"  -> Exported coverage report: {coverage_path}")

        return exported_files
