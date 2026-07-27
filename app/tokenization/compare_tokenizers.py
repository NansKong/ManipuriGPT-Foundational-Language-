"""
Tokenizer v1 vs Candidate Tokenizer v2 Evaluator (`app/tokenization/compare_tokenizers.py`).

Provides side-by-side objective evaluation of Tokenizer v1 vs Tokenizer v2 across:
  - Avg tokens per sequence
  - Compression ratio (bytes/token & chars/token)
  - Unknown rate (`<unk>%`)
  - Vocabulary coverage %
  - Vocabulary utilization (`used_vocab / total_vocab`)
  - Token entropy $H$ (bits)
  - Decision recommendation logic
"""

import os
import math
import json
from collections import Counter
from typing import List, Dict, Any, Optional

from app.utils.logger import logger


class TokenizerComparator:
    """Evaluates two SentencePiece or HuggingFace tokenizers on a validation corpus."""

    def evaluate_tokenizer(
        self,
        tokenizer_path: str,
        texts: List[str]
    ) -> Dict[str, Any]:
        """Compute strict metrics for a single tokenizer model path on a list of texts."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")

        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentencePiece model at {tokenizer_path}: {e}")

        total_seqs = len(texts)
        total_chars = 0
        total_bytes = 0
        total_tokens = 0
        total_unk = 0

        vocab_size = sp.GetPieceSize()
        unk_id = sp.unk_id()

        used_vocab_ids: set = set()
        token_counts: Counter = Counter()

        import re
        for text in texts:
            if not text:
                continue
            text_str = re.sub(r"\s+", " ", str(text)).strip()
            if not text_str:
                continue

            b_len = len(text_str.encode("utf-8"))
            c_len = len(text_str)
            total_bytes += b_len
            total_chars += c_len

            token_ids = sp.Encode(text_str)
            t_len = len(token_ids)
            total_tokens += t_len

            for tid in token_ids:
                used_vocab_ids.add(tid)
                token_counts[tid] += 1
                if tid == unk_id:
                    total_unk += 1

        avg_tokens_per_seq = total_tokens / max(total_seqs, 1)
        bytes_per_token = total_bytes / max(total_tokens, 1)
        chars_per_token = total_chars / max(total_tokens, 1)
        unk_rate_pct = (total_unk / max(total_tokens, 1)) * 100.0

        used_vocab_size = len(used_vocab_ids)
        vocab_utilization_pct = (used_vocab_size / max(vocab_size, 1)) * 100.0

        # Token Shannon Entropy calculation H = - sum(p * log2(p))
        token_entropy_bits = 0.0
        if total_tokens > 0:
            for count in token_counts.values():
                p = count / total_tokens
                if p > 0:
                    token_entropy_bits -= p * math.log2(p)

        return {
            "tokenizer_path": tokenizer_path,
            "vocab_size": vocab_size,
            "used_vocab_size": used_vocab_size,
            "vocab_utilization_pct": round(vocab_utilization_pct, 2),
            "total_sequences": total_seqs,
            "total_characters": total_chars,
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "total_unk_tokens": total_unk,
            "avg_tokens_per_sequence": round(avg_tokens_per_seq, 2),
            "compression_ratio_bytes_per_token": round(bytes_per_token, 3),
            "compression_ratio_chars_per_token": round(chars_per_token, 3),
            "unknown_rate_pct": round(unk_rate_pct, 4),
            "token_entropy_bits": round(token_entropy_bits, 4),
            "max_possible_entropy_bits": round(math.log2(max(used_vocab_size, 1)), 4),
        }

    def compare(
        self,
        v1_path: str,
        v2_path: str,
        texts: List[str]
    ) -> Dict[str, Any]:
        """Compare Tokenizer v1 vs Tokenizer v2 side-by-side."""
        logger.info(f"Evaluating Tokenizer v1: {v1_path}...")
        res_v1 = self.evaluate_tokenizer(v1_path, texts)

        logger.info(f"Evaluating Tokenizer v2: {v2_path}...")
        res_v2 = self.evaluate_tokenizer(v2_path, texts)

        # Objective Decision Logic
        bytes_ratio_diff = (res_v2["compression_ratio_bytes_per_token"] - res_v1["compression_ratio_bytes_per_token"]) / max(res_v1["compression_ratio_bytes_per_token"], 0.001)
        unk_diff = res_v2["unknown_rate_pct"] - res_v1["unknown_rate_pct"]
        vocab_util_diff = res_v2["vocab_utilization_pct"] - res_v1["vocab_utilization_pct"]

        recommendation = "KEEP Tokenizer v1"
        reason = "Tokenizer v2 did not show significant improvement (>5% compression efficiency boost) over v1."

        if bytes_ratio_diff > 0.05 and unk_diff <= 0.001:
            recommendation = "UPGRADE to Candidate Tokenizer v2"
            reason = f"Tokenizer v2 improves bytes-per-token compression ratio by {bytes_ratio_diff * 100:.2f}% without increasing unknown token rate."
        elif res_v1["vocab_utilization_pct"] < 30.0 and res_v2["vocab_utilization_pct"] > 60.0 and unk_diff <= 0.0:
            recommendation = "UPGRADE to Candidate Tokenizer v2"
            reason = f"Tokenizer v2 vastly improves vocabulary utilization ({res_v2['vocab_utilization_pct']}% vs {res_v1['vocab_utilization_pct']}%)."

        comparison = {
            "tokenizer_v1": res_v1,
            "tokenizer_v2": res_v2,
            "diff_metrics": {
                "compression_ratio_improvement_pct": round(bytes_ratio_diff * 100, 2),
                "unk_rate_diff_pct": round(unk_diff, 4),
                "vocab_utilization_diff_pct": round(vocab_util_diff, 2),
                "entropy_diff_bits": round(res_v2["token_entropy_bits"] - res_v1["token_entropy_bits"], 4),
            },
            "recommendation": recommendation,
            "decision_reason": reason,
        }

        return comparison
