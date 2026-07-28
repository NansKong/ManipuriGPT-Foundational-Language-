"""
TokenizerEvaluator module (`app/evaluation/tokenizer_eval.py`).
Performs detailed diagnostics on tokenizer efficiency: average tokens/sentence, <unk> count,
longest token, script distribution, and byte compression ratio.
"""

import re
from typing import Dict, Any, List
from app.utils.logger import logger


class TokenizerEvaluator:
    """Diagnostic suite for evaluating tokenizer performance and health."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.unk_token_id = getattr(tokenizer, "unk_token_id", None)

    def evaluate_corpus_tokenization(self, texts: List[str]) -> Dict[str, Any]:
        """Computes comprehensive tokenizer diagnostic metrics over evaluation texts."""
        if not texts:
            return {"error": "Empty text corpus"}

        total_bytes = 0
        total_tokens = 0
        total_unk = 0
        tokens_per_sentence = []

        longest_token_str = ""
        longest_token_id = -1

        for text in texts:
            if not text or not text.strip():
                continue
            text_bytes = len(text.encode("utf-8"))
            total_bytes += text_bytes

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            num_toks = len(token_ids)
            total_tokens += num_toks
            tokens_per_sentence.append(num_toks)

            if self.unk_token_id is not None:
                total_unk += token_ids.count(self.unk_token_id)

            for tid in token_ids:
                decoded = self.tokenizer.decode([tid])
                if len(decoded) > len(longest_token_str):
                    longest_token_str = decoded
                    longest_token_id = tid

        avg_tokens_per_sent = round(total_tokens / len(tokens_per_sentence), 2) if tokens_per_sentence else 0.0
        compression_ratio = round(total_bytes / max(total_tokens, 1), 3)  # Bytes per token

        return {
            "total_sentences_evaluated": len(tokens_per_sentence),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "average_tokens_per_sentence": avg_tokens_per_sent,
            "total_unk_tokens": total_unk,
            "unknown_token_rate": round(total_unk / max(total_tokens, 1), 6),
            "compression_ratio_bytes_per_token": compression_ratio,
            "longest_token": {
                "token_id": longest_token_id,
                "token_str": longest_token_str,
                "length": len(longest_token_str)
            }
        }
