"""
TokenizerBenchmarker module for evaluating and comparing tokenizers across multiple
critical performance, entropy, script preservation, and qualitative human evaluation metrics.

Phase 5.3 enhancements:
- Fertility metric (tokens per word) — the primary quality indicator
- Per-script breakdown (Meitei Mayek, Bengali, Latin, Mixed)
- Improved Manipuri boundary evaluation using Unicode range detection
- Clearer compression ratio labeling (bytes per token)
"""

import os
import math
import json
import re
from typing import Dict, Any, List, Union, Optional, Tuple, Iterator
from app.utils.logger import logger

# Unicode range patterns for script classification
_MEITEI_PATTERN = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAFF]')
_BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]')
_LATIN_PATTERN = re.compile(r'[a-zA-Z]')


def _classify_script(text: str) -> str:
    """Classifies the dominant script of a text sample."""
    meitei_count = len(_MEITEI_PATTERN.findall(text))
    bengali_count = len(_BENGALI_PATTERN.findall(text))
    latin_count = len(_LATIN_PATTERN.findall(text))

    total = meitei_count + bengali_count + latin_count
    if total == 0:
        return "unknown"

    counts = {"meitei_mayek": meitei_count, "bengali": bengali_count, "latin": latin_count}
    dominant = max(counts, key=counts.get)
    dominant_ratio = counts[dominant] / total

    # If second-highest script > 15%, classify as mixed
    sorted_counts = sorted(counts.values(), reverse=True)
    if len(sorted_counts) > 1 and sorted_counts[1] / total > 0.15:
        return "mixed"

    return dominant


class TokenizerBenchmarker:
    """
    Evaluates tokenizer quality across compression ratio, vocabulary coverage, OOV rate,
    fertility, token entropy, round-trip accuracy, script preservation, and generates
    qualitative human reports with per-script breakdowns.
    """
    def __init__(self, tokenizer_wrapper: Any):
        self.tokenizer = tokenizer_wrapper

    def evaluate_corpus(self, text_samples: List[str]) -> Dict[str, Any]:
        """
        Runs comprehensive quantitative benchmark evaluation across text samples.
        Includes per-script breakdown for Meitei Mayek, Bengali, Latin, and Mixed.
        """
        if not text_samples:
            return {
                "compression_ratio": 0.0,
                "vocabulary_coverage": 0.0,
                "average_sequence_length": 0.0,
                "oov_rate": 0.0,
                "fertility": 0.0,
                "manipuri_token_quality": 0.0,
                "avg_tokens_per_sentence": 0.0,
                "avg_chars_per_token": 0.0,
                "token_entropy": 0.0,
                "unknown_char_count": 0,
                "script_preservation": 1.0,
                "round_trip_accuracy": 1.0,
                "total_samples_evaluated": 0,
                "total_bytes": 0,
                "total_tokens": 0,
                "total_words": 0,
                "token_frequency_histogram": {},
                "per_script": {}
            }

        total_bytes = 0
        total_chars = 0
        total_tokens = 0
        total_words = 0
        total_unk = 0
        unknown_char_count = 0
        exact_round_trip = 0
        token_id_counts: Dict[int, int] = {}
        unique_chars_observed = set()
        chars_covered = set()
        manipuri_boundary_scores: List[float] = []

        # Per-script accumulators
        script_accumulators: Dict[str, Dict[str, Any]] = {}

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if unk_token_id is None and hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
            unk_token_id = vocab.get("<unk>")

        for text in text_samples:
            if not text or not isinstance(text, str):
                continue

            text_clean = text.strip()
            text_bytes = len(text_clean.encode("utf-8"))
            text_char_len = len(text_clean)
            text_word_len = len(text_clean.split())
            total_bytes += text_bytes
            total_chars += text_char_len
            total_words += text_word_len
            unique_chars_observed.update(text_clean)

            # Tokenize encode
            token_ids: List[int] = []
            if hasattr(self.tokenizer, "encode"):
                encoded = self.tokenizer.encode(text_clean)
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    token_ids = encoded["input_ids"]
                elif isinstance(encoded, list):
                    token_ids = encoded
            elif hasattr(self.tokenizer, "count_tokens"):
                token_ids = [0] * self.tokenizer.count_tokens(text_clean)

            seq_tokens = len(token_ids)
            total_tokens += seq_tokens

            for tid in token_ids:
                token_id_counts[tid] = token_id_counts.get(tid, 0) + 1

            unk_in_seq = 0
            if unk_token_id is not None:
                unk_in_seq = sum(1 for tid in token_ids if tid == unk_token_id)
                total_unk += unk_in_seq
                unknown_char_count += unk_in_seq

            # Round-trip decode check
            is_round_trip = False
            if hasattr(self.tokenizer, "decode"):
                try:
                    decoded = self.tokenizer.decode(token_ids)
                    if decoded.strip() == text_clean:
                        exact_round_trip += 1
                        is_round_trip = True
                except Exception:
                    pass
            else:
                if total_unk == 0:
                    exact_round_trip += 1
                    is_round_trip = True

            manipuri_boundary_scores.append(self._evaluate_manipuri_boundaries(text_clean, token_ids))

            # Accumulate per-script metrics
            script = _classify_script(text_clean)
            if script not in script_accumulators:
                script_accumulators[script] = {
                    "tokens": 0, "words": 0, "bytes": 0, "chars": 0,
                    "unk": 0, "round_trips": 0, "samples": 0
                }
            acc = script_accumulators[script]
            acc["tokens"] += seq_tokens
            acc["words"] += text_word_len
            acc["bytes"] += text_bytes
            acc["chars"] += text_char_len
            acc["unk"] += unk_in_seq
            acc["round_trips"] += 1 if is_round_trip else 0
            acc["samples"] += 1

        # Calculate overall metrics
        num_samples = len(text_samples)
        avg_seq_len = total_tokens / num_samples if num_samples else 0.0
        avg_tokens_per_sentence = avg_seq_len
        avg_chars_per_token = total_chars / max(total_tokens, 1)
        compression_ratio = total_bytes / max(total_tokens, 1)
        fertility = total_tokens / max(total_words, 1)
        oov_rate = total_unk / max(total_tokens, 1)
        round_trip_accuracy = exact_round_trip / max(num_samples, 1)
        sequence_length_inflation = total_tokens / max(total_words, 1)

        # Shannon token entropy: - \sum p_i \log_2(p_i)
        token_entropy = 0.0
        if total_tokens > 0:
            for count in token_id_counts.values():
                p = count / total_tokens
                if p > 0:
                    token_entropy -= p * math.log2(p)

        # Approximate vocab coverage
        if hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
            for c in unique_chars_observed:
                if c in vocab or any(c in tok for tok in vocab.keys()):
                    chars_covered.add(c)
        else:
            chars_covered = unique_chars_observed if oov_rate < 0.1 else set(list(unique_chars_observed)[:max(len(unique_chars_observed)//2, 1)])

        vocab_coverage = len(chars_covered) / max(len(unique_chars_observed), 1)
        manipuri_quality = sum(manipuri_boundary_scores) / max(len(manipuri_boundary_scores), 1)
        script_preservation = max(1.0 - oov_rate * 2.0, 0.0)

        # Top 20 token frequency histogram
        top_tokens = sorted(token_id_counts.items(), key=lambda item: item[1], reverse=True)[:20]
        histogram = {f"token_{tid}": count for tid, count in top_tokens}

        # Compute per-script metrics
        per_script_metrics: Dict[str, Dict[str, Any]] = {}
        for script, acc in script_accumulators.items():
            if acc["samples"] > 0 and acc["tokens"] > 0:
                per_script_metrics[script] = {
                    "fertility": round(acc["tokens"] / max(acc["words"], 1), 3),
                    "compression_ratio": round(acc["bytes"] / max(acc["tokens"], 1), 3),
                    "avg_tokens_per_sentence": round(acc["tokens"] / acc["samples"], 2),
                    "avg_chars_per_token": round(acc["chars"] / acc["tokens"], 2),
                    "unknown_rate": round(acc["unk"] / acc["tokens"], 6),
                    "round_trip_accuracy": round(acc["round_trips"] / acc["samples"], 4),
                    "total_samples": acc["samples"],
                    "total_tokens": acc["tokens"],
                }

        metrics = {
            "compression_ratio": round(compression_ratio, 3),
            "vocabulary_coverage": round(vocab_coverage, 4),
            "average_sequence_length": round(avg_seq_len, 2),
            "avg_tokens_per_sentence": round(avg_tokens_per_sentence, 2),
            "avg_chars_per_token": round(avg_chars_per_token, 2),
            "fertility": round(fertility, 3),
            "oov_rate": round(oov_rate, 4),
            "unknown_char_count": unknown_char_count,
            "token_entropy": round(token_entropy, 3),
            "script_preservation": round(script_preservation, 3),
            "round_trip_accuracy": round(round_trip_accuracy, 4),
            "manipuri_token_quality": round(manipuri_quality, 3),
            "sequence_length_inflation": round(sequence_length_inflation, 3),
            "total_samples_evaluated": num_samples,
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "token_frequency_histogram": histogram,
            "per_script": per_script_metrics
        }
        logger.info(f"TokenizerBenchmarker: Evaluation results -> {metrics}")
        return metrics

    def evaluate_by_domain(self, domain_samples: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates tokenizer performance grouped by distinct domains (e.g. Wikipedia, News, Code).
        """
        results: Dict[str, Dict[str, Any]] = {}
        for domain, samples in domain_samples.items():
            logger.info(f"TokenizerBenchmarker: Evaluating domain '{domain}' ({len(samples)} samples)...")
            results[domain] = self.evaluate_corpus(samples)
        return results

    def _evaluate_manipuri_boundaries(self, text: str, token_ids: List[int]) -> float:
        """
        Evaluates subword fragmentation quality for Meitei Mayek or Bengali scripts.

        Uses Unicode range detection to identify Manipuri text and measures
        whether the tokenizer produces reasonable subword boundaries.
        A fertility (tokens/word) of ≤2.2 is excellent for agglutinative languages;
        >5.0 indicates excessive fragmentation.
        """
        words = text.split()
        if not words or not token_ids:
            return 1.0

        # Check if text contains Manipuri script characters
        has_meitei = bool(_MEITEI_PATTERN.search(text))
        has_bengali = bool(_BENGALI_PATTERN.search(text))

        if not has_meitei and not has_bengali:
            # Not Manipuri text, return neutral score
            return 0.85

        avg_tokens_per_word = len(token_ids) / len(words)
        if avg_tokens_per_word <= 2.2:
            return 1.0
        elif avg_tokens_per_word <= 3.0:
            return 0.90
        elif avg_tokens_per_word <= 3.5:
            return 0.80
        elif avg_tokens_per_word <= 5.0:
            return 0.60
        else:
            return 0.35

    @classmethod
    def generate_human_evaluation_report(
        cls,
        models_dict: Dict[str, Any],
        output_path: str = "cache/benchmarks/tokenizer_examples.md"
    ) -> str:
        """
        Generates qualitative side-by-side human evaluation markdown comparison across candidate models.
        """
        test_sentences = [
            ("Bengali Script Manipuri", "আমি স্কুলে যাচ্ছি।"),
            ("Bengali Script Manipuri 2", "Manipuri language is written in Bengali and Meitei Mayek scripts."),
            ("Meitei Mayek Script", "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯑꯩꯈꯣꯏꯒꯤ ꯏꯃꯥ ꯂꯣꯟꯅꯤ꯫"),
            ("Meitei Mayek Script 2", "ꯑꯩꯍꯥꯛ ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯇꯝꯂꯤ꯫"),
            ("Mixed Script Manipuri", "ꯑꯩꯍꯥꯛ school ꯗꯥ ꯆꯠꯂꯤ। Artificial Intelligence advances rapidly."),
            ("English Reference", "Deep learning models require balanced multilingual pretraining corpora.")
        ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = [
            "# Phase 5.3 Qualitative Tokenizer Human Evaluation Report",
            "",
            "This report presents side-by-side subword segmentation examples across candidate tokenizers to verify visual boundary preservation for Manipuri Meitei Mayek, Bengali script, and English.",
            ""
        ]

        for label, sentence in test_sentences:
            lines.append(f"## {label}")
            lines.append(f"**Original**: `{sentence}`\n")
            lines.append("| Tokenizer Model | Subword Segmentation Example | Tokens Count |")
            lines.append("| :--- | :--- | :--- |")

            for model_name, tok_wrapper in sorted(models_dict.items()):
                segmentation_str = ""
                token_count = 0
                try:
                    if hasattr(tok_wrapper, "tokenize"):
                        tokens = tok_wrapper.tokenize(sentence)
                        segmentation_str = " ".join([f"▁{t.replace(' ', '')}" if t.startswith(" ") else t for t in tokens])
                        token_count = len(tokens)
                    elif hasattr(tok_wrapper, "encode"):
                        enc = tok_wrapper.encode(sentence)
                        tids = enc["input_ids"] if isinstance(enc, dict) and "input_ids" in enc else enc
                        token_count = len(tids)
                        if hasattr(tok_wrapper, "decode"):
                            # Approximate visual segmentation if tokenize method isn't exposed directly
                            subwords = [tok_wrapper.decode([tid]) for tid in tids]
                            segmentation_str = " ".join([f"▁{sw}" for sw in subwords])
                        else:
                            segmentation_str = f"[Token IDs: {tids[:10]}...]"
                    else:
                        segmentation_str = "[Unsupported wrapper format]"
                except Exception as e:
                    segmentation_str = f"[Error: {e}]"

                lines.append(f"| `{model_name}` | {segmentation_str} | {token_count} |")
            lines.append("")

        report_md = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info(f"TokenizerBenchmarker: Generated human evaluation report at '{output_path}'")
        return output_path
