"""
TokenizerEvaluator module for systematic comparison of trained tokenizer candidates
and automated winner selection based on quantitative metrics.

Evaluates candidates on:
- Fertility (tokens per word) — lower is better
- Compression ratio (bytes per token) — higher is better
- Unknown/OOV rate — lower is better
- Round-trip accuracy — higher is better
- Per-script performance (Meitei Mayek, Bengali, Latin, Mixed)

Produces a structured comparison report and selects the best candidate
based on configurable thresholds from tokenizer.yaml.
"""

import os
import json
import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from app.utils.logger import logger


# Standard test sentences for per-script evaluation
EVALUATION_SENTENCES = {
    "meitei_mayek": [
        "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯑꯁꯤ ꯑꯩꯈꯣꯏꯒꯤ ꯏꯃꯥ ꯂꯣꯟꯅꯤ꯫",
        "ꯑꯩꯍꯥꯛ ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ ꯇꯝꯂꯤ꯫",
        "ꯃꯤꯇꯩ ꯂꯣꯟ ꯑꯁꯤ ꯑꯃꯨꯛꯇꯥ ꯃꯁꯛ ꯅꯥꯏꯕꯥ ꯂꯣꯟꯅꯤ꯫",
        "ꯑꯗꯨꯒꯨꯝꯕꯅꯤ ꯃꯇꯝ ꯑꯗꯨꯗꯥ ꯂꯩꯕꯥ ꯃꯤꯑꯣꯏ ꯈꯨꯗꯤꯡꯃꯛꯅꯥ ꯃꯤꯟꯅꯩ꯫",
    ],
    "bengali": [
        "মণিপুরী ভাষা ভারতের অন্যতম প্রধান ভাষা।",
        "আমি স্কুলে যাচ্ছি।",
        "মৈতৈ ভাষা একটি সমৃদ্ধ ভাষা।",
        "এই ভাষার সাহিত্য অনেক পুরানো।",
    ],
    "latin": [
        "Deep learning models require balanced multilingual pretraining corpora.",
        "The Manipuri language is spoken by approximately two million people.",
        "Natural language processing has advanced significantly in recent years.",
        "Tokenizer quality directly affects downstream model performance.",
    ],
    "mixed": [
        "ꯑꯩꯍꯥꯛ school ꯗꯥ ꯆꯠꯂꯤ। Artificial Intelligence advances rapidly.",
        "Manipuri ꯂꯣꯟ has a rich literary tradition since ancient times.",
        "মণিপুরী ভাষা is recognized in the Eighth Schedule of India.",
    ],
}


def _load_eval_thresholds() -> Dict[str, float]:
    """Loads evaluation thresholds from tokenizer.yaml."""
    try:
        from app.configs.loader import load_config
        cfg = load_config("tokenizer.yaml")
        return cfg.get("tokenizer", {}).get("training", {}).get("evaluation", {})
    except Exception:
        return {}


class TokenizerEvaluator:
    """
    Evaluates multiple trained tokenizer candidates and selects the best one
    based on fertility, compression ratio, unknown rate, round-trip accuracy,
    and per-script performance.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        cfg_thresholds = _load_eval_thresholds()
        self.thresholds = thresholds or {
            "max_acceptable_fertility": cfg_thresholds.get("max_acceptable_fertility", 3.5),
            "max_acceptable_unknown_rate": cfg_thresholds.get("max_acceptable_unknown_rate", 0.005),
            "min_compression_ratio": cfg_thresholds.get("min_compression_ratio", 3.0),
            "min_round_trip_accuracy": cfg_thresholds.get("min_round_trip_accuracy", 0.95),
        }

    def evaluate_candidate(
        self,
        model_path: str,
        test_samples: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single tokenizer candidate across all scripts.

        Args:
            model_path: Path to the .model file
            test_samples: Dict of {script_name: [sentences]}. Defaults to EVALUATION_SENTENCES.

        Returns:
            Dict with overall metrics and per-script breakdown.
        """
        samples = test_samples or EVALUATION_SENTENCES

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("TokenizerEvaluator: sentencepiece not installed, returning empty evaluation.")
            return {"error": "sentencepiece not installed", "model_path": model_path}

        if not os.path.exists(model_path):
            logger.warning(f"TokenizerEvaluator: Model not found at {model_path}")
            return {"error": f"model not found: {model_path}", "model_path": model_path}

        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

        vocab_size = sp.GetPieceSize()
        unk_id = sp.unk_id()

        # Aggregate metrics
        overall_metrics = self._compute_metrics(sp, unk_id, _flatten_samples(samples), vocab_size)
        overall_metrics["vocab_size"] = vocab_size
        overall_metrics["model_path"] = model_path

        # Per-script breakdown
        per_script: Dict[str, Dict[str, Any]] = {}
        for script_name, script_samples in samples.items():
            if script_samples:
                per_script[script_name] = self._compute_metrics(sp, unk_id, script_samples, vocab_size)

        overall_metrics["per_script"] = per_script

        # Quality assessment against thresholds
        overall_metrics["passes_thresholds"] = self._check_thresholds(overall_metrics)

        return overall_metrics

    def _compute_metrics(
        self,
        sp: Any,
        unk_id: int,
        sentences: List[str],
        vocab_size: int = 32768
    ) -> Dict[str, Any]:
        """Computes quantitative tokenizer metrics for a list of sentences."""
        total_tokens = 0
        total_words = 0
        total_bytes = 0
        total_chars = 0
        total_unk = 0
        exact_round_trip = 0
        token_counts_per_sentence: List[int] = []
        piece_counts: Dict[int, int] = {}
        all_words: List[str] = []

        for text in sentences:
            text = text.strip()
            if not text:
                continue

            ids = sp.Encode(text, out_type=int)
            pieces = sp.Encode(text, out_type=str)
            decoded = sp.Decode(ids)

            for tid in ids:
                piece_counts[tid] = piece_counts.get(tid, 0) + 1

            words = text.split()
            all_words.extend(words)

            n_tokens = len(ids)
            n_words = len(words)
            n_bytes = len(text.encode("utf-8"))
            n_chars = len(text)
            n_unk = sum(1 for tid in ids if tid == unk_id)

            total_tokens += n_tokens
            total_words += n_words
            total_bytes += n_bytes
            total_chars += n_chars
            total_unk += n_unk
            token_counts_per_sentence.append(n_tokens)

            if decoded.strip() == text:
                exact_round_trip += 1

        n_sentences = len(sentences)
        if n_sentences == 0 or total_tokens == 0:
            return {
                "fertility": 0.0,
                "avg_tokens_per_sentence": 0.0,
                "compression_ratio": 0.0,
                "unknown_rate": 0.0,
                "round_trip_accuracy": 0.0,
                "vocab_utilization": 0.0,
                "entropy": 0.0,
                "agglutination_fragmentation": 0.0,
                "total_tokens": 0,
                "total_words": 0,
                "total_sentences": 0,
            }

        # Compute Vocab Utilization & Entropy
        unique_pieces_used = len(piece_counts)
        vocab_utilization = round(unique_pieces_used / max(vocab_size, 1), 4)

        entropy = 0.0
        if total_tokens > 0:
            for count in piece_counts.values():
                p = count / total_tokens
                if p > 0:
                    entropy -= p * math.log2(p)
        entropy = round(entropy, 4)

        # Compute Agglutination Fragmentation on top 100 longest words
        all_words.sort(key=len, reverse=True)
        top_long_words = all_words[:100]
        long_word_pieces = 0
        if top_long_words:
            for w in top_long_words:
                long_word_pieces += len(sp.Encode(w, out_type=int))
            agglutination_fragmentation = round(long_word_pieces / len(top_long_words), 2)
        else:
            agglutination_fragmentation = 0.0

        return {
            "fertility": round(total_tokens / max(total_words, 1), 3),
            "avg_tokens_per_sentence": round(total_tokens / n_sentences, 2),
            "avg_chars_per_token": round(total_chars / total_tokens, 2),
            "compression_ratio": round(total_bytes / total_tokens, 3),
            "unknown_rate": round(total_unk / total_tokens, 6),
            "unknown_count": total_unk,
            "round_trip_accuracy": round(exact_round_trip / n_sentences, 4),
            "vocab_utilization": vocab_utilization,
            "entropy": entropy,
            "agglutination_fragmentation": agglutination_fragmentation,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "total_sentences": n_sentences,
            "total_bytes": total_bytes,
        }

    def _check_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Checks whether the candidate passes each evaluation threshold."""
        return {
            "fertility_ok": metrics.get("fertility", 999) <= self.thresholds["max_acceptable_fertility"],
            "unknown_rate_ok": metrics.get("unknown_rate", 1.0) <= self.thresholds["max_acceptable_unknown_rate"],
            "compression_ok": metrics.get("compression_ratio", 0.0) >= self.thresholds["min_compression_ratio"],
            "round_trip_ok": metrics.get("round_trip_accuracy", 0.0) >= self.thresholds["min_round_trip_accuracy"],
        }

    def compare_candidates(
        self,
        candidates_dir: str,
        test_samples: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Discovers and evaluates all tokenizer candidates under a directory structure like:
            candidates_dir/
                sentencepiece_unigram/16384/tokenizer.model
                sentencepiece_unigram/24576/tokenizer.model
                sentencepiece_bpe/32768/tokenizer.model
                ...

        Returns a dict of {candidate_name: evaluation_results}.
        """
        results: Dict[str, Any] = {}

        for alg_dir in sorted(os.listdir(candidates_dir)):
            alg_path = os.path.join(candidates_dir, alg_dir)
            if not os.path.isdir(alg_path):
                continue
            # Skip non-algorithm directories (e.g. hf_fast)
            if not alg_dir.startswith("sentencepiece"):
                continue

            for vocab_dir in sorted(os.listdir(alg_path)):
                vocab_path = os.path.join(alg_path, vocab_dir)
                if not os.path.isdir(vocab_path):
                    continue

                model_path = os.path.join(vocab_path, "tokenizer.model")
                if not os.path.exists(model_path):
                    logger.warning(f"TokenizerEvaluator: No model found at {model_path}, skipping.")
                    continue

                candidate_name = f"{alg_dir}/{vocab_dir}"
                logger.info(f"TokenizerEvaluator: Evaluating candidate '{candidate_name}'...")
                results[candidate_name] = self.evaluate_candidate(model_path, test_samples)

        return results

    def select_winner(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Selects the best tokenizer candidate based on a weighted scoring function.

        Scoring formula (lower is better):
            score = fertility * 0.35 + unknown_rate * 100 * 0.25
                  + (1 - round_trip_accuracy) * 0.25
                  + (1 / compression_ratio) * 0.15

        Prefers Unigram over BPE when scores are equal (better for agglutinative languages).

        Returns:
            Tuple of (winner_name, winner_results)
        """
        if not results:
            raise ValueError("No candidates to evaluate.")

        scored: List[Tuple[str, float, Dict[str, Any]]] = []

        for name, metrics in results.items():
            if "error" in metrics:
                logger.warning(f"TokenizerEvaluator: Skipping '{name}' due to error: {metrics['error']}")
                continue

            fertility = metrics.get("fertility", 999.0)
            unknown_rate = metrics.get("unknown_rate", 1.0)
            round_trip = metrics.get("round_trip_accuracy", 0.0)
            compression = metrics.get("compression_ratio", 0.01)

            # Weighted score (lower is better)
            score = (
                fertility * 0.35
                + unknown_rate * 100 * 0.25
                + (1.0 - round_trip) * 0.25
                + (1.0 / max(compression, 0.01)) * 0.15
            )

            # Small bonus for Unigram (better for agglutinative languages)
            if "unigram" in name.lower():
                score -= 0.01

            metrics["selection_score"] = round(score, 6)
            scored.append((name, score, metrics))

        if not scored:
            raise ValueError("All candidates had errors. Cannot select a winner.")

        scored.sort(key=lambda x: x[1])
        winner_name, winner_score, winner_metrics = scored[0]

        logger.info(f"TokenizerEvaluator: Selected winner -> '{winner_name}' (score={winner_score:.4f})")
        return winner_name, winner_metrics

    def compute_stability_score(
        self,
        training_texts: List[str],
        algorithm: str,
        vocab_size: int,
        num_seeds: int = 5,
        output_dir: str = "cache/tokenizers/tmp_stability"
    ) -> Dict[str, Any]:
        """
        Computes the stability score of a candidate algorithm and vocabulary size by training
        models across multiple random seeds and measuring the pairwise vocabulary overlap percentage.
        - High Stability: >= 98% overlap across seeds
        - Medium Stability: 85% - 98% overlap
        - Low Stability (Unstable): < 80% overlap
        """
        from app.tokenization.trainer import TokenizerTrainer

        os.makedirs(output_dir, exist_ok=True)
        vocab_sets = []
        seeds = [42 + i for i in range(num_seeds)]

        for seed in seeds:
            model_prefix = os.path.join(output_dir, f"{algorithm}_{vocab_size}_seed{seed}")
            cfg = {
                "vocab_size": vocab_size,
                "character_coverage": 1.0,
                "model_prefix": model_prefix,
                "seed": seed,
            }
            trainer = TokenizerTrainer(cfg)
            try:
                meta = trainer.train_from_iterator(iter(training_texts), algorithm=algorithm, vocab_size=vocab_size, seed=seed)
                model_path = meta.get("model_file") or f"{model_prefix}.model"
                if os.path.exists(model_path):
                    import sentencepiece as spm
                    sp = spm.SentencePieceProcessor()
                    sp.Load(model_path)
                    pieces = set(sp.IdToPiece(i) for i in range(sp.GetPieceSize()))
                    vocab_sets.append(pieces)
            except Exception as e:
                logger.warning(f"TokenizerEvaluator: Stability training failed for seed {seed}: {e}")

        if len(vocab_sets) < 2:
            return {
                "stability_score": 0.0,
                "stability_rating": "N/A (Insufficient models trained)",
                "seeds_evaluated": len(vocab_sets)
            }

        overlaps = []
        for i in range(len(vocab_sets)):
            for j in range(i + 1, len(vocab_sets)):
                s1, s2 = vocab_sets[i], vocab_sets[j]
                if s1 and s2:
                    overlap = len(s1.intersection(s2)) / len(s1.union(s2))
                    overlaps.append(overlap)

        avg_overlap = round((sum(overlaps) / len(overlaps)) * 100, 2) if overlaps else 0.0

        if avg_overlap >= 98.0:
            rating = "High (>=98%)"
        elif avg_overlap >= 85.0:
            rating = "Medium (85-98%)"
        else:
            rating = "Low (<80% - Unstable)"

        return {
            "stability_score": avg_overlap,
            "stability_rating": rating,
            "seeds_evaluated": len(vocab_sets)
        }

    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, Any]],
        winner_name: str,
        output_path: str = "cache/benchmarks/tokenizer_evaluation.md"
    ) -> str:
        """
        Generates a detailed markdown comparison report across all candidates.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines = [
            "# ManipuriGPT Tokenizer Candidate Evaluation Report",
            "",
            f"**Generated**: {datetime.utcnow().isoformat()}Z",
            f"**Winner**: `{winner_name}`",
            "",
            "## Overall Comparison",
            "",
            "| Candidate | Vocab Size | Fertility | Compression | Unknown Rate | Round-Trip | Vocab Util | Entropy | Agglutination | Stability | Score |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for name in sorted(results.keys()):
            m = results[name]
            if "error" in m:
                lines.append(f"| `{name}` | — | ERROR | — | — | — | — | — | — | — | — |")
                continue

            marker = " **★**" if name == winner_name else ""
            stab_str = f"{m.get('stability_score', '?')}%" if isinstance(m.get('stability_score'), (int, float)) else str(m.get('stability_score', '?'))
            lines.append(
                f"| `{name}`{marker} "
                f"| {m.get('vocab_size', '?')} "
                f"| {m.get('fertility', '?')} "
                f"| {m.get('compression_ratio', '?')} "
                f"| {m.get('unknown_rate', '?')} "
                f"| {m.get('round_trip_accuracy', '?')} "
                f"| {round(m.get('vocab_utilization', 0.0) * 100, 1)}% "
                f"| {m.get('entropy', '?')} "
                f"| {m.get('agglutination_fragmentation', '?')} "
                f"| {stab_str} "
                f"| {m.get('selection_score', '?')} |"
            )

        lines.append("")

        # Per-script breakdown for each candidate
        lines.append("## Per-Script Breakdown")
        lines.append("")

        for name in sorted(results.keys()):
            m = results[name]
            per_script = m.get("per_script", {})
            if not per_script:
                continue

            marker = " (★ Winner)" if name == winner_name else ""
            lines.append(f"### `{name}`{marker}")
            lines.append("")
            lines.append("| Script | Fertility | Compression | Unknown Rate | Round-Trip | Avg Chars/Token | Agglutination |")
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")

            for script_name in ["meitei_mayek", "bengali", "latin", "mixed"]:
                sm = per_script.get(script_name, {})
                if sm:
                    lines.append(
                        f"| {script_name} "
                        f"| {sm.get('fertility', '?')} "
                        f"| {sm.get('compression_ratio', '?')} "
                        f"| {sm.get('unknown_rate', '?')} "
                        f"| {sm.get('round_trip_accuracy', '?')} "
                        f"| {sm.get('avg_chars_per_token', '?')} "
                        f"| {sm.get('agglutination_fragmentation', '?')} |"
                    )

            lines.append("")

        # Threshold check summary
        lines.append("## Threshold Compliance")
        lines.append("")
        lines.append("| Candidate | Fertility ≤ {:.1f} | Unknown ≤ {:.3f} | Compression ≥ {:.1f} | Round-Trip ≥ {:.2f} |".format(
            self.thresholds["max_acceptable_fertility"],
            self.thresholds["max_acceptable_unknown_rate"],
            self.thresholds["min_compression_ratio"],
            self.thresholds["min_round_trip_accuracy"]
        ))
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

        for name in sorted(results.keys()):
            m = results[name]
            passes = m.get("passes_thresholds", {})
            if not passes:
                continue
            lines.append(
                f"| `{name}` "
                f"| {'✅' if passes.get('fertility_ok') else '❌'} "
                f"| {'✅' if passes.get('unknown_rate_ok') else '❌'} "
                f"| {'✅' if passes.get('compression_ok') else '❌'} "
                f"| {'✅' if passes.get('round_trip_ok') else '❌'} |"
            )

        lines.append("")

        # Segmentation examples for the winner
        lines.append("## Segmentation Examples (Winner)")
        lines.append("")

        winner_metrics = results.get(winner_name, {})
        winner_model_path = winner_metrics.get("model_path")
        if winner_model_path and os.path.exists(winner_model_path):
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.Load(winner_model_path)

                for script_name, script_sentences in EVALUATION_SENTENCES.items():
                    lines.append(f"### {script_name}")
                    for sent in script_sentences[:2]:
                        pieces = sp.Encode(sent, out_type=str)
                        lines.append(f"- **Input**: `{sent}`")
                        lines.append(f"  **Pieces**: `{' | '.join(pieces)}`")
                        lines.append(f"  **Count**: {len(pieces)}")
                    lines.append("")
            except Exception as e:
                lines.append(f"*(Could not load winner model for segmentation examples: {e})*")
                lines.append("")

        report_md = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info(f"TokenizerEvaluator: Comparison report saved to '{output_path}'")
        return output_path


def _flatten_samples(samples: Dict[str, List[str]]) -> List[str]:
    """Flattens a dict of {script: [sentences]} into a single list."""
    flat: List[str] = []
    for script_sentences in samples.values():
        flat.extend(script_sentences)
    return flat
