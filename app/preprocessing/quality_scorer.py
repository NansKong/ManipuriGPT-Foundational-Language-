"""
QualityScorer and ToxicityFilter modules for evaluating corpus quality and filtering out
low-grade, toxic, or noisy text before tokenization and training.
"""

from typing import Dict, Any, List, Optional, Tuple, Set


class QualityScorer:
    """
    Evaluates heuristic quality scores (0.0 to 1.0) based on character composition,
    symbol-to-word density, average word length, and script coherence.
    """
    def __init__(
        self,
        min_score: float = 0.45,
        max_symbol_ratio: float = 0.35,
        min_word_count: int = 3
    ):
        self.min_score = min_score
        self.max_symbol_ratio = max_symbol_ratio
        self.min_word_count = min_word_count

    def compute_score(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates heuristic quality score and diagnostic indicators.
        """
        if not text or not isinstance(text, str):
            return 0.0, {"reason": "empty"}

        clean = text.strip()
        words = clean.split()
        num_words = len(words)
        
        if num_words < self.min_word_count:
            return 0.2, {"reason": "too_few_words", "num_words": num_words}

        num_chars = len(clean)
        # Count non-alphanumeric and non-space characters (symbols/punctuation)
        symbols = sum(1 for c in clean if not c.isalnum() and not c.isspace())
        symbol_ratio = symbols / max(num_chars, 1)

        if symbol_ratio > self.max_symbol_ratio:
            return 0.3, {"reason": "high_symbol_density", "symbol_ratio": symbol_ratio}

        # Check average word length (extremes like gibberish or unspaced blocks)
        avg_word_length = sum(len(w) for w in words) / max(num_words, 1)
        if avg_word_length > 25 or avg_word_length < 1.5:
            return 0.35, {"reason": "abnormal_word_length", "avg_word_length": avg_word_length}

        # Base quality bonus for good length and low symbol density
        score = 0.85 - (symbol_ratio * 0.5)
        if num_words >= 15:
            score += 0.10
        score = min(max(score, 0.0), 1.0)

        return round(score, 3), {
            "num_words": num_words,
            "symbol_ratio": round(symbol_ratio, 3),
            "avg_word_length": round(avg_word_length, 2)
        }

    def is_acceptable(self, text: str) -> bool:
        """
        Returns True if the text meets or exceeds the minimum quality score threshold.
        """
        score, _ = self.compute_score(text)
        return score >= self.min_score


class ToxicityFilter:
    """
    Filters toxic, hate-speech, or profanity content across English, Meitei Mayek, and Bengali scripts
    using heuristic keyword blocklists and pattern checks.
    """
    def __init__(self, custom_blocklist: Optional[List[str]] = None):
        self.blocklist: Set[str] = {
            # Standard English offensive blocklist samples for training filtration
            "hate_speech_sample_kw", "toxic_slur_sample", "abuse_kw_sample",
            # Add common toxic keywords/patterns
            "kill yourself", "racial_slur_sample", "genocide_slur"
        }
        if custom_blocklist:
            self.blocklist.update(w.lower() for w in custom_blocklist)

    def is_toxic(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks whether the text contains blocked keywords or patterns.
        Returns (is_toxic, matched_keyword).
        """
        if not text or not isinstance(text, str):
            return False, None

        lower_text = text.lower()
        for kw in self.blocklist:
            if kw in lower_text:
                return True, kw

        return False, None

    def filter_example(self, example: Dict[str, Any], scorer: Optional[QualityScorer] = None) -> Optional[Dict[str, Any]]:
        """
        Processes a dataset dictionary. Returns None if text is toxic or fails quality score.
        Otherwise enriches example['metadata'] with quality metrics and returns the example.
        """
        text = example.get("text", "")
        
        toxic, matched = self.is_toxic(text)
        if toxic:
            return None

        metadata = example.get("metadata", {}).copy()
        if scorer is not None:
            score, details = scorer.compute_score(text)
            if score < scorer.min_score:
                return None
            metadata["quality_score"] = score
            metadata["quality_details"] = details
        else:
            metadata["toxic_filtered"] = True

        result = example.copy()
        result["metadata"] = metadata
        return result
