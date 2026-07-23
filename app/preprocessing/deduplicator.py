import re
from typing import Dict, Any, Set, List
from collections import deque
from app.utils.logger import logger

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

class Deduplicator:
    """
    Component for removing duplicate and near-duplicate sentences.
    Supports three levels: Exact, Normalized, and Fuzzy/Near-duplicate matching.

    Optimizations:
    - Uses deque instead of list for O(1) fuzzy window rotation (was O(n) with list.pop(0))
    - Pre-computes normalized text to avoid redundant normalization
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.exact_enabled = config.get("exact", True)
        self.normalized_enabled = config.get("normalized", True)
        self.fuzzy_enabled = config.get("fuzzy", True)
        self.fuzzy_threshold = config.get("fuzzy_threshold", 90.0)
        self.window_size = config.get("window_size", 1000)  # Sliding window size for fuzzy matching to avoid O(N^2) slowdowns

        self.exact_seen: Set[str] = set()
        self.normalized_seen: Set[str] = set()
        # Use deque with maxlen for O(1) rotation instead of list.pop(0) which is O(n)
        self.fuzzy_window: deque = deque(maxlen=self.window_size)

        if self.fuzzy_enabled and not HAS_RAPIDFUZZ:
            logger.warning("rapidfuzz is not installed. Falling back to difflib.SequenceMatcher for fuzzy matching (slower).")

    def reset(self) -> None:
        """Resets the deduplication state."""
        self.exact_seen.clear()
        self.normalized_seen.clear()
        self.fuzzy_window.clear()

    def _normalize(self, text: str) -> str:
        """Simple normalization for duplicate checks (lowercase, strip, collapse spacing)."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def is_duplicate(self, text: str) -> bool:
        """
        Checks if a string is a duplicate or near-duplicate.
        If it's not a duplicate, it gets added to the tracking sets/window and returns False.
        If it is a duplicate, returns True.
        """
        if not self.enabled or not text:
            return False

        # 1. Exact match check
        if self.exact_enabled:
            if text in self.exact_seen:
                return True

        # 2. Normalized match check
        norm_text = self._normalize(text)
        if self.normalized_enabled:
            if norm_text in self.normalized_seen:
                return True

        # 3. Fuzzy/Near-duplicate match check (against rolling window of recent samples)
        if self.fuzzy_enabled:
            for seen_norm in self.fuzzy_window:
                if HAS_RAPIDFUZZ:
                    score = fuzz.ratio(norm_text, seen_norm)
                else:
                    # Fallback to difflib
                    score = difflib.SequenceMatcher(None, norm_text, seen_norm).ratio() * 100.0
                
                if score >= self.fuzzy_threshold:
                    return True

        # If not a duplicate, register it
        if self.exact_enabled:
            self.exact_seen.add(text)
        if self.normalized_enabled:
            self.normalized_seen.add(norm_text)
        if self.fuzzy_enabled:
            # deque with maxlen auto-evicts oldest entry — O(1) vs list.pop(0) O(n)
            self.fuzzy_window.append(norm_text)

        return False
