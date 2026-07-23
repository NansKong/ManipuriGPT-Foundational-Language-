"""
ScriptCanonicalizer module for canonicalizing multi-script Manipuri text into canonical Meitei Mayek.
As established in Phase 5.3, we canonicalize Bengali script (0x0980-0x09FF) and Romanized Manipuri into
canonical Meitei Mayek (0xABC0-0xABFF) before subword tokenization, halving vocabulary requirements
and concentrating embeddings on one unified script representation.
"""

import re
from typing import Dict, Any, Optional, Tuple


class ScriptCanonicalizer:
    """
    Canonicalizes Manipuri text across different scripts (Bengali, Romanized, Meitei Mayek)
    into a unified Meitei Mayek representation, or preserves multi-script representations cleanly.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.enabled = config.get("enabled", False)
        self.mode = config.get("mode", "preserve").lower()
        self.target_script = config.get("target_script", "meitei")
        self.use_neural_fallback = config.get("use_neural_fallback", False)

        # Unicode ranges
        self.bengali_range = re.compile(r'[\u0980-\u09FF]')
        self.mayek_range = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAFF]')

        # Rule-based Bengali -> Meitei Mayek mapping table for Manipuri (Meitei Lon)
        # Standard consonant & vowel correspondences between Bengali/Assamese script and Meitei Mayek.
        self.bengali_to_mayek_map = {
            # Vowels & Vowel Signs
            "অ": "ꯑ", "আ": "ꯑꯥ", "ই": "ꯏ", "ঈ": "ꯏ", "উ": "ꯎ", "ঊ": "ꯎ",
            "ঋ": "ꯔꯤ", "এ": "ꯑꯦ", "ঐ": "ꯑꯩ", "ও": "ꯑꯣ", "ঔ": "ꯑꯧ",
            "া": "ꯥ", "ি": "ꯤ", "ী": "ꯤ", "ু": "ꯨ", "ূ": "ꯨ",
            "ৃ": "꯭ꯔꯤ", "ে": "ꯦ", "ৈ": "ꯩ", "ো": "ꯣ", "ৌ": "ꯧ",

            # Consonants
            "ক": "ꯀ", "খ": "ꯈ", "গ": "ꯒ", "ঘ": "ꯘ", "ঙ": "ꯉ",
            "চ": "ꯆ", "ছ": "ꯆ", "জ": "ꯖ", "ঝ": "ꯓ", "ঞ": "ꯅ",
            "ট": "ꯇ", "ঠ": "ꯊ", "ড": "ꯗ", "ঢ": "ꯙ", "ণ": "ꯅ",
            "ত": "ꯇ", "থ": "ꯊ", "দ": "ꯗ", "ধ": "ꯙ", "ন": "ꯅ",
            "প": "ꯄ", "ফ": "ꯐ", "ব": "ꯕ", "ভ": "ꯚ", "ম": "ꯃ",
            "য": "ꯌ", "র": "ꯔ", "ল": "ꯂ", "শ": "ꯁ", "ষ": "ꯁ",
            "স": "ꯁ", "হ": "ꯍ", "ড়": "ꯔ", "ঢ়": "ꯔ", "য়": "ꯌ",

            # Anusvara, Visarga, Chandrabindu, Virama
            "ং": "ꯡ", "ঃ": "ꯍ", "ঁ": "ꯪ", "্": "꯭",

            # Digits
            "০": "꯰", "১": "꯱", "২": "꯲", "৩": "꯳", "৪": "꯴",
            "৫": "꯵", "৬": "꯶", "৭": "꯷", "৮": "꯸", "৯": "꯹"
        }

        # Sort keys by length descending to handle compound characters first
        self._sorted_bengali_keys = sorted(self.bengali_to_mayek_map.keys(), key=len, reverse=True)

    def detect_script_distribution(self, text: str) -> Dict[str, float]:
        """
        Calculates the proportion of characters belonging to Meitei Mayek, Bengali, Latin, and Others.
        """
        if not text:
            return {"meitei": 0.0, "bengali": 0.0, "latin": 0.0, "other": 0.0}

        chars = [c for c in text if not c.isspace()]
        if not chars:
            return {"meitei": 0.0, "bengali": 0.0, "latin": 0.0, "other": 0.0}

        total = len(chars)
        meitei_c = sum(1 for c in chars if self.mayek_range.match(c))
        bengali_c = sum(1 for c in chars if self.bengali_range.match(c))
        latin_c = sum(1 for c in chars if 'a' <= c.lower() <= 'z')
        other_c = total - (meitei_c + bengali_c + latin_c)

        return {
            "meitei": round(meitei_c / total, 4),
            "bengali": round(bengali_c / total, 4),
            "latin": round(latin_c / total, 4),
            "other": round(other_c / total, 4)
        }

    def canonicalize_to_mayek(self, text: str, source_script: Optional[str] = None) -> str:
        """
        Transliterates and canonicalizes text to Meitei Mayek.
        If the text is in Bengali script, transliterates it to Meitei Mayek.
        If already in Meitei Mayek or English/Latin code/metadata, preserves non-Bengali characters.
        """
        if not text:
            return text

        # Check if Bengali characters are present
        if not self.bengali_range.search(text):
            return text

        # Perform dictionary-based transliteration of Bengali characters to Meitei Mayek
        canonical_text = text
        for bn_char in self._sorted_bengali_keys:
            canonical_text = canonical_text.replace(bn_char, self.bengali_to_mayek_map[bn_char])

        return canonical_text

    def process_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Processes input text, canonicalizing or preserving according to configured mode.
        Returns the primary representation along with non-destructive metadata containing original and canonical forms.
        """
        orig_dist = self.detect_script_distribution(text)
        has_bengali = orig_dist.get("bengali", 0.0) > 0.05

        if not self.enabled or self.mode == "preserve" or not has_bengali:
            processed_text = text
            final_dist = orig_dist
            was_canonicalized = False
            canonical_text = text if not has_bengali else self.canonicalize_to_mayek(text)
        else:
            canonical_text = self.canonicalize_to_mayek(text)
            final_dist = self.detect_script_distribution(canonical_text)
            was_canonicalized = True
            processed_text = canonical_text if self.mode in ("canonical", "hybrid") else text

        meta = {
            "canonicalized": was_canonicalized,
            "canonicalization_mode": self.mode if self.enabled else "disabled",
            "original_text": text,
            "canonical_text": canonical_text,
            "original_script_dist": orig_dist,
            "final_script_dist": final_dist
        }
        return processed_text, meta
