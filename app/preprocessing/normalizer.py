import unicodedata
import re
from typing import Dict, Any

class UnicodeNormalizer:
    """
    Component for normalizing Unicode text, removing control characters,
    and standardizing quotes and punctuation.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.form = config.get("form", "NFC").upper()
        self.normalize_punctuation = config.get("normalize_punctuation", True)
        self.normalize_quotes = config.get("normalize_quotes", True)
        self.remove_zero_width = config.get("remove_zero_width", True)
        self.remove_control_chars = config.get("remove_control_chars", True)

        # Quotes mapping
        self.double_quotes_regex = re.compile(r'[\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033\u2036]')
        self.single_quotes_regex = re.compile(r'[\u2018\u2019\u201a\u201b\u2032\u2035\u0060\u00b4]')
        
        # Zero-width spaces and Joiners/Non-Joiners
        # \u200b: Zero-width space, \ufeff: Byte order mark
        # \u200c (ZWNJ) and \u200d (ZWJ) might be needed for Indic scripts (like Bengali), 
        # but let's make ZWNJ/ZWJ optional or keep them if they are required for Bengali script shaping.
        # Zero-width space (\u200b) is usually safe to remove.
        self.zero_width_regex = re.compile(r'[\u200b\ufeff]')

        # Control characters regex: Unicode category Cc (other, control) and Cf (other, format)
        # excluding ZWNJ (\u200c) and ZWJ (\u200d) since they are used in Bengali/Indic text layout.
        self.control_chars_regex = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xad]')

    def normalize(self, text: str) -> str:
        """
        Normalizes a string according to configuration.
        """
        if not self.enabled or not text:
            return text

        # 1. Unicode normalisation (NFC or NFKC)
        text = unicodedata.normalize(self.form, text)

        # 2. Remove zero-width spaces/BOM
        if self.remove_zero_width:
            text = self.zero_width_regex.sub('', text)

        # 3. Remove other unwanted hidden control characters
        if self.remove_control_chars:
            text = self.control_chars_regex.sub('', text)

        # 4. Standardise quotes
        if self.normalize_quotes:
            text = self.double_quotes_regex.sub('"', text)
            text = self.single_quotes_regex.sub("'", text)

        # 5. Standardise punctuation (e.g. normalize dashes/ellipses)
        if self.normalize_punctuation:
            # normalize multiple hyphens/dashes to a single hyphen, or keep as is
            text = re.sub(r'\s+([.,!?])', r'\1', text)  # remove spaces before common punctuation
            text = re.sub(r'\.\.\.+', '...', text)       # standardize ellipses
            text = re.sub(r' +', ' ', text)              # collapse multiple spaces

        return text
