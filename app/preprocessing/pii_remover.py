"""
PIIRemover module for detecting and stripping/masking Personally Identifiable Information (PII)
from raw and preprocessed corpora across multilingual text.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class PIIRemover:
    """
    Detects and masks/removes PII entities such as email addresses, phone numbers,
    IP addresses, credit card numbers, URLs, and government identifiers (e.g., Aadhaar/PAN format).
    """
    def __init__(
        self,
        mask_replacement: str = "<PII>",
        remove_pii: bool = False,
        enabled_patterns: Optional[List[str]] = None
    ):
        self.mask_replacement = mask_replacement
        self.remove_pii = remove_pii
        
        # Compiled regex patterns for PII detection
        self.patterns: Dict[str, re.Pattern] = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'),
            "phone": re.compile(r'\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b|\b(?:\+?1[\-\s]?)?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}\b'),
            "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            "credit_card": re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'),
            "pan_card": re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'),
            "aadhaar": re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
            "url": re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*')
        }
        
        if enabled_patterns:
            self.active_patterns = {k: v for k, v in self.patterns.items() if k in enabled_patterns}
        else:
            self.active_patterns = self.patterns

    def clean_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Cleans input text by masking or stripping matched PII patterns.
        Returns cleaned text and summary counts of masked entities.
        """
        if not text or not isinstance(text, str):
            return text, {}

        counts: Dict[str, int] = {}
        cleaned = text

        for entity_type, pattern in self.active_patterns.items():
            matches = pattern.findall(cleaned)
            if matches:
                counts[entity_type] = len(matches)
                replacement = "" if self.remove_pii else f"<{entity_type.upper()}>"
                cleaned = pattern.sub(replacement, cleaned)

        # Collapse extra spaces if we stripped text
        if self.remove_pii:
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned, counts

    def process(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a dictionary example containing a 'text' field.
        Appends PII detection metadata to example['metadata'].
        """
        text = example.get("text", "")
        cleaned_text, pii_counts = self.clean_text(text)
        
        metadata = example.get("metadata", {}).copy()
        if pii_counts:
            metadata["pii_detected"] = pii_counts
            metadata["pii_cleaned"] = True
        else:
            metadata["pii_cleaned"] = False

        result = example.copy()
        result["text"] = cleaned_text
        result["metadata"] = metadata
        return result
