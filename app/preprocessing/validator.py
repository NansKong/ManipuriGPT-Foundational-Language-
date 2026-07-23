import re
from typing import Dict, Any

class SentenceValidator:
    """
    Component for validating sentence quality. Rejects garbage, empty, punctuation-only,
    number-only, or heavily repeated sequences.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.reject_empty = config.get("reject_empty", True)
        self.reject_only_punctuation = config.get("reject_only_punctuation", True)
        self.reject_only_numbers = config.get("reject_only_numbers", True)
        self.reject_corrupted_unicode = config.get("reject_corrupted_unicode", True)
        self.max_repeated_chars = config.get("max_repeated_chars", 4)

        # Indic punctuation includes danda (\u0964) and double danda (\u0965)
        # Let's write a standard python regex for punctuation
        self.punctuation_only_regex = re.compile(
            r'^[\s!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~\u0964\u0965\u201c\u201d\u2018\u2019]+$'
        )
        
        # Only digits (including standard Indic digits if any, but let's cover 0-9 and Unicode digits)
        self.digits_only_regex = re.compile(r'^[\s0-9০-৯꯰-꯹]+$')

    def validate(self, text: str) -> bool:
        """
        Validates the sentence. Returns True if valid, False if it should be rejected.
        """
        if not self.enabled:
            return True

        if not text:
            return not self.reject_empty

        # 1. Reject empty/whitespace only strings
        if self.reject_empty and not text.strip():
            return False

        # 2. Reject corrupted Unicode (contains U+FFFD replacement char)
        if self.reject_corrupted_unicode and '\uFFFD' in text:
            return False

        # 3. Reject only punctuation
        if self.reject_only_punctuation and self.punctuation_only_regex.match(text):
            return False

        # 4. Reject only numbers
        if self.reject_only_numbers and self.digits_only_regex.match(text):
            return False

        # 5. Reject repeated characters (e.g. "aaaaa" or "!!!!!")
        if self.max_repeated_chars > 0:
            # Match any character repeating more than max_repeated_chars times consecutively
            # e.g., if max_repeated_chars = 4, match 5 or more times
            pattern = r'(.)\1{' + str(self.max_repeated_chars) + r',}'
            if re.search(pattern, text):
                return False

        return True
