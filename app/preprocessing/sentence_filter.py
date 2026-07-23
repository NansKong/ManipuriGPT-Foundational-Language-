from typing import Dict, Any

class SentenceFilter:
    """
    Component for filtering sentences based on length (character count or word count).
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.min_length = config.get("min_length", 3)
        self.max_length = config.get("max_length", 512)
        self.length_unit = config.get("length_unit", "characters")  # "characters" or "words"

    def filter(self, text: str) -> bool:
        """
        Filters a sentence. Returns True if it matches length constraints, False if rejected.
        """
        if not self.enabled:
            return True

        if not text:
            return self.min_length <= 0

        # Calculate length based on unit
        if self.length_unit == "words":
            length = len(text.split())
        else:
            length = len(text)

        return self.min_length <= length <= self.max_length
