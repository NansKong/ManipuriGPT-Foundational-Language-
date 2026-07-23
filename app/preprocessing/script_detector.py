import re
from typing import Dict, Any

class ScriptDetector:
    """
    Component for detecting the writing script of a text (Meitei Mayek, Bengali, Latin, Devanagari, or Mixed).
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.target_script = config.get("target_script", "any")

        # Compile regex ranges for script detection
        # Meitei Mayek block: U+ABC0 to U+ABFF, extension block: U+AAE0 to U+AAFF
        self.meitei_pattern = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAFF]')
        # Bengali block: U+0980 to U+09FF
        self.bengali_pattern = re.compile(r'[\u0980-\u09FF]')
        # Devanagari block: U+0900 to U+097F
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        # Latin characters: basic letters
        self.latin_pattern = re.compile(r'[a-zA-Z]')
        
        # Ignorable characters (spaces, punctuation, digits)
        self.ignorable_pattern = re.compile(r'[\s\d.,\/#!$%\^&\*;:{}=\-_`~()?"\'\[\]\\|<>+@\u0964\u0965]')

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detects the dominant script in the given text.
        Returns:
            dict: {
                "script": str ("meitei", "bengali", "latin", "devanagari", "mixed", or "unknown"),
                "confidence": float (0.0 to 1.0)
            }
        """
        if not self.enabled or not text:
            return {"script": "unknown", "confidence": 0.0}

        # Count character frequencies for each script category
        meitei_count = len(self.meitei_pattern.findall(text))
        bengali_count = len(self.bengali_pattern.findall(text))
        devanagari_count = len(self.devanagari_pattern.findall(text))
        latin_count = len(self.latin_pattern.findall(text))

        total_script_chars = meitei_count + bengali_count + devanagari_count + latin_count

        if total_script_chars == 0:
            return {"script": "unknown", "confidence": 0.0}

        counts = {
            "meitei": meitei_count,
            "bengali": bengali_count,
            "devanagari": devanagari_count,
            "latin": latin_count
        }

        # Find the dominant script
        sorted_scripts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        dominant_script, dominant_count = sorted_scripts[0]
        
        # Calculate confidence as percentage of script characters
        confidence = dominant_count / total_script_chars

        # Determine if it's mixed. If the second most dominant script has more than 15% representation, 
        # or if the dominant script doesn't cover at least 80% of script characters.
        second_script, second_count = sorted_scripts[1]
        second_ratio = second_count / total_script_chars if total_script_chars > 0 else 0.0

        if second_ratio > 0.15:
            return {"script": "mixed", "confidence": confidence}

        return {
            "script": dominant_script,
            "confidence": round(confidence, 4)
        }
