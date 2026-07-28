"""
ScriptEvaluator module (`app/evaluation/script_eval.py`).
Audits generated text for script consistency, unwanted script switching, and invalid Unicode sequences.
"""

import re
from typing import Dict, Any, List
from app.utils.logger import logger


class ScriptEvaluator:
    """Audits Meitei Mayek vs Bengali script consistency and Unicode health."""

    def __init__(self):
        self.meitei_pat = re.compile(r'[\uABC0-\uABFF\u1C80-\u1C8F]')
        self.bengali_pat = re.compile(r'[\u0980-\u09FF]')
        self.invalid_unicode_pat = re.compile(r'[\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F]')

    def evaluate_sequence(self, text: str, target_script: str = "meitei") -> Dict[str, Any]:
        """Audits a single generated sequence for script fidelity and invalid characters."""
        if not text:
            return {"text": text, "valid": False, "reason": "empty"}

        meitei_chars = len(self.meitei_pat.findall(text))
        bengali_chars = len(self.bengali_pat.findall(text))
        invalid_unicodes = len(self.invalid_unicode_pat.findall(text))

        total_chars = max(len(text), 1)

        has_unwanted_switch = False
        if target_script == "meitei" and bengali_chars > 3:
            has_unwanted_switch = True
        elif target_script == "bengali" and meitei_chars > 3:
            has_unwanted_switch = True

        script_confusion = False
        if meitei_chars > 0 and bengali_chars > 0 and abs(meitei_chars - bengali_chars) / total_chars < 0.2:
            script_confusion = True

        return {
            "text": text,
            "target_script": target_script,
            "meitei_char_count": meitei_chars,
            "bengali_char_count": bengali_chars,
            "invalid_unicode_count": invalid_unicodes,
            "has_unwanted_switch": has_unwanted_switch,
            "script_confusion": script_confusion
        }

    def evaluate_corpus(self, texts: List[str], target_script: str = "meitei") -> Dict[str, Any]:
        """Audits a list of generated texts and calculates overall script metrics."""
        results = [self.evaluate_sequence(t, target_script=target_script) for t in texts]

        total_samples = max(len(results), 1)
        unwanted_switches = sum(1 for r in results if r["has_unwanted_switch"])
        confusions = sum(1 for r in results if r["script_confusion"])
        invalid_unicodes = sum(r["invalid_unicode_count"] for r in results)

        return {
            "target_script": target_script,
            "total_evaluated": total_samples,
            "unwanted_script_switch_rate": round(unwanted_switches / total_samples, 4),
            "script_confusion_rate": round(confusions / total_samples, 4),
            "total_invalid_unicodes": invalid_unicodes,
            "sample_audits": results[:10]
        }
