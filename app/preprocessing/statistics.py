import json
from typing import Dict, Any, List

class StatisticsTracker:
    """
    Component for tracking preprocessing metrics, filters, and generating analytical reports.
    """
    def __init__(self):
        self.total_processed = 0
        self.unicode_fixed = 0
        self.cleaner_fixed = 0
        
        # Rejected counts
        self.empty_removed = 0
        self.invalid_unicode_removed = 0
        self.only_punctuation_removed = 0
        self.only_numbers_removed = 0
        self.repeated_chars_removed = 0
        self.length_filtered_removed = 0
        self.duplicates_removed = 0
        
        # Distributions and lengths
        self.languages: Dict[str, int] = {}
        self.scripts: Dict[str, int] = {}
        self.final_lengths: List[int] = []
        self.final_count = 0

    def record_unicode_fix(self, before: str, after: str):
        self.total_processed += 1
        if before != after:
            self.unicode_fixed += 1

    def record_cleaner_fix(self, before: str, after: str):
        if before != after:
            self.cleaner_fixed += 1

    def record_language(self, lang: str):
        self.languages[lang] = self.languages.get(lang, 0) + 1

    def record_script(self, script: str):
        self.scripts[script] = self.scripts.get(script, 0) + 1

    def record_final_sample(self, text: str):
        self.final_count += 1
        self.final_lengths.append(len(text))

    def generate_report(self) -> Dict[str, Any]:
        """Generates a dictionary report of preprocessing statistics."""
        avg_len = sum(self.final_lengths) / len(self.final_lengths) if self.final_lengths else 0.0
        min_len = min(self.final_lengths) if self.final_lengths else 0
        max_len = max(self.final_lengths) if self.final_lengths else 0
        dup_pct = (self.duplicates_removed / self.total_processed) * 100 if self.total_processed > 0 else 0.0
        
        # Convert absolute counts to percentages for distributions
        total_langs = sum(self.languages.values()) or 1
        total_scripts = sum(self.scripts.values()) or 1
        
        lang_percentages = {k: round((v / total_langs) * 100, 2) for k, v in self.languages.items()}
        script_percentages = {k: round((v / total_scripts) * 100, 2) for k, v in self.scripts.items()}

        return {
            "total_processed": self.total_processed,
            "final_accepted": self.final_count,
            "duplicate_percentage": round(dup_pct, 2),
            "unicode_fixed": self.unicode_fixed,
            "cleaner_fixed": self.cleaner_fixed,
            "avg_length_chars": round(avg_len, 2),
            "min_length_chars": min_len,
            "max_length_chars": max_len,
            "skipped": {
                "empty_removed": self.empty_removed,
                "invalid_unicode_removed": self.invalid_unicode_removed,
                "only_punctuation_removed": self.only_punctuation_removed,
                "only_numbers_removed": self.only_numbers_removed,
                "repeated_chars_removed": self.repeated_chars_removed,
                "length_filtered_removed": self.length_filtered_removed,
                "duplicates_removed": self.duplicates_removed,
            },
            "languages": lang_percentages,
            "scripts": script_percentages
        }

    def generate_markdown(self) -> str:
        """Generates a Markdown string representing the preprocessing report."""
        report = self.generate_report()
        skipped = report["skipped"]
        total_skipped = sum(skipped.values())

        md = []
        md.append("# Preprocessing Statistics Report\n")
        md.append("## Overview")
        md.append(f"- **Total Raw Samples Processed**: {report['total_processed']}")
        md.append(f"- **Final Training-Ready Samples**: {report['final_accepted']}")
        md.append(f"- **Total Samples Dropped**: {total_skipped}")
        md.append(f"- **Duplicate Percentage**: {report['duplicate_percentage']}%")
        md.append(f"- **Average Clean Sentence Length**: {report['avg_length_chars']} chars")
        md.append(f"- **Min/Max Clean Sentence Length**: {report['min_length_chars']} / {report['max_length_chars']} chars\n")

        md.append("## Corrections Applied")
        md.append(f"- **Unicode Fixes (NFC/NFKC/Quotes)**: {report['unicode_fixed']}")
        md.append(f"- **HTML/Markdown/URL/Whitespace Cleans**: {report['cleaner_fixed']}\n")

        md.append("## Rejection breakdown")
        md.append("| Rejection Reason | Count |")
        md.append("| --- | --- |")
        md.append(f"| Empty or Whitespace-only | {skipped['empty_removed']} |")
        md.append(f"| Corrupted Unicode (U+FFFD) | {skipped['invalid_unicode_removed']} |")
        md.append(f"| Punctuation-only sentences | {skipped['only_punctuation_removed']} |")
        md.append(f"| Digit-only sentences | {skipped['only_numbers_removed']} |")
        md.append(f"| Excessive character repetition | {skipped['repeated_chars_removed']} |")
        md.append(f"| Out of length bounds | {skipped['length_filtered_removed']} |")
        md.append(f"| Duplicate / Near-duplicate samples | {skipped['duplicates_removed']} |")
        md.append(f"| **Total Skipped** | **{total_skipped}** |\n")

        md.append("## Language Distribution (Percentage)")
        for lang, pct in report["languages"].items():
            md.append(f"- **{lang}**: {pct}%")
        md.append("")

        md.append("## Writing Script Distribution (Percentage)")
        for script, pct in report["scripts"].items():
            md.append(f"- **{script}**: {pct}%")
        md.append("")

        return "\n".join(md)

    def save_markdown_report(self, path: str) -> None:
        """Saves the Markdown report to a file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())

    def save_json_report(self, path: str) -> None:
        """Saves the JSON report to a file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate_report(), f, indent=2)
