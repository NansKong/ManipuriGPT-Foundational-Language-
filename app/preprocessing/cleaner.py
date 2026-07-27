import re
from collections import Counter
from typing import Dict, Any, Tuple

class TextCleaner:
    """
    Component for cleaning text from markup tags, markdown symbols, URLs, email addresses,
    unprintable control characters, foreign script leakage, and extra whitespace.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.remove_html = config.get("remove_html", True)
        self.remove_xml = config.get("remove_xml", True)
        self.remove_markdown = config.get("remove_markdown", True)
        self.remove_urls = config.get("remove_urls", True)
        self.remove_emails = config.get("remove_emails", True)
        self.remove_control_chars = config.get("remove_control_chars", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        self.remove_foreign_scripts = config.get("remove_foreign_scripts", True)

        # Removal statistics tracking counter
        self.removal_stats = Counter()

        # Regex patterns
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.xml_tag_pattern = re.compile(r'<\?[^?>]+\?>|<\![^>]+>')
        self.control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
        
        # URLs and Emails
        self.url_pattern = re.compile(r'https?://\S+?(?=[.,;:?!]*(?:\s|$))|www\.\S+?(?=[.,;:?!]*(?:\s|$))|ftp://\S+?(?=[.,;:?!]*(?:\s|$))')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Foreign script regex definitions
        self.script_patterns = {
            "gujarati": re.compile(r'[\u0A80-\u0AFF]'),
            "odia": re.compile(r'[\u0B00-\u0B7F]'),
            "tamil": re.compile(r'[\u0B80-\u0BFF]'),
            "telugu": re.compile(r'[\u0C00-\u0C7F]'),
            "kannada": re.compile(r'[\u0C80-\u0CFF]'),
            "malayalam": re.compile(r'[\u0D00-\u0D7F]'),
            "gurmukhi": re.compile(r'[\u0A00-\u0A7F]')
        }

    def clean(self, text: str) -> str:
        """
        Cleans text according to configured rules.
        """
        if not self.enabled or not text:
            return text

        # 1. Remove XML declarations/doctypes
        if self.remove_xml:
            text = self.xml_tag_pattern.sub('', text)

        # 2. Remove HTML tags
        if self.remove_html:
            text = self.html_tag_pattern.sub('', text)

        # 3. Remove Markdown artifacts
        if self.remove_markdown:
            text = re.sub(r'\!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\*\*|__|\*|_|~~|`', '', text)

        # 4. Remove URLs & Emails
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        if self.remove_emails:
            text = self.email_pattern.sub('', text)

        # 5. Remove Unprintable Control Characters
        if self.remove_control_chars:
            cnt = len(self.control_char_pattern.findall(text))
            if cnt > 0:
                self.removal_stats["control_characters_removed"] += cnt
                text = self.control_char_pattern.sub('', text)

        # 6. Remove Foreign Script Characters (Odia, Gujarati, Tamil, Telugu, etc.) with stats tracking
        if self.remove_foreign_scripts:
            text = self.filter_foreign_script_chars(text)

        # 7. Normalize Whitespace
        if self.normalize_whitespace:
            text = text.replace('\t', ' ')
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            lines = [re.sub(r' +', ' ', line) for line in lines]
            text = '\n'.join(lines)

        # 8. Clean OCR Noise
        text = self.clean_ocr_noise(text)
        return text

    def filter_foreign_script_chars(self, text: str) -> str:
        """
        Strips characters from foreign non-Manipuri scripts (Gujarati, Odia, Tamil, Telugu, Malayalam, Gurmukhi)
        that leak into raw documents, and tracks character removal statistics.
        """
        if not text:
            return text

        for script_name, pat in self.script_patterns.items():
            matches = len(pat.findall(text))
            if matches > 0:
                self.removal_stats[f"{script_name}_removed"] += matches
                text = pat.sub('', text)

        return text

    def get_removal_summary(self) -> Dict[str, int]:
        """Returns the summary dict of removed characters across all cleaning operations."""
        return dict(self.removal_stats)

    def clean_ocr_noise(self, text: str) -> str:
        """
        Cleans OCR noise typical in scanned Manipuri/Bengali PDFs.
        """
        if not text:
            return text
        text = re.sub(r'^\s*\([ivxIVX0-9]+\)\s*', '', text)
        text = re.sub(r'[\.\-\_=~]{3,}', ' ', text)
        text = re.sub(r' +', ' ', text).strip()
        return text

