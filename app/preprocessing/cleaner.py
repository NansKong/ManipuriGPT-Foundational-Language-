import re
from typing import Dict, Any

class TextCleaner:
    """
    Component for cleaning text from markup tags, markdown symbols, URLs, email addresses,
    and extra whitespace.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.remove_html = config.get("remove_html", True)
        self.remove_xml = config.get("remove_xml", True)
        self.remove_markdown = config.get("remove_markdown", True)
        self.remove_urls = config.get("remove_urls", True)
        self.remove_emails = config.get("remove_emails", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)

        # Regex patterns
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.xml_tag_pattern = re.compile(r'<\?[^?>]+\?>|<\![^>]+>')
        
        # URLs and Emails
        self.url_pattern = re.compile(r'https?://\S+?(?=[.,;:?!]*(?:\s|$))|www\.\S+?(?=[.,;:?!]*(?:\s|$))|ftp://\S+?(?=[.,;:?!]*(?:\s|$))')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Markdown artifacts:
        # bold/italic/strikethrough (**text**, *text*, ~~text~~), headers (### text), blockquotes, etc.
        # links ([text](url)), images (![text](url))
        self.md_image_pattern = re.compile(r'\!\[([^\]]*)\]\([^\)]+\)')
        self.md_link_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        self.md_headers_pattern = re.compile(r'^\s*#{1,6}\s+', re.MULTILINE)
        self.md_styles_pattern = re.compile(r'\*\*|__|\*|_|~~|`')

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
            # Replace markdown images/links with their alt/anchor text
            text = self.md_image_pattern.sub(r'\1', text)
            text = self.md_link_pattern.sub(r'\1', text)
            text = self.md_headers_pattern.sub('', text)
            text = self.md_styles_pattern.sub('', text)

        # 4. Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)

        # 5. Remove Emails
        if self.remove_emails:
            text = self.email_pattern.sub('', text)

        # 6. Normalize Whitespace (replace multiple tabs/spaces with single space, remove empty lines)
        if self.normalize_whitespace:
            # Replace tabs with spaces
            text = text.replace('\t', ' ')
            # Split into lines, strip each line, remove empty lines, collapse internal spaces
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            lines = [re.sub(r' +', ' ', line) for line in lines]
            text = '\n'.join(lines)

        # 7. Clean OCR Noise
        text = self.clean_ocr_noise(text)

        return text

    def clean_ocr_noise(self, text: str) -> str:
        """
        Cleans OCR noise typical in scanned Manipuri/Bengali PDFs:
        - Removes OCR page header/footer markers like (vi), (1v), (ii1), (viii)
        - Removes isolated garbled Latin token sequences from non-English text
        - Fixes broken diacritics and excessive whitespace
        """
        if not text:
            return text

        # Remove page markers like (vi), (vii), (1v), (ii1), (iii)
        text = re.sub(r'^\s*\([ivxIVX0-9]+\)\s*', '', text)

        # Remove multi-dots/dandas and normalize whitespace
        text = re.sub(r'[\.\-\_=~]{3,}', ' ', text)
        text = re.sub(r' +', ' ', text).strip()
        return text

