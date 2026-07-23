import pytest
from app.preprocessing.cleaner import TextCleaner

def test_remove_html_xml():
    cleaner = TextCleaner({"remove_html": True, "remove_xml": True})
    text = "<html><body><?xml version='1.0'?><p>Hello World</p></body></html>"
    cleaned = cleaner.clean(text)
    assert cleaned == "Hello World"

def test_remove_markdown():
    cleaner = TextCleaner({"remove_markdown": True})
    text = "### Header\nThis is **bold** and *italic* text with a [link](http://abc.com) and an ![image](img.jpg)."
    cleaned = cleaner.clean(text)
    # Check that styling, links, images, and headers are removed but anchor/alt text is kept
    assert "Header" in cleaned
    assert "bold" in cleaned
    assert "italic" in cleaned
    assert "link" in cleaned
    assert "image" in cleaned
    assert "http://abc.com" not in cleaned

def test_remove_urls_emails():
    cleaner = TextCleaner({"remove_urls": True, "remove_emails": True})
    text = "Contact support@example.com or visit https://example.com/help."
    cleaned = cleaner.clean(text)
    assert "support@example.com" not in cleaned
    assert "https://example.com/help" not in cleaned
    assert cleaned.strip() == "Contact or visit ."

def test_normalize_whitespace():
    cleaner = TextCleaner({"normalize_whitespace": True})
    text = "  Hello \t   World  \n\n\n  How  are  you?  "
    cleaned = cleaner.clean(text)
    assert cleaned == "Hello World\nHow are you?"
