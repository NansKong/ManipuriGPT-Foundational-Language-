"""Page-level review rules based on comparisons with the source scans."""
import hashlib
import re


# Fingerprint the observed text so a fresh transcription can be ingested.
REVIEWED_ERRORS = {
    ("M001", 1): "c5c721b59bf0b792ab9ac36ea71f4d3bb2ed823eaaad3918a27dc9ce20db431d",
    ("M020", 1): "2e438c300c4ebbfc99fda6908ce08edf21d5839e0ebb7f6c735c7c418a42b7f9",
    ("M025", 22): "037c757ce92979413434bb4ecc4f4d7b0dbca22605d68b6ad7ac873a677b1f9c",
}
# On M013 p21 these runs replace printed Bengali and English entries.
# Flag matching pages for inspection. Other books legitimately use Mayek.
MAYEK_RUN = re.compile(r"[\uabc0-\uabff\uaae0-\uaaff]{12,}")


def review_reason(book_id: str, page: dict) -> str | None:
    text = page.get("text", "")
    fingerprint = REVIEWED_ERRORS.get((book_id, page.get("page")))
    if fingerprint and hashlib.sha256(text.encode("utf-8")).hexdigest() == fingerprint:
        return "scan_mismatch"
    if book_id == "M013" and MAYEK_RUN.search(text):
        return "unexpected_mayek_run"
    return None
