"""
data_preprocessor.py
--------------------
Cleans and normalises raw text extracted from PDFs before chunking.
"""

import logging
import re
import unicodedata
from typing import List

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """Normalise unicode characters to NFC form and strip surrogates."""
    return unicodedata.normalize("NFC", text)


def remove_excessive_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs; preserve single newlines."""
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_artifacts(text: str) -> str:
    """
    Remove common PDF extraction artefacts such as:
    - Page numbers standing alone on a line  (e.g. "  3  " or "Page 3 of 10")
    - Repeated header/footer noise
    - Null bytes and control characters
    """
    # Strip null bytes and non-printable control chars (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Remove lines that are purely numeric (lone page numbers)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Remove "Page X of Y" patterns
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
    return text


def fix_hyphenation(text: str) -> str:
    """
    Rejoin words that were hyphenated across a line break, e.g.
    "pro-\ncess" → "process".
    """
    return re.sub(r"-\n(\w)", r"\1", text)


def preprocess_text(raw_text: str) -> str:
    """
    Full preprocessing pipeline applied to raw PDF text.

    Steps
    -----
    1. Unicode normalisation
    2. Remove PDF artefacts
    3. Fix hyphenation
    4. Collapse excessive whitespace

    Returns empty string for blank/whitespace-only input.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = normalize_unicode(raw_text)
    text = remove_page_artifacts(text)
    text = fix_hyphenation(text)
    text = remove_excessive_whitespace(text)

    return text


def preprocess_pages(pages: List[str]) -> List[str]:
    """
    Apply ``preprocess_text`` to a list of per-page strings.
    Empty pages (after cleaning) are discarded.

    Parameters
    ----------
    pages:
        List of raw page strings extracted from a PDF.

    Returns
    -------
    List of cleaned, non-empty page strings.
    """
    cleaned: List[str] = []
    for i, page_text in enumerate(pages):
        cleaned_text = preprocess_text(page_text)
        if cleaned_text:
            cleaned.append(cleaned_text)
        else:
            logger.debug("Page %d is empty after preprocessing — skipped.", i)

    logger.info(
        "Preprocessed %d pages → %d non-empty pages.", len(pages), len(cleaned)
    )
    return cleaned


def merge_pages(pages: List[str], separator: str = "\n\n") -> str:
    """Join a list of cleaned page strings into a single document string."""
    return separator.join(pages)
