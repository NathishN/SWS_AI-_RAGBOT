"""
pdf_reader.py
-------------
Reads PDF files and extracts text per-page using PyMuPDF (primary)
with a PyPDF fallback.  Handles corrupted files gracefully.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentContent:
    """Structured output of a PDF extraction."""

    filename: str
    pages: List[str] = field(default_factory=list)
    page_count: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)


def _extract_with_pymupdf(path: Path) -> List[str]:
    """Extract text per-page using PyMuPDF (fitz)."""
    import fitz  # pymupdf

    pages: List[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages.append(page.get_text("text") or "")
    return pages


def _extract_with_pypdf(path: Path) -> List[str]:
    """Fallback extraction using PyPDF."""
    from pypdf import PdfReader

    pages: List[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def read_pdf(file_path: str | Path) -> DocumentContent:
    """
    Extract text from a single PDF file.

    Tries PyMuPDF first; falls back to PyPDF on failure.
    On total failure returns a ``DocumentContent`` with ``error`` set.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the PDF file.
    """
    path = Path(file_path)
    filename = path.name

    if not path.exists():
        logger.error("PDF not found: %s", path)
        return DocumentContent(filename=filename, error=f"File not found: {path}")

    if path.stat().st_size == 0:
        logger.warning("Empty file: %s", path)
        return DocumentContent(filename=filename, error="File is empty.")

    # --- Attempt PyMuPDF ---
    try:
        pages = _extract_with_pymupdf(path)
        logger.info("PyMuPDF extracted %d pages from %s", len(pages), filename)
        return DocumentContent(
            filename=filename, pages=pages, page_count=len(pages)
        )
    except Exception as exc:
        logger.warning(
            "PyMuPDF failed for %s (%s). Trying PyPDF fallback.", filename, exc
        )

    # --- Fallback: PyPDF ---
    try:
        pages = _extract_with_pypdf(path)
        logger.info(
            "PyPDF extracted %d pages from %s", len(pages), filename
        )
        return DocumentContent(
            filename=filename, pages=pages, page_count=len(pages)
        )
    except Exception as exc:
        logger.error("Both extractors failed for %s: %s", filename, exc)
        return DocumentContent(
            filename=filename,
            error=f"Could not extract text: {exc}",
        )


def read_pdfs(file_paths: List[str | Path]) -> List[DocumentContent]:
    """
    Extract text from multiple PDF files.

    Returns a list of ``DocumentContent`` objects (one per file).
    Failed documents have ``error`` set and ``success == False``.

    Parameters
    ----------
    file_paths:
        List of paths to PDF files.
    """
    results: List[DocumentContent] = []
    for fp in file_paths:
        doc = read_pdf(fp)
        results.append(doc)
        status = "OK" if doc.success else f"FAILED: {doc.error}"
        logger.info("Read %s → %s", fp, status)
    return results
