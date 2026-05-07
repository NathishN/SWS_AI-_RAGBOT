"""
chunker.py
----------
Splits cleaned document text into overlapping semantic chunks using
LangChain's RecursiveCharacterTextSplitter.
"""

import logging
import os
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


@dataclass
class TextChunk:
    """A single text chunk with metadata."""

    text: str
    source: str          # filename or document identifier
    chunk_index: int     # zero-based position within the source document


def _build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Instantiate the text splitter with the given configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Split a single document string into ``TextChunk`` objects.

    Parameters
    ----------
    text:
        Cleaned document text (output of ``data_preprocessor``).
    source:
        Document identifier (filename, URL, etc.).
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Overlap between consecutive chunks in characters.

    Returns
    -------
    List of ``TextChunk`` objects ordered by position in the document.
    """
    if not text or not text.strip():
        logger.warning("Empty text for source=%r — no chunks produced.", source)
        return []

    splitter = _build_splitter(chunk_size, chunk_overlap)
    raw_chunks: List[str] = splitter.split_text(text)

    chunks = [
        TextChunk(text=raw, source=source, chunk_index=i)
        for i, raw in enumerate(raw_chunks)
        if raw.strip()
    ]

    logger.info(
        "Chunked %r → %d chunks (size=%d, overlap=%d).",
        source,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def chunk_documents(
    documents: List[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Chunk multiple documents at once.

    Parameters
    ----------
    documents:
        List of ``{"source": str, "text": str}`` dicts.

    Returns
    -------
    Flat list of all ``TextChunk`` objects across all documents.
    """
    all_chunks: List[TextChunk] = []
    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        chunks = chunk_text(text, source=source, chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(
        "Total chunks produced from %d documents: %d",
        len(documents),
        len(all_chunks),
    )
    return all_chunks
