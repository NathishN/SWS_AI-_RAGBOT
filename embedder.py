"""
embedder.py
-----------
Generates dense vector embeddings using Sentence Transformers.
Provides a singleton model instance and batch embedding helpers.
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Module-level singleton (loaded once, reused across requests)
_model = None


def _get_model():
    """Lazy-load the Sentence Transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for a list of strings.

    Parameters
    ----------
    texts:
        List of non-empty strings to embed.
    batch_size:
        Number of texts processed per forward pass.

    Returns
    -------
    List of embedding vectors (list of floats) in the same order as *texts*.
    """
    if not texts:
        return []

    model = _get_model()
    logger.debug("Embedding %d texts (batch_size=%d).", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity ready
    )

    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Parameters
    ----------
    query:
        The user's search string.

    Returns
    -------
    Single embedding vector as a list of floats.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    result = embed_texts([query])
    return result[0]


def get_embedding_dimension() -> int:
    """Return the dimensionality of the current embedding model's output."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
