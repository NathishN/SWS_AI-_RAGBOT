"""
vector_store.py
---------------
ChromaDB vector store: adding documents, similarity search, and persistence.
Uses a pre-computed embedding function (sentence-transformers via embedder.py).
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from chunker import TextChunk
from embedder import embed_texts, embed_query

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv(
    "CHROMA_COLLECTION_NAME", "rag_documents"
)
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# Singleton client and collection
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    """Lazy-initialise the ChromaDB client and collection."""
    global _client, _collection

    if _collection is not None:
        return _collection

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    _client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    logger.info(
        "ChromaDB collection '%s' ready (%d docs).",
        CHROMA_COLLECTION_NAME,
        _collection.count(),
    )
    return _collection


def add_documents(chunks: List[TextChunk]) -> int:
    """
    Embed *chunks* and upsert them into ChromaDB.

    Parameters
    ----------
    chunks:
        List of ``TextChunk`` objects to store.

    Returns
    -------
    Number of chunks successfully added.
    """
    if not chunks:
        logger.warning("add_documents called with empty chunk list.")
        return 0

    collection = _get_collection()

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas: List[Dict[str, Any]] = [
        {"source": c.source, "chunk_index": c.chunk_index} for c in chunks
    ]

    # Upsert in batches of 500 (ChromaDB limit per call)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info("Added %d chunks to ChromaDB.", len(chunks))
    return len(chunks)


def similarity_search(
    query: str,
    top_k: int = TOP_K_RESULTS,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the *top_k* most relevant chunks for a query.

    Parameters
    ----------
    query:
        The user's question / search string.
    top_k:
        Number of results to return.
    where:
        Optional ChromaDB metadata filter (e.g. ``{"source": "doc.pdf"}``).

    Returns
    -------
    List of dicts with keys ``text``, ``source``, ``chunk_index``,
    and ``distance``.
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("Vector store is empty — no results returned.")
        return []

    query_embedding = embed_query(query)

    kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
            }
        )

    return output


def get_collection_stats() -> Dict[str, Any]:
    """Return basic statistics about the vector store."""
    collection = _get_collection()
    return {
        "collection_name": CHROMA_COLLECTION_NAME,
        "document_count": collection.count(),
        "persist_dir": CHROMA_PERSIST_DIR,
    }


def delete_documents_by_source(source: str) -> None:
    """Remove all chunks that originated from *source* (filename)."""
    collection = _get_collection()
    collection.delete(where={"source": source})
    logger.info("Deleted all chunks for source=%r.", source)


def persist_vectorstore() -> None:
    """
    Explicitly persist the vector store to disk.
    (ChromaDB PersistentClient auto-persists; this is a no-op kept for API
    compatibility with older ChromaDB versions.)
    """
    logger.debug("ChromaDB PersistentClient auto-persists — no action needed.")
