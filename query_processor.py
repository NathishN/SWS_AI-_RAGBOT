"""
query_processor.py
------------------
Orchestrates the full RAG pipeline:
  User Query → Embedding → Vector Search → Prompt Augmentation → Gemini → Response
"""

import logging
import os
from typing import List, Optional

from vector_store import similarity_search
from llm import generate_response

logger = logging.getLogger(__name__)

TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))


def process_query(
    query: str,
    chat_history: Optional[List[dict]] = None,
    top_k: int = TOP_K_RESULTS,
    source_filter: Optional[str] = None,
) -> dict:
    """
    Run the end-to-end RAG pipeline for a single user query.

    Pipeline
    --------
    1. Validate query
    2. Embed the query
    3. Retrieve top-k similar chunks from ChromaDB
    4. Build augmented prompt
    5. Call Gemini LLM
    6. Return structured response

    Parameters
    ----------
    query:
        The user's question.
    chat_history:
        Previous ``{role, content}`` messages for conversation continuity.
    top_k:
        Number of context chunks to retrieve.
    source_filter:
        Optional filename to restrict retrieval to a single document.

    Returns
    -------
    Dict with keys:
        - ``answer``: str — the LLM response
        - ``sources``: list[str] — deduplicated source filenames
        - ``chunks_used``: int — number of context chunks
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string.")

    logger.info("Processing query: %r (top_k=%d)", query[:80], top_k)

    # Step 1: Retrieve relevant chunks
    where_filter = {"source": source_filter} if source_filter else None
    chunks = similarity_search(query, top_k=top_k, where=where_filter)

    logger.info("Retrieved %d chunks for query.", len(chunks))

    # Step 2: Generate answer
    answer = generate_response(
        query=query,
        context_chunks=chunks,
        chat_history=chat_history,
    )

    # Step 3: Build source list
    sources = list({c["source"] for c in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }
