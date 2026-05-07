"""
llm.py
------
Gemini API integration: prompt engineering, context injection, and
hallucination-reduction system prompt.
"""

import logging
import os
from typing import List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Lazy-initialised model instance
_model = None


def _configure_gemini() -> None:
    """Configure the Gemini SDK with the API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )
    genai.configure(api_key=api_key)


def _get_model():
    """Lazy-initialise the Gemini GenerativeModel."""
    global _model
    if _model is None:
        _configure_gemini()
        _model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.2,          # lower → more factual
                top_p=0.9,
                max_output_tokens=2048,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )
        logger.info("Gemini model '%s' initialised.", GEMINI_MODEL)
    return _model


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert enterprise AI assistant that answers questions \
strictly based on the provided document context.

Rules you MUST follow:
1. Answer ONLY using information present in the <context> section below.
2. If the answer cannot be found in the context, respond with:
   "I could not find this information in the uploaded documents."
3. Do NOT fabricate, hallucinate, or extrapolate facts.
4. Cite the source document name when referencing specific information.
5. Be concise, accurate, and professional.
6. If the question is ambiguous, ask a clarifying question.
7. Format responses clearly; use bullet points or numbered lists when appropriate.
"""


def build_rag_prompt(
    query: str,
    context_chunks: List[dict],
    chat_history: Optional[List[dict]] = None,
) -> str:
    """
    Build the full prompt for the Gemini API.

    Parameters
    ----------
    query:
        The user's current question.
    context_chunks:
        Retrieved chunks from vector_store.similarity_search().
    chat_history:
        Optional list of ``{role, content}`` dicts for multi-turn context.
    """
    # Format retrieved context
    if context_chunks:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[{i}] Source: {chunk['source']}\n{chunk['text']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)
    else:
        context_str = "No relevant context found in the uploaded documents."

    # Format recent chat history (last 6 turns)
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_str = "\n".join(history_lines)

    prompt = f"""{SYSTEM_PROMPT}

<context>
{context_str}
</context>
"""
    if history_str:
        prompt += f"\n<conversation_history>\n{history_str}\n</conversation_history>\n"

    prompt += f"\nUser question: {query}\n\nAnswer:"
    return prompt


def generate_response(
    query: str,
    context_chunks: List[dict],
    chat_history: Optional[List[dict]] = None,
) -> str:
    """
    Generate a grounded answer via Gemini.

    Parameters
    ----------
    query:
        Current user question.
    context_chunks:
        Relevant chunks retrieved from the vector store.
    chat_history:
        Previous conversation messages for multi-turn context.

    Returns
    -------
    Model-generated response string.
    """
    model = _get_model()
    prompt = build_rag_prompt(query, context_chunks, chat_history)

    logger.debug("Sending prompt to Gemini (%d chars).", len(prompt))
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.debug("Gemini response (%d chars).", len(answer))
        return answer
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise RuntimeError(f"LLM generation failed: {exc}") from exc
