from __future__ import annotations

from threading import Lock
from typing import Iterator

import httpx
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import (
    LLM_MODEL,
    HISTORY_MAX_MESSAGES,
    RETRIEVAL_K
)
from core.search import search_documents
from chat.schema import (
    DETAILED_RESPONSE_PATTERNS,
    PROCEDURAL_PATTERNS,
    ResponseDetail,
    RetrievalMode,
)
from chat.prompts import (
    RAG_PROMPT,
    RETRIEVAL_QUERY_PROMPT,
)
from logger import get_logger

logger = get_logger(__name__)

_conversation_store: dict[str, list[BaseMessage]] = {}
_history_lock = Lock()


def _normalize_query(query: str) -> str:
    """Normalize a query for deterministic phrase matching."""
    return " ".join(query.casefold().strip().split())


def classify_retrieval_mode(query: str) -> RetrievalMode:
    """
    Choose complete-guide retrieval for procedural questions.

    Unknown queries default to focused top-K retrieval.
    """
    normalized_query = _normalize_query(query)

    if any(pattern in normalized_query for pattern in PROCEDURAL_PATTERNS):
        return RetrievalMode.PROCEDURAL

    return RetrievalMode.FOCUSED


def classify_response_detail(question: str) -> ResponseDetail:
    """
    Choose detailed output only when the user explicitly requests it.

    Ordinary how-to questions default to a concise operational answer even
    though the complete guide is retrieved as evidence.
    """
    normalized_question = _normalize_query(question)

    if any(
        pattern in normalized_question
        for pattern in DETAILED_RESPONSE_PATTERNS
    ):
        return ResponseDetail.DETAILED

    return ResponseDetail.CONCISE


def get_conversation_history(session_id: str) -> list[BaseMessage]:
    """Return a copy of the stored history for one chat session."""
    with _history_lock:
        return list(_conversation_store.get(session_id, []))


def clear_conversation_history(session_id: str) -> None:
    """Clear the history for one chat session."""
    with _history_lock:
        _conversation_store.pop(session_id, None)

    logger.info("Conversation history cleared for session_id=%s", session_id)


def _save_turn(
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """Save a completed user/assistant turn and retain the last 10 messages."""
    with _history_lock:
        history = _conversation_store.setdefault(session_id, [])
        history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )

        # Ten messages means approximately five complete user/assistant turns.
        _conversation_store[session_id] = history[-HISTORY_MAX_MESSAGES:]


def get_rag_response(
    vector_store: Chroma,
    question: str,
    session_id: str,
    model_name: str,
) -> Iterator[str]:
    """
    Retrieve ServiceNow guide chunks and stream a response using Ollama.

    Procedural questions expand the matched guide. Focused questions use
    normal top-K retrieval. Detailed presentation is used only when requested.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")

    question = question.strip()
    if not question:
        yield "Please enter a question."
        return

    logger.info(
        "Generating streaming RAG response for session_id=%s",
        session_id,
    )

    history = get_conversation_history(session_id)

    retrieval_query = question
    if model_name.__contains__("gpt"):
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            top_p=0.95,
        )
    else:
        llm = ChatOllama(
            model=model_name,
            temperature=0.2,
            top_p=0.95,
        )

    if history:
        try:
            retrieval_query_chain = RETRIEVAL_QUERY_PROMPT | llm | StrOutputParser()
            rewritten_query = retrieval_query_chain.invoke(
                {
                    "history": history,
                    "question": question,
                }
            ).strip()

            if rewritten_query:
                retrieval_query = rewritten_query

            logger.info(
                "Rewrote retrieval query session_id=%s original=%r rewritten=%r",
                session_id,
                question,
                retrieval_query,
            )
        except Exception:
            # Do not block the request if query rewriting fails.
            logger.exception(
                "Follow-up query rewriting failed; using original question."
            )
            retrieval_query = question

    retrieval_mode = classify_retrieval_mode(retrieval_query)
    response_detail = classify_response_detail(question)
    expand_guide = retrieval_mode == RetrievalMode.PROCEDURAL

    logger.info(
        "RAG_ROUTE session_id=%s retrieval_mode=%s response_detail=%s "
        "expand_guide=%s original_question=%r retrieval_query=%r",
        session_id,
        retrieval_mode.value,
        response_detail.value,
        expand_guide,
        question,
        retrieval_query,
    )

    try:
        search_results = search_documents(
            vector_store=vector_store,
            question=retrieval_query,
            k=RETRIEVAL_K,
            expand_guide=expand_guide,
        )
    except Exception:
        logger.exception("Failed to query vector database.")
        yield "[Error: Could not retrieve relevant chunks from ChromaDB.]"
        return

    logger.info(
        "RAG_RETRIEVAL_COMPLETE session_id=%s retrieval_mode=%s "
        "expand_guide=%s results=%d",
        session_id,
        retrieval_mode.value,
        expand_guide,
        len(search_results),
    )

    if not search_results:
        answer = (
            "I couldn't find any relevant guides or documentation matching "
            "your request in the ServiceNow knowledge base."
        )
        _save_turn(session_id, question, answer)
        yield answer
        return

    context_blocks: list[str] = []

    for result in search_results:
        content = result.get("content", "")
        if not content:
            continue

        metadata = result.get("metadata", {})
        guide_title = metadata.get("guide_title", "")
        section_path = metadata.get("section_path", "")
        chunk_index = metadata.get("chunk_index")

        header_parts: list[str] = []
        if guide_title:
            header_parts.append(f"Guide: {guide_title}")
        if section_path:
            header_parts.append(f"Section: {section_path}")
        if chunk_index is not None:
            header_parts.append(f"Chunk: {chunk_index}")

        header = "\n".join(header_parts)
        if header:
            context_blocks.append(
                f"--- Retrieved guide chunk ---\n{header}\n\n{content}"
            )
        else:
            context_blocks.append(
                f"--- Retrieved guide chunk ---\n{content}"
            )

    formatted_context = "\n\n".join(context_blocks)

    if not formatted_context.strip():
        answer = (
            "I couldn't find any usable guide content for this question "
            "in the ServiceNow knowledge base."
        )
        _save_turn(session_id, question, answer)
        yield answer
        return

    logger.info(
        "RAG_CONTEXT_READY session_id=%s retrieval_mode=%s "
        "response_detail=%s chunks=%d chars=%d",
        session_id,
        retrieval_mode.value,
        response_detail.value,
        len(context_blocks),
        len(formatted_context),
    )

    answer_parts: list[str] = []

    try:
        answer_chain = RAG_PROMPT | llm | StrOutputParser()
        stream = answer_chain.stream(
            {
                "context": formatted_context,
                "history": history,
                "question": question,
                "response_detail": response_detail.value,
            }
        )

        for chunk in stream:
            if chunk:
                answer_parts.append(chunk)
                yield chunk

        # Store only after the stream finishes normally.
        complete_answer = "".join(answer_parts).strip()
        _save_turn(session_id, question, complete_answer)
        print("\n")

        logger.info(
            "RAG response completed session_id=%s retrieval_mode=%s "
            "response_detail=%s",
            session_id,
            retrieval_mode.value,
            response_detail.value,
        )
    except (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout):
        logger.exception("Connection to Ollama failed.")
        yield (
            "\n\n[Connection Error: Could not connect to Ollama. "
            "Please ensure Ollama is running and reachable on port 11434.]"
        )
    except Exception as exc:
        logger.exception("Unexpected error during LLM stream generation.")
        yield f"\n\n[Error: Failed to generate response -> {exc}]"
