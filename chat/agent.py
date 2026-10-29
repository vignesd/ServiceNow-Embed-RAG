from __future__ import annotations

from enum import Enum
from threading import Lock
from typing import Iterator

import httpx

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.search import search_documents
from config import LLM_MODEL
from logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HISTORY_MAX_MESSAGES = 10
RETRIEVAL_K = 5


class RetrievalMode(str, Enum):
    """
    Retrieval strategy used for a user question.

    FOCUSED:
        Normal top-K semantic retrieval.

    PROCEDURAL:
        Semantic retrieval followed by complete guide expansion.
    """

    FOCUSED = "focused"
    PROCEDURAL = "procedural"


PROCEDURAL_PATTERNS = (
    "how do i",
    "how to",
    "how can i",
    "steps to",
    "steps for",
    "procedure",
    "process for",
    "guide me",
    "walk me through",
    "full procedure",
    "complete procedure",
    "full requirements",
    "complete requirements",
    "what information do i need",
    "what details do i need",
    "what fields do i need",
)


def classify_retrieval_mode(query: str) -> RetrievalMode:
    """
    Classify whether the retrieval query needs complete guide coverage
    or normal focused semantic retrieval.

    Unknown queries intentionally default to focused retrieval so that
    full-guide expansion is used only when there is a clear procedural
    intent.
    """
    normalized_query = " ".join(
        query.casefold().strip().split()
    )

    if any(
        pattern in normalized_query
        for pattern in PROCEDURAL_PATTERNS
    ):
        return RetrievalMode.PROCEDURAL

    return RetrievalMode.FOCUSED


llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.2,
    top_p=0.95,
)

# llm = ChatOpenAI(
#     model=LLM_MODEL,
#     temperature=0.2,
#     top_p=0.95,
# )


# ---------------------------------------------------------------------
# In-memory, per-session conversation history
# ---------------------------------------------------------------------

# Each session_id maps to its own sequence of HumanMessage / AIMessage.
# This is in-memory storage: restarting the application clears all history.

_conversation_store: dict[str, list[BaseMessage]] = {}
_history_lock = Lock()


def get_conversation_history(
    session_id: str,
) -> list[BaseMessage]:
    """Return a copy of the stored history for one chat session."""
    with _history_lock:
        return list(
            _conversation_store.get(session_id, [])
        )


def clear_conversation_history(
    session_id: str,
) -> None:
    """Clear the history for one chat session."""
    with _history_lock:
        _conversation_store.pop(session_id, None)

    logger.info(
        "Conversation history cleared for session_id=%s",
        session_id,
    )


def _save_turn(
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """
    Save a completed user/assistant turn and retain
    the last 10 messages.
    """
    with _history_lock:
        history = _conversation_store.setdefault(
            session_id,
            [],
        )

        history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )

        # Ten messages means approximately five complete
        # user/assistant turns.
        _conversation_store[session_id] = (
            history[-HISTORY_MAX_MESSAGES:]
        )


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

# Use history to turn follow-up questions into standalone retrieval queries.
# This is a query-rewriting step, not the final answer-generation step.

RETRIEVAL_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the latest user question as a standalone search query
for a ServiceNow IT Service Management (ITSM) catalog guide.

Use the conversation history only to resolve references such as "it",
"that request", "what about approvals", or "how long does it take".

Keep the user's intent unchanged.

If the latest user question asks for a procedure, process, steps,
requirements, or how to perform/request something, preserve that
procedural intent in the rewritten query.

Do not answer the question, invent facts, or add details that the
conversation does not support.

Return only the rewritten search query.""",
        ),
        MessagesPlaceholder(
            variable_name="history"
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


retrieval_query_chain = (
    RETRIEVAL_QUERY_PROMPT
    | llm
    | StrOutputParser()
)


# Preserve ServiceNow RAG grounding rules while distinguishing
# focused questions from complete procedural questions.

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert ITSM Helpdesk RAG Assistant.

Answer questions using only facts explicitly supported by the retrieved
knowledge-base context.

Match the scope of the answer to the user's question.

For focused questions asking for one specific fact, such as:
- an SLA,
- approval requirement,
- mandatory status,
- system/environment option,
- escalation contact,
- URL,
- or an individual form field,

answer that specific question directly using the relevant retrieved
evidence.

Include closely related qualifications only when they are necessary
for accuracy.

Do not reproduce the full catalog procedure for a focused question.


For procedural or "how do I raise/request" questions:

1. Identify the catalog item and provide its direct URL when it exists
   in the retrieved context.

2. Include EVERY field explicitly marked Mandatory: Yes.
   Never omit a mandatory field in favor of an optional field.

3. For each mandatory field, state whether it is:
   - auto-populated and should be verified, or
   - manually entered/selected.

4. Cover all mandatory fields first.

   Mention optional fields concisely when they are useful for completing
   the request.

   Do not reproduce every metadata attribute of every field unless the
   user explicitly asks for a detailed field-by-field guide.

5. Preserve explicitly stated validation rules and restrictions that
   could prevent successful submission.

6. If the retrieved context specifies available dropdown choices,
   list only those choices.

7. State approval requirements exactly as described by the retrieved
   context.

8. State SLA wording faithfully, including trigger conditions such as
   "after approval".

   Do not convert a conditional SLA into an unconditional SLA.

9. Include available submission options if provided by the retrieved
   context.

10. Include escalation/contact information when available and relevant.

11. If the retrieved context explicitly contains a security warning,
    prominently include it.

    Do not invent security policies or warnings that are absent from
    retrieved evidence.

12. Do not repeat the same instruction in multiple sections.

13. If information needed to answer the question is missing from the
    retrieved context, say which information is unavailable rather
    than guessing.


CRITICAL GROUNDING RULE:

1. Every URL, email address, contact, approver, form field,
   submission action, SLA, security rule, and procedural step
   in the response MUST appear explicitly in the retrieved context.

2. If a requested detail is absent from the retrieved context,
   state that it is not available in the retrieved knowledge.

3. Never generate placeholder URLs, example URLs, generic helpdesk
   addresses, assumed approvers, or typical ServiceNow actions.


Before finalizing a procedural answer, internally verify:

- All mandatory fields included?
- URL given if available?
- Required manual-entry fields clearly identified?
- Validation rules retained?
- Approval included?
- SLA condition preserved?
- Restrictions included?
- Any factual statement unsupported by retrieved context?


Retrieved knowledge-base context:

{context}""",
        ),
        MessagesPlaceholder(
            variable_name="history"
        ),
        (
            "human",
            "User Question: {question}\n\nAnswer:",
        ),
    ]
)


answer_chain = (
    RAG_PROMPT
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------------------
# Streaming RAG response
# ---------------------------------------------------------------------

def get_rag_response(
    vector_store: Chroma,
    question: str,
    session_id: str,
) -> Iterator[str]:
    """
    Retrieve ServiceNow guide chunks and stream a response using Ollama.

    Args:
        vector_store:
            Initialized Chroma vector store.

        question:
            Latest user question.

        session_id:
            Unique ID for the current chat session.

    Yields:
        Response text in streamed chunks.

    Notes:
        - History is isolated by session_id.
        - Only the latest 10 messages are retained.
        - Follow-up questions are rewritten into standalone
          retrieval queries.
        - Procedural questions expand the matched guide.
        - Focused questions use normal top-K retrieval.
        - A turn is saved after a successful, complete response.
        - The caller must reuse the same session_id for
          follow-up questions.
    """

    if not session_id or not session_id.strip():
        raise ValueError(
            "session_id must be a non-empty string."
        )

    question = question.strip()

    if not question:
        yield "Please enter a question."
        return

    logger.info(
        "Generating streaming RAG response for session_id=%s",
        session_id,
    )

    # Snapshot history so this request uses a consistent
    # conversation state.
    history = get_conversation_history(
        session_id
    )

    # -----------------------------------------------------------------
    # 1. Rewrite follow-up questions for better retrieval.
    # -----------------------------------------------------------------

    retrieval_query = question

    if history:
        try:
            rewritten_query = (
                retrieval_query_chain.invoke(
                    {
                        "history": history,
                        "question": question,
                    }
                ).strip()
            )

            if rewritten_query:
                retrieval_query = rewritten_query

            logger.info(
                "Rewrote retrieval query "
                "session_id=%s original=%r rewritten=%r",
                session_id,
                question,
                retrieval_query,
            )

        except Exception:
            # Query rewriting is an enhancement.
            # Do not block retrieval if it fails.
            logger.exception(
                "Follow-up query rewriting failed; "
                "using original question."
            )

            retrieval_query = question

    # -----------------------------------------------------------------
    # 2. Determine retrieval strategy.
    # -----------------------------------------------------------------

    retrieval_mode = classify_retrieval_mode(
        retrieval_query
    )

    expand_guide = (
        retrieval_mode
        == RetrievalMode.PROCEDURAL
    )

    logger.info(
        "RAG_RETRIEVAL_ROUTE "
        "session_id=%s mode=%s expand_guide=%s "
        "original_question=%r retrieval_query=%r",
        session_id,
        retrieval_mode.value,
        expand_guide,
        question,
        retrieval_query,
    )

    # -----------------------------------------------------------------
    # 3. Retrieve relevant guide chunks from ChromaDB.
    # -----------------------------------------------------------------

    try:
        search_results = search_documents(
            vector_store=vector_store,
            question=retrieval_query,
            k=RETRIEVAL_K,
            expand_guide=expand_guide,
        )

    except Exception:
        logger.exception(
            "Failed to query vector database."
        )

        yield (
            "[Error: Could not retrieve relevant chunks "
            "from ChromaDB.]"
        )

        return

    logger.info(
        "RAG_RETRIEVAL_COMPLETE "
        "session_id=%s mode=%s "
        "expand_guide=%s results=%d",
        session_id,
        retrieval_mode.value,
        expand_guide,
        len(search_results),
    )

    if not search_results:
        answer = (
            "I couldn't find any relevant guides or documentation "
            "matching your request in the ServiceNow knowledge base."
        )

        _save_turn(
            session_id,
            question,
            answer,
        )

        yield answer
        return

    # -----------------------------------------------------------------
    # 4. Format retrieved chunks into the RAG context.
    # -----------------------------------------------------------------

    context_blocks: list[str] = []

    for result in search_results:
        content = result.get(
            "content",
            "",
        )

        if not content:
            continue

        metadata = result.get(
            "metadata",
            {},
        )

        section_path = metadata.get(
            "section_path",
            "",
        )

        chunk_index = metadata.get(
            "chunk_index",
        )

        guide_title = metadata.get(
            "guide_title",
            "",
        )

        header_parts: list[str] = []

        if guide_title:
            header_parts.append(
                f"Guide: {guide_title}"
            )

        if section_path:
            header_parts.append(
                f"Section: {section_path}"
            )

        if chunk_index is not None:
            header_parts.append(
                f"Chunk: {chunk_index}"
            )

        if header_parts:
            context_blocks.append(
                "--- Retrieved guide chunk ---\n"
                + "\n".join(header_parts)
                + "\n\n"
                + content
            )
        else:
            context_blocks.append(
                "--- Retrieved guide chunk ---\n"
                + content
            )

    formatted_context = "\n\n".join(
        context_blocks
    )

    if not formatted_context.strip():
        answer = (
            "I couldn't find any usable guide content for this "
            "question in the ServiceNow knowledge base."
        )

        _save_turn(
            session_id,
            question,
            answer,
        )

        yield answer
        return

    logger.info(
        "RAG_CONTEXT_READY "
        "session_id=%s mode=%s chunks=%d chars=%d",
        session_id,
        retrieval_mode.value,
        len(context_blocks),
        len(formatted_context),
    )

    # -----------------------------------------------------------------
    # 5. Stream answer using retrieved context + conversation history.
    # -----------------------------------------------------------------

    answer_parts: list[str] = []

    try:
        stream = answer_chain.stream(
            {
                "context": formatted_context,
                "history": history,
                "question": question,
            }
        )

        for chunk in stream:
            if chunk:
                answer_parts.append(chunk)
                yield chunk

        # Store history only after streaming finishes successfully.
        complete_answer = "".join(
            answer_parts
        ).strip()

        _save_turn(
            session_id,
            question,
            complete_answer,
        )

        print("\n")

        logger.info(
            "RAG response completed "
            "session_id=%s mode=%s",
            session_id,
            retrieval_mode.value,
        )

    except (
        ConnectionError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
    ):
        logger.exception(
            "Connection to Ollama failed."
        )

        yield (
            "\n\n[Connection Error: Could not connect to Ollama. "
            "Please ensure Ollama is running and reachable "
            "on port 11434.]"
        )

    except Exception as exc:
        logger.exception(
            "Unexpected error during LLM stream generation."
        )

        yield (
            "\n\n[Error: Failed to generate response -> "
            f"{exc}]"
        )
