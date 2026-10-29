
from core.rag_pipeline import initialize_rag_system

from core.vector_stores import (
    get_vector_store,
    build_vector_store,
)
from core.chunk_documents import chunk_documents
from chat.agent import (
    get_rag_response,
    clear_conversation_history,  # Keep if your agent supports history tracking
)
from logger import setup_logging, get_logger
import uuid

setup_logging()
logger = get_logger(__name__)


def chain_rag() -> None:
    # Ask whether to rebuild the vector database.
    rebuild_choice = input(
        "Do you want to force a full re-index of all documents? (Y/N): "
    ).strip().lower()
    force_rebuild = rebuild_choice in {"y", "yes"}

    logger.info("Loading and chunking documents...")
    chunks = chunk_documents()

    logger.info("Syncing vector database...")
    build_vector_store(chunks, force_rebuild=force_rebuild)
    logger.info("RAG indexing and sync completed.")

    # Create ONE session ID for this terminal conversation.
    session_id = str(uuid.uuid4())
    logger.info("Started conversation session: %s", session_id)

    # Load the vector store once before the conversation loop.
    vector_store = get_vector_store()

    try:
        while True:
            question = input(
                "\nAsk a question (or type 'new', 'exit'): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            # Optionally start a fresh conversation.
            if question.lower() == "new":
                if "clear_conversation_history" in globals():
                    clear_conversation_history(session_id)
                session_id = str(uuid.uuid4())
                logger.info("Started new conversation session: %s", session_id)
                print("Started a new conversation.")
                continue

            print("\nAssistant: ", end="", flush=True)

            # Pass parameters depending on your core/agent.py signature
            for token in get_rag_response(
                vector_store=vector_store,
                question=question,
                session_id=session_id,  # Uncomment if your agent uses session memory
            ):
                print(token, end="", flush=True)

            print()

    except KeyboardInterrupt:
        print("\nConversation interrupted.")

    finally:
        # Remove this session's in-memory history when exiting (if supported).
        if "clear_conversation_history" in globals():
            clear_conversation_history(session_id)
        logger.info("Conversation ended.")


def graph_rag() -> None:
    # Ask whether to rebuild the vector database.
    rebuild_choice = input(
        "Do you want to force a full re-index of all documents? (Y/N): "
    ).strip().lower()
    force_rebuild = rebuild_choice in {"y", "yes"}

    # Execute the LangGraph pipeline
    vector_store = initialize_rag_system(force_rebuild=force_rebuild)

    if not vector_store:
        print("\n[Error: Failed to initialize the RAG pipeline. Check logs.]")
        return

    logger.info("RAG pipeline initialization completed via LangGraph.")

    # Create ONE session ID for this terminal conversation.
    session_id = str(uuid.uuid4())
    logger.info("Started conversation session: %s", session_id)

    try:
        while True:
            question = input(
                "\nAsk a question (or type 'new', 'exit'): "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            if question.lower() == "new":
                session_id = str(uuid.uuid4())
                logger.info("Started new conversation session: %s", session_id)
                print("Started a new conversation.")
                continue

            print("\nAssistant: ", end="", flush=True)

            for token in get_rag_response(
                vector_store=vector_store,
                question=question,
                session_id=session_id,
            ):
                print(token, end="", flush=True)

            print()

    except KeyboardInterrupt:
        print("\nConversation interrupted.")

    finally:
        logger.info("Conversation ended.")


if __name__ == "__main__":
    type = int(input("Enter your option to run type (1 for chain,2 for graph)"))
    if type == 1:
        chain_rag()
    else:
        graph_rag()
