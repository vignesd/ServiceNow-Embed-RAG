
import uuid

from core.embedding import (
    build_vector_store,
    get_vector_store,
)
from core.splitter import chunk_documents
from chat.agent import (
    generate_rag_response_stream,
    clear_conversation_history,
)
from logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def main() -> None:
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
                clear_conversation_history(session_id)
                session_id = str(uuid.uuid4())
                logger.info("Started new conversation session: %s", session_id)
                print("Started a new conversation.")
                continue

            print("\nAssistant: ", end="", flush=True)

            for token in generate_rag_response_stream(
                vector_store=vector_store,
                question=question,
                session_id=session_id,
            ):
                print(token, end="", flush=True)

            print()

    except KeyboardInterrupt:
        print("\nConversation interrupted.")

    finally:
        # Remove this session's in-memory history when exiting.
        clear_conversation_history(session_id)
        logger.info("Conversation ended.")


if __name__ == "__main__":
    main()
