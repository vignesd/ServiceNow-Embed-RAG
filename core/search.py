from typing import Any, Dict, List, Optional
from langchain_chroma import Chroma
from logger import get_logger

logger = get_logger(__name__)


def search_documents(
    vector_store: Chroma,
    question: str,
    guide_id: Optional[str] = None,
    k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Find the most relevant chunks for a question from the Chroma vector store.

    Args:
        vector_store: The Chroma vector store instance.
        question: The search query string.
        guide_id: Optional filter string for a specific guide.
        k: Number of top results to retrieve.

    Returns:
        List of dictionaries containing document content, metadata, and distance score.
    """
    search_filter = {"guide_id": guide_id} if guide_id else None

    logger.info(f"Executing similarity search for query: '{question}' (k={k})")

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
        filter=search_filter
    )

    if not results:
        logger.info("No matching documents found.")
        return []

    formatted_results = []

    print(f"\nQuestion: {question}")
    print("=" * 50)

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        guide_title = metadata.get("guide_title", "N/A")
        source = metadata.get("source", "N/A")

        # logger.info(f"Result {rank}")
        # logger.info(f"Guide: {guide_title}")
        # logger.info(f"Source: {source}")
        # logger.info(f"Distance: {distance:.4f}")

        # Print structured details to console
        # print(f"\n--- Result {rank} ---")
        # print(f"Guide: {guide_title}")
        # print(f"Source: {source}")

        # if "page" in metadata:
        #     print(f"PDF Page: {metadata['page'] + 1}")

        # for key in ("h1", "h2", "h3"):
        #     if metadata.get(key):
        #         print(f"{key.upper()}: {metadata[key]}")

        # print(f"Content:\n{document.page_content}")
        # print(f"Distance: {round(distance, 4)}")

        # Append structured data for programmatic usage
        formatted_results.append({
            "rank": rank,
            "content": document.page_content,
            "metadata": metadata,
            "distance": float(distance)
        })

    # print("=" * 50)
    return formatted_results
