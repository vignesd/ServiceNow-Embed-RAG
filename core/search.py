from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from logger import get_logger


logger = get_logger(__name__)


SearchResult = Dict[str, Any]
RawSearchResult = Tuple[Document, float]


def _format_similarity_results(
    results: List[RawSearchResult],
) -> List[SearchResult]:
    """
    Convert Chroma similarity-search results into the structure expected
    by the RAG pipeline.
    """
    formatted_results: List[SearchResult] = []

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata or {}

        logger.info(
            "RAG_RESULT rank=%d distance=%.4f "
            "guide=%s guide_id=%s section=%s chunk=%s source=%s",
            rank,
            distance,
            metadata.get("guide_title", "N/A"),
            metadata.get("guide_id", "N/A"),
            metadata.get("section_path", "N/A"),
            metadata.get("chunk_index", "N/A"),
            metadata.get("source", "N/A"),
        )

        logger.debug(
            "RAG_CONTENT rank=%d content=%s",
            rank,
            document.page_content,
        )

        formatted_results.append(
            {
                "rank": rank,
                "content": document.page_content,
                "metadata": metadata,
                "distance": float(distance),
                "retrieval_type": "similarity",
            }
        )

    return formatted_results


def _get_expansion_guide_id(
    results: List[RawSearchResult],
    explicit_guide_id: Optional[str],
) -> Optional[str]:
    """
    Determine which guide should be expanded.

    An explicitly supplied guide_id takes priority. Otherwise, the guide_id
    from the highest-ranked similarity result is used.
    """
    if explicit_guide_id:
        return explicit_guide_id

    if not results:
        return None

    top_document = results[0][0]
    return top_document.metadata.get("guide_id")


def _get_initial_distance_map(
    results: List[RawSearchResult],
) -> Dict[Tuple[str, Any], float]:
    """
    Preserve similarity distances for chunks that appeared in the initial
    semantic search.

    The map key uses source + chunk_index because those values identify a
    chunk within the current metadata structure.
    """
    distance_map: Dict[Tuple[str, Any], float] = {}

    for document, distance in results:
        metadata = document.metadata or {}

        key = (
            str(metadata.get("source", "")),
            metadata.get("chunk_index"),
        )

        distance_map[key] = float(distance)

    return distance_map


def _expand_guide(
    vector_store: Chroma,
    guide_id: str,
    initial_results: List[RawSearchResult],
    max_guide_chunks: Optional[int] = None,
) -> List[SearchResult]:
    """
    Retrieve all indexed chunks belonging to one guide and return them in
    original document order.

    Similarity distance is preserved for chunks that appeared in the initial
    semantic search. Expanded-only chunks have distance=None because they
    were retrieved by metadata, not by vector similarity.
    """
    logger.info(
        "Expanding guide context: guide_id=%s max_guide_chunks=%s",
        guide_id,
        max_guide_chunks,
    )

    collection_results = vector_store.get(
        where={"guide_id": guide_id},
        include=["documents", "metadatas"],
    )

    documents = collection_results.get("documents") or []
    metadatas = collection_results.get("metadatas") or []

    if not documents:
        logger.warning(
            "Guide expansion returned no chunks: guide_id=%s",
            guide_id,
        )
        return _format_similarity_results(initial_results)

    distance_map = _get_initial_distance_map(initial_results)

    expanded_chunks: List[Tuple[str, Dict[str, Any]]] = []

    for content, metadata in zip(documents, metadatas):
        if not content or not str(content).strip():
            continue

        expanded_chunks.append(
            (
                str(content),
                metadata or {},
            )
        )

    expanded_chunks.sort(
        key=lambda item: (
            _safe_chunk_index(item[1].get("chunk_index")),
            str(item[1].get("section_path", "")),
        )
    )

    if max_guide_chunks is not None:
        max_guide_chunks = max(1, max_guide_chunks)
        expanded_chunks = expanded_chunks[:max_guide_chunks]

    formatted_results: List[SearchResult] = []

    for rank, (content, metadata) in enumerate(
        expanded_chunks,
        start=1,
    ):
        key = (
            str(metadata.get("source", "")),
            metadata.get("chunk_index"),
        )

        initial_distance = distance_map.get(key)

        logger.info(
            "RAG_EXPANDED_RESULT rank=%d initial_distance=%s "
            "guide=%s guide_id=%s section=%s chunk=%s source=%s",
            rank,
            (
                f"{initial_distance:.4f}"
                if initial_distance is not None
                else "N/A"
            ),
            metadata.get("guide_title", "N/A"),
            metadata.get("guide_id", "N/A"),
            metadata.get("section_path", "N/A"),
            metadata.get("chunk_index", "N/A"),
            metadata.get("source", "N/A"),
        )

        logger.debug(
            "RAG_EXPANDED_CONTENT rank=%d content=%s",
            rank,
            content,
        )

        formatted_results.append(
            {
                "rank": rank,
                "content": content,
                "metadata": metadata,

                # None means the chunk was loaded through guide metadata
                # expansion rather than returned by similarity search.
                "distance": initial_distance,

                "retrieval_type": (
                    "similarity"
                    if initial_distance is not None
                    else "guide_expansion"
                ),
            }
        )

    logger.info(
        "Guide expansion completed: guide_id=%s chunks=%d",
        guide_id,
        len(formatted_results),
    )

    return formatted_results


def _safe_chunk_index(value: Any) -> int:
    """
    Produce a sortable integer chunk index.

    Missing or invalid indexes are placed at the end rather than causing
    guide expansion to fail.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2_147_483_647


def search_documents(
    vector_store: Chroma,
    question: str,
    guide_id: Optional[str] = None,
    k: int = 5,
    expand_guide: bool = False,
    max_guide_chunks: Optional[int] = None,
) -> List[SearchResult]:
    """
    Search the Chroma vector store.

    Modes
    -----
    Focused retrieval:
        expand_guide=False

        Returns the top-K semantically similar chunks.

    Guide expansion:
        expand_guide=True

        Performs the same semantic search to identify the most relevant
        guide, then loads that guide's chunks and orders them by chunk_index.

    Args:
        vector_store:
            Chroma vector store instance.

        question:
            User's search query.

        guide_id:
            Optional guide filter. When supplied, the initial similarity
            search is restricted to that guide.

        k:
            Number of semantic candidates retrieved during the initial
            similarity search.

        expand_guide:
            Whether to expand the highest-ranked guide after semantic search.

        max_guide_chunks:
            Optional safety limit for guide expansion. None retrieves every
            chunk in the matched guide.

    Returns:
        A list containing content, metadata, rank, distance, and retrieval
        type.
    """
    cleaned_question = question.strip()

    logger.info(
        "Executing similarity search: query=%r k=%d "
        "guide_id=%s expand_guide=%s",
        cleaned_question,
        k,
        guide_id,
        expand_guide,
    )

    if not cleaned_question:
        logger.warning("Search skipped because the question was empty.")
        return []

    if k < 1:
        raise ValueError("k must be greater than or equal to 1.")

    search_filter = (
        {"guide_id": guide_id}
        if guide_id
        else None
    )

    logger.info(
        "Executing similarity search: query=%r k=%d "
        "guide_id=%s expand_guide=%s",
        cleaned_question,
        k,
        guide_id,
        expand_guide,
    )

    initial_results: List[RawSearchResult] = (
        vector_store.similarity_search_with_score(
            cleaned_question,
            k=k,
            filter=search_filter,
        )
    )

    if not initial_results:
        logger.info(
            "No matching documents found for query=%r",
            cleaned_question,
        )
        return []

    if not expand_guide:
        return _format_similarity_results(initial_results)

    matched_guide_id = _get_expansion_guide_id(
        results=initial_results,
        explicit_guide_id=guide_id,
    )

    if not matched_guide_id:
        logger.warning(
            "Guide expansion was requested, but the highest-ranked "
            "result has no guide_id. Returning similarity results."
        )
        return _format_similarity_results(initial_results)

    return _expand_guide(
        vector_store=vector_store,
        guide_id=matched_guide_id,
        initial_results=initial_results,
        max_guide_chunks=max_guide_chunks,
    )
