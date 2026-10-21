from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    EMBEDDING_BATCH_SIZE,
)
from core.embeddings import GuideEmbeddings
from logger import get_logger

logger = get_logger(__name__)


def create_chunk_id(document: Document) -> str:
    """Create a deterministic ID based on source and chunk identity/content."""
    identity = (
        f"{document.metadata.get('source', '')}|"
        f"{document.metadata.get('page', '')}|"
        f"{document.metadata.get('chunk_index', '')}|"
        f"{document.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: Path) -> str:
    """Compute a SHA-256 hash of a file's contents."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_vector_store() -> Chroma:
    """Initialize and return the Chroma vector store instance."""
    embeddings = GuideEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _add_batch_with_diagnostics(
    vector_store: Chroma,
    chunks: list[Document],
    ids: list[str],
    start_index: int,
) -> None:
    """Index one batch and log execution status."""
    batch_number = start_index // EMBEDDING_BATCH_SIZE + 1
    logger.info(f"CHROMA_BATCH_START batch={batch_number}")
    started = time.perf_counter()

    try:
        vector_store.add_documents(documents=chunks, ids=ids)
        logger.info(f"CHROMA_BATCH_SUCCESS batch={batch_number}")
    except Exception:
        logger.exception(
            "CHROMA_BATCH_FAILURE batch=%d start_index=%d count=%d elapsed_seconds=%.2f",
            batch_number,
            start_index,
            len(chunks),
            time.perf_counter() - started,
        )
        raise


def build_vector_store(
    chunks: list[Document],
    force_rebuild: bool = False,
) -> Chroma:
    """Incrementally synchronize guide chunks with ChromaDB using file hashing."""
    logger.info(
        "INDEX_START model=%s collection=%s persist_directory=%s input_chunks=%d force_rebuild=%s",
        EMBEDDING_MODEL,
        COLLECTION_NAME,
        CHROMA_DIR,
        len(chunks),
        force_rebuild,
    )

    vector_store = get_vector_store()
    existing = vector_store.get(include=["metadatas"])
    existing_ids = existing.get("ids") or []
    existing_metadatas = existing.get("metadatas") or []

    logger.info("CHROMA_EXISTING chunks=%d", len(existing_ids))

    if force_rebuild:
        logger.warning(
            "FORCE_REBUILD requested; deleting existing collection records.")
        if existing_ids:
            vector_store.delete(ids=existing_ids)
        existing_ids = []
        existing_metadatas = []

    current_files: dict[str, str] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source and source not in current_files:
            source_path = Path(source)
            if source_path.is_file():
                current_files[source] = compute_file_hash(source_path)

    indexed_sources_map: dict[str, list[str]] = defaultdict(list)
    indexed_file_hashes: dict[str, str] = {}

    for chunk_id, metadata in zip(existing_ids, existing_metadatas):
        if not metadata:
            continue
        source = metadata.get("source")
        file_hash = metadata.get("file_hash")
        if source:
            indexed_sources_map[source].append(chunk_id)
            if file_hash:
                indexed_file_hashes[source] = file_hash

    sources_on_disk = set(current_files)
    sources_indexed = set(indexed_sources_map)

    sources_to_remove = sources_indexed - sources_on_disk
    sources_to_add_or_update: set[str] = set()

    for source in sources_on_disk:
        if source not in sources_indexed:
            sources_to_add_or_update.add(source)
        elif indexed_file_hashes.get(source) != current_files[source]:
            sources_to_add_or_update.add(source)

    if sources_to_remove and not force_rebuild:
        ids_to_delete: list[str] = []
        for source in sources_to_remove:
            ids_to_delete.extend(indexed_sources_map[source])
        if ids_to_delete:
            logger.info("SYNC_DELETE deleted_sources=%d chunks=%d",
                        len(sources_to_remove), len(ids_to_delete))
            vector_store.delete(ids=ids_to_delete)

    if sources_to_add_or_update and not force_rebuild:
        ids_to_update: list[str] = []
        for source in sources_to_add_or_update:
            ids_to_update.extend(indexed_sources_map.get(source, []))
        if ids_to_update:
            logger.info("SYNC_REPLACE modified_sources=%d old_chunks=%d", len(
                sources_to_add_or_update), len(ids_to_update))
            vector_store.delete(ids=ids_to_update)

    if force_rebuild:
        chunks_to_process = list(chunks)
    else:
        chunks_to_process = [
            chunk for chunk in chunks
            if chunk.metadata.get("source") in sources_to_add_or_update
        ]

    if not chunks_to_process and not force_rebuild and not sources_to_remove:
        logger.info("INDEX_SKIP index is up to date; no changed documents.")
        return vector_store

    for chunk in chunks_to_process:
        source = chunk.metadata.get("source")
        if source in current_files:
            chunk.metadata["file_hash"] = current_files[source]

    unique_chunks: list[Document] = []
    unique_ids: list[str] = []
    seen: set[str] = set()

    for chunk in chunks_to_process:
        chunk_id = create_chunk_id(chunk)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique_chunks.append(chunk)
        unique_ids.append(chunk_id)

    total = len(unique_chunks)
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch_chunks = unique_chunks[start: start + EMBEDDING_BATCH_SIZE]
        batch_ids = unique_ids[start: start + EMBEDDING_BATCH_SIZE]
        _add_batch_with_diagnostics(
            vector_store=vector_store,
            chunks=batch_chunks,
            ids=batch_ids,
            start_index=start,
        )

    logger.info("INDEX_SUCCESS indexed_chunks=%d collection=%s",
                total, COLLECTION_NAME)
    return vector_store
