from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import COLLECTION_NAME, EMBEDDING_MODEL, CHROMA_DIR
from logger import get_logger

import hashlib
from pathlib import Path

logger = get_logger(__name__)


class GuideEmbeddings(Embeddings):
    """EmbeddingGemma embeddings for independent guide retrieval."""

    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://localhost:11434",
    ):
        self.model = OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(
            f"task: search result | query: {text}"
        )


def create_chunk_id(document: Document) -> str:
    """Create a deterministic ID based on source and content."""
    identity = (
        f"{document.metadata.get('source', '')}|"
        f"{document.metadata.get('page', '')}|"
        f"{document.metadata.get('start_index', '')}|"
        f"{document.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: Path) -> str:
    """Compute a fast sha256 hash of a file's contents."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_vector_store() -> Chroma:
    embeddings = GuideEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def build_vector_store(chunks: list[Document], force_rebuild: bool = False) -> Chroma:
    """
    Incrementally sync chunks to ChromaDB. 
    If force_rebuild=True, clears the entire collection and re-indexes everything.
    """

    logger.info(f"Embedding Model - {EMBEDDING_MODEL}.")
    vector_store = get_vector_store()
    existing = vector_store.get()

    existing_ids = existing.get("ids", [])
    existing_metadatas = existing.get("metadatas", [])

    logger.info(f"Found {len(existing_ids)} existing chunks in ChromaDB.")

    if force_rebuild:
        logger.info(
            "Force rebuild requested. Clearing all existing vector store data...")
        if existing_ids:
            vector_store.delete(ids=existing_ids)
        existing_ids = []
        existing_metadatas = []

    # Map current files on disk and compute their hashes
    current_files = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if source and source not in current_files:
            current_files[source] = compute_file_hash(Path(source))

    # Determine which sources already exist in metadata
    indexed_sources_map = {}  # source -> set of chunk ids
    indexed_file_hashes = {}  # source -> file_hash stored in metadata

    for chunk_id, meta in zip(existing_ids, existing_metadatas):
        if not meta:
            continue
        source = meta.get("source")
        f_hash = meta.get("file_hash")
        if source:
            indexed_sources_map.setdefault(source, []).append(chunk_id)
            if f_hash:
                indexed_file_hashes[source] = f_hash

    # Identify files to add, update, or remove
    sources_on_disk = set(current_files.keys())
    sources_indexed = set(indexed_sources_map.keys())

    sources_to_remove = sources_indexed - sources_on_disk
    sources_to_add_or_update = set()

    for source in sources_on_disk:
        if source not in sources_indexed:
            sources_to_add_or_update.add(source)  # New file
        elif indexed_file_hashes.get(source) != current_files[source]:
            sources_to_add_or_update.add(source)  # Modified file

    # 1. Handle Deletions
    if sources_to_remove and not force_rebuild:
        ids_to_delete = []
        for src in sources_to_remove:
            ids_to_delete.extend(indexed_sources_map[src])
        if ids_to_delete:
            logger.info(
                f"Removing {len(ids_to_delete)} stale chunks from deleted files...")
            vector_store.delete(ids=ids_to_delete)

    # 2. Handle Updates (Delete old chunks for modified files first)
    if sources_to_add_or_update and not force_rebuild:
        ids_to_update = []
        for src in sources_to_add_or_update:
            if src in indexed_sources_map:
                ids_to_update.extend(indexed_sources_map[src])
        if ids_to_update:
            logger.info(
                f"Clearing old chunks for {len(sources_to_add_or_update)} modified/updated files...")
            vector_store.delete(ids=ids_to_update)

    # Filter chunks to only those needing indexing
    if not force_rebuild and sources_to_remove | sources_to_add_or_update:
        chunks_to_process = [
            c for c in chunks if c.metadata.get("source") in sources_to_add_or_update
        ]
    elif force_rebuild:
        chunks_to_process = chunks
    else:
        # Check if any new files were added
        chunks_to_process = [
            c for c in chunks if c.metadata.get("source") in sources_to_add_or_update
        ]

    # If no files changed and not forcing, we can skip adding
    if not chunks_to_process and not force_rebuild and not sources_to_remove:
        logger.info(
            "No new or modified documents detected. Index is up to date!")
        return vector_store

    # Attach file hash to chunk metadata for future comparisons
    for chunk in chunks_to_process:
        source = chunk.metadata.get("source")
        if source in current_files:
            chunk.metadata["file_hash"] = current_files[source]

    if chunks_to_process:
        ids = [create_chunk_id(chunk) for chunk in chunks_to_process]

        # Deduplicate chunk IDs
        seen = set()
        unique_chunks = []
        unique_ids = []
        for chunk, chunk_id in zip(chunks_to_process, ids):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique_chunks.append(chunk)
            unique_ids.append(chunk_id)

        vector_store.add_documents(documents=unique_chunks, ids=unique_ids)
        logger.info("Indexed %d new/updated chunks in ChromaDB.",
                    len(unique_chunks))
    else:
        logger.info("Vector store synchronization completed.")

    return vector_store
