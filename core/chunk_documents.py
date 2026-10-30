import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, MARKDOWN_HEADERS
from core.load_documents import load_all_documents
from logger import get_logger

logger = get_logger(__name__)

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=MARKDOWN_HEADERS,
    strip_headers=False,
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    length_function=len,
)


def prepare_guide_chunks(chunks: list[Document], file_path: Path) -> list[Document]:
    """Add guide identity, section context, and metadata enrichment."""
    guide_id = re.sub(r"[^a-z0-9]+", "_", file_path.stem.lower()).strip("_")
    guide_title = file_path.stem.replace("_", " ").replace("-", " ").title()

    prepared: list[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        original_text = chunk.page_content.strip()
        if not original_text:
            continue

        heading_parts = [
            str(chunk.metadata[key]).strip()
            for key in ("h1", "h2", "h3")
            if chunk.metadata.get(key)
        ]
        section_path = " > ".join(heading_parts)

        metadata = {
            **chunk.metadata,
            "guide_id": guide_id,
            "guide_title": guide_title,
            "section_path": section_path,
            "source": str(file_path),
            "chunk_index": chunk_index,
        }

        # Provide richer context for the embedding model
        embedding_parts = [f"Guide: {guide_title}"]
        if section_path:
            embedding_parts.append(f"Section: {section_path}")
        embedding_parts.append(original_text)

        embedding_text = "\n\n".join(embedding_parts)

        prepared.append(
            Document(page_content=embedding_text, metadata=metadata)
        )

    return prepared


def chunk_documents() -> list[Document]:
    """Load raw documents, split them by headings/size, and attach metadata."""
    raw_documents = load_all_documents()
    all_chunks = []

    for doc in raw_documents:
        file_path = Path(doc.metadata["source"])
        file_type = doc.metadata.get("file_type")

        try:
            if file_type == "md":
                sections = markdown_splitter.split_text(doc.page_content)
                if not sections:
                    sections = [
                        Document(page_content=doc.page_content, metadata={})]

                markdown_chunks = []
                for section in sections:
                    section.metadata.update(doc.metadata)
                    section_text = section.page_content.strip()
                    if not section_text:
                        continue

                    if len(section_text) <= recursive_splitter._chunk_size:
                        markdown_chunks.append(section)
                    else:
                        markdown_chunks.extend(
                            recursive_splitter.split_documents([section]))

                chunks = prepare_guide_chunks(markdown_chunks, file_path)
            else:
                # TXT and PDFs
                chunks = recursive_splitter.split_documents([doc])
                # Ensure metadata format consistency
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i

            chunks = [c for c in chunks if c.page_content.strip()]
            all_chunks.extend(chunks)
            logger.debug(f"Chunked {file_path.name}: {len(chunks)} chunks")

        except Exception:
            logger.exception(f"Failed to chunk document {file_path}")

    if not all_chunks:
        raise ValueError(
            "No usable text chunks were extracted from documents.")

    logger.info(f"Total generated chunks: {len(all_chunks)}")
    return all_chunks
