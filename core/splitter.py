
import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader
from pathlib import Path

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    MARKDOWN_HEADERS,
    DOCUMENT_DIR,
    SUPPORTED_FILE_EXTENSIONS,
)
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


def prepare_guide_chunks(
    chunks: list[Document],
    file_path: Path,
) -> list[Document]:
    """Add guide identity, section context, and retrieval metadata."""

    guide_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        file_path.stem.lower(),
    ).strip("_")

    guide_title = (
        file_path.stem
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )

    prepared: list[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        original_text = chunk.page_content.strip()

        if not original_text:
            continue

        # Construct a section label from the Markdown heading metadata.
        heading_parts = [
            str(chunk.metadata[key]).strip()
            for key in ("h1", "h2", "h3")
            if chunk.metadata.get(key)
        ]

        section_path = " > ".join(heading_parts)

        # Preserve existing metadata and add guide-specific metadata.
        metadata = {
            **chunk.metadata,
            "guide_id": guide_id,
            "guide_title": guide_title,
            "section_path": section_path,
            "source": str(file_path),
            "file_type": "md",
            "chunk_index": chunk_index,
        }

        # Provide context to the embedding model.
        embedding_parts = [
            f"Guide: {guide_title}",
        ]

        if section_path:
            embedding_parts.append(
                f"Section: {section_path}"
            )

        embedding_parts.append(original_text)

        embedding_text = "\n\n".join(embedding_parts)

        prepared.append(
            Document(
                page_content=embedding_text,
                metadata=metadata,
            )
        )

    # logger.info(f"{len(prepared)} - Guided chunks completed.")

    return prepared


def load_markdown(file_path: Path) -> list[Document]:
    """Load Markdown, split by headings, then split oversized sections."""

    text = file_path.read_text(encoding="utf-8-sig")

    if not text.strip():
        return []

    sections = markdown_splitter.split_text(text)

    if not sections:
        sections = [
            Document(
                page_content=text,
                metadata={},
            )
        ]

    markdown_chunks: list[Document] = []

    for section in sections:
        section.metadata.update({
            "source": str(file_path),
            "file_type": "md",
        })

        section_text = section.page_content.strip()

        if not section_text:
            continue

        # Keep a complete section whenever it fits.
        if len(section_text) <= recursive_splitter._chunk_size:
            markdown_chunks.append(section)
            continue

        # Split only oversized sections.
        split_chunks = recursive_splitter.split_documents(
            [section]
        )

        markdown_chunks.extend(split_chunks)

    # logger.info(f"{len(markdown_chunks)} - Markdown normal chunks completed.")

    guided_chunks = prepare_guide_chunks(markdown_chunks, file_path)

    return guided_chunks


def load_text(file_path: Path) -> list[Document]:
    """Load TXT files without a community loader."""

    text = file_path.read_text(encoding="utf-8-sig")

    if not text.strip():
        return []

    document = Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "file_type": "txt",
        },
    )
    text_chunks = recursive_splitter.split_documents([document])
    logger.info(f"{len(text_chunks)} - Text chunks completed.")

    return text_chunks


def load_pdf(file_path: Path) -> list[Document]:
    """Extract PDF pages directly using pypdf."""

    reader = PdfReader(str(file_path))
    pdf_chunks = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "pdf",
                "page": page_number,
            },
        )

        pdf_chunks.extend(
            recursive_splitter.split_documents([document])
        )

    logger.info(f"{len(pdf_chunks)} - PDF chunks completed.")

    return pdf_chunks


def chunk_documents() -> list[Document]:
    """Load supported files and return all text chunks."""

    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: "
            f"{DOCUMENT_DIR.resolve()}"
        )

    all_chunks = []
    logger.info(f"Support Files  - {SUPPORTED_FILE_EXTENSIONS}")
    supported_files = [
        path
        for path in DOCUMENT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
    ]

    if not supported_files:
        raise FileNotFoundError(
            f"No Markdown, TXT, or PDF files found in "
            f"{DOCUMENT_DIR.resolve()}"
        )

    for file_path in supported_files:
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".md":
                chunks = load_markdown(file_path)
            elif suffix == ".txt":
                chunks = load_text(file_path)
            else:
                chunks = load_pdf(file_path)

            chunks = [
                chunk for chunk in chunks
                if chunk.page_content.strip()
            ]

            all_chunks.extend(chunks)
            logger.info(
                "Loaded %s: %d chunks",
                file_path,
                len(chunks),
            )

        except Exception:
            logger.exception(
                "Failed to process %s",
                file_path,
            )

    if not all_chunks:
        raise ValueError(
            "No usable text chunks were extracted. "
            "Check the files and PDF text extraction."
        )

    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks
