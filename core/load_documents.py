from pathlib import Path
from langchain_core.documents import Document
from pypdf import PdfReader

from config import DOCUMENT_DIR, SUPPORTED_FILE_EXTENSIONS
from logger import get_logger

logger = get_logger(__name__)


def load_markdown(file_path: Path) -> list[Document]:
    """Load raw Markdown file content."""
    text = file_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "md",
            },
        )
    ]


def load_text(file_path: Path) -> list[Document]:
    """Load raw TXT file content."""
    text = file_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "txt",
            },
        )
    ]


def load_pdf(file_path: Path) -> list[Document]:
    """Extract raw pages from a PDF file."""
    reader = PdfReader(str(file_path))
    pdf_docs = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        pdf_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_type": "pdf",
                    "page": page_number,
                },
            )
        )

    return pdf_docs


def load_all_documents() -> list[Document]:
    """Scan DOCUMENT_DIR and load all supported raw documents."""
    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENT_DIR.resolve()}")

    raw_documents = []
    supported_files = [
        path for path in DOCUMENT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
    ]

    if not supported_files:
        raise FileNotFoundError(
            f"No supported files found in {DOCUMENT_DIR.resolve()}")

    for file_path in supported_files:
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".md":
                docs = load_markdown(file_path)
            elif suffix == ".txt":
                docs = load_text(file_path)
            else:
                docs = load_pdf(file_path)

            raw_documents.extend(docs)
            logger.debug("Loaded raw file: %s (%d items/pages)",
                         file_path.name, len(docs))

        except Exception:
            logger.exception("Failed to load raw file %s", file_path)

    logger.info(f"Total loaded documents: {len(raw_documents)}")

    return raw_documents
