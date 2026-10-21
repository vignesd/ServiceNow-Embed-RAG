from pathlib import Path
from dotenv import load_dotenv
import os
from logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Configure logging

logger = get_logger(__name__)

# --- Directory & Storage Paths ---
DOCUMENT_DIR = Path(os.getenv("DOCUMENT_DIR", "documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))

# --- Vector Store & Embedding Settings ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ai_knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma-2")

LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")

# --- Chunking Parameters ---
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2400))
except ValueError:
    CHUNK_SIZE = 2400
    logger.warning(
        "Invalid CHUNK_SIZE in environment variables. Falling back to default: 800")

try:
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 250))
except ValueError:
    CHUNK_OVERLAP = 250
    logger.warning(
        "Invalid CHUNK_OVERLAP in environment variables. Falling back to default: 120")

try:
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 16))
except ValueError:
    EMBEDDING_BATCH_SIZE = 16
    logger.warning(
        "Invalid EMBEDDING_BATCH_SIZE in environment variables. Falling back to default: 16")

try:
    OLLAMA_BASE_URL = os.getenv("LLM_MODEL", "http://127.0.0.1:11434")
except ValueError:
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    logger.warning(
        "Invalid OLLAMA_BASE_URL in environment variables. Falling back to default: http://127.0.0.1:11434")


# --- Text Splitting / Parsing Configurations ---
SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

SUPPORTED_FILE_EXTENSIONS = {".md", ".txt", ".pdf"}

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# --- Startup Configuration Logs ---
logger.info(f"Document Path - {DOCUMENT_DIR.resolve()}")
logger.info(f"ChromaDB Path - {CHROMA_DIR.resolve()}")
logger.info(f"Collection Name - {COLLECTION_NAME}")
logger.info(f"Embedding Model - {EMBEDDING_MODEL}")
logger.info(f"Chunking Config - Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
