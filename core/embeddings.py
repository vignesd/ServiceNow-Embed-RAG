from __future__ import annotations

import time
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL, OLLAMA_BASE_URL
from logger import get_logger

logger = get_logger(__name__)


class GuideEmbeddings(Embeddings):
    """Ollama embeddings wrapper with diagnostic logging around embedding calls."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model_name = model
        self.client = OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )
        logger.info(
            "Initialized Ollama embeddings: model=%s, base_url=%s",
            model,
            base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.info(
            f"EMBED_START operation=embed_documents model={self.model_name}")
        started = time.perf_counter()

        try:
            vectors = self.client.embed_documents(texts)
            logger.info(
                f"EMBED_SUCCESS operation=embed_documents model={self.model_name}")
            return vectors
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(
                "EMBED_FAILURE operation=embed_documents model=%s count=%d elapsed_seconds=%.2f",
                self.model_name,
                len(texts),
                elapsed,
            )
            raise

    def embed_query(self, text: str) -> list[float]:
        logger.info(
            f"EMBED_START operation=embed_query model={self.model_name}")
        started = time.perf_counter()

        try:
            vector = self.client.embed_query(text)
            logger.info(
                f"EMBED_SUCCESS operation=embed_query model={self.model_name}")
            return vector
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(
                "EMBED_FAILURE operation=embed_query model=%s chars=%d elapsed_seconds=%.2f",
                self.model_name,
                len(text),
                elapsed,
            )
            raise
