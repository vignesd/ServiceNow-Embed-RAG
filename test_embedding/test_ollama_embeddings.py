"""Standalone diagnostic for Ollama embeddings; does not touch ChromaDB."""

import logging
import time

from langchain_ollama import OllamaEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

MODEL = "nomic-embed-text"
BASE_URL = "http://127.0.0.1:11434"

def main() -> None:
    client = OllamaEmbeddings(model=MODEL, base_url=BASE_URL)

    tests = [
        ("single_query", lambda: client.embed_query("ServiceNow catalog guide")),
        (
            "three_documents",
            lambda: client.embed_documents(
                [
                    "How to request software access.",
                    "How to configure a catalog item.",
                    "How to troubleshoot access issues.",
                ]
            ),
        ),
    ]

    for name, operation in tests:
        started = time.perf_counter()
        logging.info("TEST_START name=%s", name)
        try:
            result = operation()
            if name == "single_query":
                dimensions = len(result)
                count = 1
            else:
                count = len(result)
                dimensions = len(result[0]) if result else 0
            logging.info(
                "TEST_SUCCESS name=%s count=%d dimensions=%d elapsed_seconds=%.2f",
                name,
                count,
                dimensions,
                time.perf_counter() - started,
            )
        except Exception:
            logging.exception(
                "TEST_FAILURE name=%s elapsed_seconds=%.2f",
                name,
                time.perf_counter() - started,
            )
            raise

if __name__ == "__main__":
    main()
