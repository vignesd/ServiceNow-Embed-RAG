# ServiceNow-Embed-RAG

A command-line Retrieval-Augmented Generation (RAG) assistant for answering questions using ServiceNow IT service management guides. It indexes Markdown, text, and PDF files from a local document folder, stores their embeddings in ChromaDB, and retrieves relevant guide content when generating answers.

Ollama supplies embeddings and can also run the chat model. The chat code additionally supports OpenAI models when an OpenAI model is selected.

## How it works

1. Documents in `documents/` are loaded and split into chunks. Markdown headings and guide metadata are retained to provide useful context during retrieval.
2. The chunks are synchronized with a persistent ChromaDB collection in `chroma_db/`. By default, unchanged files are skipped, changed files are re-indexed, and removed files are deleted from the collection.
3. Questions are matched against the indexed content. The assistant streams an answer grounded in the retrieved guides and retains a short conversation history for follow-up questions.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally for Ollama chat or embedding models
- An OpenAI API key if you choose an OpenAI chat model such as `gpt-4o-mini`

## Installation

Clone the repository and install its dependencies:

```shell
git clone https://github.com/vignesd/ServiceNow-Embed-RAG.git
cd ServiceNow-Embed-RAG
uv sync
```

Create a `.env` file by copying `.env.example` (for example, `Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux), then adjust the values as needed.

Pull the Ollama models specified by your configuration. For the models listed in `.env.example`, run:

```shell
ollama pull nomic-embed-text
ollama pull gemma2:9b
```

The application prompts you to select a chat model when it starts. The menu offers `llama3.2L3b`, `qwen2.5-coder:7b-instruct`, and `gpt-4o-mini`. Pull any Ollama model you intend to use before selecting it. To use `gpt-4o-mini`, set `OPENAI_API_KEY` in your environment.

## Configuration

The checked-in `.env.example` values are shown below. The application also has fallback values in `config.py` for settings that are not set:

| Variable | Purpose | Example value |
| --- | --- | --- |
| `LLM_MODEL` | Fallback chat model if model selection is unavailable | `gpt-4o-mini` |
| `DOCUMENT_DIR` | Directory scanned for source documents | `documents` |
| `CHROMA_DIR` | Persistent ChromaDB storage directory | `chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `ai_knowledge` |
| `EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `CHUNK_SIZE` | Maximum chunk size in characters | `2400` |
| `CHUNK_OVERLAP` | Overlap between text chunks | `250` |
| `EMBEDDING_BATCH_SIZE` | Number of chunks indexed per batch | `16` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://127.0.0.1:11434` |
| `ENABLE_CONSOLE_LOGGING` | Enable console logs | `false` |
| `ENABLE_FILE_LOGGING` | Enable rotating logs in `logs/app.log` | `true` |

Additional settings recognized by the application include `HISTORY_MAX_MESSAGES` (fallback `10`) and `RETRIEVAL_K` (fallback `5`).

## Run

Start the interactive assistant:

```shell
uv run python main.py
```

At startup, choose whether to force a full re-index, then select a chat model. Ask questions at the prompt:

```text
Ask a question (or type 'new', 'exit'):
```

Enter `new` to start a fresh conversation, or `exit`/`quit` to end the session. Answers are streamed in the terminal.

## Add or update knowledge

Place `.md`, `.txt`, or text-based `.pdf` guides in the configured document directory (by default, `documents/`). The next run detects added, modified, and deleted files and synchronizes their chunks. Choose the full re-index option at startup to rebuild the collection from all current documents.

## Classifier tests

Run the retrieval and response classifier tests with:

```shell
uv run python -m pytest testing/test_classifier.py
```

## Project layout

```text
ServiceNow-Embed-RAG/
├── chat/                    # Chat agent, prompts, schemas, and RAG pipeline
│   ├── __init__.py
│   ├── agent.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   └── schema.py
├── core/                    # Document processing, embeddings, retrieval, and vector store
│   ├── __init__.py
│   ├── chunk_documents.py
│   ├── embeddings.py
│   ├── load_documents.py
│   ├── search.py
│   └── vector_stores.py
├── documents/               # Source ServiceNow guides (.md, .txt, and .pdf)
├── miscellaneous/           # Older experimental scripts
├── prompts/                 # Prompt text files
├── testing/                 # Tests
│   ├── test_classifier.py
│   ├── test_embedding.py
│   └── test_ollama_embeddings.py
├── chroma_db/               # Persistent vector database (created/updated at runtime)
├── logs/                    # Application logs (created when file logging is enabled)
├── .env.example             # Example environment configuration
├── .gitignore
├── .python-version
├── config.py                # Environment-based configuration
├── logger.py                # Console and rotating-file logging
├── main.py                  # Interactive command-line entry point
├── pyproject.toml           # Project metadata and dependencies
├── README.md
├── run_test.py              # Classifier test runner
└── uv.lock                  # Locked dependency versions
```
