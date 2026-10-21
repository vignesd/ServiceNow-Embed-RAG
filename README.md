# ServiceNow-Embed-RAG 🚀

A lightweight, local Retrieval-Augmented Generation (RAG) assistant designed for ServiceNow IT Service Management (ITSM). Built specifically for **16GB RAM machines**, it runs entirely locally using **Ollama**, **ChromaDB**, and **LangChain** with fast Python dependency management via **`uv`**.

---

## 🏗️ Project Architecture & Modules

```
ServiceNow-Embed-RAG
├── core
│   ├── __init__.py
│   ├── agent.py         # LCEL streaming agent using ChatOllama
│   ├── embeddings.py    # Incremental vector store sync & file hashing
│   ├── search.py        # Semantic similarity retrieval engine
│   └── splitter.py      # Markdown, TXT, and PDF chunkers with metadata injection
├── documents            # Local IT service request guides (Markdown/PDF/TXT)
├── config.py            # Centralized project configuration (.env driven)
├── logger.py            # Centralized logging with toggleable file/console output
├── main.py              # CLI entrypoint and interactive loop
├── pyproject.toml       # Project metadata & dependencies
└── .env.example         # Environment variable template

```

### Core Components:

- **`config.py`**: Centralized configuration reading directly from environment variables.
- **`logger.py`**: Customizable logging setup with console output and optional rotating file logging.
- **`core/splitter.py`**: Parses and structures service request guides (splitting Markdown by headers, handling TXT and PDFs with custom metadata context).
- **`core/embeddings.py`**: Features an **incremental sync engine** that computes file hashes to add only new/modified files and prune deleted ones without full re-indexing (plus an optional forced re-build flag).
- **`core/search.py`**: Performs semantic search across the ChromaDB vector database.
- **`core/agent.py`**: Leverages LangChain Expression Language (LCEL) and ChatOllama to stream accurate answers based strictly on retrieved guidelines.

---

## 🛠️ Tech Stack & Local Models

- **Language/Runtime**: Python 3.10+
- **Dependency Manager**: [`uv`](https://github.com/astral-sh/uv)
- **Orchestration**: LangChain & LangChain Community / Chroma
- **Vector Store**: ChromaDB (Persisted locally in `chroma_db/`)
- **Local LLM**: `qwen2.5-coder:7b-instruct` (Optimized for strict instruction following, formatting, and 16GB RAM constraints)
- **Embeddings**: `nomic-embed-text` (Lightweight, efficient, 8k context embedding model)

---

## 📋 Prerequisites & Installation

### 1. Install Ollama & Models

Ensure you have [Ollama](https://ollama.com/) installed and running on your machine. Then, pull the required local models:

```bash
# Pull the LLM for reasoning and response generation
ollama pull qwen2.5-coder:7b-instruct

# Pull the embedding model for semantic search
ollama pull nomic-embed-text

```

### 2. Clone and Setup Project using `uv`

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/vignesd/ServiceNow-Embed-RAG.git
cd ServiceNow-Embed-RAG

```

Create and configure your environment file:

```bash
cp .env.example .env

```

Sync dependencies using **`uv`**:

```bash
uv sync

```

---

## ⚙️ Configuration (`.env`)

Configure your environment variables in the `.env` file:

```env
LLM_MODEL=qwen2.5-coder:7b-instruct

DOCUMENT_DIR=documents
CHROMA_DIR=chroma_db
COLLECTION_NAME=ai_knowledge
EMBEDDING_MODEL=nomic-embed-text

CHUNK_SIZE=2400
CHUNK_OVERLAP=250

ENABLE_FILE_LOGGING=true

```

---

## 🚀 Running the Assistant

Start the application using `uv`:

```bash
uv run python main.py

```

### What happens on startup:

1. The app prompts if you want to force a full re-index (`y/N`) or perform a fast incremental sync (checking for new or modified markdown/PDF guides).
2. It loads documentation chunks and updates ChromaDB.
3. Once ready, you can type your IT service desk questions interactively in the terminal!

```text
Ask a question (or type 'exit'): password reset

Assistant: To raise a password reset request in ServiceNow, please follow these guidelines...

```
