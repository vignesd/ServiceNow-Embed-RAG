from __future__ import annotations

from typing import List, TypedDict, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

from core.chunk_documents import chunk_documents
from core.vector_stores import (
    build_vector_store,
    get_vector_store,
)
from logger import get_logger

logger = get_logger(__name__)


# Define the state schema for our LangGraph pipeline
class RAGPipelineState(TypedDict):
    force_rebuild: bool
    raw_chunks: Optional[List[Document]]
    vector_store: Optional[Chroma]
    status: str
    error: Optional[str]


def node_load_and_chunk(state: RAGPipelineState) -> RAGPipelineState:
    """Node 1: Load and chunk documents from the document directory."""
    logger.info("GRAPH_NODE: Starting document loading and chunking...")
    try:
        chunks = chunk_documents()
        logger.info(
            f"GRAPH_NODE: Successfully generated {len(chunks)} chunks.")
        return {
            **state,
            "raw_chunks": chunks,
            "status": "chunked",
        }
    except Exception as e:
        logger.exception("GRAPH_NODE: Error during loading and chunking.")
        return {
            **state,
            "status": "error",
            "error": str(e),
        }


def node_sync_vector_store(state: RAGPipelineState) -> RAGPipelineState:
    """Node 2: Incrementally sync chunks with ChromaDB vector store."""
    logger.info("GRAPH_NODE: Starting vector store synchronization...")
    try:
        chunks = state.get("raw_chunks", [])
        force_rebuild = state.get("force_rebuild", False)

        vector_store = build_vector_store(
            chunks, force_rebuild=force_rebuild)  # type:ignore
        logger.info("GRAPH_NODE: Vector store sync completed successfully.")
        return {
            **state,
            "vector_store": vector_store,
            "status": "synchronized",
        }
    except Exception as e:
        logger.exception(
            "GRAPH_NODE: Error during vector store synchronization.")
        return {
            **state,
            "status": "error",
            "error": str(e),
        }


def check_status(state: RAGPipelineState) -> str:
    """Conditional edge router based on execution status."""
    if state.get("status") == "error":
        return "error"
    return "continue"


def create_rag_graph() -> StateGraph:
    """Builds and compiles the LangGraph state machine for initialization & indexing."""
    workflow = StateGraph(RAGPipelineState)

    # Add graph nodes
    workflow.add_node("load_and_chunk", node_load_and_chunk)
    workflow.add_node("sync_vector_store", node_sync_vector_store)

    # Define edges and conditional routing
    workflow.set_entry_point("load_and_chunk")

    workflow.add_conditional_edges(
        "load_and_chunk",
        check_status,
        {
            "continue": "sync_vector_store",
            "error": END,
        },
    )

    workflow.add_edge("sync_vector_store", END)

    return workflow.compile()  # type:ignore


def initialize_rag_system(force_rebuild: bool = False) -> Optional[Chroma]:
    """
    Executes the compiled LangGraph pipeline and returns the active vector store.
    """
    app = create_rag_graph()

    initial_state: RAGPipelineState = {
        "force_rebuild": force_rebuild,
        "raw_chunks": None,
        "vector_store": None,
        "status": "started",
        "error": None,
    }

    logger.info("Invoking LangGraph RAG Initialization Pipeline...")
    final_state = app.invoke(initial_state)  # type:ignore

    if final_state.get("status") == "error":
        logger.error(
            f"RAG Pipeline failed with error: {final_state.get('error')}")
        return None

    return final_state.get("vector_store")
