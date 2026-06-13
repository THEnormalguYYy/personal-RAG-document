"""personal-document-rag :: backend engine package.

This package bundles the three pillars of the layout-aware Retrieval-Augmented
Generation (RAG) backend and re-exports their primary public callables so that
the Streamlit frontend (``app.py``) can import them directly from ``src`` with
a flat, stable interface:

    from src import (
        parse_document_with_unstructured,
        get_chroma_index,
        get_chat_engine,
    )

Modules
-------
ingest
    Layout-aware visual chunking of source documents via the Unstructured
    framework; maps tables to raw HTML strings and produces LlamaIndex
    ``Document`` objects enriched with structural metadata.
database
    Persistent ChromaDB collection management, index load/build routing, and
    on-disk persistence using a LlamaIndex ``StorageContext``.
engine
    Conversational brain implementing multi-query expansion, hybrid
    (vector + BM25) retrieval, FlashRank cross-encoder re-ranking, and a
    ``CondenseQuestionChatEngine`` for multi-turn dialogue.
"""

from __future__ import annotations

from .database import get_chroma_index
from .engine import get_chat_engine
from .ingest import parse_document_with_unstructured

__version__: str = "1.0.0"

__author__: str = "personal-document-rag"

__all__: list[str] = [
    "parse_document_with_unstructured",
    "get_chroma_index",
    "get_chat_engine",
    "__version__",
]
