"""personal-document-rag :: Streamlit frontend.

A production-grade dashboard for a layout-aware Retrieval-Augmented Generation
system. It provides:

* A sidebar document uploader that ingests PDFs through the Unstructured-backed
  pipeline, with database-level de-duplication to avoid redundant embedding
  spend.
* A streaming conversational chat tab with per-answer citation expanders that
  surface raw source snippets and the exact retrieved HTML tables.
* A system & pipeline analytics tab summarizing processing metrics, chunk
  boundaries, and database content.

State is managed through ``st.session_state`` for conversational continuity and
``st.cache_resource`` for the heavyweight vector index handle.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

import streamlit as st
from dotenv import load_dotenv

from src.database import (
    file_already_indexed,
    get_chroma_index,
    get_collection_size,
    get_indexed_file_names,
    reset_index_cache,
)
from src.engine import get_chat_engine
from src.ingest import parse_document_with_unstructured

# ---------------------------------------------------------------------------
# Environment & logging bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

logger: logging.Logger = logging.getLogger("personal_document_rag.app")
if not logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ---------------------------------------------------------------------------
# Page configuration & global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS: str = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .app-hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 100%);
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .app-hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .app-hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.6rem;
        padding: 1rem 1.2rem;
    }
    .source-snippet {
        background-color: #f1f5f9;
        border-left: 3px solid #6d28d9;
        border-radius: 0.4rem;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ok { background-color: #dcfce7; color: #166534; }
    .status-warn { background-color: #fef9c3; color: #854d0e; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Initialize all session-state keys exactly once per browser session.

    Returns
    -------
    None
    """
    defaults: Dict[str, Any] = {
        "messages": [],
        "chat_engine": None,
        "index_ready": False,
        "indexed_files": set(),
        "processed_signatures": set(),
        "analytics": {
            "documents_processed": 0,
            "total_chunks_created": 0,
            "text_chunks": 0,
            "table_chunks": 0,
            "last_ingest_seconds": 0.0,
            "per_file": [],
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ---------------------------------------------------------------------------
# Cached backend resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_index() -> Any:
    """Load (or initialize) the persistent vector index a single time per process.

    Returns
    -------
    Any
        The live ``VectorStoreIndex`` bound to the persistent Chroma collection.
    """
    logger.info("Loading vector index into cache_resource.")
    return get_chroma_index(documents=None)


def ensure_chat_engine(force_rebuild: bool = False) -> Optional[Any]:
    """Ensure a chat engine exists in session state, building it if needed.

    Parameters
    ----------
    force_rebuild:
        When ``True`` the cached index is cleared and the engine is rebuilt so
        newly ingested nodes are reflected in the BM25 corpus.

    Returns
    -------
    Any | None
        The conversational chat engine, or ``None`` when no documents have been
        indexed yet.
    """
    if get_collection_size() == 0:
        st.session_state["chat_engine"] = None
        st.session_state["index_ready"] = False
        return None

    if force_rebuild:
        st.session_state["chat_engine"] = None
        load_vector_index.clear()

    if st.session_state["chat_engine"] is None:
        with st.spinner("Initializing the conversational engine…"):
            index = load_vector_index()
            st.session_state["chat_engine"] = get_chat_engine(index)
            st.session_state["index_ready"] = True
            st.session_state["indexed_files"] = get_indexed_file_names()
        logger.info("Chat engine initialized and cached in session state.")

    return st.session_state["chat_engine"]


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------
def _persist_upload_to_temp(uploaded_file: Any) -> str:
    """Write an uploaded file to a temporary path for partitioning.

    Parameters
    ----------
    uploaded_file:
        A Streamlit ``UploadedFile`` instance.

    Returns
    -------
    str
        The absolute path to the written temporary file.
    """
    suffix: str = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path: str = tmp.name
    logger.debug("Persisted upload '%s' to temp path %s.", uploaded_file.name, temp_path)
    return temp_path


def _signature(uploaded_file: Any) -> str:
    """Compute a lightweight session signature for an uploaded file.

    Parameters
    ----------
    uploaded_file:
        A Streamlit ``UploadedFile`` instance.

    Returns
    -------
    str
        A ``name:size`` signature used to avoid reprocessing on Streamlit reruns.
    """
    return f"{uploaded_file.name}:{uploaded_file.size}"


def process_uploaded_files(uploaded_files: List[Any]) -> None:
    """Ingest newly uploaded files, skipping any already indexed in the database.

    For each file this performs database-level de-duplication (via
    :func:`file_already_indexed`) before partitioning, then persists the produced
    documents into the Chroma collection and updates session analytics.

    Parameters
    ----------
    uploaded_files:
        The list of files returned by the sidebar ``st.file_uploader`` widget.

    Returns
    -------
    None
    """
    if not uploaded_files:
        return

    newly_indexed: int = 0

    for uploaded_file in uploaded_files:
        signature: str = _signature(uploaded_file)
        if signature in st.session_state["processed_signatures"]:
            continue

        file_name: str = uploaded_file.name

        # Authoritative de-duplication gate: never re-embed an existing file.
        if file_already_indexed(file_name):
            st.session_state["processed_signatures"].add(signature)
            st.toast(f"'{file_name}' is already indexed — skipping.", icon="✅")
            logger.info("Skipping already-indexed file: %s", file_name)
            continue

        temp_path: str = ""
        try:
            with st.spinner(f"Parsing '{file_name}' with layout-aware ingestion…"):
                temp_path = _persist_upload_to_temp(uploaded_file)
                start_time: float = time.perf_counter()
                documents = parse_document_with_unstructured(temp_path)
                # Restore the true file name (temp path basename differs).
                for document in documents:
                    document.metadata["file_name"] = file_name

            if not documents:
                st.warning(f"No extractable content found in '{file_name}'.")
                st.session_state["processed_signatures"].add(signature)
                continue

            with st.spinner(f"Embedding & persisting '{file_name}'…"):
                get_chroma_index(documents=documents)
                elapsed: float = time.perf_counter() - start_time

            table_chunks: int = sum(
                1 for doc in documents if doc.metadata.get("element_type") == "table"
            )
            text_chunks: int = len(documents) - table_chunks

            analytics: Dict[str, Any] = st.session_state["analytics"]
            analytics["documents_processed"] += 1
            analytics["total_chunks_created"] += len(documents)
            analytics["text_chunks"] += text_chunks
            analytics["table_chunks"] += table_chunks
            analytics["last_ingest_seconds"] = elapsed
            analytics["per_file"].append(
                {
                    "file_name": file_name,
                    "chunks": len(documents),
                    "text_chunks": text_chunks,
                    "table_chunks": table_chunks,
                    "seconds": round(elapsed, 2),
                }
            )

            st.session_state["processed_signatures"].add(signature)
            newly_indexed += 1
            st.toast(
                f"Indexed '{file_name}' ({len(documents)} chunks).", icon="📚"
            )
            logger.info(
                "Indexed '%s': %d chunks (%d text, %d table) in %.2fs.",
                file_name,
                len(documents),
                text_chunks,
                table_chunks,
                elapsed,
            )
        except FileNotFoundError as exc:
            st.error(f"File error while processing '{file_name}': {exc}")
            logger.error("FileNotFoundError for '%s': %s", file_name, exc)
        except ValueError as exc:
            st.error(f"Unsupported file '{file_name}': {exc}")
            logger.error("ValueError for '%s': %s", file_name, exc)
        except Exception as exc:  # noqa: BLE001 - surface ingestion faults to UI
            st.error(f"Failed to process '{file_name}': {exc}")
            logger.error("Ingestion failure for '%s': %s", file_name, exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as cleanup_exc:
                    logger.warning(
                        "Could not remove temp file %s: %s", temp_path, cleanup_exc
                    )

    if newly_indexed > 0:
        reset_index_cache()
        ensure_chat_engine(force_rebuild=True)
        st.session_state["indexed_files"] = get_indexed_file_names()


# ---------------------------------------------------------------------------
# Source citation extraction
# ---------------------------------------------------------------------------
def _extract_sources(response: Any) -> List[Dict[str, Any]]:
    """Extract a structured citation list from a chat engine response object.

    Parameters
    ----------
    response:
        The response object returned by the chat engine (``chat`` or
        ``stream_chat``), which may expose ``source_nodes``.

    Returns
    -------
    list[dict]
        One entry per source node with text, score, and layout metadata.
    """
    sources: List[Dict[str, Any]] = []
    source_nodes = getattr(response, "source_nodes", None) or []

    for source_node in source_nodes:
        node = getattr(source_node, "node", source_node)
        metadata: Dict[str, Any] = dict(getattr(node, "metadata", {}) or {})
        try:
            content: str = node.get_content()
        except Exception:  # noqa: BLE001 - fall back to text attribute
            content = getattr(node, "text", "") or ""

        sources.append(
            {
                "file_name": metadata.get("file_name", "unknown"),
                "page_number": metadata.get("page_number", "—"),
                "element_type": metadata.get("element_type", "text"),
                "text_as_html": metadata.get("text_as_html"),
                "score": getattr(source_node, "score", None),
                "content": content,
            }
        )
    return sources


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render the citation expander for a single assistant reply.

    Parameters
    ----------
    sources:
        The structured source list produced by :func:`_extract_sources`.

    Returns
    -------
    None
    """
    if not sources:
        return

    with st.expander("🔍 Retained Source Citations & Layout Structures"):
        for position, source in enumerate(sources, start=1):
            score_text: str = (
                f"{source['score']:.4f}" if source.get("score") is not None else "n/a"
            )
            st.markdown(
                f"**Source {position}** · `{source['file_name']}` · "
                f"page {source['page_number']} · type `{source['element_type']}` · "
                f"relevance {score_text}"
            )

            if source.get("element_type") == "table" and source.get("text_as_html"):
                st.markdown("Retrieved table (rendered from source HTML):")
                st.markdown(source["text_as_html"], unsafe_allow_html=True)
            else:
                snippet: str = source.get("content", "")
                st.markdown(
                    f"<div class='source-snippet'>{snippet}</div>",
                    unsafe_allow_html=True,
                )
            if position < len(sources):
                st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the document-management sidebar.

    Returns
    -------
    None
    """
    with st.sidebar:
        st.header("📂 Document Library")

        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            st.error(
                "OPENAI_API_KEY is not set. Add a valid key to your `.env` file "
                "before uploading documents."
            )

        uploaded_files: List[Any] = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files already indexed are skipped automatically.",
        )

        if st.button("⚙️ Process Uploaded Documents", use_container_width=True):
            process_uploaded_files(uploaded_files)

        st.divider()
        st.subheader("🗂️ Indexed Documents")
        indexed_files: Set[str] = get_indexed_file_names()
        st.session_state["indexed_files"] = indexed_files
        if indexed_files:
            for file_name in sorted(indexed_files):
                st.markdown(f"- `{file_name}`")
        else:
            st.caption("No documents indexed yet. Upload a PDF to begin.")

        st.divider()
        st.metric("Total Chunks On Disk", get_collection_size())

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            engine = st.session_state.get("chat_engine")
            if engine is not None and hasattr(engine, "reset"):
                engine.reset()
            st.toast("Chat history cleared.", icon="🧹")


# ---------------------------------------------------------------------------
# Tab 1 :: Chat
# ---------------------------------------------------------------------------
def render_chat_tab() -> None:
    """Render the conversational chat interface tab.

    Returns
    -------
    None
    """
    st.subheader("💬 Smart Study Chat")

    chat_engine = ensure_chat_engine(force_rebuild=False)

    if chat_engine is None:
        st.info(
            "Upload and process at least one document in the sidebar to start "
            "chatting with your knowledge base."
        )
        return

    # Replay prior conversation turns.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    prompt: Optional[str] = st.chat_input("Ask a question about your documents…")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response: str = ""
        sources: List[Dict[str, Any]] = []

        try:
            with st.spinner("Retrieving, re-ranking, and synthesizing…"):
                streaming_response = chat_engine.stream_chat(prompt)
                for token in streaming_response.response_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                sources = _extract_sources(streaming_response)
        except Exception as exc:  # noqa: BLE001 - surface chat faults to UI
            full_response = (
                "I encountered an error while generating a response. "
                f"Details: {exc}"
            )
            response_placeholder.error(full_response)
            logger.error("Chat generation failed: %s", exc)

        _render_sources(sources)

    st.session_state["messages"].append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )


# ---------------------------------------------------------------------------
# Tab 2 :: Analytics
# ---------------------------------------------------------------------------
def render_analytics_tab() -> None:
    """Render the system & pipeline analytics dashboard tab.

    Returns
    -------
    None
    """
    st.subheader("📊 System & Pipeline Analytics")

    analytics: Dict[str, Any] = st.session_state["analytics"]
    collection_size: int = get_collection_size()
    indexed_files: Set[str] = get_indexed_file_names()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents Indexed", len(indexed_files))
    with col2:
        st.metric("Total Chunks On Disk", collection_size)
    with col3:
        st.metric("Tables Captured", analytics["table_chunks"])
    with col4:
        st.metric("Last Ingest (s)", f"{analytics['last_ingest_seconds']:.2f}")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🧩 Chunk Composition (this session)")
        composition: Dict[str, int] = {
            "Text Chunks": analytics["text_chunks"],
            "Table Chunks": analytics["table_chunks"],
        }
        if analytics["text_chunks"] + analytics["table_chunks"] > 0:
            st.bar_chart(composition)
        else:
            st.caption("No documents processed in this session yet.")

    with right:
        st.markdown("#### ⚙️ Pipeline Configuration")
        st.markdown(
            f"""
- **Embedding model:** `{os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")}`
- **LLM model:** `{os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")}`
- **Retrieval top-k:** `{os.getenv("RETRIEVAL_TOP_K", "10")}`
- **Re-rank top-n:** `{os.getenv("RERANK_TOP_N", "3")}`
- **Query variants:** `{os.getenv("MULTI_QUERY_VARIANTS", "3")}`
- **Re-ranker:** `{os.getenv("FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2")}`
"""
        )

    st.divider()
    st.markdown("#### 📑 Per-Document Processing Log")
    per_file: List[Dict[str, Any]] = analytics["per_file"]
    if per_file:
        st.dataframe(per_file, use_container_width=True, hide_index=True)
    else:
        st.caption("Process a document to populate the per-document log.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Application entry point: render the hero, sidebar, and workspace tabs.

    Returns
    -------
    None
    """
    st.markdown(
        """
<div class="app-hero">
    <h1>📚 Personal Document RAG</h1>
    <p>Layout-aware ingestion · Hybrid search · Cross-encoder re-ranking · Conversational memory</p>
</div>
""",
        unsafe_allow_html=True,
    )

    render_sidebar()

    chat_tab, analytics_tab = st.tabs(
        ["💬 Smart Study Chat", "📊 System & Pipeline Analytics"]
    )
    with chat_tab:
        render_chat_tab()
    with analytics_tab:
        render_analytics_tab()


if __name__ == "__main__":
    main()
