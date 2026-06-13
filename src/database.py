"""personal-document-rag :: persistence layer.

This module owns all interaction with the local, on-disk ChromaDB vector store
and the LlamaIndex objects that wrap it. It is responsible for:

* Initializing a singleton ``chromadb.PersistentClient`` rooted at ``./chroma_db``.
* Creating (or reusing) a single named collection that holds every embedded
  document chunk produced by :mod:`src.ingest`.
* Wrapping that collection in a LlamaIndex ``ChromaVectorStore`` bound to a
  ``StorageContext``.
* Routing between *loading* an existing index from disk and *building* a new one
  when fresh documents are supplied — without re-instantiating clients or
  duplicating embeddings across repeated runs.

State strategy
--------------
All heavyweight handles (the Chroma client, the collection, the vector store,
the storage context, and the materialized ``VectorStoreIndex``) are memoized at
module scope. The first call constructs them; every subsequent call reuses the
cached objects. Chroma's ``get_or_create_collection`` guarantees the on-disk
collection is created exactly once and transparently reopened thereafter, so we
never re-embed content that already exists.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Final, List, Optional, Set

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger: Final[logging.Logger] = logging.getLogger(__name__)
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
# Configuration constants (sourced from the environment with safe defaults)
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: Final[str] = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME: Final[str] = os.getenv(
    "CHROMA_COLLECTION_NAME", "personal_documents"
)
OPENAI_EMBED_MODEL: Final[str] = os.getenv(
    "OPENAI_EMBED_MODEL", "text-embedding-3-small"
)

# Distance metric used by the Chroma collection's HNSW index. Cosine is the
# natural choice for normalized OpenAI embeddings.
CHROMA_DISTANCE_METRIC: Final[str] = "cosine"

# ---------------------------------------------------------------------------
# Module-level singletons (lazily initialized, guarded by a lock)
# ---------------------------------------------------------------------------
_INIT_LOCK: Final[threading.Lock] = threading.Lock()
_chroma_client: Optional[ClientAPI] = None
_chroma_collection: Optional[Collection] = None
_vector_store: Optional[ChromaVectorStore] = None
_storage_context: Optional[StorageContext] = None
_vector_index: Optional[VectorStoreIndex] = None


def _configure_embeddings() -> None:
    """Bind the OpenAI embedding model onto the global LlamaIndex ``Settings``.

    Ensures every build and query path computes embeddings with the exact same
    model (``text-embedding-3-small``), which is mandatory for vector-space
    consistency between stored chunks and incoming queries.

    Returns
    -------
    None

    Raises
    ------
    EnvironmentError
        If the ``OPENAI_API_KEY`` environment variable is not configured.
    """
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_openai_api_key_here":
        raise EnvironmentError(
            "OPENAI_API_KEY is not configured. Set a valid key in your .env "
            "file before initializing the vector store."
        )

    if not isinstance(Settings.embed_model, OpenAIEmbedding) or (
        getattr(Settings.embed_model, "model_name", None) != OPENAI_EMBED_MODEL
    ):
        Settings.embed_model = OpenAIEmbedding(
            model=OPENAI_EMBED_MODEL,
            api_key=api_key,
        )
        logger.info("Configured embedding model: %s", OPENAI_EMBED_MODEL)


def _get_chroma_client() -> ClientAPI:
    """Return the singleton persistent ChromaDB client, creating it if needed.

    Returns
    -------
    chromadb.api.ClientAPI
        The process-wide persistent client rooted at :data:`CHROMA_DB_PATH`.
    """
    global _chroma_client
    if _chroma_client is None:
        absolute_path: str = os.path.abspath(CHROMA_DB_PATH)
        os.makedirs(absolute_path, exist_ok=True)
        logger.info("Initializing persistent ChromaDB client at: %s", absolute_path)
        _chroma_client = chromadb.PersistentClient(path=absolute_path)
    return _chroma_client


def _get_collection() -> Collection:
    """Return the singleton Chroma collection, creating it on first access.

    Uses ``get_or_create_collection`` so the on-disk collection is created
    exactly once and reopened transparently on every later run.

    Returns
    -------
    chromadb.api.models.Collection.Collection
        The collection that stores all embedded document chunks.
    """
    global _chroma_collection
    if _chroma_collection is None:
        client: ClientAPI = _get_chroma_client()
        logger.info("Opening Chroma collection: '%s'", CHROMA_COLLECTION_NAME)
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
        )
    return _chroma_collection


def _get_storage_context() -> StorageContext:
    """Return the singleton LlamaIndex ``StorageContext`` bound to Chroma.

    Returns
    -------
    llama_index.core.StorageContext
        A storage context whose vector store is the persisted Chroma collection.
    """
    global _vector_store, _storage_context
    if _storage_context is None:
        collection: Collection = _get_collection()
        _vector_store = ChromaVectorStore(chroma_collection=collection)
        _storage_context = StorageContext.from_defaults(vector_store=_vector_store)
        logger.debug("StorageContext bound to ChromaVectorStore.")
    return _storage_context


def get_indexed_file_names() -> Set[str]:
    """Return the set of distinct ``file_name`` values already in the collection.

    This enables the frontend to skip documents that have already been processed,
    preventing duplicate indexing and redundant embedding API spend.

    Returns
    -------
    set[str]
        Distinct source file names currently present in the vector store. Empty
        if the collection has no entries yet.
    """
    collection: Collection = _get_collection()
    try:
        total: int = collection.count()
        if total == 0:
            return set()
        records = collection.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001 - normalize Chroma read faults
        logger.error("Failed to read collection metadata: %s", exc)
        return set()

    metadatas = records.get("metadatas") or []
    file_names: Set[str] = set()
    for metadata in metadatas:
        if not metadata:
            continue
        file_name = metadata.get("file_name")
        if isinstance(file_name, str) and file_name:
            file_names.add(file_name)
    return file_names


def file_already_indexed(file_name: str) -> bool:
    """Check whether a given source file has already been indexed.

    Parameters
    ----------
    file_name:
        The base file name to look up (e.g. ``"report.pdf"``).

    Returns
    -------
    bool
        ``True`` if at least one chunk with this ``file_name`` exists in the
        collection, otherwise ``False``.
    """
    already_present: bool = file_name in get_indexed_file_names()
    logger.debug(
        "Index check for '%s': %s.",
        file_name,
        "already indexed" if already_present else "not indexed",
    )
    return already_present


def get_collection_size() -> int:
    """Return the total number of embedded chunks stored in the collection.

    Returns
    -------
    int
        The chunk/vector count currently persisted on disk.
    """
    collection: Collection = _get_collection()
    try:
        return int(collection.count())
    except Exception as exc:  # noqa: BLE001 - normalize Chroma read faults
        logger.error("Failed to count collection entries: %s", exc)
        return 0


def get_chroma_index(documents: Optional[List[Document]] = None) -> VectorStoreIndex:
    """Load an existing vector index from disk or build/extend it with documents.

    Behavior
    --------
    * **No documents, populated collection** → load the existing index instantly
      from the persisted vectors (no re-embedding).
    * **Documents provided** → embed and insert the new chunks into the persisted
      collection, then return the live index. Only the supplied chunks are
      embedded; previously stored vectors are untouched.
    * **No documents, empty collection** → return an empty index bound to the
      store, ready to receive future inserts.

    The materialized index is memoized at module scope so repeated calls within a
    single process reuse the same object rather than rebuilding it.

    Parameters
    ----------
    documents:
        Optional list of LlamaIndex ``Document`` objects to embed and persist. If
        ``None`` (or empty), the function operates in pure load mode.

    Returns
    -------
    llama_index.core.VectorStoreIndex
        The live, persistence-backed vector index.

    Raises
    ------
    EnvironmentError
        If the OpenAI API key is not configured.
    RuntimeError
        If the index cannot be built or loaded.
    """
    global _vector_index

    with _INIT_LOCK:
        _configure_embeddings()
        storage_context: StorageContext = _get_storage_context()
        existing_count: int = get_collection_size()

        # ---------------------------------------------------------------
        # Path A: new documents supplied -> embed + persist them.
        # ---------------------------------------------------------------
        if documents:
            logger.info(
                "Building/extending index with %d new document chunks "
                "(existing chunks on disk: %d).",
                len(documents),
                existing_count,
            )
            try:
                if _vector_index is None and existing_count > 0:
                    # Load the existing index first, then insert the new docs so
                    # we extend the collection rather than rebuilding it.
                    _vector_index = VectorStoreIndex.from_vector_store(
                        vector_store=storage_context.vector_store,
                        storage_context=storage_context,
                    )
                if _vector_index is None:
                    _vector_index = VectorStoreIndex.from_documents(
                        documents=documents,
                        storage_context=storage_context,
                        show_progress=True,
                    )
                else:
                    for document in documents:
                        _vector_index.insert(document)
            except Exception as exc:  # noqa: BLE001 - normalize build faults
                message: str = (
                    "Failed to build or extend the vector index with the "
                    f"supplied documents. Underlying error: {exc}"
                )
                logger.error(message)
                raise RuntimeError(message) from exc

            logger.info(
                "Index build/extend complete. Total chunks on disk: %d.",
                get_collection_size(),
            )
            return _vector_index

        # ---------------------------------------------------------------
        # Path B: no documents, but the collection already holds vectors.
        # ---------------------------------------------------------------
        if existing_count > 0:
            if _vector_index is None:
                logger.info(
                    "Loading existing index from disk (%d chunks persisted).",
                    existing_count,
                )
                try:
                    _vector_index = VectorStoreIndex.from_vector_store(
                        vector_store=storage_context.vector_store,
                        storage_context=storage_context,
                    )
                except Exception as exc:  # noqa: BLE001 - normalize load faults
                    message = (
                        "Failed to load the existing vector index from disk. "
                        f"Underlying error: {exc}"
                    )
                    logger.error(message)
                    raise RuntimeError(message) from exc
            else:
                logger.debug("Reusing cached in-memory vector index.")
            return _vector_index

        # ---------------------------------------------------------------
        # Path C: no documents and an empty collection -> empty index.
        # ---------------------------------------------------------------
        logger.info(
            "No documents supplied and the collection is empty. Returning an "
            "empty index bound to the persistent store."
        )
        if _vector_index is None:
            try:
                _vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=storage_context.vector_store,
                    storage_context=storage_context,
                )
            except Exception as exc:  # noqa: BLE001 - normalize init faults
                message = (
                    "Failed to initialize an empty vector index bound to the "
                    f"persistent store. Underlying error: {exc}"
                )
                logger.error(message)
                raise RuntimeError(message) from exc
        return _vector_index


def reset_index_cache() -> None:
    """Clear the memoized in-memory index handle.

    Forces the next :func:`get_chroma_index` call to re-load from disk. Useful
    after a fresh ingestion so the cached index reflects newly persisted data.
    The on-disk collection itself is left completely intact.

    Returns
    -------
    None
    """
    global _vector_index
    _vector_index = None
    logger.debug("In-memory vector index cache cleared.")
