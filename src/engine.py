"""personal-document-rag :: conversational retrieval engine.

This module assembles the advanced "brain" of the RAG system. It composes four
cooperating stages into a single conversational interface:

1. **Conversational condensing** — a :class:`CondenseQuestionChatEngine` folds the
   running chat history (held in a sliding-window :class:`ChatMemoryBuffer`) into
   a single standalone question.
2. **Multi-query expansion** — the standalone question is expanded by an internal
   ``gpt-4o-mini`` call into several alternative semantic phrasings to broaden
   recall.
3. **Hybrid retrieval** — every expanded query is run, in parallel, against both a
   semantic vector retriever (ChromaDB) and a lexical BM25 retriever; the pooled
   results are de-duplicated.
4. **Cross-encoder re-ranking** — the de-duplicated candidate pool is scored by a
   local FlashRank cross-encoder, and only the top-N highest-quality nodes are
   handed to the LLM for answer synthesis.

The public entry point is :func:`get_chat_engine`.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Final, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.retrievers.bm25 import BM25Retriever

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
# Configuration constants (environment-driven with safe defaults)
# ---------------------------------------------------------------------------
OPENAI_LLM_MODEL: Final[str] = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_LLM_TEMPERATURE: Final[float] = float(
    os.getenv("OPENAI_LLM_TEMPERATURE", "0.1")
)
OPENAI_LLM_MAX_TOKENS: Final[int] = int(os.getenv("OPENAI_LLM_MAX_TOKENS", "1024"))

RETRIEVAL_TOP_K: Final[int] = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_N: Final[int] = int(os.getenv("RERANK_TOP_N", "3"))
MULTI_QUERY_VARIANTS: Final[int] = int(os.getenv("MULTI_QUERY_VARIANTS", "3"))
FLASHRANK_MODEL: Final[str] = os.getenv(
    "FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2"
)
CHAT_MEMORY_TOKEN_LIMIT: Final[int] = int(
    os.getenv("CHAT_MEMORY_TOKEN_LIMIT", "3000")
)

# Maximum worker threads used to fan out hybrid retrieval across query variants.
_MAX_RETRIEVAL_WORKERS: Final[int] = 8

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
MULTI_QUERY_SYSTEM_PROMPT: Final[str] = (
    "You are an expert search query generator for a document retrieval system. "
    "Given a single user question, produce alternative phrasings that capture "
    "the same information need from different angles (synonyms, broader terms, "
    "and more specific terms). Each variant must be self-contained and "
    "answerable independently."
)

MULTI_QUERY_USER_PROMPT: Final[str] = (
    "Generate exactly {num_variants} alternative search queries for the question "
    "below. Return ONLY the queries, one per line, with no numbering, bullets, "
    "quotation marks, or commentary.\n\n"
    "Original question: {query}"
)


def _configure_models() -> None:
    """Bind the OpenAI chat model onto the global LlamaIndex ``Settings``.

    Ensures the conversational synthesizer, the condensing step, and the
    multi-query expander all share one consistently configured ``gpt-4o-mini``
    instance.

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
            "file before initializing the chat engine."
        )

    if not isinstance(Settings.llm, OpenAI) or (
        getattr(Settings.llm, "model", None) != OPENAI_LLM_MODEL
    ):
        Settings.llm = OpenAI(
            model=OPENAI_LLM_MODEL,
            temperature=OPENAI_LLM_TEMPERATURE,
            max_tokens=OPENAI_LLM_MAX_TOKENS,
            api_key=api_key,
        )
        logger.info(
            "Configured LLM: %s (temperature=%.2f, max_tokens=%d).",
            OPENAI_LLM_MODEL,
            OPENAI_LLM_TEMPERATURE,
            OPENAI_LLM_MAX_TOKENS,
        )


def _generate_query_variants(query: str, num_variants: int) -> List[str]:
    """Expand a single query into multiple alternative semantic phrasings.

    Parameters
    ----------
    query:
        The standalone question to expand.
    num_variants:
        The number of *additional* variant queries to request from the LLM.

    Returns
    -------
    list[str]
        The original query followed by the de-duplicated generated variants. The
        original query is always the first element so recall never regresses even
        if generation fails.
    """
    cleaned_query: str = query.strip()
    variants: List[str] = [cleaned_query] if cleaned_query else []

    if num_variants <= 0 or not cleaned_query:
        return variants

    prompt: str = MULTI_QUERY_USER_PROMPT.format(
        num_variants=num_variants, query=cleaned_query
    )

    try:
        response = Settings.llm.complete(
            f"{MULTI_QUERY_SYSTEM_PROMPT}\n\n{prompt}"
        )
        raw_text: str = str(response).strip()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on LLM faults
        logger.warning(
            "Multi-query expansion failed; falling back to the original query "
            "only. Underlying error: %s",
            exc,
        )
        return variants

    for line in raw_text.splitlines():
        candidate: str = line.strip().lstrip("0123456789.-)• ").strip().strip('"')
        if candidate and candidate.lower() not in {v.lower() for v in variants}:
            variants.append(candidate)

    logger.info(
        "Multi-query expansion produced %d total queries (1 original + %d variants).",
        len(variants),
        len(variants) - 1,
    )
    return variants


class HybridMultiQueryRetriever(BaseRetriever):
    """Fan-out retriever combining multi-query expansion with hybrid search.

    For each incoming query bundle this retriever:

    1. Expands the query into multiple semantic variants via an LLM.
    2. Runs every variant through both a dense vector retriever and a lexical
       BM25 retriever, executing the retrievals concurrently.
    3. Pools and de-duplicates the resulting nodes by node id, retaining the
       highest score observed for each unique node.

    The fused, de-duplicated node set is then returned for downstream
    re-ranking by the query engine's node post-processors.
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        num_query_variants: int = MULTI_QUERY_VARIANTS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize the hybrid, multi-query retriever.

        Parameters
        ----------
        vector_retriever:
            Dense semantic retriever backed by the Chroma vector store.
        bm25_retriever:
            Sparse lexical retriever operating over the same node corpus.
        num_query_variants:
            Number of additional query variants to generate per request.
        callback_manager:
            Optional LlamaIndex callback manager for instrumentation.
        """
        self._vector_retriever: VectorIndexRetriever = vector_retriever
        self._bm25_retriever: BM25Retriever = bm25_retriever
        self._num_query_variants: int = num_query_variants
        super().__init__(callback_manager=callback_manager)

    def _retrieve_single(self, query_text: str) -> List[NodeWithScore]:
        """Run both retrievers for one query string and merge their results.

        Parameters
        ----------
        query_text:
            A single query string (original or expanded variant).

        Returns
        -------
        list[NodeWithScore]
            The combined vector + BM25 hits for this query. Failures in either
            retriever are logged and treated as empty result sets so a single
            fault never aborts the whole fan-out.
        """
        bundle: QueryBundle = QueryBundle(query_str=query_text)
        combined: List[NodeWithScore] = []

        try:
            combined.extend(self._vector_retriever.retrieve(bundle))
        except Exception as exc:  # noqa: BLE001 - isolate per-retriever faults
            logger.warning(
                "Vector retrieval failed for query '%s': %s", query_text, exc
            )

        try:
            combined.extend(self._bm25_retriever.retrieve(bundle))
        except Exception as exc:  # noqa: BLE001 - isolate per-retriever faults
            logger.warning(
                "BM25 retrieval failed for query '%s': %s", query_text, exc
            )

        return combined

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Expand, fan out, and fuse retrieval across all query variants.

        Parameters
        ----------
        query_bundle:
            The incoming query produced by the chat engine's condensing step.

        Returns
        -------
        list[NodeWithScore]
            The de-duplicated union of all retrieved nodes, sorted by descending
            score, ready for cross-encoder re-ranking.
        """
        queries: List[str] = _generate_query_variants(
            query_bundle.query_str, self._num_query_variants
        )
        logger.info("Executing hybrid retrieval across %d queries.", len(queries))

        pooled: List[NodeWithScore] = []
        worker_count: int = min(_MAX_RETRIEVAL_WORKERS, max(1, len(queries)))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_query = {
                executor.submit(self._retrieve_single, query): query
                for query in queries
            }
            for future in as_completed(future_to_query):
                query_text: str = future_to_query[future]
                try:
                    pooled.extend(future.result())
                except Exception as exc:  # noqa: BLE001 - isolate worker faults
                    logger.warning(
                        "Retrieval worker failed for query '%s': %s",
                        query_text,
                        exc,
                    )

        deduplicated: Dict[str, NodeWithScore] = {}
        for node_with_score in pooled:
            node_id: str = node_with_score.node.node_id
            incoming_score: float = (
                node_with_score.score if node_with_score.score is not None else 0.0
            )
            existing: Optional[NodeWithScore] = deduplicated.get(node_id)
            if existing is None:
                deduplicated[node_id] = node_with_score
            else:
                existing_score: float = (
                    existing.score if existing.score is not None else 0.0
                )
                if incoming_score > existing_score:
                    deduplicated[node_id] = node_with_score

        fused: List[NodeWithScore] = sorted(
            deduplicated.values(),
            key=lambda nws: nws.score if nws.score is not None else 0.0,
            reverse=True,
        )
        logger.info(
            "Hybrid retrieval pooled %d hits, de-duplicated to %d unique nodes.",
            len(pooled),
            len(fused),
        )
        return fused


def _build_bm25_retriever(index: VectorStoreIndex) -> Optional[BM25Retriever]:
    """Construct a BM25 lexical retriever from the index's stored nodes.

    Parameters
    ----------
    index:
        The vector index whose underlying nodes will seed the BM25 corpus.

    Returns
    -------
    BM25Retriever | None
        A configured BM25 retriever, or ``None`` if no nodes are available yet
        (e.g. an empty collection), in which case the engine degrades to pure
        vector retrieval.
    """
    try:
        node_dict = index.docstore.docs
        nodes: List[TextNode] = [
            node for node in node_dict.values() if isinstance(node, TextNode)
        ]
    except Exception as exc:  # noqa: BLE001 - normalize docstore access faults
        logger.warning("Unable to read nodes for BM25 corpus: %s", exc)
        nodes = []

    if not nodes:
        logger.warning(
            "No nodes available for BM25; the engine will rely on vector "
            "retrieval only until documents are indexed."
        )
        return None

    bm25_retriever: BM25Retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=RETRIEVAL_TOP_K,
    )
    logger.info("BM25 lexical retriever built over %d nodes.", len(nodes))
    return bm25_retriever


def _build_reranker() -> FlashRankRerank:
    """Instantiate the local FlashRank cross-encoder re-ranker.

    Returns
    -------
    FlashRankRerank
        A re-ranker configured to emit the top :data:`RERANK_TOP_N` nodes.
    """
    reranker: FlashRankRerank = FlashRankRerank(
        model=FLASHRANK_MODEL,
        top_n=RERANK_TOP_N,
    )
    logger.info(
        "FlashRank re-ranker initialized (model=%s, top_n=%d).",
        FLASHRANK_MODEL,
        RERANK_TOP_N,
    )
    return reranker


def get_chat_engine(index: VectorStoreIndex) -> CondenseQuestionChatEngine:
    """Assemble the full conversational retrieval engine over an index.

    Wires together multi-query hybrid retrieval, FlashRank re-ranking, response
    synthesis, and conversational memory into a single
    :class:`CondenseQuestionChatEngine`.

    Parameters
    ----------
    index:
        The persistence-backed vector index (see :func:`src.database.get_chroma_index`).

    Returns
    -------
    llama_index.core.chat_engine.CondenseQuestionChatEngine
        A ready-to-use chat engine supporting both ``chat`` and ``stream_chat``.

    Raises
    ------
    EnvironmentError
        If the OpenAI API key is not configured.
    RuntimeError
        If the underlying query engine cannot be assembled.
    """
    logger.info("Assembling conversational retrieval engine.")
    _configure_models()

    try:
        vector_retriever: VectorIndexRetriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=RETRIEVAL_TOP_K,
        )

        bm25_retriever: Optional[BM25Retriever] = _build_bm25_retriever(index)

        if bm25_retriever is not None:
            hybrid_retriever: BaseRetriever = HybridMultiQueryRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                num_query_variants=MULTI_QUERY_VARIANTS,
            )
        else:
            # Degrade gracefully to dense-only retrieval when the corpus is empty.
            hybrid_retriever = vector_retriever

        reranker: FlashRankRerank = _build_reranker()

        query_engine: RetrieverQueryEngine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[reranker],
            streaming=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize assembly faults
        message: str = (
            "Failed to assemble the retrieval query engine. "
            f"Underlying error: {exc}"
        )
        logger.error(message)
        raise RuntimeError(message) from exc

    memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    chat_engine: CondenseQuestionChatEngine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        memory=memory,
        llm=Settings.llm,
        verbose=True,
    )

    logger.info(
        "Chat engine ready (memory token limit=%d).", CHAT_MEMORY_TOKEN_LIMIT
    )
    return chat_engine
