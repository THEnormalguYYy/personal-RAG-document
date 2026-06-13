"""personal-document-rag :: ingestion layer.

This module converts raw source documents into layout-aware, embedding-ready
LlamaIndex :class:`~llama_index.core.schema.Document` objects.

Pipeline overview
-----------------
1. Validate that the requested file exists on disk.
2. Partition the document with Unstructured's local ``partition_pdf`` using the
   high-resolution (``strategy="hi_res"``) visual model with table-structure
   inference enabled (``pdf_infer_table_structure=True``).
3. Apply element-aware, title-based chunking (``chunk_by_title``) so that
   semantically related layout blocks are grouped beneath their headings.
4. Isolate ``Table`` elements as indivisible chunks whose body is the raw HTML
   representation (``el.metadata.text_as_html``) for faithful tabular grounding.
5. Wrap each resulting chunk into a LlamaIndex ``Document`` enriched with a deep
   metadata array (``file_name``, ``page_number``, ``element_type`` and more).

All host-binary dependent operations (which transitively require ``tesseract``
and ``poppler``) are wrapped in descriptive exception handling that surfaces an
actionable message when those system binaries are missing from ``PATH``.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Final, List, Optional
from urllib.error import HTTPError, URLError

from llama_index.core import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Table, Title
from unstructured.partition.pdf import partition_pdf

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
# Chunking configuration constants
# ---------------------------------------------------------------------------
# Soft target (in characters) for the size of a composite text chunk. Unstructured
# will try to keep aggregated content beneath this ceiling while still respecting
# title boundaries.
MAX_CHUNK_CHARACTERS: Final[int] = 2000

# Once a chunk crosses this character count, a new chunk may be started even when
# the next element shares the same title context. Provides a hard safety valve.
NEW_AFTER_N_CHARS: Final[int] = 1500

# Minimum overlap (in characters) carried between consecutive composite chunks to
# preserve cross-boundary context for retrieval.
CHUNK_OVERLAP: Final[int] = 150

# Supported source file extensions for layout-aware partitioning.
SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf",)


class IngestionError(RuntimeError):
    """Raised when a document cannot be partitioned or converted.

    This wraps lower-level Unstructured / system-binary failures into a single,
    explicit exception type that the calling layer (``app.py``) can catch and
    surface to the end user without leaking raw stack traces.
    """


def _verify_system_binaries() -> None:
    """Probe the host ``PATH`` for the binaries Unstructured ``hi_res`` requires.

    The high-resolution strategy shells out to Poppler (PDF rasterization) and
    Tesseract (OCR). When either is absent the underlying error messages can be
    cryptic, so we log an explicit, actionable warning up-front.

    Returns
    -------
    None
        This function only emits warnings; it never raises. The actual partition
        call remains the authoritative point of failure.
    """
    missing: List[str] = []

    if shutil.which("tesseract") is None:
        missing.append("tesseract")
    if shutil.which("pdfinfo") is None and shutil.which("pdftoppm") is None:
        missing.append("poppler")

    if missing:
        logger.warning(
            "Required system binaries not found on PATH: %s. "
            "The 'hi_res' strategy depends on them. Install with "
            "`brew install tesseract poppler` (macOS) or "
            "`apt-get install -y tesseract-ocr poppler-utils` (Debian/Ubuntu).",
            ", ".join(missing),
        )
    else:
        logger.debug("System binaries 'tesseract' and 'poppler' located on PATH.")


def _partition_pdf_hi_res(file_path: str) -> List[Element]:
    """Run Unstructured's high-resolution PDF partitioner with table inference.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the PDF document on the local filesystem.

    Returns
    -------
    list[unstructured.documents.elements.Element]
        The ordered list of raw layout elements extracted from the document.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not point to an existing file.
    IngestionError
        If partitioning fails, including the common case where the ``tesseract``
        or ``poppler`` system binaries are missing from the host environment.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Source document not found at path: '{file_path}'."
        )

    logger.info("Partitioning document with hi_res strategy: %s", file_path)

    try:
        elements: List[Element] = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            pdf_infer_table_structure=True,
            infer_table_structure=True,
            include_page_breaks=True,
        )
    except FileNotFoundError as exc:
        # A FileNotFoundError raised from *inside* partition_pdf almost always
        # means a required system binary (poppler/tesseract) is not on PATH,
        # rather than the source document being absent (we validated that above).
        message: str = (
            "Partitioning failed due to a missing executable. The 'hi_res' "
            "strategy requires the 'poppler' and 'tesseract' system binaries. "
            "Install them with `brew install tesseract poppler` (macOS) or "
            "`apt-get install -y tesseract-ocr poppler-utils` (Debian/Ubuntu). "
            f"Underlying error: {exc}"
        )
        logger.error(message)
        raise IngestionError(message) from exc
    except HTTPError as exc:
        # The hi_res strategy and pdf_infer_table_structure download layout and
        # table-transformer models from the Hugging Face Hub on first use. An
        # HTTP error here (e.g. 403/429 rate limiting) is a network/download
        # problem, NOT a missing system binary.
        message = (
            "Partitioning failed while downloading a required model from the "
            "Hugging Face Hub (HTTP "
            f"{getattr(exc, 'code', 'error')}). This is a network/rate-limit "
            "issue, not a missing system binary. Retry in a moment; if it "
            "persists, set a HF_TOKEN environment variable to raise the "
            "download rate limit, or pre-download the models on a connected "
            f"network. Underlying error: {exc}"
        )
        logger.error(message)
        raise IngestionError(message) from exc
    except URLError as exc:
        # Generic network failure (no connectivity, DNS, TLS) while fetching
        # models or resources required by the hi_res pipeline.
        message = (
            "Partitioning failed due to a network error while fetching a "
            "required model or resource. Check your internet connection and "
            f"retry. Underlying error: {exc}"
        )
        logger.error(message)
        raise IngestionError(message) from exc
    except OSError as exc:
        # Covers permission errors, subprocess execution failures, and other
        # operating-system level faults encountered while invoking binaries.
        message = (
            "An operating-system error occurred while partitioning the "
            "document. This frequently indicates that 'tesseract' or 'poppler' "
            "is installed incorrectly or not executable on the current PATH. "
            f"Underlying error: {exc}"
        )
        logger.error(message)
        raise IngestionError(message) from exc
    except Exception as exc:  # noqa: BLE001 - we intentionally normalize all faults
        message = (
            "An unexpected error occurred during document partitioning. "
            "Verify that the file is a valid, non-corrupt PDF and that all "
            "Unstructured system dependencies are installed correctly. "
            f"Underlying error: {exc}"
        )
        logger.error(message)
        raise IngestionError(message) from exc

    logger.info(
        "Partitioning complete: %d raw layout elements extracted from %s.",
        len(elements),
        os.path.basename(file_path),
    )
    return elements


def _chunk_elements(elements: List[Element]) -> List[Element]:
    """Group raw elements into layout-coherent chunks via title segmentation.

    Uses Unstructured's ``chunk_by_title`` so that body content is aggregated
    beneath the heading (``Title``) that introduces it. ``Table`` elements are
    preserved as their own indivisible chunks by the chunker.

    Parameters
    ----------
    elements:
        The raw layout elements returned by :func:`_partition_pdf_hi_res`.

    Returns
    -------
    list[unstructured.documents.elements.Element]
        The list of chunked elements (``CompositeElement`` and ``Table``).

    Raises
    ------
    IngestionError
        If the chunking routine fails for any reason.
    """
    logger.info("Chunking %d elements with title-based segmentation.", len(elements))

    try:
        chunks: List[Element] = chunk_by_title(
            elements,
            max_characters=MAX_CHUNK_CHARACTERS,
            new_after_n_chars=NEW_AFTER_N_CHARS,
            overlap=CHUNK_OVERLAP,
            combine_text_under_n_chars=0,
            multipage_sections=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize chunking faults
        message: str = (
            "Failed to chunk document elements with 'chunk_by_title'. "
            f"Underlying error: {exc}"
        )
        logger.error(message)
        raise IngestionError(message) from exc

    logger.info("Chunking complete: %d composite chunks produced.", len(chunks))
    return chunks


def _resolve_page_number(element: Element) -> Optional[int]:
    """Safely extract the originating page number from an element's metadata.

    Parameters
    ----------
    element:
        An Unstructured element (raw or chunked).

    Returns
    -------
    int | None
        The 1-based page number if present, otherwise ``None``.
    """
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return None
    page_number = getattr(metadata, "page_number", None)
    if isinstance(page_number, int):
        return page_number
    return None


def _build_document_from_chunk(
    chunk: Element,
    file_name: str,
    chunk_index: int,
) -> Optional[Document]:
    """Convert a single Unstructured chunk into a LlamaIndex ``Document``.

    Two distinct paths are handled:

    * **Table chunks** — The body becomes the raw HTML representation
      (``metadata.text_as_html``) so the tabular structure is preserved exactly.
      ``element_type`` is recorded as ``"table"`` and the HTML is also stored in
      metadata for downstream citation rendering.
    * **Text / composite chunks** — The body is the aggregated plain text of the
      chunk (titles are already folded in by ``chunk_by_title`` for structural
      grounding). ``element_type`` reflects the underlying element category.

    Parameters
    ----------
    chunk:
        The chunked Unstructured element to convert.
    file_name:
        The base file name of the source document (used for metadata + filtering).
    chunk_index:
        The positional index of this chunk within the document (for stable IDs).

    Returns
    -------
    llama_index.core.schema.Document | None
        A populated ``Document``, or ``None`` when the chunk holds no usable
        content and should be skipped.
    """
    page_number: Optional[int] = _resolve_page_number(chunk)
    element_type: str = type(chunk).__name__
    text_as_html: Optional[str] = None

    if isinstance(chunk, Table):
        # Tables are isolated and represented by their HTML for faithful layout.
        metadata = getattr(chunk, "metadata", None)
        text_as_html = getattr(metadata, "text_as_html", None) if metadata else None
        body: str = text_as_html if text_as_html else (chunk.text or "")
        element_type = "table"
        if not body.strip():
            logger.debug(
                "Skipping empty table chunk #%d on page %s.",
                chunk_index,
                page_number,
            )
            return None
    else:
        body = chunk.text or ""
        # Normalize the reported element type for composite/title/text chunks.
        if isinstance(chunk, Title):
            element_type = "title"
        else:
            element_type = element_type.lower()
        if not body.strip():
            logger.debug(
                "Skipping empty text chunk #%d on page %s.",
                chunk_index,
                page_number,
            )
            return None

    metadata_payload: dict[str, object] = {
        "file_name": file_name,
        "page_number": page_number if page_number is not None else -1,
        "element_type": element_type,
        "chunk_index": chunk_index,
    }
    if text_as_html:
        metadata_payload["text_as_html"] = text_as_html

    # Keys that should not pollute the embedding text or the LLM prompt context.
    excluded_keys: List[str] = ["chunk_index", "text_as_html"]

    document: Document = Document(
        text=body,
        metadata=metadata_payload,
        excluded_embed_metadata_keys=excluded_keys,
        excluded_llm_metadata_keys=excluded_keys,
        metadata_separator="\n",
        metadata_template="{key}: {value}",
        text_template="[{metadata_str}]\n\n{content}",
    )

    logger.debug(
        "Built Document from chunk #%d (type=%s, page=%s, chars=%d).",
        chunk_index,
        element_type,
        page_number,
        len(body),
    )
    return document


def parse_document_with_unstructured(file_path: str) -> List[Document]:
    """Parse a source document into layout-aware LlamaIndex ``Document`` objects.

    This is the public entry point of the ingestion layer. It orchestrates the
    full pipeline: binary verification, high-resolution partitioning, title-based
    chunking, and chunk-to-``Document`` conversion with rich metadata.

    Parameters
    ----------
    file_path:
        Path to the source document on the local filesystem. Currently only PDF
        documents are supported by the high-resolution strategy.

    Returns
    -------
    list[llama_index.core.schema.Document]
        The fully populated, embedding-ready documents. May be empty if the
        source contained no extractable content.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If the file extension is not supported.
    IngestionError
        If partitioning or chunking fails (including missing system binaries).
    """
    logger.info("Starting ingestion for: %s", file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Source document not found at path: '{file_path}'."
        )

    file_name: str = os.path.basename(file_path)
    _, extension = os.path.splitext(file_name)
    if extension.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported extensions are: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )

    # Probe the environment up-front so users get an actionable warning early.
    _verify_system_binaries()

    raw_elements: List[Element] = _partition_pdf_hi_res(file_path)
    if not raw_elements:
        logger.warning(
            "No layout elements were extracted from '%s'. Returning empty list.",
            file_name,
        )
        return []

    chunked_elements: List[Element] = _chunk_elements(raw_elements)

    documents: List[Document] = []
    table_count: int = 0
    text_count: int = 0

    for chunk_index, chunk in enumerate(chunked_elements):
        document: Optional[Document] = _build_document_from_chunk(
            chunk=chunk,
            file_name=file_name,
            chunk_index=chunk_index,
        )
        if document is None:
            continue
        if document.metadata.get("element_type") == "table":
            table_count += 1
        else:
            text_count += 1
        documents.append(document)

    logger.info(
        "Ingestion complete for '%s': %d documents created "
        "(%d text chunks, %d table chunks).",
        file_name,
        len(documents),
        text_count,
        table_count,
    )
    return documents
