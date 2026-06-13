# 📚 Personal Document RAG

> **A production-grade, layout-aware Retrieval-Augmented Generation workstation for your private documents.**
> Upload PDFs, preserve their visual structure (headings, paragraphs, and tables), and chat with a hybrid-search, cross-encoder–reranked knowledge base — all running locally except for the OpenAI model calls.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white">
  <img alt="LlamaIndex" src="https://img.shields.io/badge/Orchestration-LlamaIndex-7C3AED">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/Vector%20Store-ChromaDB-1f6feb">
  <img alt="OpenAI" src="https://img.shields.io/badge/Models-OpenAI-412991?logo=openai&logoColor=white">
</p>

---

## 🧭 Table of Contents

1. [Overview](#-overview)
2. [Why This Architecture Matters](#-why-this-architecture-matters)
3. [System Architecture](#-system-architecture)
4. [End-to-End Data Flow](#-end-to-end-data-flow)
5. [Repository Structure](#-repository-structure)
6. [Prerequisites](#-prerequisites)
7. [Installation Guide](#-installation-guide)
8. [Configuration](#-configuration)
9. [Running the Application](#-running-the-application)
10. [Usage Walkthrough](#-usage-walkthrough)
11. [Tech Stack](#-tech-stack)
12. [Troubleshooting](#-troubleshooting)
13. [License](#-license)

---

## 🔭 Overview

**Personal Document RAG** is a self-hosted dashboard that turns a folder of unstructured PDFs into a queryable, conversational knowledge base. Unlike naive RAG pipelines that flatten a document into a stream of characters, this system is **layout-aware**: it understands the difference between a heading, a paragraph, and a table, and it preserves tabular data as raw HTML so that numerical and structured content survives the journey from page to prompt.

The retrieval stack goes well beyond a single vector lookup. Every question is condensed against conversation history, expanded into multiple semantic variants, searched across **both** dense embeddings and lexical keywords in parallel, and finally re-scored by a **local cross-encoder** that selects only the most relevant context before the language model ever sees it.

---

## 🧠 Why This Architecture Matters

### 1. Layout-Aware Parsing with Unstructured (`hi_res`)

Most RAG failures begin at ingestion. A standard text extractor will happily turn a financial table into a meaningless run-on string of numbers, destroying the row/column relationships that give the data meaning.

This project uses the **Unstructured** framework's high-resolution (`hi_res`) visual strategy:

- It runs an **object-detection model** over a rasterized render of each page to recognize titles, narrative text, lists, and tables by their *visual* position — not just their text stream.
- With `pdf_infer_table_structure=True`, detected tables are reconstructed into **HTML** (`text_as_html`), preserving cells, rows, and headers.
- `chunk_by_title` segmentation groups body content beneath the heading that introduces it, so each chunk carries its own structural context instead of being split mid-thought.

**The result:** chunks that are semantically coherent and structurally faithful, which directly improves retrieval precision and answer grounding.

### 2. Hybrid Search (Dense + Sparse)

Vector search excels at *meaning* ("what does this paragraph talk about?") but can miss exact identifiers, product codes, or rare terminology. Lexical **BM25** search excels at *precision* on exact tokens but is blind to synonyms.

By running **both retrievers in parallel** and fusing their results, the system captures conceptually similar passages **and** keyword-exact matches — covering each method's blind spots.

### 3. Multi-Query Expansion

A single phrasing of a question only probes one corner of the embedding space. Before retrieval, the standalone query is expanded by `gpt-4o-mini` into several alternative phrasings (synonyms, broader terms, more specific terms). Each variant is searched independently, dramatically widening recall so the right passage is far less likely to be missed.

### 4. Cross-Encoder Re-ranking with FlashRank

Wide recall produces a large, noisy candidate pool. A **cross-encoder** re-ranker (run locally via **FlashRank**) reads each candidate *together with* the query and assigns a true relevance score — far more accurate than the approximate distances used during initial retrieval. Only the **top 3** highest-quality blocks are forwarded to the LLM, which keeps the final prompt focused, cheap, and accurate.

### 5. Conversational Memory

A `CondenseQuestionChatEngine` with a sliding-window `ChatMemoryBuffer` rewrites follow-up questions ("what about the second one?") into fully standalone queries, so multi-turn conversations retrieve the right context every time.

---

## 🏗 System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Frontend (app.py)                       │
│   Sidebar Uploader  │  💬 Smart Study Chat  │  📊 Pipeline Analytics        │
└───────────────┬───────────────────────┬──────────────────────────────────┘
                │                        │
       upload + dedup            stream_chat()
                │                        │
                ▼                        ▼
   ┌─────────────────────┐    ┌────────────────────────────────────────────┐
   │   src/ingest.py     │    │              src/engine.py                  │
   │  Unstructured hi_res│    │  CondenseQuestionChatEngine (history)       │
   │  chunk_by_title     │    │   └─ Multi-Query Expansion (gpt-4o-mini)    │
   │  Table → HTML       │    │       └─ HybridMultiQueryRetriever          │
   │  LlamaIndex Docs    │    │           ├─ Vector (Chroma, cosine)        │
   └──────────┬──────────┘    │           └─ BM25 (lexical)                 │
              │               │       └─ FlashRank Rerank (top 3)           │
              ▼               │           └─ gpt-4o-mini synthesis          │
   ┌─────────────────────┐    └───────────────────────┬────────────────────┘
   │   src/database.py   │◄───────────────────────────┘
   │  ChromaVectorStore  │      load / persist embeddings
   │  StorageContext     │
   │  PersistentClient   │──────► ./chroma_db (on-disk, persistent)
   └─────────────────────┘
```

---

## 🔁 End-to-End Data Flow

1. **Upload & De-duplicate** — A PDF is uploaded in the sidebar. Before any work is done, `database.file_already_indexed()` checks the collection by file name to avoid re-embedding and wasting tokens.
2. **Partition** — `ingest.parse_document_with_unstructured()` runs `partition_pdf(strategy="hi_res", pdf_infer_table_structure=True)`, guarded against missing `tesseract`/`poppler` binaries.
3. **Chunk** — Elements are grouped with `chunk_by_title`; tables are isolated and stored as raw HTML.
4. **Embed & Persist** — Chunks become LlamaIndex `Document`s with rich metadata (`file_name`, `page_number`, `element_type`) and are embedded with `text-embedding-3-small` into the persistent ChromaDB collection.
5. **Condense** — At query time, chat history is folded into a single standalone question.
6. **Expand** — The question is expanded into multiple semantic variants by `gpt-4o-mini`.
7. **Hybrid Retrieve** — Every variant is searched in parallel across vector + BM25 retrievers; results are pooled and de-duplicated.
8. **Re-rank** — FlashRank scores the candidate pool and keeps the top 3.
9. **Synthesize & Stream** — `gpt-4o-mini` composes an answer from the top context, streamed token-by-token to the UI with citation expanders showing source snippets and HTML tables.

---

## 🗂 Repository Structure

```text
personal-document-rag/
├── app.py                  # Streamlit dashboard: uploads, chat, analytics
├── requirements.txt        # Pinned, mutually compatible dependencies
├── README.md               # You are here
├── .env                    # Local secrets & tuning (git-ignored)
├── .gitignore              # Blocks secrets, raw docs, chroma_db, caches
├── chroma_db/              # Persistent vector store (auto-created, git-ignored)
└── src/
    ├── __init__.py         # Package exports: ingest / database / engine
    ├── ingest.py           # Layout-aware Unstructured parsing → LlamaIndex Docs
    ├── database.py         # Persistent ChromaDB index: load vs. build routing
    └── engine.py           # Multi-query + hybrid retrieval + FlashRank rerank
```

---

## ✅ Prerequisites

| Requirement | Version / Notes |
| --- | --- |
| **Python** | 3.10 or 3.11 (recommended) |
| **OpenAI API key** | Required for embeddings and the LLM |
| **Tesseract OCR** | System binary — OCR for the `hi_res` strategy |
| **Poppler** | System binary — PDF rasterization (`pdftoppm`/`pdfinfo`) |
| **libmagic** | System library — robust file-type detection |

> ⚠️ **Important:** `tesseract`, `poppler`, and `libmagic` are **operating-system binaries**, not Python packages. They must be installed with your system package manager *before* the high-resolution ingestion pipeline can run.

### Install System Binaries

#### 🍎 macOS (Homebrew)

```bash
brew install tesseract poppler libmagic
```

#### 🐧 Debian / Ubuntu Linux

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libmagic1
```

#### 🐧 Fedora / RHEL / CentOS

```bash
sudo dnf install -y tesseract poppler-utils file-libs
```

#### 🪟 Windows

1. Install **Tesseract** from the UB-Mannheim build: <https://github.com/UB-Mannheim/tesseract/wiki> and add it to your `PATH`.
2. Install **Poppler for Windows** from <https://github.com/oschwartz10612/poppler-windows/releases> and add its `bin/` folder to your `PATH`.
3. `libmagic` is bundled via the `python-magic-bin` wheel pulled in by `unstructured[all-docs]`.

### Verify the Binaries

```bash
tesseract --version
pdfinfo -v
```

Both commands should print version information. If either reports "command not found", re-check your installation and `PATH`.

---

## ⚙️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/THEnormalguYYy/personal-RAG-document.git
cd personal-document-rag
```

### 2. Create and Activate a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Upgrade Pip and Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> ℹ️ The first install is sizeable: `unstructured[all-docs]` pulls in document-parsing and computer-vision models. Allow several minutes depending on your connection.

---

## 🔐 Configuration

All runtime configuration lives in a local, git-ignored `.env` file at the project root. Open it and replace the placeholder API key with your own:

```dotenv
# --- REQUIRED ---
OPENAI_API_KEY=your_openai_api_key_here

# --- Models ---
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini

# --- Optional tuning (safe defaults shown) ---
OPENAI_LLM_TEMPERATURE=0.1
OPENAI_LLM_MAX_TOKENS=1024
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=personal_documents
RETRIEVAL_TOP_K=10
RERANK_TOP_N=3
MULTI_QUERY_VARIANTS=3
FLASHRANK_MODEL=ms-marco-MiniLM-L-12-v2
CHAT_MEMORY_TOKEN_LIMIT=3000
LOG_LEVEL=INFO
```

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | **Required.** Your secret OpenAI key (`sk-…`). |
| `OPENAI_EMBED_MODEL` | Embedding model for vectorizing chunks. |
| `OPENAI_LLM_MODEL` | Chat/reasoning model (the "brain"). |
| `OPENAI_LLM_TEMPERATURE` | Sampling temperature; lower is more factual. |
| `OPENAI_LLM_MAX_TOKENS` | Max tokens generated per reply. |
| `CHROMA_DB_PATH` | On-disk location of the persistent vector store. |
| `CHROMA_COLLECTION_NAME` | Name of the Chroma collection. |
| `RETRIEVAL_TOP_K` | Candidates fetched per retriever per query. |
| `RERANK_TOP_N` | Final context blocks kept after re-ranking. |
| `MULTI_QUERY_VARIANTS` | Extra query phrasings generated per question. |
| `FLASHRANK_MODEL` | Local cross-encoder used for re-ranking. |
| `CHAT_MEMORY_TOKEN_LIMIT` | Sliding-window memory budget for chat history. |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`/`INFO`/`WARNING`/…). |

---

## ▶️ Running the Application

With the virtual environment active and `.env` configured, launch the dashboard from the project root:

```bash
streamlit run app.py
```

Streamlit will print a local URL (default <http://localhost:8501>). Open it in your browser.

To run on a custom port:

```bash
streamlit run app.py --server.port 8080
```

---

## 🧪 Usage Walkthrough

1. **Add your API key** to `.env` and restart the app if it was already running.
2. **Upload PDFs** in the sidebar using the file uploader (multiple files supported).
3. **Click "⚙️ Process Uploaded Documents."** Already-indexed files are skipped automatically to avoid duplicate token spend. Watch the toasts for per-file indexing confirmations.
4. **Open the "💬 Smart Study Chat" tab** and ask questions in natural language. Responses stream in token-by-token.
5. **Expand "🔍 Retained Source Citations & Layout Structures"** beneath any answer to inspect the exact source snippets — including reconstructed HTML tables — that grounded the response.
6. **Open the "📊 System & Pipeline Analytics" tab** to review documents indexed, total chunks, tables captured, chunk composition, the live pipeline configuration, and a per-document processing log.

---

## 🧰 Tech Stack

| Layer | Technology |
| --- | --- |
| **Orchestration** | LlamaIndex |
| **UI Framework** | Streamlit |
| **Ingestion** | Unstructured (`hi_res`, `pdf_infer_table_structure`, `chunk_by_title`) |
| **Vector Store** | ChromaDB (local, persistent) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Lexical Search** | BM25 (`rank-bm25`) |
| **Re-ranking** | FlashRank cross-encoder |
| **Config** | python-dotenv |

---

## 🛠 Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `IngestionError: ... missing executable` | `tesseract`/`poppler` not on `PATH` | Reinstall the system binaries (see [Prerequisites](#-prerequisites)) and verify with `tesseract --version` / `pdfinfo -v`. |
| `EnvironmentError: OPENAI_API_KEY is not configured` | `.env` still holds the placeholder | Paste a valid `sk-…` key into `.env` and restart. |
| `libmagic`-related import error | `libmagic` missing | macOS: `brew install libmagic`; Debian/Ubuntu: `sudo apt-get install -y libmagic1`. |
| Slow first response | FlashRank model downloading on first run | The cross-encoder is cached locally after the initial download; subsequent runs are fast. |
| Empty answers / "no documents" | Collection is empty | Upload and process at least one PDF in the sidebar. |
| Table renders as plain text | Source PDF table was image-only or low quality | Ensure `hi_res` is active and that OCR (`tesseract`) is installed; image-only tables depend on OCR quality. |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center"><em>Built with layout-aware ingestion, hybrid retrieval, and cross-encoder precision.</em></p>
