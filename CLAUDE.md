# CLAUDE.md

This file provides guidance to Claude Code
(claude.ai/code) when working with code in this
repository.

## Project Overview

Multimodal RAG application that embeds text, images,
video, audio, and PDFs using Google's Gemini Embedding
model, stores vectors in Supabase (pgvector), and uses
Gemini 3.1 Flash Lite for reasoning. Single Streamlit
monolith — no separate backend. Only Google APIs are
used (no OpenAI dependency).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run the MCP server (stdio transport)
python mcp_server.py
```

## Architecture

- **app.py** — Streamlit GUI with three tabs:
  Upload & Embed (multi-file, per-collection),
  Search (with image/video display, collection
  filter), Browse (table view, delete by filename
  or ID). Sidebar shows live DB stats and settings.
- **mcp_server.py** — FastMCP server (stdio) exposing
  four tools: `search_documents`,
  `search_and_reason`, `list_collections`,
  `document_stats`. Configured in `.mcp.json` for
  Claude Desktop.
- **lib/gemini_client.py** — Module-level singleton
  (`get_client()`) for `google.genai.Client`. Both
  `embedder.py` and `reasoning.py` import from here
  to share one authenticated client.
- **lib/embedder.py** — Wraps `google-genai` SDK.
  Uses `gemini-embedding-2-preview` model (3072 dims,
  L2-normalized). All content types go through
  `embed_content()` with `Part.from_bytes` for binary
  data. Query embeddings use
  `task_type="RETRIEVAL_QUERY"`, documents use
  `"RETRIEVAL_DOCUMENT"`. Retries on
  `ResourceExhausted`, `ServiceUnavailable`,
  `DeadlineExceeded`, `InternalServerError`, and
  `ConnectionError` with exponential backoff (base
  60 s, up to 3 attempts).
- **lib/chunker.py** — Splits oversized content:
  text (~6000 token chunks with 500 token overlap),
  PDF (5-page sub-PDFs via PyMuPDF), audio (75 s via
  pydub), video (120 s via moviepy). Also exposes
  `extract_pdf_text()` for text extraction alongside
  embedding.
- **lib/db.py** — Supabase client singleton. Key
  functions: `insert_document()`,
  `search_documents()` (via `match_documents()` RPC),
  `get_all_documents()`, `get_collections()`,
  `get_existing_chunks()` (duplicate detection),
  `delete_document()` (by ID),
  `delete_by_filename()` (batch delete),
  `get_stats()`.
- **lib/rag.py** — Orchestrates ingest (detect type
  → get existing chunks → chunk → embed → store,
  skipping duplicates) and query (embed query →
  vector search → optional reasoning). Accepts
  `on_progress` callback for UI progress bars.
  Images and videos are stored as base64 in the
  `file_data` column.
- **lib/reasoning.py** — Sends query + retrieved
  context chunks to Gemini 3.1 Flash Lite
  (`gemini-3.1-flash-lite-preview`) via
  `get_client().models.generate_content()` with a
  system instruction that requires numbered source
  citations.

## Database

Supabase project `lgllivbqhqmpkcbmacyy`.
The `documents` table schema:

| Column | Type | Notes |
| ------ | ---- | ----- |
| `id` | uuid | Primary key |
| `title` | text | User-supplied title |
| `content_type` | text | `text`, `image`, `pdf`, `audio`, `video` |
| `original_filename` | text | Used for duplicate detection and batch delete |
| `chunk_index` | int | 0-based chunk position |
| `chunk_total` | int | Total chunks for this file |
| `text_content` | text | Extracted text (NULL for binary-only) |
| `file_data` | text | Base64-encoded binary (images/videos) |
| `metadata` | jsonb | Type-specific metadata (mime, size, etc.) |
| `embedding` | vector(3072) | L2-normalized; exact search (no HNSW) |
| `collection` | text | Collection name, default `"default"` |
| `created_at` | timestamptz | Auto-set |

The `match_documents()` RPC performs cosine similarity
search with optional `filter_type` and
`filter_collection` parameters.

## Environment

Requires `.env` with: `GEMINI_API_KEY`,
`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.

## Key Design Decisions

- **No HNSW index**: pgvector's HNSW is limited to
  2000 dimensions; at 3072 dims the query falls back
  to exact (sequential) scan.
- **Duplicate detection**: `get_existing_chunks()`
  queries by `original_filename` before embedding;
  already-stored chunk indices are skipped. This
  allows resuming interrupted uploads without
  re-embedding.
- **Shared Gemini client**: `gemini_client.py`
  prevents multiple `google.genai.Client` instances
  from being created across modules.
- **`mcp` package**: `mcp_server.py` requires the
  `mcp` package (FastMCP). It is not listed in
  `requirements.txt` — install separately with
  `pip install mcp` if using the MCP server.
