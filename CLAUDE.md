# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal RAG application that embeds text, images, video, audio, and PDFs using Google's Gemini Embedding model, stores vectors in Supabase (pgvector), and uses OpenAI o4-mini for reasoning. Single Streamlit monolith — no separate backend.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Architecture

- **app.py** — Streamlit GUI with three tabs: Upload & Embed (multi-file), Search (with image display), Browse (with delete)
- **lib/embedder.py** — Wraps `google-genai` SDK. Uses `gemini-embedding-2-preview` model (3072 dims, L2-normalized). All content types go through `embed_content()` with `Part.from_bytes` for binary data. Query embeddings use `task_type="RETRIEVAL_QUERY"`, documents use `"RETRIEVAL_DOCUMENT"`.
- **lib/chunker.py** — Splits oversized content: text (~6000 token chunks), PDF (5-page chunks via PyMuPDF), audio (75s via pydub), video (120s via moviepy)
- **lib/db.py** — Supabase client. Inserts documents with embeddings, runs vector search via `match_documents()` RPC, stats, delete.
- **lib/rag.py** — Orchestrates ingest (detect type → chunk → embed → store) and query (embed query → vector search → Codex reasoning). Images are stored as base64 in `file_data` column.
- **lib/codex.py** — Sends query + retrieved context to OpenAI o4-mini with source citation instructions.

## Database

Supabase project `lgllivbqhqmpkcbmacyy`. The `documents` table has a `vector(3072)` embedding column (no HNSW — pgvector's limit is 2000 dims, so exact search is used). The `match_documents()` RPC performs cosine similarity search with optional content type filtering. Images are stored as base64 in the `file_data` TEXT column.

## Environment

Requires `.env` with: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.
