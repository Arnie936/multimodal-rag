# Multimodal RAG with Gemini Embedding

A Retrieval-Augmented Generation application that
embeds multiple content types — text, images, video,
audio, and PDFs — using Google's Gemini Embedding 2
model, stores vectors in Supabase (pgvector), and
uses Gemini 3.1 Flash Lite for reasoning.
Built as a single Streamlit app with an optional MCP
server for programmatic access.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys
   (see `.env.example`):

   ```text
   GEMINI_API_KEY=your-key
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-key
   ```

3. Set up the Supabase database — run `schema.sql`
   in the Supabase SQL Editor (or via `psql`).
   This creates the `documents` table, enables
   pgvector, and adds the required RPC functions.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

## Features

### Upload & Embed

- Upload one or multiple files at once (text,
  Markdown, images, PDFs, audio, video)
- Assign each batch to a named **collection** for
  thematic grouping
- Files that exceed size limits are automatically
  chunked:
  - **Text / Markdown**: ~6000 token chunks with
    500 token overlap
  - **PDF**: 5-page chunks via PyMuPDF
  - **Audio**: 75-second segments via pydub
  - **Video**: 120-second segments via moviepy
- **Duplicate detection**: chunks already stored for
  a given filename are skipped, allowing interrupted
  uploads to resume without re-embedding
- Per-file progress bar and status text during
  embedding
- All embeddings are L2-normalized before storage

### Search

- Natural language queries embedded with
  `RETRIEVAL_QUERY` task type
- Vector similarity search via Supabase RPC
  (cosine distance)
- Configurable top-k (1–50) and similarity threshold
  (0.0–1.0, default 0.3)
- Filter results by content type and by collection
- Images and videos displayed inline in search
  results
- Optional reasoning via Gemini 3.1 Flash Lite with
  numbered source citations

### Browse

- View all stored documents in a table (ID, title,
  type, filename, chunk, collection, created)
- Delete all chunks of a file by filename
- Delete a single chunk by document ID

### Sidebar

- Live database statistics (total chunks, counts
  per content type)
- Collection filter applied to searches
- Stats refresh on demand

## Architecture

```text
app.py            Streamlit GUI
                  (upload, search, browse + sidebar)
mcp_server.py     MCP server exposing RAG search
lib/
├── gemini_client.py  Shared google-genai client
├── embedder.py       Gemini Embedding 2 (3072d)
├── chunker.py        Content-aware chunking
├── db.py             Supabase vector operations
├── rag.py            RAG pipeline orchestration
└── reasoning.py      Gemini Flash Lite reasoning
```

## MCP Server

`mcp_server.py` exposes four tools over the MCP
protocol (stdio transport), allowing any
MCP-compatible client (e.g. Claude Desktop) to query
the RAG index directly.

### Available tools

| Tool | Description |
| ---- | ----------- |
| `search_documents` | Semantic vector search; returns ranked excerpts |
| `search_and_reason` | Vector search + Gemini reasoning; returns answer |
| `list_collections` | Lists all available document collections |
| `document_stats` | Returns chunk count and per-type breakdown |

### Tool parameters (search tools)

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `query` | string | required | Search query |
| `top_k` | int | 10 | Max results (1–50) |
| `threshold` | float | 0.3 | Min similarity |
| `content_type` | string | `"all"` | Type filter |
| `collection` | string | `"all"` | Collection filter |

### Running the MCP server

```bash
python mcp_server.py
```

The server runs over stdio by default.
The `.mcp.json` in the project root pre-configures
it for use with Claude Desktop using the local
`.venv` interpreter.

## Tech Stack

| Component | Technology |
| --------- | ---------- |
| Embeddings | Gemini Embedding 2 Preview (3072d) |
| Reasoning | Gemini 3.1 Flash Lite |
| Vector DB | Supabase + pgvector |
| GUI | Streamlit |
| MCP server | FastMCP (stdio) |
| PDF | PyMuPDF |
| Audio | pydub |
| Video | moviepy |
