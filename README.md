# Multimodal RAG with Gemini Embedding

A Retrieval-Augmented Generation application that embeds multiple content types — text, images, video, audio, and PDFs — using Google's Gemini Embedding 2 model, stores vectors in Supabase (pgvector), and uses Gemini 3.1 Flash Lite for reasoning. Built as a single Streamlit app.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your-key
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-key
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features

### Upload & Embed
- Upload one or multiple files at once (text, images, PDFs, audio, video)
- Files that exceed size limits are automatically chunked:
  - **Text**: ~6000 token chunks with 500 token overlap
  - **PDF**: 5-page chunks via PyMuPDF
  - **Audio**: 75-second segments via pydub
  - **Video**: 120-second segments via moviepy
- All embeddings are L2-normalized before storage

### Search
- Natural language queries embedded with `RETRIEVAL_QUERY` task type
- Vector similarity search via Supabase RPC (cosine distance)
- Configurable top-k and similarity threshold
- Filter results by content type
- Images are displayed inline in search results
- Optional reasoning via Gemini 3.1 Flash Lite with source citations

### Browse
- View all stored documents in a table
- Delete documents by ID

## Architecture

```
app.py              Streamlit GUI (upload, search, browse tabs)
lib/
├── embedder.py     Gemini Embedding 2 (3072 dims) for all content types
├── chunker.py      Content-aware chunking (text, PDF, audio, video)
├── db.py           Supabase vector operations (insert, search, stats)
├── rag.py          RAG pipeline orchestration (ingest + query)
└── reasoning.py    Gemini 3.1 Flash Lite reasoning with source citations
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Gemini Embedding 2 Preview (3072 dims) |
| Vector DB | Supabase + pgvector |
| Reasoning | Gemini 3.1 Flash Lite |
| GUI | Streamlit |
| PDF | PyMuPDF |
| Audio | pydub |
| Video | moviepy |
