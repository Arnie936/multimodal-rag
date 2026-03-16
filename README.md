# Multimodal RAG with Gemini Embedding

A SaaS-ready Retrieval-Augmented Generation application that embeds multiple content types — text, images, video, audio, and PDFs — using Google's Gemini Embedding 2 model, stores vectors in Supabase (pgvector), and supports multiple LLM providers for reasoning. Built as a single Streamlit app with multi-tenant authentication.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your-key
   OPENAI_API_KEY=your-key          # optional, for OpenAI provider
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-service-role-key
   SUPABASE_ANON_KEY=your-anon-key  # required for user auth
   ```

3. Add `DATABASE_URL` to your `.env` file (needed once for the setup script):

   ```
   DATABASE_URL=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
   ```

   > Find it at: **Supabase Dashboard → Settings → Database → Connection string (URI)**

4. Run the database setup script (creates all tables, indexes, RLS policies, and the RPC):

   ```bash
   python setup_db.py
   ```

   The script is idempotent — safe to run multiple times.

5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features

### Authentication
- User registration and login via Supabase Auth
- Each user's documents are fully isolated via Row Level Security
- LLM settings are stored and persisted per user

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
- Images and videos are displayed inline in search results
- Optional reasoning via your configured LLM provider with source citations

### Browse
- View all your stored documents in a table
- Delete documents by ID

### LLM Settings
Configure your preferred reasoning provider in the sidebar — settings are saved per user:

| Provider | Default model | Notes |
|----------|--------------|-------|
| Gemini | `gemini-2.0-flash-lite` | Uses `GEMINI_API_KEY` if no key entered |
| OpenAI | `o4-mini` | Requires API key |
| Anthropic | `claude-sonnet-4-5` | Requires API key |
| Ollama | `llama3` | Local — no API key needed, configure URL |

## Architecture

```
app.py              Streamlit GUI (auth gate, upload, search, browse tabs)
lib/
├── embedder.py     Gemini Embedding 2 (3072 dims) for all content types
├── chunker.py      Content-aware chunking (text, PDF, audio, video)
├── db.py           Supabase vector operations + user_settings CRUD
├── rag.py          RAG pipeline orchestration (ingest + query)
├── auth.py         Supabase Auth wrapper (login, register, JWT client)
├── llm.py          Multi-provider LLM abstraction (factory + implementations)
└── codex.py        Backward-compatible reasoning wrapper → llm.GeminiProvider
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Gemini Embedding 2 Preview (3072 dims) |
| Vector DB | Supabase + pgvector |
| Auth | Supabase Auth + Row Level Security |
| Reasoning | OpenAI / Anthropic / Gemini / Ollama (user-configurable) |
| GUI | Streamlit |
| PDF | PyMuPDF |
| Audio | pydub |
| Video | moviepy |
