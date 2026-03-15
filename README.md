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

3. Run the Supabase migrations (Supabase Dashboard → SQL Editor):

   ```sql
   -- Add user_id to documents
   ALTER TABLE documents
     ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
   CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

   -- User settings table
   CREATE TABLE IF NOT EXISTS user_settings (
     user_id      UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
     llm_provider TEXT NOT NULL DEFAULT 'gemini',
     llm_model    TEXT NOT NULL DEFAULT 'gemini-2.0-flash-lite',
     llm_api_key  TEXT NOT NULL DEFAULT '',
     ollama_url   TEXT NOT NULL DEFAULT 'http://localhost:11434',
     updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
   );

   -- Enable Row Level Security
   ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
   ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;

   CREATE POLICY "users_own_documents" ON documents
     FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
   CREATE POLICY "users_own_settings" ON user_settings
     FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

   -- Update RPC to respect RLS
   CREATE OR REPLACE FUNCTION match_documents(
     query_embedding vector(3072), match_threshold float,
     match_count int, filter_type text DEFAULT NULL
   )
   RETURNS TABLE (id uuid, title text, content_type text, original_filename text,
     chunk_index int, chunk_total int, text_content text,
     metadata jsonb, file_data text, similarity float)
   LANGUAGE sql SECURITY INVOKER AS $$
     SELECT id, title, content_type, original_filename,
            chunk_index, chunk_total, text_content, metadata, file_data,
            1 - (embedding <=> query_embedding) AS similarity
     FROM documents
     WHERE 1 - (embedding <=> query_embedding) > match_threshold
       AND (filter_type IS NULL OR content_type = filter_type)
     ORDER BY embedding <=> query_embedding LIMIT match_count;
   $$;
   ```

4. Run the app:
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
