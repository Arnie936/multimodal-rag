# Multimodal RAG with Gemini Embedding

A RAG application that embeds text, images, video, audio, and PDFs using Google's Gemini Embedding model, stores vectors in Supabase (pgvector), and uses OpenAI o4-mini for reasoning.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Fill in your API keys in `.env`:
   ```
   GEMINI_API_KEY=your-key
   OPENAI_API_KEY=your-key
   SUPABASE_SERVICE_KEY=your-key
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

## Features

- **Upload & Embed** — Upload text, images, PDFs, audio, or video. Files are automatically chunked if they exceed size limits and embedded via Gemini.
- **Search** — Enter a natural language query. Results are retrieved by vector similarity and optionally passed to OpenAI o4-mini for a synthesized answer with source citations.
- **Browse** — View and delete all stored documents.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Gemini Embedding (768 dims) |
| Vector DB | Supabase + pgvector |
| Reasoning | OpenAI o4-mini |
| GUI | Streamlit |
| PDF | PyMuPDF |
| Audio | pydub |
| Video | moviepy |
