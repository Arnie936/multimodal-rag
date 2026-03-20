# Important Info — Scaling the Multimodal RAG Prototype

## Data Storage

- Images and videos are stored as **Base64 strings**
  directly in the `file_data` column of the
  Supabase DB — no paths or links to original sources
- Base64 inflates data by ~33%
- Audio bytes are stored as Base64 in `file_data`
  (same as images and videos)
- Text and PDF store the extracted text in
  `text_content`

## Supabase Storage Limits

- **Free plan**: 500 MB — sufficient for ~700 images
  or ~20–70 video chunks
- **Pro plan**: 8 GB included ($25/mo),
  then $0.125/GB
- Video-heavy usage fills the DB quickly / gets
  expensive

## Recommendation for Scaling

- Move binary data (images, videos, audio) to
  **Supabase Storage** or an S3 bucket
- Store only the **storage path/URL** + embedding
  in the DB
- On search: take the URL from the result, load the
  file from storage, and display it

## Performance

- **No HNSW index** possible — pgvector only supports
  HNSW up to 2000 dims, our embeddings have 3072
- Search is therefore **brute-force (exact)** and
  scales linearly with the number of documents
- With thousands of docs, search becomes noticeably
  slower

## Security

- `SUPABASE_SERVICE_KEY` bypasses Row Level
  Security — fine for a local prototype, but a
  security risk in public deployments

## Cost Pitfalls

- Every video/audio chunk = one Gemini Embedding
  API call
- Every search with reasoning = one additional
  LLM call
- API costs can rise quickly during bulk ingests

## Chunking Limitations

- Video chunks are fixed at 120s, audio at 75s — no
  intelligent splitting (e.g., at scene changes or
  silence)
- Relevant moments can be split at chunk boundaries
  and become harder to find
