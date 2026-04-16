-- Schema for multimodal-rag
-- Run this once against a fresh Supabase project.

-- 1. Enable pgvector
create extension if not exists vector with schema extensions;

-- 2. Documents table
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  content_type text not null,
  original_filename text not null,
  chunk_index integer not null default 0,
  chunk_total integer not null default 1,
  text_content text,
  metadata jsonb default '{}'::jsonb,
  embedding vector(3072) not null,
  file_data text,
  collection text not null default 'default',
  created_at timestamptz not null default now()
);

-- 3. Row Level Security
alter table public.documents enable row level security;

create policy "Service key full access"
  on public.documents
  for all
  using (true)
  with check (true);

-- 4. Vector search RPC
create or replace function public.match_documents(
  query_embedding vector(3072),
  match_threshold float default 0.5,
  match_count int default 10,
  filter_type text default null,
  filter_collection text default null
)
returns table (
  id uuid,
  title text,
  content_type text,
  original_filename text,
  chunk_index integer,
  chunk_total integer,
  text_content text,
  metadata jsonb,
  file_data text,
  collection text,
  similarity float
)
language plpgsql stable as $$
begin
  return query
  select
    d.id,
    d.title,
    d.content_type,
    d.original_filename,
    d.chunk_index,
    d.chunk_total,
    d.text_content,
    d.metadata,
    d.file_data,
    d.collection,
    1 - (d.embedding <=> query_embedding)::float as similarity
  from public.documents d
  where
    (filter_type is null or d.content_type = filter_type)
    and (filter_collection is null
         or d.collection = filter_collection)
    and 1 - (d.embedding <=> query_embedding)
        >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 5. Full-text search column and index
alter table public.documents
  add column if not exists fts tsvector
  generated always as (
    to_tsvector('english',
      coalesce(text_content, '') || ' '
      || coalesce(title, ''))
  ) stored;

create index if not exists idx_documents_fts
  on public.documents using gin(fts);

-- 6. Hybrid search RPC (vector + full-text)
create or replace function public.hybrid_search(
  query_text text,
  query_embedding vector(3072),
  match_threshold float default 0.3,
  match_count int default 10,
  filter_type text default null,
  filter_collection text default null,
  vector_weight float default 0.7,
  text_weight float default 0.3
)
returns table (
  id uuid,
  title text,
  content_type text,
  original_filename text,
  chunk_index integer,
  chunk_total integer,
  text_content text,
  metadata jsonb,
  file_data text,
  collection text,
  similarity float,
  text_rank float,
  hybrid_score float
)
language plpgsql stable as $$
declare
  ts_query tsquery;
begin
  ts_query := plainto_tsquery('english', query_text);
  return query
  select
    d.id,
    d.title,
    d.content_type,
    d.original_filename,
    d.chunk_index,
    d.chunk_total,
    d.text_content,
    d.metadata,
    d.file_data,
    d.collection,
    (1 - (d.embedding <=> query_embedding))::float
      as similarity,
    coalesce(ts_rank_cd(d.fts, ts_query), 0)::float
      as text_rank,
    (
      vector_weight
        * (1 - (d.embedding <=> query_embedding))
      + text_weight
        * coalesce(ts_rank_cd(d.fts, ts_query), 0)
    )::float as hybrid_score
  from public.documents d
  where
    (filter_type is null
     or d.content_type = filter_type)
    and (filter_collection is null
         or d.collection = filter_collection)
    and (
      1 - (d.embedding <=> query_embedding)
        >= match_threshold
      or d.fts @@ ts_query
    )
  order by hybrid_score desc
  limit match_count;
end;
$$;

-- 7. Distinct collections RPC
create or replace function public.get_distinct_collections()
returns table(collection text)
language sql stable as $$
  select distinct d.collection
  from public.documents d
  order by d.collection;
$$;

-- 6. Document stats RPC
create or replace function public.get_document_stats()
returns table(content_type text, cnt bigint)
language sql stable as $$
  select d.content_type, count(*) as cnt
  from public.documents d
  group by d.content_type;
$$;
