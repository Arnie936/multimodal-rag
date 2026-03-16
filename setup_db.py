#!/usr/bin/env python3
"""
Supabase database setup script for Multimodal RAG.

Usage:
    python setup_db.py

Requires in .env:
    DATABASE_URL=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres

Get the DATABASE_URL from:
    Supabase Dashboard → Settings → Database → Connection string (URI)
"""

import os
import sys
import textwrap

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

MIGRATION_SQL = textwrap.dedent("""\
    -- ── 1. Extensions ────────────────────────────────────────────────────────────
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- ── 2. documents table (full schema, idempotent) ──────────────────────────────
    CREATE TABLE IF NOT EXISTS documents (
        id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        title             TEXT        NOT NULL,
        content_type      TEXT        NOT NULL,
        original_filename TEXT        NOT NULL,
        chunk_index       INT         NOT NULL DEFAULT 0,
        chunk_total       INT         NOT NULL DEFAULT 1,
        text_content      TEXT,
        metadata          JSONB       NOT NULL DEFAULT '{}',
        embedding         VECTOR(3072) NOT NULL,
        file_data         TEXT,
        user_id           UUID        REFERENCES auth.users(id) ON DELETE CASCADE,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    -- ── 3. Add user_id if missing (migration for existing tables) ─────────────────
    ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

    -- ── 4. Indexes ────────────────────────────────────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_documents_user_id    ON documents(user_id);
    CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);
    CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

    -- ── 5. user_settings table ────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id      UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
        llm_provider TEXT        NOT NULL DEFAULT 'gemini',
        llm_model    TEXT        NOT NULL DEFAULT 'gemini-2.0-flash-lite',
        llm_api_key  TEXT        NOT NULL DEFAULT '',
        ollama_url   TEXT        NOT NULL DEFAULT 'http://localhost:11434',
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    -- ── 6. Row Level Security ─────────────────────────────────────────────────────
    ALTER TABLE documents     ENABLE ROW LEVEL SECURITY;
    ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;

    -- ── 7. RLS Policies (drop-then-create for idempotency) ────────────────────────
    DROP POLICY IF EXISTS users_own_documents ON documents;
    CREATE POLICY users_own_documents ON documents
        FOR ALL
        USING     (auth.uid() = user_id)
        WITH CHECK (auth.uid() = user_id);

    DROP POLICY IF EXISTS users_own_settings ON user_settings;
    CREATE POLICY users_own_settings ON user_settings
        FOR ALL
        USING     (auth.uid() = user_id)
        WITH CHECK (auth.uid() = user_id);

    -- ── 8. match_documents RPC ────────────────────────────────────────────────────
    CREATE OR REPLACE FUNCTION match_documents(
        query_embedding  vector(3072),
        match_threshold  float,
        match_count      int,
        filter_type      text DEFAULT NULL
    )
    RETURNS TABLE (
        id                uuid,
        title             text,
        content_type      text,
        original_filename text,
        chunk_index       int,
        chunk_total       int,
        text_content      text,
        metadata          jsonb,
        file_data         text,
        similarity        float
    )
    LANGUAGE sql SECURITY INVOKER AS $$
        SELECT
            id, title, content_type, original_filename,
            chunk_index, chunk_total, text_content, metadata, file_data,
            1 - (embedding <=> query_embedding) AS similarity
        FROM documents
        WHERE 1 - (embedding <=> query_embedding) > match_threshold
          AND (filter_type IS NULL OR content_type = filter_type)
        ORDER BY embedding <=> query_embedding
        LIMIT match_count;
    $$;
""")


STEPS = [
    ("Extensions",              "CREATE EXTENSION IF NOT EXISTS vector"),
    ("documents table",         "CREATE TABLE IF NOT EXISTS documents"),
    ("user_id column",          "ADD COLUMN IF NOT EXISTS user_id"),
    ("Indexes",                 "CREATE INDEX IF NOT EXISTS idx_documents_user_id"),
    ("user_settings table",     "CREATE TABLE IF NOT EXISTS user_settings"),
    ("Row Level Security",      "ALTER TABLE documents     ENABLE ROW LEVEL SECURITY"),
    ("RLS Policies",            "DROP POLICY IF EXISTS users_own_documents"),
    ("match_documents RPC",     "CREATE OR REPLACE FUNCTION match_documents"),
]


def run():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print(
            "\nERROR: DATABASE_URL not set.\n\n"
            "Add it to your .env file:\n"
            "  DATABASE_URL=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres\n\n"
            "Find it at: Supabase Dashboard → Settings → Database → Connection string (URI)\n"
        )
        sys.exit(1)

    print("Connecting to Supabase PostgreSQL …")
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Could not connect to database:\n  {e}\n")
        sys.exit(1)

    print("Connected.\n")
    print("Running migrations …\n")

    cur = conn.cursor()
    try:
        cur.execute(MIGRATION_SQL)
    except psycopg2.Error as e:
        print(f"ERROR during migration:\n  {e}\n")
        conn.close()
        sys.exit(1)
    finally:
        cur.close()

    conn.close()

    print("  ✓ Extensions (pgvector, uuid-ossp)")
    print("  ✓ documents table created / verified")
    print("  ✓ user_id column added / verified")
    print("  ✓ Indexes created")
    print("  ✓ user_settings table created / verified")
    print("  ✓ Row Level Security enabled")
    print("  ✓ RLS policies applied")
    print("  ✓ match_documents() RPC created / updated")
    print("\nDatabase setup complete. You can now run: streamlit run app.py\n")


if __name__ == "__main__":
    run()
