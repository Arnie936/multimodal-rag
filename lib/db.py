import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )
    return _client


def insert_document(
    title: str,
    content_type: str,
    original_filename: str,
    chunk_index: int,
    chunk_total: int,
    text_content: str | None,
    metadata: dict,
    embedding: list[float],
    file_data: str | None = None,
    user_id: str | None = None,
    authed_client: Client | None = None,
) -> dict:
    client = authed_client or get_client()
    row = {
        "title": title,
        "content_type": content_type,
        "original_filename": original_filename,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "text_content": text_content,
        "metadata": metadata,
        "embedding": embedding,
        "file_data": file_data,
        "user_id": user_id,
    }
    result = client.table("documents").insert(row).execute()
    return result.data[0]


def search_documents(
    query_embedding: list[float],
    match_threshold: float = 0.5,
    match_count: int = 10,
    filter_type: str | None = None,
    authed_client: Client | None = None,
) -> list[dict]:
    client = authed_client or get_client()
    result = client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_type": filter_type,
        },
    ).execute()
    return result.data


def get_all_documents(authed_client: Client | None = None) -> list[dict]:
    client = authed_client or get_client()
    result = (
        client
        .table("documents")
        .select("id, title, content_type, original_filename, chunk_index, chunk_total, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def delete_document(doc_id: str, authed_client: Client | None = None) -> None:
    client = authed_client or get_client()
    client.table("documents").delete().eq("id", doc_id).execute()


def get_stats(authed_client: Client | None = None) -> dict:
    rows = get_all_documents(authed_client=authed_client)
    total = len(rows)
    by_type: dict[str, int] = {}
    for r in rows:
        ct = r["content_type"]
        by_type[ct] = by_type.get(ct, 0) + 1
    return {"total": total, "by_type": by_type}


def get_user_settings(authed_client: Client, user_id: str) -> dict | None:
    result = (
        authed_client
        .table("user_settings")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def upsert_user_settings(authed_client: Client, user_id: str, settings: dict) -> dict:
    row = {
        "user_id": user_id,
        "llm_provider": settings.get("provider", "gemini"),
        "llm_model": settings.get("model", "gemini-2.0-flash-lite"),
        "llm_api_key": settings.get("api_key", ""),
        "ollama_url": settings.get("ollama_url", "http://localhost:11434"),
    }
    result = (
        authed_client
        .table("user_settings")
        .upsert(row, on_conflict="user_id")
        .execute()
    )
    return result.data[0]
