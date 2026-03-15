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
    collection: str = "default",
) -> dict:
    row = {
        "title": title,
        "content_type": content_type,
        "original_filename": original_filename,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "text_content": text_content.replace("\x00", "") if text_content else text_content,
        "metadata": metadata,
        "embedding": embedding,
        "file_data": file_data,
        "collection": collection,
    }
    result = get_client().table("documents").insert(row).execute()
    return result.data[0]


def search_documents(
    query_embedding: list[float],
    match_threshold: float = 0.5,
    match_count: int = 10,
    filter_type: str | None = None,
    filter_collection: str | None = None,
) -> list[dict]:
    result = get_client().rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_type": filter_type,
            "filter_collection": filter_collection,
        },
    ).execute()
    return result.data


def get_all_documents() -> list[dict]:
    result = (
        get_client()
        .table("documents")
        .select("id, title, content_type, original_filename, chunk_index, chunk_total, collection, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def get_collections() -> list[str]:
    """Return sorted list of distinct collection names."""
    result = (
        get_client()
        .table("documents")
        .select("collection")
        .execute()
    )
    return sorted(set(r["collection"] for r in result.data))


def get_existing_chunks(original_filename: str) -> set[int]:
    """Return set of chunk_indices already stored for this filename."""
    result = (
        get_client()
        .table("documents")
        .select("chunk_index")
        .eq("original_filename", original_filename)
        .execute()
    )
    return {r["chunk_index"] for r in result.data}


def delete_document(doc_id: str) -> None:
    get_client().table("documents").delete().eq("id", doc_id).execute()


def delete_by_filename(original_filename: str) -> int:
    """Delete all chunks for a given filename. Returns count of deleted rows."""
    result = (
        get_client()
        .table("documents")
        .delete()
        .eq("original_filename", original_filename)
        .execute()
    )
    return len(result.data)


def get_stats() -> dict:
    rows = get_all_documents()
    total = len(rows)
    by_type: dict[str, int] = {}
    for r in rows:
        ct = r["content_type"]
        by_type[ct] = by_type.get(ct, 0) + 1
    return {"total": total, "by_type": by_type}
