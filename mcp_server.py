"""MCP server exposing RAG search over embedded documents."""
from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP
from lib import rag, db

mcp = FastMCP("multimodal-rag")


@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 10,
    threshold: float = 0.3,
    content_type: str = "all",
    collection: str = "all",
) -> str:
    """Search embedded documents using semantic vector search.

    Args:
        query: Natural language search query
        top_k: Maximum number of results to return (1-50)
        threshold: Minimum similarity threshold (0.0-1.0)
        content_type: Filter by type: all, text, image, pdf, audio, video
        collection: Filter by collection name, or "all" for all collections
    """
    result = rag.query(
        query_text=query,
        top_k=top_k,
        threshold=threshold,
        filter_type=content_type,
        filter_collection=collection,
        use_reasoning=False,
    )

    sources = result["sources"]
    if not sources:
        return "No matching documents found."

    parts = []
    for src in sources:
        sim = src.get("similarity", 0)
        title = src.get("title", "")
        filename = src.get("original_filename", "")
        ctype = src.get("content_type", "")
        chunk = src.get("chunk_index", 0)
        total = src.get("chunk_total", 1)
        collection_name = src.get("collection", "")
        text = src.get("text_content") or "(non-text content)"

        parts.append(
            f"[{sim:.3f}] {title} — {filename} "
            f"(chunk {chunk}/{total}, {ctype}, collection: {collection_name})\n"
            f"{text[:3000]}"
        )

    return f"Found {len(sources)} results:\n\n" + "\n\n---\n\n".join(parts)


@mcp.tool()
def search_and_reason(
    query: str,
    top_k: int = 10,
    threshold: float = 0.3,
    content_type: str = "all",
    collection: str = "all",
) -> str:
    """Search documents and generate a reasoned answer using Gemini Flash Lite.

    Args:
        query: Natural language question
        top_k: Maximum number of source documents (1-50)
        threshold: Minimum similarity threshold (0.0-1.0)
        content_type: Filter by type: all, text, image, pdf, audio, video
        collection: Filter by collection name, or "all" for all collections
    """
    result = rag.query(
        query_text=query,
        top_k=top_k,
        threshold=threshold,
        filter_type=content_type,
        filter_collection=collection,
        use_reasoning=True,
    )

    answer = result.get("answer") or "No answer generated."
    sources = result["sources"]

    source_list = "\n".join(
        f"- [{s.get('similarity', 0):.3f}] {s.get('title', '')} — {s.get('original_filename', '')} "
        f"(chunk {s.get('chunk_index', 0)}/{s.get('chunk_total', 1)}, {s.get('collection', '')})"
        for s in sources
    )

    return f"Answer:\n{answer}\n\nSources ({len(sources)}):\n{source_list}"


@mcp.tool()
def list_collections() -> str:
    """List all available document collections."""
    collections = db.get_collections()
    if not collections:
        return "No collections found."
    return "Collections:\n" + "\n".join(f"- {c}" for c in collections)


@mcp.tool()
def document_stats() -> str:
    """Show database statistics: total documents and counts per content type."""
    stats = db.get_stats()
    lines = [f"Total chunks: {stats['total']}"]
    for ctype, count in stats["by_type"].items():
        lines.append(f"- {ctype}: {count}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
