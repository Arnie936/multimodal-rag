import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def reason(query: str, context_chunks: list[dict]) -> str:
    """Send query + retrieved context to OpenAI o4-mini for reasoning."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("original_filename", "unknown")
        ctype = chunk.get("content_type", "unknown")
        sim = chunk.get("similarity", 0)
        text = chunk.get("text_content") or "(non-text content)"
        context_parts.append(
            f"[Source {i}] {source} ({ctype}, similarity: {sim:.3f})\n{text}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Cite your sources by referencing the source numbers [Source N]. "
                "If the context doesn't contain enough information, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}",
        },
    ]

    response = get_client().chat.completions.create(
        model="o4-mini",
        messages=messages,
    )
    return response.choices[0].message.content
