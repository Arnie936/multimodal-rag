import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def reason(query: str, context_chunks: list[dict]) -> str:
    """Send query + retrieved context to Gemini Flash Lite for reasoning."""
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

    system_instruction = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Cite your sources by referencing the source numbers [Source N]. "
        "If the context doesn't contain enough information, say so clearly."
    )

    response = get_client().models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=f"Context:\n{context_str}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    return response.text
