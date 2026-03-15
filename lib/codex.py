import os
from lib.llm import GeminiProvider


def reason(query: str, context_chunks: list[dict]) -> str:
    """Backward-compatible wrapper — uses Gemini as default provider."""
    provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    return provider.reason(query, context_chunks)
