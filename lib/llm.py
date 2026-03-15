from __future__ import annotations
from typing import Protocol, runtime_checkable


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Cite your sources by referencing the source numbers [Source N]. "
    "If the context doesn't contain enough information, say so clearly."
)


def _build_user_message(query: str, context_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("original_filename", "unknown")
        ctype = chunk.get("content_type", "unknown")
        sim = chunk.get("similarity", 0)
        text = chunk.get("text_content") or "(non-text content)"
        parts.append(f"[Source {i}] {source} ({ctype}, similarity: {sim:.3f})\n{text}")
    context_str = "\n\n---\n\n".join(parts)
    return f"Context:\n{context_str}\n\nQuestion: {query}"


@runtime_checkable
class LLMProvider(Protocol):
    def reason(self, query: str, context_chunks: list[dict]) -> str: ...


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "o4-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def reason(self, query: str, context_chunks: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, context_chunks)},
            ],
        )
        return response.choices[0].message.content


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def reason(self, query: str, context_chunks: list[dict]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query, context_chunks)}],
        )
        return response.content[0].text


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def reason(self, query: str, context_chunks: list[dict]) -> str:
        from google.genai import types
        response = self._client.models.generate_content(
            model=self.model,
            contents=_build_user_message(query, context_chunks),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return response.text


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def reason(self, query: str, context_chunks: list[dict]) -> str:
        import httpx
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, context_chunks)},
            ],
        }
        response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_provider(settings: dict) -> LLMProvider:
    """Factory: settings = {provider, model, api_key, ollama_url}"""
    provider = settings.get("provider", "gemini")
    model = settings.get("model", "")
    api_key = settings.get("api_key", "")

    if provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model or "o4-mini")
    if provider == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-5")
    if provider == "gemini":
        return GeminiProvider(api_key=api_key, model=model or "gemini-2.0-flash-lite")
    if provider == "ollama":
        return OllamaProvider(
            base_url=settings.get("ollama_url", "http://localhost:11434"),
            model=model or "llama3",
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")
