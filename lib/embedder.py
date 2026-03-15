import os
import time
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-embedding-2-preview"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 60  # seconds

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _normalize(vec: list[float]) -> list[float]:
    a = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm > 0:
        a = a / norm
    return a.tolist()


def _embed_with_retry(contents, task_type: str) -> list[float]:
    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().models.embed_content(
                model=MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return _normalize(response.embeddings[0].values)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)  # 60s, 120s, 240s
            print(f"Embedding failed ({e}), retrying in {delay}s...")
            time.sleep(delay)


def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    return _embed_with_retry(text, task_type)


def embed_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return _embed_with_retry(part, task_type)


def embed_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/mp3",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    return _embed_with_retry(part, task_type)


def embed_video(
    video_bytes: bytes,
    mime_type: str = "video/mp4",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
    return _embed_with_retry(part, task_type)


def embed_pdf_page_bytes(
    pdf_bytes: bytes,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    return _embed_with_retry(part, task_type)


def embed_query(text: str) -> list[float]:
    return embed_text(text, task_type="RETRIEVAL_QUERY")
