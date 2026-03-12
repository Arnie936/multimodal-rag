import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-embedding-2-preview"

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


def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    response = get_client().models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = get_client().models.embed_content(
        model=MODEL,
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/mp3",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    response = get_client().models.embed_content(
        model=MODEL,
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_video(
    video_bytes: bytes,
    mime_type: str = "video/mp4",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
    response = get_client().models.embed_content(
        model=MODEL,
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_pdf_page_bytes(
    pdf_bytes: bytes,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    response = get_client().models.embed_content(
        model=MODEL,
        contents=part,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return _normalize(response.embeddings[0].values)


def embed_query(text: str) -> list[float]:
    return embed_text(text, task_type="RETRIEVAL_QUERY")
