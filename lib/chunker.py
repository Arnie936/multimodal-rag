from __future__ import annotations
import io
import tempfile
import os
import fitz  # PyMuPDF


def chunk_text(text: str, max_tokens: int = 6000, overlap: int = 500) -> list[str]:
    """Split text into chunks by approximate token count (1 token ~ 4 chars)."""
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap_chars
    return chunks


def chunk_pdf(pdf_bytes: bytes, max_pages: int = 5) -> list[bytes]:
    """Split a PDF into sub-PDFs of at most max_pages pages. Returns list of PDF bytes."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        total = len(src)
        if total <= max_pages:
            return [pdf_bytes]
        chunks = []
        for start in range(0, total, max_pages):
            end = min(start + max_pages, total)
            with fitz.open() as dst:
                dst.insert_pdf(src, from_page=start, to_page=end - 1)
                chunks.append(dst.tobytes())
        return chunks


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract all text and page count from a PDF."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
        return text, len(doc)


def chunk_audio(audio_bytes: bytes, fmt: str = "mp3", max_seconds: int = 75) -> list[bytes]:
    """Split audio into segments of max_seconds. Returns list of audio bytes."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    duration_ms = len(seg)
    max_ms = max_seconds * 1000
    if duration_ms <= max_ms:
        return [audio_bytes]
    chunks = []
    for start_ms in range(0, duration_ms, max_ms):
        end_ms = min(start_ms + max_ms, duration_ms)
        part = seg[start_ms:end_ms]
        buf = io.BytesIO()
        part.export(buf, format=fmt)
        chunks.append(buf.getvalue())
    return chunks


def chunk_video(video_bytes: bytes, suffix: str = ".mp4", max_seconds: int = 120) -> list[bytes]:
    """Split video into segments of max_seconds. Returns list of video bytes."""
    from moviepy import VideoFileClip

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_in.write(video_bytes)
    tmp_in.close()

    try:
        clip = VideoFileClip(tmp_in.name)
        duration = clip.duration
        if duration <= max_seconds:
            clip.close()
            return [video_bytes]
        chunks = []
        t = 0
        while t < duration:
            end = min(t + max_seconds, duration)
            sub = clip.subclipped(t, end)
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_out_name = tmp_out.name
            tmp_out.close()
            try:
                sub.write_videofile(tmp_out_name, logger=None)
                with open(tmp_out_name, "rb") as f:
                    chunks.append(f.read())
            finally:
                os.unlink(tmp_out_name)
            sub.close()
            t = end
        clip.close()
        return chunks
    finally:
        os.unlink(tmp_in.name)
