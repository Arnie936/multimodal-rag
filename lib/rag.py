from __future__ import annotations
import base64
import fitz
from lib import embedder, db, reasoning, chunker

MIME_MAP = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/webp": "image",
    "image/gif": "image",
    "application/pdf": "pdf",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "video/mp4": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
}

AUDIO_FMT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}

VIDEO_SUFFIX = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
}


def detect_content_type(mime: str, filename: str) -> str:
    if mime in MIME_MAP:
        return MIME_MAP[mime]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "txt":
        return "text"
    if ext in ("png", "jpg", "jpeg", "webp", "gif"):
        return "image"
    if ext == "pdf":
        return "pdf"
    if ext in ("mp3", "wav", "ogg"):
        return "audio"
    if ext in ("mp4", "mov", "avi"):
        return "video"
    return "text"


def ingest(
    file_bytes: bytes,
    filename: str,
    title: str,
    mime_type: str,
    on_progress=None,
) -> list[dict]:
    """Embed and store a file. Returns list of inserted document rows."""
    content_type = detect_content_type(mime_type, filename)
    existing = db.get_existing_chunks(filename)
    results = []

    def _progress(msg: str, current: int = 0, total: int = 1):
        if on_progress:
            on_progress(msg, current, total)

    if content_type == "text":
        text = file_bytes.decode("utf-8", errors="replace")
        chunks = chunker.chunk_text(text)
        total = len(chunks)
        for i, chunk_text in enumerate(chunks):
            if i in existing:
                _progress(f"Skipping text chunk {i+1}/{total} (exists)", i + 1, total)
                continue
            _progress(f"Embedding text chunk {i+1}/{total}", i + 1, total)
            vec = embedder.embed_text(chunk_text)
            row = db.insert_document(
                title=title,
                content_type="text",
                original_filename=filename,
                chunk_index=i,
                chunk_total=total,
                text_content=chunk_text,
                metadata={"char_count": len(chunk_text)},
                embedding=vec,
            )
            results.append(row)

    elif content_type == "image":
        if 0 in existing:
            _progress("Skipping image (exists)", 1, 1)
        else:
            _progress("Embedding image", 1, 1)
            vec = embedder.embed_image(file_bytes, mime_type=mime_type)
            b64 = base64.b64encode(file_bytes).decode("ascii")
            row = db.insert_document(
                title=title,
                content_type="image",
                original_filename=filename,
                chunk_index=0,
                chunk_total=1,
                text_content=None,
                metadata={"mime_type": mime_type, "size_bytes": len(file_bytes)},
                embedding=vec,
                file_data=b64,
            )
            results.append(row)

    elif content_type == "pdf":
        pdf_chunks = chunker.chunk_pdf(file_bytes)
        total = len(pdf_chunks)
        for i, pdf_bytes in enumerate(pdf_chunks):
            if i in existing:
                _progress(f"Skipping PDF chunk {i+1}/{total} (exists)", i + 1, total)
                continue
            _progress(f"Embedding PDF chunk {i+1}/{total}", i + 1, total)
            text = chunker.extract_pdf_text(pdf_bytes)
            page_count = len(fitz.open(stream=pdf_bytes, filetype="pdf"))
            vec = embedder.embed_pdf_page_bytes(pdf_bytes)
            row = db.insert_document(
                title=title,
                content_type="pdf",
                original_filename=filename,
                chunk_index=i,
                chunk_total=total,
                text_content=text[:10000] if text else None,
                metadata={"chunk_pages": page_count},
                embedding=vec,
            )
            results.append(row)

    elif content_type == "audio":
        fmt = AUDIO_FMT.get(mime_type, "mp3")
        audio_chunks = chunker.chunk_audio(file_bytes, fmt=fmt)
        total = len(audio_chunks)
        for i, chunk_bytes in enumerate(audio_chunks):
            if i in existing:
                _progress(f"Skipping audio chunk {i+1}/{total} (exists)", i + 1, total)
                continue
            _progress(f"Embedding audio chunk {i+1}/{total}", i + 1, total)
            vec = embedder.embed_audio(chunk_bytes, mime_type=mime_type)
            row = db.insert_document(
                title=title,
                content_type="audio",
                original_filename=filename,
                chunk_index=i,
                chunk_total=total,
                text_content=None,
                metadata={"format": fmt, "chunk_seconds": 75},
                embedding=vec,
            )
            results.append(row)

    elif content_type == "video":
        suffix = VIDEO_SUFFIX.get(mime_type, ".mp4")
        video_chunks = chunker.chunk_video(file_bytes, suffix=suffix)
        total = len(video_chunks)
        for i, chunk_bytes in enumerate(video_chunks):
            if i in existing:
                _progress(f"Skipping video chunk {i+1}/{total} (exists)", i + 1, total)
                continue
            _progress(f"Embedding video chunk {i+1}/{total}", i + 1, total)
            vec = embedder.embed_video(chunk_bytes, mime_type=mime_type)
            b64 = base64.b64encode(chunk_bytes).decode("ascii")
            row = db.insert_document(
                title=title,
                content_type="video",
                original_filename=filename,
                chunk_index=i,
                chunk_total=total,
                text_content=None,
                metadata={"format": suffix, "chunk_seconds": 120, "mime_type": mime_type},
                embedding=vec,
                file_data=b64,
            )
            results.append(row)

    return results


def query(
    query_text: str,
    top_k: int = 10,
    threshold: float = 0.5,
    filter_type: str | None = None,
    use_reasoning: bool = True,
) -> dict:
    """Run the full RAG pipeline: embed query -> search -> reason."""
    query_vec = embedder.embed_query(query_text)
    matches = db.search_documents(
        query_embedding=query_vec,
        match_threshold=threshold,
        match_count=top_k,
        filter_type=filter_type if filter_type != "all" else None,
    )
    answer = None
    if use_reasoning and matches:
        answer = reasoning.reason(query_text, matches)
    return {"answer": answer, "sources": matches}
