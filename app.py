import base64
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from lib import db, rag

st.set_page_config(page_title="Multimodal RAG", page_icon="🔍", layout="wide")
st.title("Multimodal RAG with Gemini Embedding")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K results", 1, 50, 10)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.05)
    filter_type = st.selectbox(
        "Content type filter",
        ["all", "text", "image", "pdf", "audio", "video"],
    )

    @st.cache_data(ttl=60)
    def _collections():
        return db.get_collections()

    try:
        collections = _collections()
    except Exception:
        collections = []
    filter_collection = st.selectbox(
        "Collection filter",
        ["all"] + collections,
    )

    use_reasoning = st.checkbox("Use reasoning", value=True)

    st.divider()
    st.header("Database Stats")
    if st.button("Refresh stats"):
        st.cache_data.clear()

    @st.cache_data(ttl=30)
    def _stats():
        return db.get_stats()

    try:
        stats = _stats()
        st.metric("Total documents", stats["total"])
        for ctype, count in stats["by_type"].items():
            st.metric(ctype.capitalize(), count)
    except Exception as e:
        st.error(f"Could not load stats: {e}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_search, tab_browse = st.tabs(["Upload & Embed", "Search", "Browse"])

# ── Tab 1: Upload & Embed ───────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload files to embed")
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["txt", "md", "png", "jpg", "jpeg", "webp", "gif", "pdf", "mp3", "wav", "mp4", "mov", "avi"],
        accept_multiple_files=True,
    )
    title = st.text_input("Document title (applied to all files)", placeholder="My document")
    col_choice = st.selectbox("Collection", collections + ["+ New collection..."], key="upload_col")
    if col_choice == "+ New collection...":
        col_choice = st.text_input("New collection name")

    if uploaded_files and title and col_choice:
        if st.button("Embed & Store", type="primary"):
            total_stored = 0
            for file_idx, uploaded in enumerate(uploaded_files):
                st.write(f"**Processing {file_idx+1}/{len(uploaded_files)}: {uploaded.name}**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                def on_progress(msg: str, current: int, total: int, _bar=progress_bar, _st=status_text):
                    _st.text(msg)
                    _bar.progress(min(current / total, 1.0) if total > 0 else 0)

                try:
                    file_bytes = uploaded.read()
                    mime = uploaded.type or "application/octet-stream"
                    status_text.text("Chunking...")
                    results = rag.ingest(
                        file_bytes=file_bytes,
                        filename=uploaded.name,
                        title=title,
                        mime_type=mime,
                        collection=col_choice,
                        on_progress=on_progress,
                    )
                    progress_bar.progress(100)
                    status_text.text("Done!")
                    total_stored += len(results)
                except Exception as e:
                    st.error(f"Error processing {uploaded.name}: {e}")
                    raise
            st.success(f"Stored {total_stored} chunk(s) across {len(uploaded_files)} file(s)")
            st.cache_data.clear()
            st.rerun()
    elif uploaded_files and (not title or not col_choice):
        st.warning("Please enter a document title and select a collection.")

# ── Tab 2: Search ───────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Query your documents")
    query_text = st.text_area("Enter your query", height=100)

    if st.button("Search", type="primary", key="search_btn"):
        if not query_text.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                try:
                    result = rag.query(
                        query_text=query_text.strip(),
                        top_k=top_k,
                        threshold=threshold,
                        filter_type=filter_type,
                        filter_collection=filter_collection,
                        use_reasoning=use_reasoning,
                    )
                except Exception as e:
                    st.error(f"Search error: {e}")
                    raise

            if result["answer"]:
                st.subheader("Answer")
                st.markdown(result["answer"])

            st.subheader(f"Sources ({len(result['sources'])} matches)")
            if not result["sources"]:
                st.info("No matching documents found. Try lowering the similarity threshold.")

            for src in result["sources"]:
                sim = src.get("similarity", 0)
                with st.expander(
                    f"[{sim:.3f}] {src['title']} — {src['original_filename']} "
                    f"(chunk {src['chunk_index']}/{src['chunk_total']}, {src['content_type']})",
                    expanded=src["content_type"] in ("image", "video"),
                ):
                    if src["content_type"] == "image" and src.get("file_data"):
                        img_bytes = base64.b64decode(src["file_data"])
                        st.image(img_bytes, caption=src["original_filename"], width="stretch")
                    elif src["content_type"] == "video" and src.get("file_data"):
                        vid_bytes = base64.b64decode(src["file_data"])
                        mime = (src.get("metadata") or {}).get("mime_type", "video/mp4")
                        st.video(vid_bytes, format=mime)
                    elif src.get("text_content"):
                        st.text(src["text_content"][:2000])
                    else:
                        st.caption(f"Non-text content ({src['content_type']})")
                    if src.get("metadata"):
                        st.json(src["metadata"])

# ── Tab 3: Browse ───────────────────────────────────────────────────────────
with tab_browse:
    st.subheader("All documents")

    try:
        docs = db.get_all_documents()
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        docs = []

    if not docs:
        st.info("No documents yet. Upload something in the first tab.")
    else:
        st.dataframe(
            docs,
            width="stretch",
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "title": st.column_config.TextColumn("Title"),
                "content_type": st.column_config.TextColumn("Type"),
                "original_filename": st.column_config.TextColumn("Filename"),
                "chunk_index": st.column_config.NumberColumn("Chunk"),
                "chunk_total": st.column_config.NumberColumn("Total"),
                "collection": st.column_config.TextColumn("Collection"),
                "created_at": st.column_config.TextColumn("Created"),
            },
        )

        st.divider()
        st.subheader("Delete documents")

        filenames = sorted(set(d["original_filename"] for d in docs))
        delete_file = st.selectbox("Delete all chunks of a file", [""] + filenames)
        if st.button("Delete file", type="secondary"):
            if delete_file:
                try:
                    count = db.delete_by_filename(delete_file)
                    st.success(f"Deleted {count} chunk(s) of {delete_file}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete error: {e}")

        delete_id = st.text_input("Or delete single chunk by ID")
        if st.button("Delete by ID", type="secondary"):
            if delete_id.strip():
                try:
                    db.delete_document(delete_id.strip())
                    st.success(f"Deleted {delete_id}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete error: {e}")
            else:
                st.warning("Enter a document ID.")
