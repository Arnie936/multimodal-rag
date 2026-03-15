import base64
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from lib import db, rag, auth

st.set_page_config(page_title="Multimodal RAG", page_icon="🔍", layout="wide")

# ── Auth Gate ─────────────────────────────────────────────────────────────────

def render_auth_page():
    st.title("Multimodal RAG — Login")
    tab_login, tab_register = st.tabs(["Login", "Registrieren"])

    with tab_login:
        email = st.text_input("E-Mail", key="login_email")
        password = st.text_input("Passwort", type="password", key="login_pw")
        if st.button("Login", type="primary"):
            if email and password:
                result = auth.login(email, password)
                if result["error"]:
                    st.error(result["error"])
                else:
                    st.session_state["access_token"] = result["session"].access_token
                    st.session_state["user_id"] = result["user"].id
                    st.session_state["user_email"] = result["user"].email
                    st.rerun()
            else:
                st.warning("Bitte E-Mail und Passwort eingeben.")

    with tab_register:
        reg_email = st.text_input("E-Mail", key="reg_email")
        reg_pw = st.text_input("Passwort", type="password", key="reg_pw")
        if st.button("Registrieren", type="primary"):
            if reg_email and reg_pw:
                result = auth.register(reg_email, reg_pw)
                if result["error"]:
                    st.error(result["error"])
                else:
                    st.success("Registrierung erfolgreich! Bitte E-Mail bestätigen und dann einloggen.")
            else:
                st.warning("Bitte E-Mail und Passwort eingeben.")


if "access_token" not in st.session_state:
    render_auth_page()
    st.stop()

# ── Authenticated context ─────────────────────────────────────────────────────

access_token = st.session_state["access_token"]
user_id = st.session_state["user_id"]
authed_client = auth.get_authed_db_client(access_token)

# Load LLM settings once per session
if "llm_settings" not in st.session_state:
    saved = db.get_user_settings(authed_client, user_id)
    if saved:
        st.session_state["llm_settings"] = {
            "provider": saved["llm_provider"],
            "model": saved["llm_model"],
            "api_key": saved["llm_api_key"],
            "ollama_url": saved["ollama_url"],
        }
    else:
        st.session_state["llm_settings"] = {
            "provider": "gemini",
            "model": "gemini-2.0-flash-lite",
            "api_key": "",
            "ollama_url": "http://localhost:11434",
        }

st.title("Multimodal RAG with Gemini Embedding")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Eingeloggt als **{st.session_state['user_email']}**")
    if st.button("Logout"):
        auth.logout(access_token)
        for key in ["access_token", "user_id", "user_email", "llm_settings"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.header("LLM-Einstellungen")

    current = st.session_state["llm_settings"]
    providers = ["gemini", "openai", "anthropic", "ollama"]
    provider = st.selectbox(
        "Provider",
        providers,
        index=providers.index(current["provider"]) if current["provider"] in providers else 0,
    )
    model_defaults = {
        "gemini": "gemini-2.0-flash-lite",
        "openai": "o4-mini",
        "anthropic": "claude-sonnet-4-5",
        "ollama": "llama3",
    }
    model = st.text_input("Modell", value=current.get("model") or model_defaults.get(provider, ""))
    api_key_input = st.text_input(
        "API Key",
        type="password",
        value=current.get("api_key", ""),
        help="Nicht erforderlich für Ollama",
    )
    ollama_url = ""
    if provider == "ollama":
        ollama_url = st.text_input(
            "Ollama URL",
            value=current.get("ollama_url", "http://localhost:11434"),
        )

    if st.button("Einstellungen speichern"):
        new_settings = {
            "provider": provider,
            "model": model,
            "api_key": api_key_input,
            "ollama_url": ollama_url,
        }
        try:
            db.upsert_user_settings(authed_client, user_id, new_settings)
            st.session_state["llm_settings"] = new_settings
            st.success("Gespeichert")
        except Exception as e:
            st.error(f"Fehler beim Speichern: {e}")

    st.divider()
    st.header("Settings")
    top_k = st.slider("Top K results", 1, 50, 10)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.05)
    filter_type = st.selectbox(
        "Content type filter",
        ["all", "text", "image", "pdf", "audio", "video"],
    )
    use_codex = st.checkbox("Use LLM reasoning", value=True)

    st.divider()
    st.header("Database Stats")
    if st.button("Refresh stats"):
        st.cache_data.clear()

    @st.cache_data(ttl=30)
    def _stats(uid: str):
        return db.get_stats(authed_client=auth.get_authed_db_client(
            st.session_state.get("access_token", "")
        ))

    try:
        stats = _stats(user_id)
        st.metric("Total documents", stats["total"])
        for ctype, count in stats["by_type"].items():
            st.metric(ctype.capitalize(), count)
    except Exception as e:
        st.error(f"Could not load stats: {e}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_search, tab_browse = st.tabs(["Upload & Embed", "Search", "Browse"])

# ── Tab 1: Upload & Embed ─────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload files to embed")
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["txt", "png", "jpg", "jpeg", "webp", "gif", "pdf", "mp3", "wav", "mp4", "mov", "avi"],
        accept_multiple_files=True,
    )
    title = st.text_input("Document title (applied to all files)", placeholder="My document")

    if uploaded_files and title:
        if st.button("Embed & Store", type="primary"):
            total_stored = 0
            for file_idx, uploaded in enumerate(uploaded_files):
                st.write(f"**Processing {file_idx+1}/{len(uploaded_files)}: {uploaded.name}**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                chunks_done = [0]

                def on_progress(msg: str, _bar=progress_bar, _st=status_text, _c=chunks_done):
                    _c[0] += 1
                    _st.text(msg)
                    _bar.progress(min(_c[0] * 10, 100))

                try:
                    file_bytes = uploaded.read()
                    mime = uploaded.type or "application/octet-stream"
                    status_text.text("Processing...")
                    results = rag.ingest(
                        file_bytes=file_bytes,
                        filename=uploaded.name,
                        title=title,
                        mime_type=mime,
                        on_progress=on_progress,
                        authed_client=authed_client,
                        user_id=user_id,
                    )
                    progress_bar.progress(100)
                    status_text.text("Done!")
                    total_stored += len(results)
                except Exception as e:
                    st.error(f"Error processing {uploaded.name}: {e}")
                    raise
            st.success(f"Stored {total_stored} chunk(s) across {len(uploaded_files)} file(s)")
            st.cache_data.clear()
    elif uploaded_files and not title:
        st.warning("Please enter a document title.")

# ── Tab 2: Search ─────────────────────────────────────────────────────────────
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
                        use_codex=use_codex,
                        authed_client=authed_client,
                        llm_settings=st.session_state["llm_settings"] if use_codex else None,
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
                        st.image(img_bytes, caption=src["original_filename"], use_container_width=True)
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

# ── Tab 3: Browse ─────────────────────────────────────────────────────────────
with tab_browse:
    st.subheader("All documents")

    try:
        docs = db.get_all_documents(authed_client=authed_client)
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        docs = []

    if not docs:
        st.info("No documents yet. Upload something in the first tab.")
    else:
        st.dataframe(
            docs,
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "title": st.column_config.TextColumn("Title"),
                "content_type": st.column_config.TextColumn("Type"),
                "original_filename": st.column_config.TextColumn("Filename"),
                "chunk_index": st.column_config.NumberColumn("Chunk"),
                "chunk_total": st.column_config.NumberColumn("Total"),
                "created_at": st.column_config.TextColumn("Created"),
            },
        )

        st.divider()
        st.subheader("Delete documents")
        delete_id = st.text_input("Document ID to delete")
        if st.button("Delete", type="secondary"):
            if delete_id.strip():
                try:
                    db.delete_document(delete_id.strip(), authed_client=authed_client)
                    st.success(f"Deleted {delete_id}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete error: {e}")
            else:
                st.warning("Enter a document ID.")
