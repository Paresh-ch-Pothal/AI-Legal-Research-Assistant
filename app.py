import os
import uuid
import tempfile
import time
import re

import streamlit as st
from dotenv import load_dotenv

from pdf_report import generate_chat_pdf
from web_search import search_web

load_dotenv()

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Legal AI Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pipeline Initialisation (Cached) ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(gemini_key: str, kanoon_token: str, device: str):
    """Load the LegalAIPipeline once and cache it for the session."""
    from pipeline import LegalAIPipeline  # your existing module
    return LegalAIPipeline(
        google_api_key   = gemini_key,
        kanoon_api_token = kanoon_token if kanoon_token else "free_mode",
        db_path          = "./legal_db",
        gemini_model     = "gemini-2.5-flash",
        device           = device,
    )

# ── Session State Initialisation ────────────────────────────────
def init_state():
    defaults = {
        "messages":        [],          # [{role, content, sources}]
        "session_id":      None,        # pipeline session
        "uploaded_files":  [],          # [{name, meta, status}]
        "pipeline":        None,
        "pipeline_ready":  False,
        "api_key":         os.getenv("GEMINI_API_KEY", ""),
        "kanoon_token":    os.getenv("INDIAN_KANOON_API_KEY", ""),
        "device":          os.getenv("EMBEDDING_DEVICE", "cpu"),
        "last_sources":    [],
        "precedent_data":  {},
        "find_precedents": False,
        "web_sources":        [],       # results from the standalone web-search box
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helper: Process Uploaded Document ───────────────────────────
def process_upload(uploaded_file):
    """Save to temp file, call pipeline.upload_document, return result dict."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        result = st.session_state.pipeline.upload_document(
            filepath   = tmp_path,
            session_id = st.session_state.session_id,
        )
    finally:
        os.unlink(tmp_path)
    return result

# ── SIDEBAR: Configuration & Document Controls ──────────────────
with st.sidebar:
    st.title("⚖️ LegalAI RAG")
    st.caption("Advanced Indian Legal Research Assistant")
    st.divider()

    # ── Pipeline Configurations ──
    st.subheader("Configuration")
    gemini_key = st.text_input(
        "Gemini API Key",
        value    = st.session_state.api_key,
        type     = "password",
        help     = "Get yours at aistudio.google.com",
    )
    kanoon_tok = st.text_input(
        "Indian Kanoon Token (optional)",
        value    = st.session_state.kanoon_token,
        type     = "password",
        help     = "Leave blank to use free scraper",
    )
    device_opt = st.selectbox(
        "Embedding Device",
        ["cpu", "cuda"],
        index  = 0 if st.session_state.device == "cpu" else 1,
    )

    if st.button("🚀 Load Pipeline", use_container_width=True, type="primary"):
        if not gemini_key.strip():
            st.error("Gemini API key required.")
        else:
            with st.spinner("Loading InLegalBERT + ChromaDB…"):
                try:
                    st.session_state.api_key       = gemini_key.strip()
                    st.session_state.kanoon_token = kanoon_tok.strip()
                    st.session_state.device       = device_opt
                    
                    pipeline = load_pipeline(
                        gemini_key.strip(),
                        kanoon_tok.strip(),
                        device_opt,
                    )
                    st.session_state.pipeline       = pipeline
                    st.session_state.session_id    = pipeline.new_session()
                    st.session_state.pipeline_ready = True
                    st.session_state.messages       = []
                    st.session_state.uploaded_files = []
                    st.success("Pipeline ready ✅")
                except Exception as e:
                    st.error(f"Failed to load pipeline:\n{e}")

    st.divider()

    # ── Document Management ──
    st.subheader("Upload Documents")
    if not st.session_state.pipeline_ready:
        st.info("⬆️ Load the pipeline first to enable file uploads.")
    else:
        uploaded = st.file_uploader(
            "Upload Case Files",
            type             = ["pdf", "png", "jpg", "jpeg", "tiff","txt"],
            accept_multiple_files = True,
            label_visibility = "collapsed"
        )
        if uploaded:
            already = {f["name"] for f in st.session_state.uploaded_files}
            new_files = [f for f in uploaded if f.name not in already]
            if new_files:
                for uf in new_files:
                    with st.spinner(f"Processing {uf.name}…"):
                        try:
                            result = process_upload(uf)
                            status = "error" if "error" in result else "ready"
                            st.session_state.uploaded_files.append({
                                "name":   uf.name,
                                "meta":   result,
                                "status": status,
                            })
                        except Exception as e:
                            st.session_state.uploaded_files.append({
                                "name":   uf.name,
                                "meta":   {"error": str(e)},
                                "status": "error",
                            })

    # ── Indexed Files Registry ──
    if st.session_state.uploaded_files:
        st.subheader("Indexed Documents")
        for f in st.session_state.uploaded_files:
            meta   = f["meta"]
            status = f["status"]
            
            if status == "ready":
                chunks  = meta.get("chunks_added", "?")
                method  = meta.get("extraction_method", "?")
                court   = meta.get("court", "Unknown Case/Court")
                st.success(f"**📄 {f['name']}**\n\n`{chunks} chunks` | `{method}` | `{court[:25]}`")
            else:
                err = meta.get("error", "Unknown error")
                st.error(f"**⚠️ {f['name']}**\n\nError: {err[:50]}")

    st.divider()

    # ── Export & Share ──
    st.subheader("Export & Share")
    if st.session_state.messages:
        pdf_bytes = generate_chat_pdf(
            st.session_state.messages,
            session_id=st.session_state.session_id,
        )
        st.download_button(
            label="📥 Download Chat as PDF",
            data=pdf_bytes,
            file_name=f"legal_chat_{st.session_state.session_id or 'session'}.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Export the full conversation (with cited sources) as a shareable PDF.",
        )
    else:
        st.button("📥 Download Chat as PDF", use_container_width=True, disabled=True)
        st.caption("Start a conversation to enable PDF export.")

    st.divider()

    # ── Standalone Web Search (independent of chat) ──
    st.subheader("🌐 Web Search")
    st.caption(
        "Quick DuckDuckGo lookup for your own reference. This is completely "
        "separate from the chat — it does NOT use your chat question and the "
        "results are never added to the conversation."
    )
    web_query = st.text_input(
        "Search query",
        key              = "web_search_query_input",
        placeholder      = "e.g. Section 498A IPC recent judgments",
        label_visibility = "collapsed",
    )
    if st.button("🔍 Search", use_container_width=True):
        if web_query.strip():
            with st.spinner("Searching DuckDuckGo…"):
                st.session_state.web_sources = search_web(web_query.strip(), max_results=5)
            if not st.session_state.web_sources:
                st.warning("No results found (or the search failed).")
        else:
            st.warning("Enter a search query first.")

    # st.session_state.find_precedents = st.toggle(
    #     "Search similar precedents",
    #     value = st.session_state.find_precedents,
    #     help  = "Also query Indian Kanoon for similar past judgments",
    # )

    # ── Clear Session ──
    if st.session_state.pipeline_ready:
        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            try:
                st.session_state.pipeline.end_session(st.session_state.session_id)
            except Exception:
                pass
            st.session_state.session_id    = st.session_state.pipeline.new_session()
            st.session_state.messages       = []
            st.session_state.uploaded_files = []
            st.session_state.last_sources   = []
            st.session_state.precedent_data = {}
            st.session_state.web_sources    = []
            st.rerun()


# ════════════════════════════════════════════════════════════════
#  MAIN WINDOW LAYOUT: Split view (Chat vs Sources)
# ════════════════════════════════════════════════════════════════
col_chat, col_src = st.columns([2.2, 1], gap="medium")

# ── LEFT PANEL: Main Chat Workspace ─────────────────────────────
with col_chat:
    # Header Status Ribbon
    doc_count = len(st.session_state.uploaded_files)
    status_msg = f"🟢 **Pipeline Active** ({doc_count} document{'s' if doc_count != 1 else ''} indexed)" if st.session_state.pipeline_ready else "🔴 **Pipeline Offline**"
    
    st.subheader("Interactive Case Analysis Workspace")
    st.caption(f"{status_msg} ｜ Model: `gemini-2.5-flash` ｜ Embeddings: `InLegalBERT`")
    st.divider()

    # Stream empty state if no history exists
    if not st.session_state.messages:
        st.info(
            "💡 **Welcome to Indian Legal AI Assistant!**\n\n"
            "To begin, insert your configurations and upload your primary documents (FIRs, Petitions, Orders) in the sidebar. "
            "Once indexed, use the interface below to cross-examine files, cite provisions, and surface matching precedents."
        )
    else:
        # Render clean, native message sequences
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Core Native Chat Input Window
    if prompt := st.chat_input("Ask a legal question about your indexed case files..."):
        if not st.session_state.pipeline_ready:
            st.warning("Please load the pipeline from the sidebar configuration first.")
        elif not st.session_state.uploaded_files:
            st.warning("Please upload and index at least one legal file to query context.")
        else:
            # Commit User Query to State
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Immediately show user question natively
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Assistant Answer Wrapper
            with st.chat_message("assistant"):
                with st.spinner("Analyzing arguments & fetching grounds..."):
                    try:
                        pipeline = st.session_state.pipeline
                        session_id = st.session_state.session_id

                        # Query pipeline execution engine
                        result = pipeline.ask(question=prompt, session_id=session_id)
                        answer = result.get("answer", "No answer returned.")
                        sources = result.get("sources", [])

                        # # Handle alternative precedent tracks
                        # if st.session_state.find_precedents:
                        #     prec = pipeline.find_similar_verdicts(
                        #         session_id        = session_id,
                        #         n_results         = 3,
                        #         fetch_from_kanoon = True,
                        #     )
                        #     st.session_state.precedent_data = prec
                        #     if prec.get("analysis"):
                        #         answer += "\n\n---\n### 📚 Precedent Analysis\n" + prec["analysis"]
                        # else:
                        #     st.session_state.precedent_data = {}

                        # Render response instantly
                        st.markdown(answer)
                        
                        # Cache properties natively
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                        st.session_state.last_sources = sources
                        st.rerun()

                    except Exception as e:
                        err_msg = f"⚠️ **Execution Error Encountered:** {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg, "sources": []})


# ── RIGHT PANEL: Source Citations & Precedents ──────────────────
with col_src:
    st.subheader("Retrieved Context")
    st.caption("Factual verification sources extracted by vector pipeline")
    st.divider()

    # Section 1: Vector Chunks
    st.markdown("### 📄 Context Chunks")
    if st.session_state.last_sources:
        for idx, src in enumerate(st.session_state.last_sources, 1):
            score = src.get("score") or 0
            score_pct = min(int(float(score) * 100), 100) if score else 0
            
            with st.expander(f"Ground {idx} — Relevance {score_pct}%", expanded=(idx == 1)):
                st.write(f"**Source File:** `{src.get('filename', 'Unknown File')}`")
                st.write(f"**Jurisdiction:** {src.get('court', '—')}")
                st.write(f"**Case reference:** {src.get('case_number', '—')}")
                if src.get("kanoon_url"):
                    st.link_button("🔗 View Record on Indian Kanoon", src["kanoon_url"])
                st.progress(score_pct / 100)
    else:
        st.caption("Active chunk segments for the latest query will generate here.")

    st.divider()

    # Section 2: Standalone Web Search Results (DuckDuckGo)
    st.markdown("### 🌐 Web Search Results")
    if st.session_state.web_sources:
        for idx, res in enumerate(st.session_state.web_sources, 1):
            with st.container(border=True):
                st.markdown(f"**{idx}. {res.get('title', 'Untitled Result')}**")
                if res.get("snippet"):
                    st.caption(res["snippet"])
                if res.get("link"):
                    st.link_button("🔗 Open Source", res["link"])
    else:
        st.caption("Use the '🌐 Web Search' box in the sidebar to look something up — results will appear here.")

    st.divider()

    # Section 3: Court Precedents
    # st.markdown("### 🏛️ Related Judicial Precedents")
    # pd = st.session_state.get("precedent_data", {})
    # if pd and pd.get("similar_cases"):
    #     for idx, case in enumerate(pd["similar_cases"], 1):
    #         with st.container(border=True):
    #             st.markdown(f"**{idx}. {case.get('title', 'Untitled Judgement')}**")
    #             st.write(f"📍 *Court:* {case.get('court', '—')} ｜ *Date:* {case.get('date', '—')}")
    #             st.write(f"🔢 *Ref:* {case.get('case_number', '—')}")
    #             if case.get("kanoon_url"):
    #                 st.link_button("View Precedent Document", case["kanoon_url"])
    # else:
    #     st.caption("Toggle 'Search similar precedents' to pipeline web judgements.")