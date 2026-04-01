import streamlit as st
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle

st.set_page_config(
    page_title="Yakima Fisheries Literature",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

/* ── Design tokens ─────────────────────────────────────────────── */
:root {
    --bg:           #1a1915;
    --bg-surface:   #1e1d19;
    --bg-input:     #2a2925;
    --bg-user:      #2f2e2a;
    --bg-chip:      #2a2925;
    --bg-chip-on:   #333029;
    --bg-hover:     #252420;
    --border:       #2e2c27;
    --border-subtle:#252420;
    --text:         #e8e4d9;
    --text-sub:     #8c8880;
    --text-muted:   #5a5650;
    --accent:       #6b6b6b;
    --accent-dim:   rgba(107, 107, 107, 0.12);
    --font:         'Inter', ui-sans-serif, system-ui, sans-serif;
    --mono:         'JetBrains Mono', ui-monospace, monospace;
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    18px;
    --radius-xl:    24px;
}

/* ── Base ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container {
    max-width: 740px !important;
    padding-top: 0 !important;
    padding-bottom: 0.5rem !important;
}


/* ── Password gate ─────────────────────────────────────────────── */
.password-gate {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
    gap: 0;
}
.password-gate-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2.5rem 2rem;
    width: 100%;
    max-width: 380px;
    text-align: center;
}
.password-gate-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
    display: block;
}
.password-gate-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.25rem;
    letter-spacing: -0.015em;
}
.password-gate-subtitle {
    font-size: 0.8rem;
    color: var(--text-sub);
    margin-bottom: 1.75rem;
}
.stTextInput > label {
    font-size: 0.75rem !important;
    color: var(--text-sub) !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
}
.stTextInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    outline: none !important;
    box-shadow: none !important;
    caret-color: var(--accent) !important;
}
.stTextInput input:focus {
    border-color: #4a4a4a !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.04) !important;
}
.stTextInput input::placeholder {
    color: var(--text-muted) !important;
}

/* ── App header ────────────────────────────────────────────────── */
.app-header {
    padding: 2rem 0 1.25rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
    text-align: center;
}
.app-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}

/* ── Chat messages ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 0 !important;
    margin-bottom: 0 !important;
    gap: 0.9rem !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td {
    color: var(--text) !important;
    font-size: 0.9375rem !important;
    line-height: 1.72 !important;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stChatMessage"] h2 { font-size: 1rem !important; }
[data-testid="stChatMessage"] h3 { font-size: 0.9rem !important; color: var(--text-sub) !important; }
[data-testid="stChatMessage"] code {
    background: var(--bg-surface) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 5px !important;
    padding: 0.1em 0.4em !important;
    font-size: 0.87em !important;
    font-family: var(--mono) !important;
    color: var(--text) !important;
}
[data-testid="stChatMessage"] pre {
    background: var(--bg-surface) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
}
[data-testid="stChatMessage"] pre code {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    font-family: var(--mono) !important;
}
[data-testid="stChatMessage"] blockquote {
    border-left: 2px solid var(--border) !important;
    padding-left: 0.85rem !important;
    color: var(--text-sub) !important;
    margin: 0.75rem 0 !important;
}
/* User message pill */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] + div {
    background: var(--bg-user) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.7rem 1.05rem !important;
    max-width: 88% !important;
}
/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: var(--bg-chip) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50% !important;
    width: 28px !important;
    height: 28px !important;
    font-size: 0.78rem !important;
    flex-shrink: 0 !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 28px !important;
    height: 28px !important;
    font-size: 0.78rem !important;
    flex-shrink: 0 !important;
    color: white !important;
}

/* ── Bottom bar ────────────────────────────────────────────────── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, var(--bg) 80%, transparent) !important;
    padding-bottom: 0.5rem !important;
    padding-top: 0.5rem !important;
}
[data-testid="stBottom"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
}
[data-testid="stChatInput"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #484848 !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.9375rem !important;
    background: transparent !important;
    padding: 0.85rem 1.1rem !important;
    caret-color: var(--accent) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 50% !important;
    color: #ffffff !important;
    transition: background 0.15s, opacity 0.15s !important;
}
[data-testid="stChatInput"] button:hover {
    opacity: 0.85 !important;
}

/* ── Bottom row: toggle left, footer center ────────────────────── */
.bottom-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.3rem 0.25rem 0.1rem;
}
.bottom-row-center {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.01em;
    white-space: nowrap;
    pointer-events: none;
}

/* ── Web search toggle ─────────────────────────────────────────── */
div[data-testid="stToggle"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    margin: 0 !important;
    padding: 0.25rem 0.65rem 0.25rem 0.5rem !important;
    background: var(--bg-chip) !important;
    border: 1px solid var(--border) !important;
    border-radius: 99px !important;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
div[data-testid="stToggle"]:hover {
    background: var(--bg-chip-on) !important;
    border-color: #444 !important;
}
div[data-testid="stToggle"] label {
    font-size: 0.75rem !important;
    color: var(--text-sub) !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    user-select: none !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stToggle"] label:hover { color: var(--text) !important; }
div[data-testid="stToggle"] [data-baseweb="toggle"] {
    transform: scale(0.85) !important;
}

/* ── Source expander ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-top: 0.75rem !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-sub) !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--text) !important;
    background: var(--bg-hover) !important;
}
[data-testid="stExpander"] > div > div {
    padding: 0 0.9rem 0.75rem !important;
}
[data-testid="stExpander"] svg { color: var(--text-muted) !important; }

/* ── Source rows ───────────────────────────────────────────────── */
.source-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border-subtle);
}
.source-row:last-child { border-bottom: none; }
.source-badge {
    font-size: 0.63rem;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--bg-chip);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.18rem 0.5rem;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 0.1rem;
    letter-spacing: 0.02em;
    font-variant-numeric: tabular-nums;
}
.source-title {
    font-size: 0.8125rem;
    color: var(--text);
    line-height: 1.5;
    font-weight: 400;
}
.source-file {
    font-size: 0.69rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
    font-family: ui-monospace, 'SF Mono', monospace;
    letter-spacing: -0.01em;
}

/* ── Status indicator ──────────────────────────────────────────── */
.status-wrap {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.75rem;
    background: var(--bg-chip);
    border: 1px solid var(--border);
    border-radius: 99px;
    margin: 0.25rem 0;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: var(--text-sub);
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 1.4s ease-in-out infinite;
}
.status-text {
    font-size: 0.78rem;
    color: var(--text-sub);
    font-weight: 400;
}
@keyframes pulse {
    0%, 100% { opacity: 0.4; transform: scale(0.9); }
    50%       { opacity: 1;   transform: scale(1.1); }
}

/* ── Empty state ───────────────────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 1rem 2rem;
    gap: 0.5rem;
    text-align: center;
}
.empty-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.5;
}
.empty-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-sub);
    letter-spacing: -0.01em;
}
.empty-hint {
    font-size: 0.8rem;
    color: var(--text-muted);
    max-width: 320px;
    line-height: 1.55;
}
.suggestion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    justify-content: center;
    margin-top: 1.25rem;
    max-width: 480px;
}
.chip {
    font-size: 0.76rem;
    color: var(--text-sub);
    background: var(--bg-chip);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.35rem 0.75rem;
    cursor: default;
    transition: background 0.15s, color 0.15s;
    line-height: 1.4;
}
.chip:hover {
    background: var(--bg-chip-on);
    color: var(--text);
}

/* ── Column gutters in bottom row ──────────────────────────────── */
[data-testid="stBottom"] [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
    align-items: center !important;
}
[data-testid="stBottom"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #484848; }

/* ── Divider ───────────────────────────────────────────────────── */
hr { border-color: var(--border-subtle) !important; }

/* ── Mobile ────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding-top: 0 !important; }
    .app-title { font-size: 0.85rem; }
    .empty-state { padding: 2rem 1rem 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Password gate ───────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div class="password-gate">
        <div class="password-gate-card">
            <div class="password-gate-title">Yakima Fisheries Literature</div>
            <div class="password-gate-subtitle">Enter your access key to continue</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    password = st.text_input("Access key", type="password", placeholder="Enter access key…", label_visibility="collapsed")
    if password == "yakima2026":
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.markdown("<p style='font-size:0.78rem;color:#9b5555;text-align:center;margin-top:0.5rem;'>Incorrect access key</p>", unsafe_allow_html=True)
    st.stop()

# ── Load resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="yakima")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic = Anthropic(api_key=api_key)
    with open("./bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    return collection, embed_model, rerank_model, anthropic, bm25_data

collection, embed_model, rerank_model, anthropic, bm25_data = load_resources()
bm25 = bm25_data["bm25"]
bm25_texts = bm25_data["texts"]
bm25_metadata = bm25_data["metadata"]

# ── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert fish biologist specializing in bull trout ecology and conservation 
in the Yakima Basin. You are assisting USFWS biologists who need detailed, technical answers.

When answering:
- Cite sources using the full title and year provided in the source label, e.g. "Smith et al. (2001) 
  found that..." or "(Jones & Brown, 1998)". Use author-style citations, not just filenames.
- Note the age of information when relevant — flag when you're drawing on older studies (pre-2000) 
  vs recent work, and highlight if findings have been updated or contradicted over time
- Be specific and quantitative — include actual numbers, measurements, temperatures, distances, 
  population counts, and dates from the literature when available
- Synthesize across multiple sources when they agree or disagree — note contradictions explicitly
- Use technical biological terminology appropriate for professional biologists
- Structure longer answers with clear sections if the topic warrants it
- If the excerpts don't contain enough to answer well, say exactly what is missing and suggest 
  what kind of source might have it

Do not give vague summaries. Give the kind of detailed answer a senior biologist would give 
when briefing a colleague. Always ground your answer in the specific literature provided."""

# ── Query expansion ─────────────────────────────────────────────────────────
def expand_query(question, anthropic_client):
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": f"""Generate 3 alternative search queries for this biology question to help find relevant scientific literature. 
Return only the queries, one per line, no numbering or explanation.
Question: {question}"""}]
    )
    alternatives = response.content[0].text.strip().split('\n')
    return [question] + [q.strip() for q in alternatives if q.strip()][:3]

# ── Hybrid retrieval ────────────────────────────────────────────────────────
def hybrid_retrieve(queries, collection, embed_model, bm25, bm25_texts, bm25_metadata, n_vector=25, n_bm25=25):
    seen_ids = set()
    candidate_docs = []
    candidate_metas = []

    for q in queries:
        q_embedding = embed_model.encode(q)
        # ids are returned automatically, do not include in include list
        results = collection.query(
            query_embeddings=[q_embedding.tolist()],
            n_results=n_vector,
            include=["documents", "metadatas"]
        )
        for doc, meta, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0]
        ):
            parent_text = meta.get("parent_text", doc)
            parent_id = meta.get("parent_id", doc_id)
            if parent_id not in seen_ids:
                seen_ids.add(parent_id)
                candidate_docs.append(parent_text)
                candidate_metas.append(meta)

    tokens = queries[0].lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_bm25]

    for idx in top_bm25_indices:
        parent_id = bm25_metadata[idx]["parent_id"]
        if parent_id not in seen_ids:
            seen_ids.add(parent_id)
            candidate_docs.append(bm25_texts[idx])
            candidate_metas.append(bm25_metadata[idx])

    return candidate_docs, candidate_metas

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Yakima Fisheries Literature</div>
</div>
""", unsafe_allow_html=True)

# ── Chat ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_history" not in st.session_state:
    st.session_state.source_history = []

# Empty state — no content shown when no messages

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i // 2 < len(st.session_state.source_history):
            sources = st.session_state.source_history[i // 2]
            if sources:
                n = len(sources)
                with st.expander(f"  {n} source{'s' if n != 1 else ''} retrieved"):
                    for s in sources:
                        title = (s['title'][:90] + '…') if len(s['title']) > 90 else s['title']
                        display = title or s['source']
                        st.markdown(
                            f"<div class='source-row'>"
                            f"<span class='source-badge'>{s['year'] or '—'}</span>"
                            f"<div><div class='source-title'>{display}</div>"
                            f"<div class='source-file'>{s['source']}</div></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

# ── Bottom input area ────────────────────────────────────────────────────────
if "use_web" not in st.session_state:
    st.session_state.use_web = False

if question := st.chat_input("Ask about Yakima fisheries…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()

        status.markdown("<div class='status-wrap'><div class='status-dot'></div><span class='status-text'>Expanding query…</span></div>", unsafe_allow_html=True)
        recent_history = st.session_state.messages[-4:]
        conversation_context = ""
        if len(recent_history) > 1:
            conversation_context = "\n".join([
                f"{m['role'].upper()}: {m['content'][:300]}"
                for m in recent_history[:-1]
            ])
        contextual_question = question
        if conversation_context:
            contextual_question = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {question}"
        queries = expand_query(contextual_question, anthropic)

        status.markdown("<div class='status-wrap'><div class='status-dot'></div><span class='status-text'>Retrieving literature…</span></div>", unsafe_allow_html=True)
        candidate_docs, candidate_metas = hybrid_retrieve(
            queries, collection, embed_model,
            bm25, bm25_texts, bm25_metadata
        )

        status.markdown("<div class='status-wrap'><div class='status-dot'></div><span class='status-text'>Reranking results…</span></div>", unsafe_allow_html=True)
        pairs = [[question, doc] for doc in candidate_docs]
        scores = rerank_model.predict(pairs)
        scores = [float(s) for s in scores]
        ranked = sorted(zip(scores, candidate_docs, candidate_metas), key=lambda x: x[0], reverse=True)
        top = ranked[:12]

        status.markdown("<div class='status-wrap'><div class='status-dot'></div><span class='status-text'>Generating answer…</span></div>", unsafe_allow_html=True)

        context_parts = []
        used_sources = []
        for score, doc, meta in top:
            title = meta.get('title', '')
            year = meta.get('year', 'unknown')
            source = meta['source']
            label = f"{title} ({year}) [{source}]" if title else f"{source} ({year})"
            context_parts.append(f"[Source: {label}]\n{doc}")
            if not any(s['source'] == source for s in used_sources):
                used_sources.append({"source": source, "title": title, "year": year})
        context = "\n\n---\n\n".join(context_parts)

        claude_messages = []
        for m in st.session_state.messages[:-1]:
            claude_messages.append({"role": m["role"], "content": m["content"]})
        claude_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Literature excerpts:\n\n{context}", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"Question: {question}"}
            ]
        })

        tools = []
        if st.session_state.use_web:
            tools.append({"type": "web_search_20250305", "name": "web_search"})

        api_kwargs = dict(
            model="claude-sonnet-4-6",
            max_tokens=6000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=claude_messages
        )
        if tools:
            api_kwargs["tools"] = tools

        response = anthropic.messages.create(**api_kwargs)
        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text

        status.empty()
        st.markdown(answer)

        n = len(used_sources)
        with st.expander(f"  {n} source{'s' if n != 1 else ''} retrieved"):
            for s in used_sources:
                title = (s['title'][:90] + '…') if len(s['title']) > 90 else s['title']
                display = title or s['source']
                st.markdown(
                    f"<div class='source-row'>"
                    f"<span class='source-badge'>{s['year'] or '—'}</span>"
                    f"<div><div class='source-title'>{display}</div>"
                    f"<div class='source-file'>{s['source']}</div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append(used_sources)

# ── Web search toggle + footer (renders inside stBottom, below chat input) ────
col_toggle, col_footer = st.columns([1, 1])
with col_toggle:
    st.session_state.use_web = st.toggle("Web search", value=st.session_state.use_web, key="web_toggle")
with col_footer:
    st.markdown("""
<div style="text-align:right; font-size:0.68rem; color:var(--text-muted); padding-top:0.45rem; letter-spacing:0.01em;">
    Connor Cunningham &nbsp;·&nbsp; 2026
</div>""", unsafe_allow_html=True)