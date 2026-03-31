import streamlit as st
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle

st.set_page_config(
    page_title="Yakima Fisheries Literature",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset & dark base ───────────────────────────────────── */
:root {
    --bg-base:      #0f1117;
    --bg-surface:   #1a1d27;
    --bg-elevated:  #21242f;
    --border:       #2d3142;
    --border-light: #363b52;
    --text-primary: #e8eaf0;
    --text-secondary: #8b90a7;
    --text-muted:   #565c7a;
    --accent:       #5b8dee;
    --accent-dim:   #5b8dee22;
    --green:        #4ade80;
    --green-dim:    #4ade8022;
    --font-sans:    'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:    'JetBrains Mono', 'Fira Code', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 900px !important;
}

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 0.75rem 0 !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Password input ────────────────────────────────────────────── */
.stTextInput input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.85rem !important;
    transition: border-color 0.15s ease;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ── Chat messages ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.1rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"] p {
    color: var(--text-primary) !important;
    line-height: 1.65 !important;
    font-size: 0.9rem !important;
}

/* ── Chat input bar ────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    transition: border-color 0.15s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.9rem !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Bottom toolbar (web toggle row) ──────────────────────────── */
.toolbar-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.6rem 0.35rem 0;
    margin-bottom: 0.3rem;
}
/* Override Streamlit toggle to look compact in the toolbar */
.toolbar-row [data-testid="stToggle"] {
    background: transparent !important;
}
.toolbar-row [data-testid="stToggle"] label {
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    cursor: pointer;
}
.toolbar-row [data-testid="stToggle"] label:hover {
    color: var(--text-primary) !important;
}

/* ── Source expanders ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0 !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--accent) !important;
    font-size: 0.8rem !important;
    font-family: var(--font-mono) !important;
    font-weight: 500 !important;
    padding: 0.6rem 0.85rem !important;
}
[data-testid="stExpander"] summary:hover {
    background: var(--bg-surface) !important;
}

/* ── Source cards inside expander ─────────────────────────────── */
.source-row {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 3.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.15rem 0.45rem;
    border-radius: 5px;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.source-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.4;
}
.source-file {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-top: 0.15rem;
}

/* ── Status steps ──────────────────────────────────────────────── */
.status-step {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-secondary);
    padding: 0.3rem 0;
}
.status-step::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── App header ────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding-bottom: 0.9rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.app-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.app-badge {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
}

/* ── Metrics ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.35rem !important;
    font-family: var(--font-mono) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: var(--font-sans) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Hide sidebar entirely ─────────────────────────────────────── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 820px !important; }

/* ── Footer ────────────────────────────────────────────────────── */
.app-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    letter-spacing: 0.03em;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Responsive ────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.75rem !important;
    }
    .app-title { font-size: 1rem; }
    [data-testid="stChatMessage"] { padding: 0.75rem 0.85rem !important; }
    .source-badge { font-size: 0.62rem; min-width: 3rem; }
    .status-step { font-size: 0.72rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Password gate ───────────────────────────────────────────────────────────
password = st.text_input("Access key:", type="password", placeholder="Enter password...")
if password != "yakima2026":
    st.stop()

# ── Load resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="yakima")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
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
    <div class="app-title">🐟 Yakima Fisheries Literature</div>
    <div class="app-badge">Bull Trout RAG</div>
</div>
""", unsafe_allow_html=True)

# ── Chat ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_history" not in st.session_state:
    st.session_state.source_history = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i // 2 < len(st.session_state.source_history):
            sources = st.session_state.source_history[i // 2]
            if sources:
                n = len(sources)
                with st.expander(f"📄 {n} source{'s' if n != 1 else ''} retrieved"):
                    for s in sources:
                        title = (s['title'][:80] + '…') if len(s['title']) > 80 else s['title']
                        display = title or s['source']
                        st.markdown(
                            f"<div class='source-row'>"
                            f"<span class='source-badge'>{s['year'] or '—'}</span>"
                            f"<div><div class='source-title'>{display}</div>"
                            f"<div class='source-file'>{s['source']}</div></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

# ── Web search toggle (rendered in Streamlit's bottom bar) ──────────────────
st.markdown("<div class='toolbar-row'>", unsafe_allow_html=True)
use_web = st.toggle("🌐 Include web search", value=False, key="web_toggle")
st.markdown("</div>", unsafe_allow_html=True)

if question := st.chat_input("Ask about Yakima fisheries..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()

        status.markdown("<div class='status-step'>Expanding query…</div>", unsafe_allow_html=True)
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

        status.markdown("<div class='status-step'>Retrieving literature (vector + keyword)…</div>", unsafe_allow_html=True)
        candidate_docs, candidate_metas = hybrid_retrieve(
            queries, collection, embed_model,
            bm25, bm25_texts, bm25_metadata
        )

        status.markdown("<div class='status-step'>Reranking results…</div>", unsafe_allow_html=True)
        pairs = [[question, doc] for doc in candidate_docs]
        scores = rerank_model.predict(pairs)
        scores = [float(s) for s in scores]
        ranked = sorted(zip(scores, candidate_docs, candidate_metas), key=lambda x: x[0], reverse=True)
        top = ranked[:12]

        status.markdown("<div class='status-step'>Generating answer…</div>", unsafe_allow_html=True)

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
        if use_web:
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
        with st.expander(f"📄 {n} source{'s' if n != 1 else ''} retrieved"):
            for s in used_sources:
                title = (s['title'][:80] + '…') if len(s['title']) > 80 else s['title']
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

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Created by Connor Cunningham · 2026
</div>
""", unsafe_allow_html=True)