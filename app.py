import streamlit as st
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle

st.set_page_config(
    page_title="Yakima Fisheries Literature",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── Design tokens ─────────────────────────────────────────────── */
:root {
    --bg:           #212121;
    --bg-input:     #2f2f2f;
    --bg-user:      #2f2f2f;
    --bg-chip:      #383838;
    --bg-chip-on:   #4a4a4a;
    --border:       #3a3a3a;
    --text:         #ececec;
    --text-sub:     #9b9b9b;
    --text-muted:   #686868;
    --accent:       #10a37f;
    --font:         'Inter', -apple-system, BlinkMacSystemFont, ui-sans-serif, sans-serif;
}

/* ── Base ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]       { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container {
    max-width: 760px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 1rem !important;
}

/* ── Password screen ───────────────────────────────────────────── */
.stTextInput > label {
    font-size: 0.8rem !important;
    color: var(--text-sub) !important;
    font-weight: 400 !important;
    margin-bottom: 0.35rem !important;
}
.stTextInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.15s;
    outline: none !important;
    box-shadow: none !important;
}
.stTextInput input:focus {
    border-color: #555 !important;
    box-shadow: none !important;
}

/* ── Chat messages ─────────────────────────────────────────────── */
/* No borders, no boxes — flat like Claude/ChatGPT */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.25rem 0 !important;
    margin-bottom: 0 !important;
    gap: 0.85rem !important;
}
/* User bubble gets a pill background */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] + div,
[data-testid="stChatMessage"][aria-label*="user"] .stMarkdown {
    background: var(--bg-user) !important;
    border-radius: 18px !important;
    padding: 0.65rem 1rem !important;
    display: inline-block;
    max-width: 85%;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td {
    color: var(--text) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: var(--text) !important;
    margin-top: 1.1rem !important;
}
/* Avatar icons */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--bg-chip) !important;
    border-radius: 50% !important;
    width: 30px !important;
    height: 30px !important;
    font-size: 0.8rem !important;
    flex-shrink: 0 !important;
}

/* ── Chat input ────────────────────────────────────────────────── */
[data-testid="stBottom"] {
    background: var(--bg) !important;
    padding-bottom: 0.5rem !important;
}
[data-testid="stChatInput"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    transition: border-color 0.15s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #555 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    background: transparent !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInput"] button {
    background: var(--text) !important;
    border-radius: 8px !important;
    color: var(--bg) !important;
}

/* ── Web toggle chip ───────────────────────────────────────────── */
/* Sits just above the input bar */
div[data-testid="stToggle"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stToggle"] label {
    font-size: 0.78rem !important;
    color: var(--text-sub) !important;
    font-weight: 400 !important;
    cursor: pointer !important;
}
div[data-testid="stToggle"] label:hover {
    color: var(--text) !important;
}
/* Style the toggle track */
div[data-testid="stToggle"] span[data-baseweb="checkbox"] {
    background: var(--bg-chip) !important;
    border: 1px solid var(--border) !important;
    border-radius: 99px !important;
}

/* ── Source expander ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-sub) !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.5rem 0.85rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--text) !important;
    background: transparent !important;
}
[data-testid="stExpander"] svg { color: var(--text-muted) !important; }

/* ── Source rows ───────────────────────────────────────────────── */
.source-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-badge {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-sub);
    background: var(--bg-chip);
    border-radius: 4px;
    padding: 0.15rem 0.45rem;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 0.15rem;
}
.source-title {
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.45;
}
.source-file {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}

/* ── Status indicator ──────────────────────────────────────────── */
.status-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: var(--text-sub);
    padding: 0.2rem 0;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.status-step::before {
    content: '';
    width: 12px; height: 12px;
    border: 1.5px solid var(--border);
    border-top-color: var(--text-sub);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}

/* ── App header ────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    margin-bottom: 2rem;
}
.app-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
}

/* ── Footer ────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 1.5rem;
    padding-top: 1rem;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Mobile ────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding-top: 1rem !important; }
    .app-title { font-size: 1rem; }
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
    <div class="app-title">Yakima Fisheries Literature</div>
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

# ── Bottom input area ────────────────────────────────────────────────────────
with st.container():
    use_web = st.toggle("Search the web", value=False, key="web_toggle")

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