import streamlit as st
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle

st.set_page_config(
    page_title="Yakima Fisheries Literature",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1100px;}

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {color: #e6edf3 !important;}

.stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stChatMessage"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    margin-bottom: 0.4rem;
}
[data-testid="stExpander"] summary {
    color: #58a6ff !important;
    font-size: 0.85rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stTabs"] button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #8b949e !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3fb950 !important;
    border-bottom: 2px solid #3fb950 !important;
}

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] {color: #8b949e !important; font-size: 0.75rem !important;}
[data-testid="stMetricValue"] {color: #3fb950 !important; font-size: 1.4rem !important; font-family: 'IBM Plex Mono', monospace !important;}

[data-testid="stCheckbox"] label {color: #8b949e !important; font-size: 0.85rem !important;}

.status-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #3fb950;
    padding: 0.25rem 0;
}

.app-header {
    border-bottom: 1px solid #21262d;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.8rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}

.source-badge {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #3fb95033;
    color: #3fb950;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    margin: 0.15rem;
}

.conf-bar {
    height: 4px;
    background: #21262d;
    border-radius: 2px;
    margin-top: 0.5rem;
}
.conf-fill {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #3fb950, #58a6ff);
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.75rem !important;
    }
    .app-title { font-size: 1.1rem; }
    .app-subtitle { font-size: 0.7rem; }
    [data-testid="stChatMessage"] { padding: 0.75rem; }
    .source-badge { font-size: 0.65rem; }
    [data-testid="stExpander"] summary { font-size: 0.78rem !important; }
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

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Tips")
    st.markdown("<span style='color:#8b949e;font-size:0.8rem'>Ask specific questions about tributaries, temperature thresholds, population counts, or management actions for best results.</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Database")
    st.markdown("<span style='color:#8b949e;font-size:0.8rem;font-family:IBM Plex Mono'>~310 papers indexed<br>Hybrid vector + BM25</span>", unsafe_allow_html=True)

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
                with st.expander(f"📄 {n} sources used"):
                    for s in sources:
                        st.markdown(f"<span class='source-badge'>{s['year']}</span> **{s['title'][:70] or s['source']}**<br><span style='color:#8b949e;font-size:0.75rem;font-family:IBM Plex Mono'>{s['source']}</span>", unsafe_allow_html=True)
                        st.markdown("---")

# ── Fixed bottom bar with web toggle ────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stBottom"] {
    padding-bottom: 0.5rem;
}
.web-toggle-bar {
    position: fixed;
    bottom: 70px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 999;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    use_web = st.toggle("🌐 Include web search", value=False, key="web_toggle")

if question := st.chat_input("Ask about Yakima fisheries..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()

        status.markdown("<div class='status-step'>⟳ Expanding query...</div>", unsafe_allow_html=True)
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

        status.markdown("<div class='status-step'>⟳ Retrieving literature (vector + keyword)...</div>", unsafe_allow_html=True)
        candidate_docs, candidate_metas = hybrid_retrieve(
            queries, collection, embed_model,
            bm25, bm25_texts, bm25_metadata
        )

        status.markdown("<div class='status-step'>⟳ Reranking results...</div>", unsafe_allow_html=True)
        pairs = [[question, doc] for doc in candidate_docs]
        scores = rerank_model.predict(pairs)
        scores = [float(s) for s in scores]
        ranked = sorted(zip(scores, candidate_docs, candidate_metas), key=lambda x: x[0], reverse=True)
        top = ranked[:12]

        status.markdown("<div class='status-step'>⟳ Generating answer...</div>", unsafe_allow_html=True)

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
        with st.expander(f"📄 {n} sources used"):
            for s in used_sources:
                st.markdown(f"<span class='source-badge'>{s['year']}</span> **{s['title'][:70] or s['source']}**<br><span style='color:#8b949e;font-size:0.75rem;font-family:IBM Plex Mono'>{s['source']}</span>", unsafe_allow_html=True)
                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append(used_sources)