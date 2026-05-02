"""
app.py — Yakima Fisheries RAG — Streamlit App
----------------------------------------------
6-step retrieval pipeline:
  1. Query expansion   (Claude Haiku → 3 alt phrasings)
  2. Vector search     (ChromaDB, 25 results × 4 queries)
  3. BM25 search       (keyword index, 25 results)
  4. Deduplication     (merge by parent_id)
  5. Reranking         (CrossEncoder ms-marco-MiniLM-L-6-v2, keep top 12)
  6. Generation        (Claude Sonnet, cited answer)

Run locally:
    py -m streamlit run app.py
"""

import os
import re
import sys
import pickle
import time
import traceback

from typing import Tuple

# Log startup progress to stderr so it shows up in Streamlit Cloud runtime logs.
print("[startup] python booting...", flush=True)

import streamlit as st
print("[startup] streamlit imported", flush=True)

try:
    import chromadb
    print("[startup] chromadb imported", flush=True)
    from sentence_transformers import CrossEncoder, SentenceTransformer
    print("[startup] sentence_transformers imported", flush=True)
    import anthropic
    print("[startup] anthropic imported", flush=True)
except Exception as e:
    print(f"[startup] IMPORT FAILED: {e}", flush=True)
    traceback.print_exc()
    raise

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Yakima Fisheries Bot",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "yakima"
CHROMA_DIR       = "./chroma_db"
BM25_PATH        = "./bm25_index.pkl"
EMBED_MODEL      = "BAAI/bge-large-en-v1.5"  # must match ingestheavy.py — 1024-dim
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_TOP_K     = 25
BM25_TOP_K       = 25
RERANK_TOP_N     = 12
HAIKU_MODEL      = "claude-haiku-4-5"
SONNET_MODEL     = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are a senior fisheries biologist with deep expertise in Bull Trout (Salvelinus confluentus) ecology, movement, and conservation in the Yakima Basin. You are assisting U.S. Fish and Wildlife Service (USFWS) biologists, researchers, and technical staff. Your role is to provide precise, literature-grounded, management-relevant answers.

CORE PRINCIPLES:
- Prioritize accuracy, specificity, and technical rigor over simplicity
- Treat all responses as if they will inform management decisions or scientific writing
- Only use information supported by the provided context (RAG excerpts). Do not invent or infer beyond the evidence

CITATION REQUIREMENTS:
- Cite sources using author-style citations with full title/year context when available (e.g., "Smith et al. (2001)" or "(Jones & Brown, 1998)")
- Tie every key claim to a citation
- If multiple sources support a claim, cite them together
- If sources disagree, explicitly describe the disagreement and cite both sides

TEMPORAL CONTEXT:
- Explicitly note when findings are based on older literature (especially pre-2000)
- Highlight when more recent studies refine, contradict, or update earlier conclusions
- When trends over time are evident, describe how understanding has changed

QUANTITATIVE DETAIL:
- Always include specific values when available:
  - Temperatures (°C)
  - Distances (km, m)
  - Flow/discharge (cfs or m³/s)
  - Elevation changes (ft or m)
  - Survival or entrainment rates
  - Sample sizes and study duration
- Avoid qualitative phrases when quantitative data exists

SYNTHESIS AND COMPARISON:
- Integrate across studies rather than summarizing them independently
- Identify consensus, uncertainty, and gaps
- Explicitly state when findings are consistent across systems vs. context-dependent (e.g., reservoir vs. riverine populations)
- When relevant, relate findings to Yakima Basin systems (Keechelus, Kachess, Cle Elum, Rimrock)

UNCERTAINTY AND LIMITATIONS:
- Clearly state when the available excerpts are insufficient
- Identify what is missing (e.g., seasonal resolution, sample size, spatial scale, life stage)
- Suggest what type of study or data would resolve the uncertainty

STYLE AND STRUCTURE:
- Use professional biological terminology appropriate for agency scientists
- Avoid conversational tone
- Organize complex responses with clear sections such as:
  - Background
  - Key Findings
  - Mechanisms
  - Management Implications
  - Uncertainty / Data Gaps
- Be concise but information-dense

MANAGEMENT RELEVANCE:
- When appropriate, translate findings into implications for:
  - Dam operations and entrainment risk
  - Habitat use and thermal refugia
  - Trap-and-haul or passage strategies
  - Monitoring (PIT, acoustic telemetry)
- Do not speculate beyond what the literature supports

STRICT GROUNDING:
- Do not use outside knowledge unless explicitly instructed
- Do not hallucinate citations or results
- If the provided excerpts do not support a claim, state that directly

FAILURE MODE:
- If the context is insufficient to answer the question:
  - Say exactly what is missing
  - Provide a minimal partial answer only if supported
  - Recommend the type of source needed (e.g., telemetry study, thermal habitat study, entrainment modeling paper)

Your responses should reflect the level of detail and rigor expected in a technical briefing or manuscript discussion section, not a general summary.
"""

WEB_SEARCH_SYSTEM = """\
You are a fisheries science assistant. The user's question has been supplemented
with recent web search results in addition to the literature library.

PRIORITY WEB SOURCE — Yakipedia (ybfwrb.org/yakipedia):
- Yakipedia is the curated MediaWiki maintained by the Yakima Basin Fish and
  Wildlife Recovery Board (YBFWRB). It is the authoritative, up-to-date
  reference for bull trout populations, FMO habitat areas, reservoirs
  (Rimrock, Bumping, Cle Elum, Kachess, Keechelus), tributaries, management
  actions, basin acronyms, and agency programs in the Yakima Basin.
- When any web snippet comes from ybfwrb.org, treat it as the highest-authority
  web source. Prefer it over general web results when they conflict, unless
  the general result is a peer-reviewed paper or a more recent official agency
  document.
- When citing Yakipedia inline, use the format:
    (Yakipedia: <Page Title>, YBFWRB 2025)
- If Yakipedia and the PDF literature disagree on a number or date, surface the
  disagreement explicitly rather than silently picking one — Yakipedia is often
  more current, the literature is often more rigorously cited.

General rules:
- Clearly distinguish between findings from the literature vs. web sources.
- Always cite sources inline.
"""

# ── CSS / Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }


  /* Center content in a readable column */
  .block-container {
    max-width: 860px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
  }

  /* Header */
  .app-header {
    padding: 1.5rem 0 0.5rem 0;
    border-bottom: 2px solid #e87722;
    margin-bottom: 1.5rem;
    text-align: center;
  }
  .app-title {
    font-size: 3.6rem;
    font-weight: 700;
    color: #e87722;
    margin: 0;
    text-align: center;
  }

  /* Chat messages */
  .user-msg {
    background: #3a3a3a;
    border-left: 3px solid #e87722;
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 1rem;
  }
  .assistant-msg {
    background: #333333;
    border-left: 3px solid #c45e0a;
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.5rem;
  }

  /* Source cards */
  .source-card {
    background: #3a3a3a;
    border: 1px solid #555555;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #c0c0c0;
  }
  .source-title { color: #e87722; font-weight: 500; }
  .source-year  { color: #999999; margin-left: 0.5rem; }

  /* Input area */
  .stTextInput > div > div > input {
    background-color: #3a3a3a !important;
    border: 1px solid #555555 !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }
  .stButton > button {
    background-color: #e87722;
    color: white;
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .stButton > button:hover { background-color: #c45e0a; }

  /* Spinner */
  .status-text { color: #6c7280; font-size: 0.85rem; font-style: italic; }

  /* Source excerpt — shown when a source card expander is opened */
  .source-excerpt {
    background: #2b2b2b;
    border-left: 3px solid #e87722;
    padding: 0.7rem 0.9rem;
    color: #d5d5d5;
    font-size: 0.85rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    border-radius: 0 4px 4px 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Password Gate ─────────────────────────────────────────────────────────────

def check_password() -> bool:
    """Returns True once the correct password has been entered.

    If no APP_PASSWORD secret is set, auth is skipped entirely so anyone
    with the URL can use the app.
    """
    expected = st.secrets.get("APP_PASSWORD", "")
    if not expected:
        return True  # no password configured → open access
    if st.session_state.get("authenticated"):
        return True

    st.markdown('<div class="app-header"><p class="app-title">Yakima Fisheries Bot</p></div>', unsafe_allow_html=True)
    st.markdown("#### Enter access password")

    pw = st.text_input("Password", type="password", key="pw_input")
    if st.button("Login"):
        if pw == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ── Resource Loading (cached) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_resources():
    """Load all models and indexes once at startup."""
    print("[load] starting load_resources()", flush=True)

    # ── Download from Hugging Face if not present locally ──────────────────
    if not os.path.exists(CHROMA_DIR) or not os.path.exists(BM25_PATH):
        print(f"[load] {CHROMA_DIR} or {BM25_PATH} missing, downloading from HF...", flush=True)
        hf_token  = st.secrets.get("HF_TOKEN", "")
        hf_repo   = st.secrets.get("HF_REPO", "")
        if hf_token and hf_repo:
            from huggingface_hub import snapshot_download
            print(f"[load] snapshot_download from {hf_repo}...", flush=True)
            snapshot_download(
                repo_id=hf_repo,
                repo_type="dataset",
                local_dir=".",
                token=hf_token,
            )
            print("[load] snapshot_download complete", flush=True)
        else:
            st.error(
                "Database not found locally and HF_TOKEN / HF_REPO not set in secrets. "
                "Run ingest.py first, or configure Hugging Face credentials."
            )
            st.stop()
    else:
        print("[load] chroma_db and bm25 already present locally", flush=True)

    # ── ChromaDB ───────────────────────────────────────────────────────────
    # The persisted collection was created without an explicit embedding function,
    # so we load it with no embedding function and embed queries ourselves.
    print("[load] opening ChromaDB PersistentClient...", flush=True)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    print("[load] loading SentenceTransformer (BGE-large, ~1.3GB)...", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("[load] SentenceTransformer ready", flush=True)

    try:
        collection = chroma.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        existing = [c.name for c in chroma.list_collections()]
        st.error(
            f"ChromaDB collection '{COLLECTION_NAME}' not found at {CHROMA_DIR}.\n\n"
            f"Collections present: {existing or '(none)'}\n"
            f"Underlying error: {e}"
        )
        st.stop()

    # ── BM25 ───────────────────────────────────────────────────────────────
    print("[load] loading BM25 pickle...", flush=True)
    with open(BM25_PATH, "rb") as f:
        bm25_payload = pickle.load(f)
    bm25          = bm25_payload["bm25"]
    bm25_texts    = bm25_payload["texts"]
    bm25_metadata = bm25_payload["metadata"]
    print(f"[load] BM25 loaded ({len(bm25_texts)} docs)", flush=True)

    # ── CrossEncoder ───────────────────────────────────────────────────────
    print("[load] loading CrossEncoder reranker...", flush=True)
    reranker = CrossEncoder(RERANK_MODEL)
    print("[load] CrossEncoder ready", flush=True)

    # ── Anthropic client ───────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    print("[load] load_resources() done", flush=True)

    return collection, embed_model, bm25, bm25_texts, bm25_metadata, reranker, client


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def expand_query(question: str, client: anthropic.Anthropic) -> Tuple[List[str], List[str]]:
    """Step 1: Use Claude Haiku to generate 3 alternative search queries and 4 HyDE answers."""
    # Generate 3 alternative questions
    alt_questions_prompt = (
        f"Generate 3 alternative search queries for a fisheries literature database. "
        f"Each query should use different terminology but retrieve relevant papers. "
        f"Return ONLY the 3 queries, one per line, no numbering or extra text.\n\n"
        f"Original question: {question}"
    )
    alt_questions_resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": alt_questions_prompt}],
    )
    alt_q_lines = alt_questions_resp.content[0].text.strip().splitlines()
    alt_queries = [q.strip() for q in alt_q_lines if q.strip()][:3]
    all_questions = [question] + alt_queries

    # Generate HyDE answers for all 4 questions
    hyde_prompt_parts = []
    for i, q in enumerate(all_questions, 1):
        hyde_prompt_parts.append(f"{i}. {q}")

    hyde_prompt = (
        f"For each of the following {len(all_questions)} questions, write a 2-3 sentence passage "
        f"that reads like a results or discussion section from a peer-reviewed fisheries paper. "
        f"Use technical fisheries terminology. Return ONLY the {len(all_questions)} passages, "
        f"numbered 1 through {len(all_questions)}, with no preamble and no extra text.\n\n"
        + "\n".join(hyde_prompt_parts)
    )

    hyde_resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": hyde_prompt}],
    )

    # Parse HyDE answers
    raw_hyde_answers = hyde_resp.content[0].text.strip()
    parsed_hyde_answers = re.split(r'^\d+\.\s*', raw_hyde_answers, flags=re.MULTILINE)
    hyde_answers = [ans.strip() for ans in parsed_hyde_answers if ans.strip()]

    # Fallback if parsing fails
    if len(hyde_answers) < len(all_questions):
        print(f"Warning: HyDE answer parsing failed, expected {len(all_questions)} but got {len(hyde_answers)}. Falling back to all_questions for HyDE.")
        hyde_answers = all_questions

    return all_questions, hyde_answers


def vector_search(
    queries: list[str],
    collection,
    embed_model: SentenceTransformer,
    top_k: int = VECTOR_TOP_K,
) -> list[dict]:
    """Step 2: Run vector search for each query, collect results."""
    # Embed all queries in one batch (faster than one-by-one).
    # normalize_embeddings=True matches ingestheavy.py so cosine similarity works correctly.
    embeddings = embed_model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    candidates = {}  # parent_id → result dict (dedup by parent)
    for q, emb in zip(queries, embeddings):
        results = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            pid = meta.get("parent_id", doc[:40])
            if pid not in candidates:
                candidates[pid] = {
                    "parent_id":   pid,
                    "parent_text": meta.get("parent_text", doc),
                    "source":      meta.get("source", ""),
                    "title":       meta.get("title", ""),
                    "year":        meta.get("year", ""),
                    "score":       1 - dist,  # cosine similarity
                }
    return list(candidates.values())


def bm25_search(
    question: str,
    bm25,
    bm25_texts: list[str],
    bm25_metadata: list[dict],
    top_k: int = BM25_TOP_K,
) -> list[dict]:
    """Step 3: BM25 keyword search on the original question."""
    tokens = question.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        if scores[i] <= 0:
            continue
        meta = bm25_metadata[i]
        results.append({
            "parent_id":   meta.get("parent_id", f"bm25_{i}"),
            "parent_text": bm25_texts[i],
            "source":      meta.get("source", ""),
            "title":       meta.get("title", ""),
            "year":        meta.get("year", ""),
            "score":       float(scores[i]),
        })
    return results


def deduplicate(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    """Step 4: Merge vector + BM25 results, deduplicate by parent_id."""
    seen = {}
    for r in vector_results + bm25_results:
        pid = r["parent_id"]
        if pid not in seen:
            seen[pid] = r
    return list(seen.values())


def rerank(
    question: str,
    candidates: list[dict],
    reranker: CrossEncoder,
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """Step 5: CrossEncoder reranking — keep top_n by relevance score."""
    if not candidates:
        return []
    pairs  = [(question, c["parent_text"]) for c in candidates]
    scores = reranker.predict(pairs)
    scores = [float(s) for s in scores]  # ensure plain floats

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for _, c in ranked[:top_n]]


def build_context(chunks: list[dict]) -> str:
    """Format reranked chunks into a numbered context block for Claude."""
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['title']} ({c['year']}) — {c['source']}"
        parts.append(f"{header}\n{c['parent_text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    context: str,
    history: list[dict],
    client: anthropic.Anthropic,
    use_web: bool = False,
    web_snippets: str = "",
) -> str:
    """Step 6: Generate a cited answer using Claude Sonnet with prompt caching."""
    system = WEB_SEARCH_SYSTEM if use_web else SYSTEM_PROMPT

    literature_block = (
        f"<literature>\n{context}\n</literature>"
    )
    web_block = (
        f"\n\n<web_results>\n{web_snippets}\n</web_results>"
        if use_web and web_snippets else ""
    )

    # Build message history (conversation memory)
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Current turn — literature context as first user message in this turn
    user_content = (
        f"{literature_block}{web_block}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    resp = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=6000,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )
    return resp.content[0].text


def web_search_snippets(question: str) -> str:
    """
    Fetch brief web search results using DuckDuckGo (no API key needed).

    Yakipedia (ybfwrb.org) is queried first as the authoritative Yakima
    Basin source; general fisheries results are appended below, clearly
    labeled so the LLM can weight them appropriately.
    """
    import urllib.parse, urllib.request, html

    def _fetch(query: str, limit: int = 5) -> list[str]:
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                body = r.read().decode("utf-8", errors="ignore")
            raw = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
            out = []
            for s in raw[:limit]:
                s = re.sub(r"<[^>]+>", "", s)
                s = html.unescape(s).strip()
                if s:
                    out.append(s)
            return out
        except Exception:
            return []

    # 1) Yakipedia-first pass (authoritative YBFWRB wiki)
    yaki = _fetch(f"site:ybfwrb.org {question}", limit=5)

    # 2) General fisheries pass
    general = _fetch(f"{question} fisheries", limit=5)

    parts = []
    if yaki:
        parts.append(
            "[Yakipedia — ybfwrb.org — AUTHORITATIVE YBFWRB SOURCE]\n"
            + "\n\n".join(yaki)
        )
    if general:
        parts.append(
            "[General web results]\n"
            + "\n\n".join(general)
        )
    return "\n\n---\n\n".join(parts)


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Kick off resource loading first so the cache warms while the user
    # is entering their password (or immediately if no password is set).
    # After the first cold start, @st.cache_resource makes this a no-op.
    with st.spinner("Loading models and database…"):
        collection, embed_model, bm25, bm25_texts, bm25_meta, reranker, client = load_resources()

    if not check_password():
        return

    # ── Session state ─────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []   # full chat history
    if "history" not in st.session_state:
        st.session_state.history  = []   # Claude message history (role/content)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-header">'
        '<p class="app-title">Yakima Fisheries Bot</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Render chat history ───────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">🧑‍🔬 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Source cards — each source is its own expander that reveals
            # the exact paragraph (parent_text) the answer drew from.
            if msg.get("sources"):
                st.markdown(
                    f'<p style="color:#999999;font-size:0.85rem;margin:0.5rem 0 0.25rem 0;">'
                    f'Sources ({len(msg["sources"])} passages) — click any to view the exact excerpt'
                    f'</p>',
                    unsafe_allow_html=True,
                )
                for i, src in enumerate(msg["sources"], 1):
                    header = (
                        f"[{i}] {src.get('title') or src.get('source', 'Untitled')}"
                        f"  ({src.get('year', 'n.d.')})"
                    )
                    with st.expander(header, expanded=False):
                        filename = src.get("source", "")
                        excerpt  = src.get("parent_text", "") or "(no excerpt available)"
                        if filename:
                            st.markdown(
                                f'<p style="color:#999999;font-size:0.75rem;margin:0 0 0.5rem 0;">'
                                f'File: <code>{filename}</code></p>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f'<div class="source-excerpt">{excerpt}</div>',
                            unsafe_allow_html=True,
                        )

    # ── Input area ────────────────────────────────────────────────────────
    st.divider()

    # ── Input area (form enables Enter-key submission) ─────────────────────
    with st.form("question_form", clear_on_submit=True):
        use_web = st.toggle(
            "🌐 Web", value=False,
            help="Supplement with live web search results",
        )
        question = st.text_input(
            "Ask a question about Yakima Basin fisheries…",
            key="question_input",
            label_visibility="collapsed",
            placeholder="e.g. What are the primary habitat requirements for bull trout spawning?",
        )
        ask_btn = st.form_submit_button("Ask →")

    if not (ask_btn and question.strip()):
        if not st.session_state.messages:
            st.markdown(
                '<p style="color:#999999;font-size:0.85rem;text-align:center;margin-top:2rem;">'
                'Connor Cunningham 2026'
                '</p>',
                unsafe_allow_html=True,
            )
        return

    question = question.strip()

    # ── Run pipeline ──────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        all_questions, hyde_answers = expand_query(question, client)
        vec_results   = vector_search(hyde_answers, collection, embed_model)
        bm25_results  = bm25_search(question, bm25, bm25_texts, bm25_meta)
        candidates    = deduplicate(vec_results, bm25_results)
        top_chunks    = rerank(question, candidates, reranker)
        snippets      = web_search_snippets(question) if use_web else ""
        context       = build_context(top_chunks)
        answer        = generate_answer(
            question,
            context,
            st.session_state.history,
            client,
            use_web=use_web,
            web_snippets=snippets,
        )

    # ── Save to history ───────────────────────────────────────────────────
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Deduplicate sources for display — keep parent_text so the UI can
    # expand each source card and show the exact paragraph used.
    seen_src = {}
    for c in top_chunks:
        pid = c["parent_id"]
        if pid not in seen_src:
            seen_src[pid] = {
                "title":       c.get("title", ""),
                "year":        c.get("year", ""),
                "source":      c.get("source", ""),
                "parent_text": c.get("parent_text", ""),
            }
    sources = list(seen_src.values())

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

    st.rerun()


if __name__ == "__main__":
    main()
