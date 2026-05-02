# Batched Graph Explorer Implementation Spec — Yakima Fisheries RAG — app.py

This spec covers four sequential phases: entity extraction, graph build, node summary generation, and the Streamlit graph explorer UI. Each phase is a discrete script or module. The chat pipeline in app.py is not touched until the final phase which adds a second page via st.navigation.

---

## Phase 1 — Entity extraction from ChromaDB

Write a standalone script called `extract_entities.py`. This script queries ChromaDB directly — do **not** read PDF files. Pull all unique parent chunks from the existing collection by querying all documents and deduplicating on `parent_id` from metadata. This gives you the full corpus as clean parent-level text without re-ingesting anything.

For each unique parent chunk, make a Haiku call asking it to extract named entities from the text. The prompt should ask for entities in these categories:
- geographic locations (rivers, lakes, reservoirs, watersheds)
- infrastructure (dams, diversion structures)
- species and life stages
- management actions and programs
- monitoring methods
- agencies and organizations

Instruct Haiku to return **only** a JSON list of entity strings, no explanation, no preamble. Set `max_tokens` to 200.

Store the result as a dict mapping `parent_id` → list of raw extracted entity strings. Persist to disk as `extracted_entities.json`. This is the raw extraction output before normalization.

After extraction is complete across all parent chunks, run a normalization pass:
1. Collect every unique raw entity string across the entire extraction output
2. For each unique string, run `rapidfuzz` `token_sort_ratio` against all other unique strings
3. Collapse any pair scoring **88 or above** into a single canonical form — use the longer or more complete string as the canonical
4. Result is a normalization lookup dict: raw string → canonical string
5. Apply this lookup to `extracted_entities.json` to produce `normalized_entities.json` — same structure but all entity strings replaced with their canonical forms

Normalize with union-find for efficiency. Note the 88 threshold in a code comment so it can be tuned later (upward if over-collapsing, downward if under-collapsing).

---

## Phase 2 — Graph build

Write a standalone script called `build_graph.py`. Reads `normalized_entities.json` and builds a NetworkX undirected weighted graph.

For each paper — identified by `source` in ChromaDB metadata — collect all canonical entities that appear in any of its parent chunks. For every pair of entities that co-occur in the same paper, add or increment an edge between them with weight = number of papers both entities co-occur in.

**Node attributes:**
- `papers`: list of source filenames for every paper mentioning this entity
- `category`: inferred entity category — use heuristic keyword matching against these sets:
  - **species**: "trout", "salmon", "bull trout", "smolt", "parr", "redd", "anadromous", "oncorhynchus", "salvelinus", "kokanee", "lamprey", etc.
  - **infrastructure**: "dam", "diversion", "weir", "fish ladder", "trap-and-haul", "spillway", "screen", "culvert", etc.
  - **location**: "river", "creek", "lake", "reservoir", "basin", "watershed", "tributary", "reach", "fork", "valley", etc.
  - **management**: "recovery", "restoration", "conservation", "habitat", "reintroduction", "supplementation", "listing", "esa", "hatchery", etc.
  - **monitoring**: "pit tag", "pit", "acoustic", "telemetry", "radio", "electrofishing", "mark-recapture", "monitoring", "survey", etc.
  - **agency**: "usfws", "wdfw", "noaa", "nmfs", "bureau of reclamation", "ybfwrb", etc.
  - If nothing matches: category = "other"

**Output files:**
- `graph.pkl` — NetworkX graph via pickle
- `graph_stats.txt` — total nodes, total edges, 10 highest-weight edges, 10 nodes with most connections

---

## Phase 3 — Node summary generation

Write a standalone script called `generate_summaries.py`. Runs after the graph is built. For each node in the graph:

1. Read the node's `papers` attribute (list of source filenames)
2. For each paper, query ChromaDB for the `abstract` field in metadata
3. If abstract is absent or empty, fall back to pulling the top 2 parent chunks for that paper from ChromaDB (highest character count as proxy for info density)
4. Assemble collected abstracts + fallback chunks into a single input block
5. Make a Haiku call asking it to write a 3–4 sentence synthesis of what the scientific literature collectively says about this entity. Prompt should specify:
   - Use technical fisheries language
   - Include specific findings where the input supports them
   - Note contradictions or gaps across studies if present
   - Write in third person present tense as if for a reference document
   - Abstracts are author-written and grounded, so no hallucination concerns

Store summaries as dict: canonical entity name → summary string. Persist as `node_summaries.pkl` alongside `graph.pkl`. Log cost estimate: number of Haiku calls, approximate token count, approximate cost.

---

## Phase 4 — Graph explorer page in app.py

This phase modifies `app.py` and creates `graph_page.py`.

### app.py modifications

1. Wire up `st.navigation` at the top of `main()` or globally — define two pages:
   - Existing chat interface (default page) — move current `main()` logic into `def chat_page()` with **zero changes** to pipeline logic
   - New graph explorer as second page — `def graph_page():` which imports and calls `graph_page.run()` (or inline)
2. Navigation appears as minimal top bar or sidebar toggle, not intrusive to existing chat layout
3. When navigating from graph to chat with "Ask about this" button, pre-fill `st.session_state["graph_query"]` and the chat page uses it if set

### graph_page.py

Create `graph_page.py` with all graph explorer UI logic:

**On load:** Read `graph.pkl` and `node_summaries.pkl`. If missing, show message: "Graph not built yet. Run: `python extract_entities.py && python build_graph.py && python generate_summaries.py`"

**Default view:** Full graph rendered with PyVis via `st.components.v1.html`. Height=700px.
- Physics layout: Barnes-Hut default
- Node size scales with number of connections (degree centrality)
- Edge thickness scales with weight
- Node color by category — distinct color per category so species, locations, management are visually distinguishable

**Node click / Ego network:**
- When user clicks a node, filter to ego network (node + immediate neighbors only)
- Display node summary in a side panel below the graph
- Below summary, list papers mentioning this entity — show title, year, source formatted like source cards in chat
- "Ask about this in chat" button → stores entity name in `st.session_state["graph_query"]` so chat page pre-fills input

**Filters:**
- Search box above graph that filters to a specific node by name and centers view
- Entity type filter (dropdown/selectbox) to show only certain categories — updates PyVis render

**IMPORTANT — PyVis click limitation:** PyVis `st.components.v1.html` cannot directly relay JavaScript click events back to Python. Implement the click/ego-network interaction using a **Streamlit selectbox** as the node selector instead of relying on JS callbacks. When user selects a node from the dropdown, re-render with ego network filtered. Flag this limitation clearly in the file header.

**IMPORTANT — Large graph performance:** If the graph has hundreds+ of nodes, full render will be slow/visually unusable. If total nodes > 100, default to showing a **subselection** — either top N connected nodes, or require user to use the entity type filter first. Flag this in file header and implement a threshold check.

### What does NOT change

- `expand_query`, `vector_search`, `bm25_search`, `deduplicate`, `rerank`, `build_context`, `generate_answer` — completely unchanged
- `ingestheavy.py` — unchanged
- `bm25_index.pkl` — unchanged
- ChromaDB collection — unchanged
- All constants (VECTOR_TOP_K, RERANK_TOP_N, HAIKU_MODEL, SONNET_MODEL, etc.)

### requirements.txt

Add three lines:
```
networkx
pyvis
rapidfuzz
```

### Run order

`extract_entities.py` → `build_graph.py` → `generate_summaries.py`. Idempotent — rebuild from whatever's in ChromaDB. For incremental ingestion, re-run all three.

### Scripts should be standalone

`extract_entities.py`, `build_graph.py`, `generate_summaries.py` must work from terminal, not depend on Streamlit. They import `anthropic` directly and read `ANTHROPIC_API_KEY` from `os.environ`.
