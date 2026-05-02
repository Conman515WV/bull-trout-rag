"""
generate_summaries.py — Phase 3: Node summary generation
=========================================================
For each node in the graph, collects paper abstracts (or fallback parent chunks)
from ChromaDB and generates a 3–4 sentence synthesis via Haiku.

Outputs:
  - node_summaries.pkl  (dict: canonical entity name → summary string)

Requires: graph.pkl (Phase 2 output), ChromaDB with documents
Run:
  python generate_summaries.py
"""

import os
import json
import math
import pickle
import time
import chromadb
import networkx as nx
from anthropic import Anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "yakima"
GRAPH_PATH = "graph.pkl"
SUMMARIES_PATH = "node_summaries.pkl"
HAIKU_MODEL = "claude-haiku-4-5"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Haiku pricing (May 2026): ~$0.25 / 1M input tokens, ~$1.25 / 1M output tokens
COST_PER_1K_INPUT_TOKENS = 0.00025
COST_PER_1K_OUTPUT_TOKENS = 0.00125


def get_chroma_client():
    """Open ChromaDB and return the collection."""
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION_NAME)
    return collection


def get_abstracts_for_papers(collection, papers):
    """
    For each paper (source filename), try to get its abstract from ChromaDB metadata.
    Falls back to top 2 parent chunks by character count if abstract is missing.
    Returns a list of text blocks (abstracts or chunks).
    """
    # Fetch all documents with metadata in batches
    total = collection.count()
    batch_size = 1000
    offset = 0

    # Group documents by source, keeping track of abstracts and parent chunks
    source_abstracts = {}  # source → abstract text (if available)
    source_chunks = {}     # source → list of (char_count, text)

    while True:
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        docs = results["documents"]
        if not docs:
            break
        metas = results["metadatas"]
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "")
            abstract = meta.get("abstract", "").strip()
            if abstract and source:
                source_abstracts.setdefault(source, abstract)
            # Always collect parent chunk for fallback
            if source and meta.get("parent_id"):
                source_chunks.setdefault(source, []).append((len(doc), doc))
        offset += batch_size
        if offset >= total:
            break

    text_blocks = []
    for paper in papers:
        if paper in source_abstracts and source_abstracts[paper]:
            text_blocks.append(f"[Abstract — {paper}]\n{source_abstracts[paper]}")
        else:
            # Fallback: top 2 parent chunks by character count
            if paper in source_chunks:
                sorted_chunks = sorted(source_chunks[paper], key=lambda x: x[0], reverse=True)[:2]
                for i, (char_len, text) in enumerate(sorted_chunks, 1):
                    text_blocks.append(f"[Parent chunk {i} — {paper}]\n{text[:3000]}")

    return text_blocks


def generate_node_summary(client, entity_name, text_blocks):
    """
    Call Haiku to write a 3–4 sentence synthesis of what the literature says
    about a given entity.
    """
    input_text = "\n\n---\n\n".join(text_blocks)
    if not input_text.strip():
        return None

    prompt = (
        f"Based on the following excerpts from fisheries literature, write a 3–4 sentence "
        f"synthesis of what the scientific literature collectively says about **{entity_name}**.\n\n"
        f"Guidelines:\n"
        f"- Use technical fisheries/biology language appropriate for agency scientists\n"
        f"- Include specific findings (temperatures, distances, rates, sample sizes) where the excerpts support them\n"
        f"- Note contradictions or gaps across studies if present\n"
        f"- Write in third person present tense, as if for a reference document\n"
        f"- Be concise but information-dense\n"
        f"- Do not invent or infer beyond what the excerpts provide\n\n"
        f"Excerpts:\n{input_text[:5000]}"
    )

    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=400,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = resp.content[0].text.strip()
    # Rough token estimate: ~1.3 chars per token
    input_tokens = math.ceil(len(prompt) / 1.3)
    output_tokens = math.ceil(len(summary) / 1.3)
    return summary, input_tokens, output_tokens


def main():
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set. Set it in your environment and re-run.")
        return

    if not os.path.exists(GRAPH_PATH):
        print(f"ERROR: {GRAPH_PATH} not found. Run build_graph.py first.")
        return

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    collection = get_chroma_client()

    summaries = {}
    total_input_tokens = 0
    total_output_tokens = 0
    call_count = 0
    start_time = time.time()

    nodes = sorted(G.nodes())
    print(f"Generating summaries for {len(nodes)} nodes...\n")

    for i, node in enumerate(nodes, 1):
        papers = G.nodes[node].get("papers", [])
        if not papers:
            summaries[node] = None
            continue

        text_blocks = get_abstracts_for_papers(collection, papers)
        if not text_blocks:
            summaries[node] = None
            continue

        try:
            result = generate_node_summary(client, node, text_blocks)
            if result:
                summary, in_tok, out_tok = result
                summaries[node] = summary
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                call_count += 1
                print(f"  [{i}/{len(nodes)}] {node} ✓")
            else:
                summaries[node] = None
                print(f"  [{i}/{len(nodes)}] {node} — No text available")
        except Exception as e:
            summaries[node] = None
            print(f"  [{i}/{len(nodes)}] {node} ✗ Error: {e}")

    # Save summaries
    with open(SUMMARIES_PATH, "wb") as f:
        pickle.dump(summaries, f)
    print(f"\nSummaries saved to {SUMMARIES_PATH}")

    # Log stats
    elapsed = time.time() - start_time
    approx_input_cost = (total_input_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
    approx_output_cost = (total_output_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    total_cost = approx_input_cost + approx_output_cost

    print(f"\n=== Cost Estimate ===")
    print(f"Haiku calls made:      {call_count}")
    print(f"Approx input tokens:   {total_input_tokens:,}")
    print(f"Approx output tokens:  {total_output_tokens:,}")
    print(f"Approx input cost:     ${approx_input_cost:.4f}")
    print(f"Approx output cost:    ${approx_output_cost:.4f}")
    print(f"Total approximate cost: ${total_cost:.4f}")
    print(f"Elapsed time:          {elapsed:.1f}s")
    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
