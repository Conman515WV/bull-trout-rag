"""
extract_entities.py — Phase 1: Entity Extraction from ChromaDB
===============================================================
Queries ChromaDB directly (no PDF reads), extracts named entities via Haiku
per parent chunk, then runs fuzzy normalization across all raw entities using
rapidfuzz token_sort_ratio (threshold = 88).

Run:
  python extract_entities.py
"""

import os
import json
import chromadb
from anthropic import Anthropic
from rapidfuzz import fuzz

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "yakima"
HAIKU_MODEL = "claude-haiku-4-5"
# FUZZY_DEDUP_THRESHOLD: pairs scoring >= this are collapsed to one canonical form.
# Tune upward (e.g. 92) if over-collapsing, downward (e.g. 82) if under-collapsing.
FUZZY_DEDUP_THRESHOLD = 88

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def get_parent_chunks():
    """Fetch all documents from ChromaDB, deduplicated by parent_id."""
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION_NAME)
    total = collection.count()
    print(f"Total chunks in collection: {total}")

    unique_parents = {}
    batch = 1000
    offset = 0
    fetched = 0

    while fetched < total:
        results = collection.get(
            limit=batch, offset=offset,
            include=["documents", "metadatas"],
        )
        docs = results["documents"]
        if not docs:
            break
        metas = results["metadatas"]
        for doc, meta in zip(docs, metas):
            pid = meta.get("parent_id")
            if pid and pid not in unique_parents:
                unique_parents[pid] = doc  # doc text keyed by parent_id
        fetched += len(docs)
        offset += batch
        print(f"  Fetched {fetched}/{total} chunks, {len(unique_parents)} unique parents...")

    print(f"Done: {len(unique_parents)} unique parent chunks.")
    return unique_parents


def extract_one(client, text):
    """Call Haiku to extract named entities from one parent chunk."""
    prompt = (
        "Extract named entities from the following text in these categories:\n"
        "- geographic locations (rivers, lakes, reservoirs, watersheds)\n"
        "- infrastructure (dams, diversion structures)\n"
        "- species and life stages\n"
        "- management actions and programs\n"
        "- monitoring methods\n"
        "- agencies and organizations\n\n"
        "Return ONLY a JSON list of entity strings, like [\"Entity 1\", \"Entity 2\"].\n"
        "Do NOT include any explanation, preamble, or markdown code fences.\n\n"
        f"Text:\n{text[:3000]}"
    )
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL, max_tokens=200, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip possible ```json ... ``` block
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        entities = json.loads(raw)
        return [str(e) for e in entities if isinstance(e, (str, int, float))]
    except Exception as e:
        print(f"    Extraction failed: {e}")
        return []


def normalize(raw_entities):
    """Union-find fuzzy dedup using token_sort_ratio >= threshold."""
    entities = list(raw_entities)
    n = len(entities)
    print(f"Fuzzy dedup on {n} unique raw entities (threshold: {FUZZY_DEDUP_THRESHOLD})...")

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            if len(entities[pi]) >= len(entities[pj]):
                parent[pj] = pi
            else:
                parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) != find(j):
                if fuzz.token_sort_ratio(entities[i], entities[j]) >= FUZZY_DEDUP_THRESHOLD:
                    union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(entities[i])

    norm_map = {}
    for members in groups.values():
        canonical = max(members, key=len)
        for m in members:
            norm_map[m] = canonical

    collapsed = n - len(set(norm_map.values()))
    print(f"Done: {n} raw → {len(set(norm_map.values()))} canonical ({collapsed} collapsed).")
    return norm_map


def main():
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.")
        return

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Step 1: Extract
    parents = get_parent_chunks()
    extracted = {}
    for i, (pid, text) in enumerate(parents.items(), 1):
        print(f"[{i}/{len(parents)}] {pid[:80]}...")
        entities = extract_one(client, text)
        if entities:
            extracted[pid] = entities

    with open("extracted_entities.json", "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)
    print(f"Raw entities saved (extracted_entities.json)")

    # Step 2: Normalize
    all_raw = set()
    for ents in extracted.values():
        all_raw.update(ents)
    print(f"\n{len(all_raw)} unique raw entity strings.")

    norm_map = normalize(all_raw)

    with open("normalization_map.json", "w", encoding="utf-8") as f:
        json.dump(norm_map, f, indent=2, ensure_ascii=False)

    normalized = {}
    for pid, ents in extracted.items():
        normalized[pid] = sorted(set(norm_map.get(e, e) for e in ents))

    with open("normalized_entities.json", "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    print(f"Normalized entities saved (normalized_entities.json)")
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
