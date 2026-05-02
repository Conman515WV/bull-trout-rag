"""
build_graph.py — Phase 2: Build NetworkX co-occurrence graph from normalized entities
=====================================================================================
Reads normalized_entities.json and ChromaDB metadata. For each paper (identified
by source), collects all canonical entities co-occurring in its parent chunks.
Edge weight = number of papers both entities appear in.

Node attributes:
  - papers: list of source filenames mentioning this entity
  - category: inferred entity category via keyword heuristics

Outputs:
  - graph.pkl  (NetworkX graph, pickled)
  - graph_stats.txt  (summary statistics)

Run:
  python build_graph.py
"""

import os
import json
import pickle
from collections import defaultdict
from itertools import combinations
import chromadb
import networkx as nx

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "yakima"
NORMALIZED_PATH = "normalized_entities.json"
GRAPH_PATH = "graph.pkl"
STATS_PATH = "graph_stats.txt"

# ── Category keyword heuristics ──────────────────────────────────────────────
CATEGORIES = {
    "species": [
        "trout", "salmon", "bull trout", "rainbow trout", "steelhead",
        "cutthroat", "brook trout", "brown trout", "sockeye", "chinook",
        "coho", "chum", "pink", "smolt", "parr", "fry", "redd", "spawner",
        "juvenile", "adult fish", "kokanee", "lamprey", "sturgeon",
        "whitefish", "sculpin", "dace", "minnow", "salvelinus",
        "oncorhynchus", "salmo", "anadromous", "landlocked",
        "bullchar", "char",
    ],
    "infrastructure": [
        "dam", "diversion", "weir", "fish ladder", "fish passage",
        "trap", "haul", "trap-and-haul", "trap and haul", "culvert",
        "screen", "spillway", "bridge", "pump", "canal", "conduit",
        "powerhouse", "turbine", "outlet", "gate", "structure",
        "roza", "cle elum dam", "kachess dam", "keechelus dam",
        "rimrock dam", "bumping dam",
    ],
    "location": [
        "river", "creek", "lake", "reservoir", "basin", "watershed",
        "tributary", "canyon", "valley", "stream", "reach", "pool",
        "fork", "fork of", "confluence", "spillway", "pass", "passage",
        "cle elum", "keechelus", "kachess", "rimrock", "washington",
        "yakima", "columbia", "naches", "tieton", "bumping", "kittitas",
        "snoqualmie", "wenatchee", "methow", "okanogan", "spokane",
        "lake creek", "upper", "lower", "north fork", "south fork",
        "east fork", "west fork", "middle fork", "subbasin", "drainage",
        "hatchery", "complex",
    ],
    "management": [
        "recovery", "restoration", "conservation", "habitat", "enhancement",
        "reintroduction", "supplementation", "translocation", "removal",
        "passage", "management plan", "action plan", "biological opinion",
        "listing", "esa", "critical habitat", "recovery plan",
        "genetic", "broodstock", "program", "project", "initiative",
        "strategy", "management", "hatchery program", "fishery",
        "harvest", "fishery management",
    ],
    "monitoring": [
        "pit tag", "pit", "acoustic", "telemetry", "radio tracking",
        "mark-recapture", "electrofishing", "seining", "gill net",
        "trapping", "monitoring", "survey", "census", "abundance",
        "density", "sonar", "didson", "ars", "video", "observation",
        "passive integrated transponder", "tag", "tagging",
    ],
    "agency": [
        "usfws", "us bor", "reclamation", "wdfw", "noaa", "nmfs", "nps",
        "bureau of reclamation", "fish and wildlife", "environmental",
        "department", "agency", "commission", "service", "army corps",
        "usgs", "epa", "fws", "us fish", "ybfwrb", "yakima basin",
        "recovery board", "mid-columbia", "fwco", "field station",
        "yakama nation", "yakama",
    ],
}


def infer_category(entity_name):
    """Heuristically map an entity string to a category."""
    lower = entity_name.lower().strip()
    for category, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "other"


def main():
    if not os.path.exists(NORMALIZED_PATH):
        print(f"ERROR: {NORMALIZED_PATH} not found. Run extract_entities.py first.")
        return

    with open(NORMALIZED_PATH, "r", encoding="utf-8") as f:
        normalized = json.load(f)
    print(f"Loaded normalized entities: {len(normalized)} parent chunks.")

    # Build source → entities mapping from ChromaDB metadata
    print("Opening ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    total = collection.count()

    source_to_entities = defaultdict(set)
    batch = 1000
    offset = 0

    while offset < total:
        results = collection.get(
            limit=batch, offset=offset,
            include=["metadatas"],
        )
        metas = results["metadatas"]
        if not metas:
            break
        for meta in metas:
            pid = meta.get("parent_id")
            source = meta.get("source", "unknown")
            if pid in normalized:
                for ent in normalized[pid]:
                    source_to_entities[source].add(ent)
        offset += batch

    print(f"Found {len(source_to_entities)} unique papers with entities.")

    # Build graph
    G = nx.Graph()

    for source, entities in source_to_entities.items():
        entities_sorted = sorted(entities)
        # Add/update node papers list and category
        for ent in entities_sorted:
            if not G.has_node(ent):
                G.add_node(ent, papers=[], category=infer_category(ent))
            if source not in G.nodes[ent]["papers"]:
                G.nodes[ent]["papers"].append(source)

        # Add/increment edge weights for every co-occurring pair
        for e1, e2 in combinations(entities_sorted, 2):
            if G.has_edge(e1, e2):
                G[e1][e2]["weight"] += 1
            else:
                G.add_edge(e1, e2, weight=1)

    # Save graph
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {GRAPH_PATH}")

    # Write stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    top_edges = sorted(G.edges(data="weight"), key=lambda x: x[2], reverse=True)[:10]
    top_nodes = sorted(G.nodes(data=True), key=lambda x: len(x[1]["papers"]), reverse=True)[:10]

    lines = []
    lines.append("Graph Statistics")
    lines.append(f"Total nodes: {n_nodes}")
    lines.append(f"Total edges: {n_edges}")
    lines.append("")
    lines.append("Top 10 highest-weight edges:")
    for e1, e2, w in top_edges:
        lines.append(f"  {e1} ↔ {e2}  (weight: {w})")
    lines.append("")
    lines.append("Top 10 most-connected nodes (by paper count):")
    for node, data in top_nodes:
        lines.append(f"  {node} ({len(data['papers'])} papers)")

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Stats saved to {STATS_PATH}")
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
