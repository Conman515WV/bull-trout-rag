"""
build_graph.py — Phase 2: Graph Build from Normalized Entities
==============================================================
Reads normalized_entities.json and builds a NetworkX undirected weighted graph.
No API calls — pure Python + NetworkX.

Nodes: canonical entity strings (appearing in >= MIN_ENTITY_PAPERS papers)
Edges: co-occurrence within the same paper, weight = number of papers both appear in
       Only edges with weight >= MIN_EDGE_WEIGHT are kept.

Tune MIN_ENTITY_PAPERS and MIN_EDGE_WEIGHT upward if graph is too large for PyVis.

Output:
  - graph.pkl       — NetworkX graph via pickle
  - graph_stats.txt — summary stats

Run:
  python build_graph.py
"""

import os
import json
import pickle
from collections import defaultdict
from itertools import combinations

import networkx as nx
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "yakima"
NORMALIZED_ENTITIES_PATH = "normalized_entities.json"

# Only include entities mentioned in at least this many papers
MIN_ENTITY_PAPERS = 10
# Only include edges where two entities co-occur in at least this many papers
MIN_EDGE_WEIGHT = 20

# Entities to exclude — category header strings returned by Haiku instead of real entities
BLACKLIST = {
    "agencies_and_organizations", "geographic locations", "infrastructure",
    "species_and_life_stages", "management_actions_and_programs",
    "Monitoring methods", "monitoring methods", "management actions and programs",
    "species and life stages",
}

# Manual normalization overrides — map bad strings to correct canonical form
# Applied before graph build so edges merge correctly
MANUAL_NORM = {
    "Yakirna River": "Yakima River",
    "Yakirna river": "Yakima River",
    "yakirna river": "Yakima River",
    "Adams": "Mount Adams",
    "adams": "Mount Adams",
}

CATEGORY_KEYWORDS = {
    "species": [
        "trout", "salmon", "bull trout", "smolt", "parr", "redd", "anadromous",
        "oncorhynchus", "salvelinus", "kokanee", "lamprey", "steelhead", "chinook",
        "coho", "sockeye", "whitefish", "sculpin", "sucker", "pikeminnow",
        "juvenile", "adult", "spawner", "fry", "alevin", "fingerling"
    ],
    "infrastructure": [
        "dam", "diversion", "weir", "fish ladder", "trap-and-haul", "spillway",
        "screen", "culvert", "impoundment", "bypass", "passage",
        "facility", "structure", "canal", "gate", "turbine"
    ],
    "location": [
        "river", "creek", "lake", "reservoir", "basin", "watershed", "tributary",
        "reach", "fork", "valley", "canyon", "confluence", "headwater", "mainstem",
        "stream", "pond", "slough", "wetland", "floodplain", "estuary", "mount", "mountain"
    ],
    "management": [
        "recovery", "restoration", "conservation", "habitat", "reintroduction",
        "supplementation", "listing", "esa", "hatchery", "stocking", "removal",
        "treatment", "enhancement", "mitigation", "protection", "plan", "strategy",
        "program", "project", "initiative", "action", "measure"
    ],
    "monitoring": [
        "pit tag", "acoustic", "telemetry", "radio", "electrofishing",
        "mark-recapture", "monitoring", "survey", "sampling", "snorkel",
        "redd count", "escapement", "detection", "antenna", "array",
        "population estimate", "abundance", "density"
    ],
    "agency": [
        "usfws", "wdfw", "noaa", "nmfs", "bureau of reclamation", "ybfwrb",
        "yakama", "fish and wildlife", "reclamation", "corps of engineers",
        "bpa", "bonneville", "epa", "usfs", "forest service", "tribal"
    ],
}


def infer_category(entity_name):
    name_lower = entity_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"


def get_source_map():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION_NAME)
    total = collection.count()
    print(f"Total chunks in collection: {total}")

    parent_to_source = {}
    batch = 1000
    offset = 0
    fetched = 0

    while fetched < total:
        results = collection.get(
            limit=batch, offset=offset,
            include=["metadatas"],
        )
        metas = results["metadatas"]
        if not metas:
            break
        for meta in metas:
            pid = meta.get("parent_id")
            source = meta.get("source", "")
            if pid and pid not in parent_to_source and source:
                parent_to_source[pid] = source
        fetched += len(metas)
        offset += batch
        if fetched % 10000 == 0:
            print(f"  Fetched {fetched}/{total} chunks, {len(parent_to_source)} unique parents...")

    print(f"Done: {len(parent_to_source)} parent->source mappings.")
    return parent_to_source


def apply_manual_norm(normalized_entities):
    """Apply manual normalization overrides to entity lists."""
    result = {}
    for pid, entities in normalized_entities.items():
        corrected = []
        for e in entities:
            corrected.append(MANUAL_NORM.get(e, e))
        # Deduplicate after normalization
        result[pid] = sorted(set(corrected))
    return result


def build_graph(normalized_entities, parent_to_source):
    source_to_entities = defaultdict(set)
    entity_to_sources = defaultdict(set)

    for pid, entities in normalized_entities.items():
        source = parent_to_source.get(pid)
        if not source:
            continue
        for entity in entities:
            source_to_entities[source].add(entity)
            entity_to_sources[entity].add(source)

    print(f"Papers with entities: {len(source_to_entities)}")
    print(f"Unique entities before filtering: {len(entity_to_sources)}")

    # Remove blacklisted entities
    entity_to_sources = {
        e: s for e, s in entity_to_sources.items()
        if e not in BLACKLIST
    }

    # Filter to entities appearing in at least MIN_ENTITY_PAPERS papers
    entity_to_sources = {
        e: s for e, s in entity_to_sources.items()
        if len(s) >= MIN_ENTITY_PAPERS
    }
    print(f"Unique entities after filtering (min {MIN_ENTITY_PAPERS} papers, blacklist removed): {len(entity_to_sources)}")

    kept_entities = set(entity_to_sources.keys())
    source_to_entities = {
        source: entities & kept_entities
        for source, entities in source_to_entities.items()
    }
    source_to_entities = {s: e for s, e in source_to_entities.items() if len(e) >= 2}

    G = nx.Graph()

    for entity, sources in entity_to_sources.items():
        G.add_node(entity, papers=sorted(sources), category=infer_category(entity))

    print(f"Building edges from co-occurrences (min weight={MIN_EDGE_WEIGHT})...")
    edge_weights = defaultdict(int)
    for source, entities in source_to_entities.items():
        entity_list = sorted(entities)
        for e1, e2 in combinations(entity_list, 2):
            key = (min(e1, e2), max(e1, e2))
            edge_weights[key] += 1

    edge_weights = {k: v for k, v in edge_weights.items() if v >= MIN_EDGE_WEIGHT}
    print(f"  {len(edge_weights)} edges after filtering.")

    for (e1, e2), weight in edge_weights.items():
        G.add_edge(e1, e2, weight=weight)

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    print(f"  Removed {len(isolated)} isolated nodes.")

    return G


def write_stats(G):
    lines = []
    lines.append(f"Total nodes: {G.number_of_nodes()}")
    lines.append(f"Total edges: {G.number_of_edges()}")
    lines.append("")

    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)
    lines.append("Top 10 highest-weight edges (most co-occurring pairs):")
    for e1, e2, data in edges_sorted[:10]:
        lines.append(f"  {e1!r} <-> {e2!r}  weight={data.get('weight', 0)}")
    lines.append("")

    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    lines.append("Top 10 most connected nodes:")
    for node, degree in degree_sorted[:10]:
        cat = G.nodes[node].get("category", "other")
        npap = len(G.nodes[node].get("papers", []))
        lines.append(f"  {node!r}  degree={degree}  category={cat}  papers={npap}")

    stats_text = "\n".join(lines)
    with open("graph_stats.txt", "w", encoding="utf-8") as f:
        f.write(stats_text)
    print("\n--- Graph Stats ---")
    print(stats_text)


def main():
    if not os.path.exists(NORMALIZED_ENTITIES_PATH):
        print(f"ERROR: {NORMALIZED_ENTITIES_PATH} not found. Run extract_entities.py first.")
        return

    print(f"Loading {NORMALIZED_ENTITIES_PATH}...")
    with open(NORMALIZED_ENTITIES_PATH, "r", encoding="utf-8") as f:
        normalized_entities = json.load(f)
    print(f"Loaded {len(normalized_entities)} parent chunks with entities.")

    # Apply manual normalization fixes before building graph
    print("Applying manual normalization overrides...")
    normalized_entities = apply_manual_norm(normalized_entities)

    parent_to_source = get_source_map()

    print("\nBuilding graph...")
    G = build_graph(normalized_entities, parent_to_source)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    with open("graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print("Graph saved -> graph.pkl")

    write_stats(G)
    print("Stats saved -> graph_stats.txt")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
