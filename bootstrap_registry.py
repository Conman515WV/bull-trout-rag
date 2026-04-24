"""
bootstrap_registry.py — one-time script
---------------------------------------
Scans the existing ChromaDB 'yakima' collection and writes every unique
`source` value (the original filename) into processed_files.json.

After running this once, ingestheavy.py will know which files are already
indexed and skip them on subsequent runs, so you only ever process NEW
papers when you add them to YakimaReferences/.

Run from the RAGV2 folder:
    py bootstrap_registry.py
"""

import json
from pathlib import Path

import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION = "yakima"
REGISTRY   = "./processed_files.json"


def main() -> None:
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        col = chroma.get_collection(COLLECTION)
    except Exception as e:
        raise SystemExit(
            f"Could not open Chroma collection '{COLLECTION}' at {CHROMA_DIR}: {e}\n"
            f"Make sure you are running this from the RAGV2 folder that contains "
            f"your chroma_db/ directory."
        )

    print(f"Scanning collection '{COLLECTION}' at {CHROMA_DIR}...")
    results = col.get(include=["metadatas"])
    metas   = results.get("metadatas", []) or []

    sources = {m["source"] for m in metas if m and m.get("source")}
    sorted_sources = sorted(sources)

    Path(REGISTRY).write_text(
        json.dumps(sorted_sources, indent=2),
        encoding="utf-8",
    )

    print(f"Scanned {len(metas)} chunk metadatas")
    print(f"Registered {len(sorted_sources)} unique source files")
    print(f"Wrote: {REGISTRY}")
    print("")
    print("Next step: run ingestheavy.py — already-indexed files will be skipped.")


if __name__ == "__main__":
    main()
