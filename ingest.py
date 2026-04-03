"""
ingest.py — Yakima Fisheries RAG Database Builder
--------------------------------------------------
Reads all PDFs from PDF_DIR, extracts text, splits into parent/child chunks,
embeds child chunks with all-MiniLM-L6-v2, stores in ChromaDB, and builds
a BM25 keyword index from parent chunks.

Run once (or again whenever you add new PDFs):
    py ingest.py

Expected output:
    Done!
      Chroma DB: ~192000 child chunks
      BM25 index: ~47000 parent chunks
      BM25 saved to: ./bm25_index.pkl
"""

import os
import re
import pickle
import hashlib
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR       = r"C:\Users\Connor\Desktop\YakimaReferences"
CHROMA_DIR    = "./chroma_db"
BM25_PATH     = "./bm25_index.pkl"
COLLECTION    = "yakima_fisheries"

# Chunk sizes (in approximate tokens; splitter uses chars, ~4 chars/token)
PARENT_CHUNK_SIZE    = 1000 * 4   # ~1000 tokens
PARENT_CHUNK_OVERLAP = 100  * 4   # ~100 token overlap
CHILD_CHUNK_SIZE     = 300  * 4   # ~300 tokens
CHILD_CHUNK_OVERLAP  = 30   * 4   # ~30 token overlap

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 256   # ChromaDB upsert batch size


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_year(filename: str, first_page_text: str) -> str:
    """Try to extract a 4-digit year from filename, then first page text."""
    # Check filename first
    m = re.search(r"\b(19|20)\d{2}\b", filename)
    if m:
        return m.group(0)
    # Fall back to first page text
    m = re.search(r"\b(19|20)\d{2}\b", first_page_text or "")
    if m:
        return m.group(0)
    return "Unknown"


def extract_title(first_page_text: str, filename: str) -> str:
    """Use the first non-empty line of first-page text as the title."""
    for line in (first_page_text or "").splitlines():
        line = line.strip()
        if len(line) > 10:
            return line[:200]
    # Fall back to filename stem
    return Path(filename).stem.replace("_", " ")


def make_id(text: str) -> str:
    """Deterministic short ID from text content (MD5)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_text(pdf_path: str) -> tuple[str, str]:
    """
    Extract full text from a PDF using pdfplumber.
    Returns (full_text, first_page_text).
    Returns ("", "") if PDF is scanned/empty.
    """
    pages = []
    first_page = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(text)
                if i == 0:
                    first_page = text
    except Exception as e:
        print(f"  [WARN] Could not read {pdf_path}: {e}")
        return "", ""
    full_text = "\n".join(pages).strip()
    return full_text, first_page


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pdf_dir = Path(PDF_DIR)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Check the path and try again.")
        return

    print(f"Found {len(pdf_files)} PDFs in {PDF_DIR}")

    # Text splitters
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=len,
    )

    # ChromaDB setup
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION)
        print("  Deleted existing ChromaDB collection for fresh rebuild.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Accumulators
    all_parent_texts: list[str] = []  # for BM25
    all_parent_meta:  list[dict] = []

    child_ids:   list[str] = []
    child_texts: list[str] = []
    child_metas: list[dict] = []

    skipped = 0
    processed = 0

    for idx, pdf_path in enumerate(pdf_files):
        filename = pdf_path.name
        print(f"[{idx+1}/{len(pdf_files)}] {filename}", end="  ")

        full_text, first_page = extract_text(str(pdf_path))

        if not full_text or len(full_text) < 100:
            print("SKIPPED (no extractable text)")
            skipped += 1
            continue

        year  = extract_year(filename, first_page)
        title = extract_title(first_page, filename)

        # ── Parent chunks ──────────────────────────────────────────────────
        parent_chunks = parent_splitter.split_text(full_text)

        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = make_id(f"{filename}::parent::{p_idx}")
            parent_meta = {
                "source":    filename,
                "title":     title,
                "year":      year,
                "parent_id": parent_id,
                "chunk_idx": p_idx,
            }
            all_parent_texts.append(parent_text)
            all_parent_meta.append(parent_meta)

            # ── Child chunks (embed these) ─────────────────────────────────
            child_chunks = child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_chunks):
                child_id = make_id(f"{filename}::child::{p_idx}::{c_idx}")
                child_meta = {
                    "source":      filename,
                    "title":       title,
                    "year":        year,
                    "parent_id":   parent_id,
                    "parent_text": parent_text,   # full parent stored inline
                }
                child_ids.append(child_id)
                child_texts.append(child_text)
                child_metas.append(child_meta)

        print(f"{len(parent_chunks)} parent chunks")
        processed += 1

    # ── Upsert child chunks into ChromaDB in batches ───────────────────────
    print(f"\nUpserting {len(child_texts)} child chunks to ChromaDB...")
    for start in range(0, len(child_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=child_ids[start:end],
            documents=child_texts[start:end],
            metadatas=child_metas[start:end],
        )
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"  {start}/{len(child_ids)}")
    print(f"  Done — {len(child_texts)} child chunks stored.")

    # ── Build and save BM25 index ──────────────────────────────────────────
    print(f"\nBuilding BM25 index from {len(all_parent_texts)} parent chunks...")
    tokenized = [text.lower().split() for text in all_parent_texts]
    bm25 = BM25Okapi(tokenized)

    bm25_payload = {
        "bm25":     bm25,
        "texts":    all_parent_texts,
        "metadata": all_parent_meta,
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_payload, f)

    print(f"\nDone!")
    print(f"  PDFs processed : {processed}")
    print(f"  PDFs skipped   : {skipped} (scanned/image-only)")
    print(f"  Chroma DB      : {len(child_texts)} child chunks")
    print(f"  BM25 index     : {len(all_parent_texts)} parent chunks")
    print(f"  BM25 saved to  : {BM25_PATH}")


if __name__ == "__main__":
    main()