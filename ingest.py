import pdfplumber
import os
import re
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

PDF_FOLDER = r"C:\Users\Connor\Desktop\YakimaReferences"
BM25_PATH = "./bm25_index.pkl"

def extract_year(filename, text):
    match = re.search(r'(19|20)\d{2}', filename)
    if match:
        return match.group()
    match = re.search(r'(19|20)\d{2}', text[:500])
    if match:
        return match.group()
    return "unknown"

def extract_title(text):
    lines = [l.strip() for l in text[:1000].split('\n') if len(l.strip()) > 20]
    if lines:
        return lines[0][:120]
    return ""

print("Setting up...")

child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection(name="yakima")
    print("Deleted old collection")
except:
    pass
collection = client.get_or_create_collection(name="yakima")

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs")

all_child_texts = []
all_child_ids = []
all_child_metadatas = []

bm25_corpus = []
bm25_texts = []
bm25_metadata = []

child_id = 0
parent_id = 0

for i, pdf_file in enumerate(pdf_files):
    print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")
    try:
        with pdfplumber.open(os.path.join(PDF_FOLDER, pdf_file)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if not text.strip():
            print(f"  Skipping (no extractable text): {pdf_file}")
            continue

        year = extract_year(pdf_file, text)
        title = extract_title(text)

        parents = parent_splitter.split_text(text)

        for parent_text in parents:
            pid = f"parent_{parent_id}"
            parent_id += 1

            tokens = parent_text.lower().split()
            bm25_corpus.append(tokens)
            bm25_texts.append(parent_text)
            bm25_metadata.append({
                "source": pdf_file,
                "year": year,
                "title": title,
                "parent_id": pid
            })

            children = child_splitter.split_text(parent_text)
            for child_text in children:
                all_child_texts.append(child_text)
                all_child_ids.append(str(child_id))
                all_child_metadatas.append({
                    "source": pdf_file,
                    "year": year,
                    "title": title,
                    "parent_text": parent_text,
                    "parent_id": pid
                })
                child_id += 1

    except Exception as e:
        print(f"  Error on {pdf_file}: {e}")

print(f"\nTotal child chunks: {len(all_child_texts)}")
print(f"Total parent chunks: {parent_id}")

print("\nEmbedding child chunks... this will take a while")
embeddings = embed_model.encode(all_child_texts, show_progress_bar=True)

print("Storing child chunks in Chroma...")
batch_size = 500
for i in range(0, len(all_child_texts), batch_size):
    collection.add(
        ids=all_child_ids[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size].tolist(),
        documents=all_child_texts[i:i+batch_size],
        metadatas=all_child_metadatas[i:i+batch_size]
    )
    print(f"  Stored {min(i+batch_size, len(all_child_texts))}/{len(all_child_texts)} chunks")

print("\nBuilding BM25 index...")
bm25 = BM25Okapi(bm25_corpus)

print("Saving BM25 index...")
with open(BM25_PATH, "wb") as f:
    pickle.dump({
        "bm25": bm25,
        "texts": bm25_texts,
        "metadata": bm25_metadata
    }, f)

print(f"\nDone!")
print(f"  Chroma DB: {len(all_child_texts)} child chunks")
print(f"  BM25 index: {parent_id} parent chunks")
print(f"  BM25 saved to: {BM25_PATH}")