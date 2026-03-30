import pdfplumber
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

PDF_FOLDER = r"C:\Users\Connor\Desktop\YakimaReferences"

def extract_year(filename, text):
    # Try filename first (e.g. Dunham_et_al_2001.pdf)
    match = re.search(r'(19|20)\d{2}', filename)
    if match:
        return match.group()
    # Try first 500 chars of text
    match = re.search(r'(19|20)\d{2}', text[:500])
    if match:
        return match.group()
    return "unknown"

def extract_title(text):
    # Take first non-empty line from PDF text as likely title
    lines = [l.strip() for l in text[:1000].split('\n') if len(l.strip()) > 20]
    if lines:
        return lines[0][:120]  # cap at 120 chars
    return ""

print("Setting up...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Delete old collection and start fresh
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection(name="yakima")
    print("Deleted old collection")
except:
    pass
collection = client.get_or_create_collection(name="yakima")

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs")

all_texts = []
all_ids = []
all_metadatas = []
chunk_id = 0

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

        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_ids.append(str(chunk_id))
            all_metadatas.append({
                "source": pdf_file,
                "year": year,
                "title": title
            })
            chunk_id += 1

    except Exception as e:
        print(f"  Error on {pdf_file}: {e}")

print(f"\nEmbedding {len(all_texts)} chunks... this will take a while")
embeddings = model.encode(all_texts, show_progress_bar=True)

print("Storing in Chroma...")
batch_size = 500
for i in range(0, len(all_texts), batch_size):
    collection.add(
        ids=all_ids[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size].tolist(),
        documents=all_texts[i:i+batch_size],
        metadatas=all_metadatas[i:i+batch_size]
    )
    print(f"  Stored {min(i+batch_size, len(all_texts))}/{len(all_texts)} chunks")

print("\nDone! Chroma DB rebuilt with title and year metadata.")