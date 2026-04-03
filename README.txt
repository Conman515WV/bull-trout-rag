================================================================================
YAKIMA FISHERIES RAG — COMPLETE BUILD GUIDE
================================================================================
OVERVIEW
--------
This app lets USFWS biologists query a library of ~310 Yakima Basin fisheries
papers using natural language. It uses a 6-step retrieval pipeline to find and
rank the most relevant literature before generating a cited answer.
PIPELINE SUMMARY
----------------
1. Query Expansion    — Claude Haiku turns your question into 4 search queries
2. Vector Search      — ChromaDB + all-MiniLM-L6-v2 retrieves 25 results per query (~100 candidates)
3. BM25 Keyword Search— Keyword index retrieves 25 more results for exact term matching
4. Deduplication      — Vector + BM25 results merged, duplicates removed (~100-110 unique candidates)
5. Reranking          — CrossEncoder (ms-marco-MiniLM-L-6-v2) scores all candidates, keeps top 12
6. Generation         — Claude Sonnet answers using top 12 parent chunks as context (~12,000 tokens)
COST PER QUERY: ~$0.04 (Haiku expansion + Sonnet generation, with prompt caching)
================================================================================
PREREQUISITES
================================================================================
Software to install before starting:
- VS Code               — code.visualstudio.com
- Python 3.11 or 3.12   — python.org (IMPORTANT: avoid 3.14, has compatibility issues)
- Git                   — git-scm.com
- Git LFS               — git-lfs.github.com
Accounts needed:
- GitHub                — github.com (free)
- Anthropic             — console.anthropic.com (pay per use)
- Streamlit Cloud       — share.streamlit.io (free)
- Hugging Face          — huggingface.co (free)
================================================================================
STEP 1 — SET UP PROJECT FOLDER
================================================================================
Open VS Code. Open a terminal (Terminal → New Terminal). Run:
    mkdir bull-trout-rag
    cd bull-trout-rag
    code .
VS Code will reopen inside the project folder. Open a new terminal inside it.
================================================================================
STEP 2 — INSTALL DEPENDENCIES
================================================================================
    py -m pip install pdfplumber langchain-text-splitters chromadb sentence-transformers anthropic streamlit rank-bm25
Note: always use "py -m pip" not just "pip" on Windows.
================================================================================
STEP 3 — CREATE ingest.py
================================================================================
Create a new file called ingest.py in VS Code and paste the full ingest script.
WHAT THE INGEST SCRIPT DOES:
- Reads all PDFs from your local folder (default: C:\Users\Connor\Desktop\YakimaReferences)
- Extracts text using pdfplumber (skips scanned/image-only PDFs automatically)
- Extracts year from filename or first page, title from first line of text
- Splits each PDF into:
    PARENT chunks: ~1000 tokens with 100 token overlap (used for context sent to Claude)
    CHILD chunks:  ~300 tokens with 30 token overlap (used for precise vector retrieval)
  Each child chunk stores its parent text inline as metadata
- Embeds all child chunks using sentence-transformers all-MiniLM-L6-v2 (free, runs locally)
- Stores child chunks + embeddings in ChromaDB (./chroma_db folder)
- Builds a BM25 keyword index from all parent chunks
- Saves BM25 index to ./bm25_index.pkl
Run it:
    py ingest.py
Takes 30-40 minutes for ~400 PDFs. Expected output when done:
    Done!
      Chroma DB: ~192000 child chunks
      BM25 index: ~47000 parent chunks
      BM25 saved to: ./bm25_index.pkl
IMPORTANT: If you add new PDFs later, re-run ingest.py to rebuild the database.
You will then need to re-upload to Hugging Face (see Step 7).
================================================================================
STEP 4 — CREATE ANTHROPIC API KEY
================================================================================
1. Go to console.anthropic.com
2. Click "API Keys" in the left sidebar
3. Click "Create Key", name it "bull-trout-rag"
4. Copy the key immediately — it starts with sk-ant-... and is only shown once
5. Set a spend limit: Billing → Usage Limits → set monthly cap to $20-50
================================================================================
STEP 5 — CREATE STREAMLIT SECRETS FILE
================================================================================
In your terminal:
    mkdir .streamlit
Create a file called secrets.toml inside the .streamlit folder with this content:
    ANTHROPIC_API_KEY = "your-key-here"
Replace "your-key-here" with your actual key from Step 4.
NEVER commit this file to GitHub. It stays local only.
================================================================================
STEP 6 — CREATE app.py
================================================================================
Create a new file called app.py and paste the full app script.
WHAT THE APP DOES:
- Password gate — reads password from st.secrets["APP_PASSWORD"], never hardcoded
- Loads ChromaDB, both embedding models, BM25 index, and Anthropic client at startup
- On each question:
    1. Uses Claude Haiku to generate 3 alternative phrasings of the question
    2. Runs vector search for each of the 4 queries (25 results each)
    3. Runs BM25 keyword search on original question (25 results)
    4. Deduplicates all results by parent_id
    5. CrossEncoder reranks all candidates against the original question
    6. Top 12 parent chunks sent to Claude Sonnet with system prompt
    7. Answer displayed with expandable source cards showing paper title and year
- Web search toggle near prompt bar for supplementing with live web results
- Dark theme UI with IBM Plex fonts
- Conversation memory within session (follow-up questions use prior context)
- Prompt caching on system prompt and literature context (reduces cost ~90% on repeat queries)
Test locally:
    py -m streamlit run app.py
App opens at http://localhost:8501
Enter password: yakima2026
================================================================================
STEP 7 — UPLOAD DATABASE TO HUGGING FACE
================================================================================
The chroma_db folder (~600MB) and bm25_index.pkl (~100MB) are too large for
standard GitHub (1GB LFS free tier fills up fast). Hugging Face Datasets is
the better solution — free, no bandwidth limits, designed for large ML files.
--- CREATE HUGGING FACE REPO ---
1. Go to huggingface.co and create a free account
2. Click your profile → "New Dataset"
3. Name it "yakima-rag-db"
4. Set visibility to Private
5. Click "Create dataset"
--- INSTALL HUGGING FACE CLI ---
    py -m pip install huggingface_hub
--- UPLOAD THE FILES ---
    py -c "
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path='./chroma_db',
        repo_id='YOURUSERNAME/yakima-rag-db',
        repo_type='dataset',
        path_in_repo='chroma_db'
    )
    api.upload_file(
        path_or_fileobj='./bm25_index.pkl',
        path_in_repo='bm25_index.pkl',
        repo_id='YOURUSERNAME/yakima-rag-db',
        repo_type='dataset'
    )
    print('Done uploading')
    "
Replace YOURUSERNAME with your Hugging Face username.
This will take several minutes depending on your connection.
--- ADD HF TOKEN TO STREAMLIT SECRETS ---
1. Go to huggingface.co → Settings → Access Tokens
2. Create a token with "read" permission
3. Add it to your .streamlit/secrets.toml:
    ANTHROPIC_API_KEY = "your-anthropic-key"
    HF_TOKEN = "your-huggingface-token"
    APP_PASSWORD = "your-chosen-password"
The APP_PASSWORD is what users will type to access the app. Choose something
memorable to share with colleagues. Keeping it in secrets (not in the code)
means the repo can be public without exposing the password.
--- UPDATE app.py TO DOWNLOAD DB AT STARTUP ---
Add this to the load_resources() function at the top, before loading ChromaDB:
    import os
    from huggingface_hub import snapshot_download
    if not os.path.exists("./chroma_db"):
        snapshot_download(
            repo_id="YOURUSERNAME/yakima-rag-db",
            repo_type="dataset",
            local_dir=".",
            token=st.secrets["HF_TOKEN"]
        )
This downloads the database once on first startup, then uses the cached version.
================================================================================
STEP 8 — PUSH CODE TO GITHUB
================================================================================
1. Go to github.com → click "+" → New repository
2. Name it "bull-trout-rag", set to PUBLIC
   (Safe to make public because all secrets are stored in Streamlit secrets,
   not in the code itself)
3. Do NOT initialize with README, .gitignore, or license
In your terminal:
    git init
    git config --global user.email "your@email.com"
    git config --global user.name "Your Name"
Create a .gitignore file with this content to exclude secrets and large files:
    .streamlit/secrets.toml
    chroma_db/
    bm25_index.pkl
    __pycache__/
    *.pyc
Then:
    git add app.py ingest.py requirements.txt .gitignore
    git commit -m "initial commit"
    git remote add origin https://github.com/YOURUSERNAME/bull-trout-rag.git
    git push -u origin master
GitHub will open a browser window for authentication.
For password use a Personal Access Token:
    github.com → Settings → Developer settings → Personal access tokens → Generate new token
    Check the "repo" scope, generate, copy and use as your password.
================================================================================
STEP 9 — GENERATE requirements.txt
================================================================================
    py -m pip freeze > requirements.txt
    git add requirements.txt
    git commit -m "add requirements"
    git push origin master
================================================================================
STEP 10 — DEPLOY TO STREAMLIT CLOUD
================================================================================
1. Go to share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your bull-trout-rag repository
5. Branch: master
6. Main file path: app.py
7. Click "Advanced settings" and paste your secrets:
    ANTHROPIC_API_KEY = "your-anthropic-key"
    HF_TOKEN = "your-huggingface-token"
    APP_PASSWORD = "your-chosen-password"
8. Click "Deploy"
First deploy takes 5-10 minutes:
- Installs all Python packages from requirements.txt
- Downloads database from Hugging Face (~700MB)
- Loads embedding model, cross-encoder model, BM25 index into memory
Subsequent boots are faster as packages are cached.
================================================================================
STEP 11 — MAKING FUTURE CHANGES
================================================================================
--- CODE CHANGES (app.py) ---
Edit app.py locally, then:
    git add app.py
    git commit -m "describe your change"
    git push origin master
Streamlit Cloud redeploys automatically within a minute.
--- ADDING NEW PDFs ---
1. Add new PDFs to C:\Users\Connor\Desktop\YakimaReferences
2. Run ingest.py again (30-40 min)
3. Re-upload to Hugging Face (repeat the upload command from Step 7)
4. Reboot the app on Streamlit Cloud (three dots menu → Reboot)
================================================================================
MODELS USED
================================================================================
EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2
- Free, runs locally
- 384-dimensional embeddings
- Used to embed child chunks during ingestion and queries at runtime
- Downloads automatically from Hugging Face on first run (~90MB)
RERANKING MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
- Free, runs locally
- Takes (question, document) pairs and scores relevance directly
- Much more accurate than vector similarity alone
- Downloads automatically from Hugging Face on first run (~70MB)
QUERY EXPANSION: claude-haiku-4-5 (Anthropic API)
- Generates 3 alternative phrasings of each question
- Cost: ~$0.0002 per query (essentially free)
ANSWER GENERATION: claude-sonnet-4-6 (Anthropic API)
- Receives top 12 reranked parent chunks + system prompt
- Generates cited, detailed technical answers
- Cost: ~$0.036 per query
- Prompt caching reduces cost ~90% on repeated similar queries
================================================================================
TROUBLESHOOTING
================================================================================
"py is not recognized"
→ Use "python" instead of "py", or reinstall Python and check "Add to PATH"
"ModuleNotFoundError"
→ Run: py -m pip install [missing module name]
"chromadb include ids error"
→ Remove "ids" from the include=[] list in collection.query() — ids are returned automatically
"reranker sort error"
→ Wrap scores in: scores = [float(s) for s in scores]
"secrets not found"
→ Check .streamlit/secrets.toml exists and is valid TOML format with quoted values
"database not found on Streamlit Cloud"
→ Check HF_TOKEN is set in Streamlit secrets and repo name matches exactly
PDF text is empty / skipped
→ PDF is a scanned image. Needs OCR software (e.g. Adobe Acrobat) to make text extractable.
   These papers simply won't be indexed.
App is slow on first load
→ Normal — models are loading into memory. After first query everything is cached.
================================================================================
FILE STRUCTURE
================================================================================
bull-trout-rag/
├── app.py                  — Streamlit app (the UI and query pipeline)
├── ingest.py               — One-time script to build the database from PDFs
├── requirements.txt        — Python package list for Streamlit Cloud
├── .gitignore              — Tells git what NOT to commit
├── .streamlit/
│   └── secrets.toml        — API keys (never committed to GitHub)
├── chroma_db/              — Vector database (uploaded to Hugging Face, not GitHub)
└── bm25_index.pkl          — BM25 keyword index (uploaded to Hugging Face, not GitHub)
================================================================================
CONTACTS / RESOURCES
================================================================================
Streamlit docs:         docs.streamlit.io
Anthropic API docs:     docs.anthropic.com
ChromaDB docs:          docs.trychroma.com
Hugging Face docs:      huggingface.co/docs
Sentence Transformers:  sbert.net
Anthropic usage/billing: console.anthropic.com
GitHub repo:            github.com/Conman515WV/bull-trout-rag
Streamlit app:          share.streamlit.io (after deployment)
================================================================================
END OF GUIDE
================================================================================