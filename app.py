import streamlit as st
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Yakima Bull Trout Literature Bot", page_icon="🐟")
password = st.text_input("Enter password to access the app:", type="password")
if password != "yakima2026":
    st.stop()
st.title("🐟 Yakima Bull Trout Literature Bot")
st.caption("Ask questions about the Yakima Basin bull trout literature library")

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="yakima")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    anthropic = Anthropic(api_key="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")
    return collection, model, anthropic

collection, model, anthropic = load_resources()

SYSTEM_PROMPT = """You are an expert fish biologist specializing in bull trout ecology and conservation 
in the Yakima Basin. You are assisting USFWS biologists who need detailed, technical answers.

When answering:
- Cite sources using the full title and year provided in the source label, e.g. "Smith et al. (2001) 
  found that..." or "(Jones & Brown, 1998)". Use author-style citations, not just filenames.
- Note the age of information when relevant — flag when you're drawing on older studies (pre-2000) 
  vs recent work, and highlight if findings have been updated or contradicted over time
- Be specific and quantitative — include actual numbers, measurements, temperatures, distances, 
  population counts, and dates from the literature when available
- Synthesize across multiple sources when they agree or disagree — note contradictions explicitly
- Use technical biological terminology appropriate for professional biologists
- Structure longer answers with clear sections if the topic warrants it
- If the excerpts don't contain enough to answer well, say exactly what is missing and suggest 
  what kind of source might have it

Do not give vague summaries. Give the kind of detailed answer a senior biologist would give 
when briefing a colleague. Always ground your answer in the specific literature provided."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

use_web = st.sidebar.checkbox("🌐 Include web search", value=False)

if question := st.chat_input("Ask about bull trout in the Yakima Basin..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    question_embedding = model.encode(question)
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=30
    )

    context_parts = []
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    title = meta.get('title', '')
    year = meta.get('year', 'unknown')
    source = meta['source']
    label = f"{title} ({year}) [{source}]" if title else f"{source} ({year})"
    context_parts.append(f"[Source: {label}]\n{doc}")
context = "\n\n---\n\n".join(context_parts)

    with st.chat_message("assistant"):
        with st.spinner("Searching literature..."):

            # Build tools list
            tools = []
            if use_web:
                tools.append({"type": "web_search_20250305", "name": "web_search"})

            # Build API call with caching
            api_kwargs = dict(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Literature excerpts:\n\n{context}",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": f"Question: {question}"
                        }
                    ]
                }]
            )

            if tools:
                api_kwargs["tools"] = tools

            response = anthropic.messages.create(**api_kwargs)

            # Extract text from response (web search returns multiple content blocks)
            answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer += block.text

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})