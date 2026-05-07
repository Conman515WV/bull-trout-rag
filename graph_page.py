"""
graph_page.py — Phase 4: Fisheries Entity Wiki
================================================
Replaces the PyVis graph visualization with a searchable wiki-style
reference tool. Each entity gets its own page showing:
  - Category and paper count
  - Haiku-written literature summary
  - Related entities (neighbors in the graph, sorted by edge weight)
  - Full list of papers mentioning this entity

Node selection is handled via search + selectbox (no JS click events needed).

KNOWN LIMITATION: PyVis click events cannot be relayed back to Python through
st.components.v1.html. Wiki approach sidesteps this entirely.
"""

import os
import pickle
import streamlit as st
import networkx as nx

# ── Configuration ──────────────────────────────────────────────────────────────
GRAPH_PATH = "graph.pkl"
SUMMARIES_PATH = "node_summaries.pkl"

CATEGORY_COLORS = {
    "species":        "#e74c3c",
    "location":       "#3498db",
    "infrastructure": "#f39c12",
    "management":     "#2ecc71",
    "monitoring":     "#9b59b6",
    "agency":         "#1abc9c",
    "other":          "#95a5a6",
}

CATEGORY_ICONS = {
    "species":        "🐟",
    "location":       "📍",
    "infrastructure": "🏗️",
    "management":     "📋",
    "monitoring":     "📡",
    "agency":         "🏛️",
    "other":          "🔹",
}


@st.cache_data
def load_graph():
    if not os.path.exists(GRAPH_PATH):
        return None
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_summaries():
    if not os.path.exists(SUMMARIES_PATH):
        return {}
    with open(SUMMARIES_PATH, "rb") as f:
        return pickle.load(f)


def category_badge(category):
    color = CATEGORY_COLORS.get(category, "#95a5a6")
    icon = CATEGORY_ICONS.get(category, "🔹")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.8rem;font-weight:600;">'
        f'{icon} {category.capitalize()}</span>'
    )


def render_entity_page(entity, G, summaries):
    """Render the full wiki page for a selected entity."""
    node_data = G.nodes[entity]
    category = node_data.get("category", "other")
    papers = node_data.get("papers", [])
    degree = G.degree(entity)
    summary = summaries.get(entity, "")

    # Header
    st.markdown(f"## {CATEGORY_ICONS.get(category, '🔹')} {entity}")
    st.markdown(category_badge(category), unsafe_allow_html=True)
    st.markdown(f"**{len(papers)} papers** · **{degree} connections**")
    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Literature summary
        st.markdown("### 📖 Literature Summary")
        if summary:
            st.info(summary)
        else:
            st.caption("No summary available for this entity.")

        # Papers
        st.markdown(f"### 📄 Papers ({len(papers)})")
        for p in sorted(papers):
            # Strip .pdf extension for cleaner display
            display = p.replace(".pdf", "").replace("_", " ")
            st.markdown(f"- `{display}`")

    with col_right:
        # Related entities sorted by edge weight
        st.markdown("### 🔗 Related Entities")
        neighbors = [
            (nb, G[entity][nb].get("weight", 1), G.nodes[nb].get("category", "other"))
            for nb in G.neighbors(entity)
        ]
        neighbors.sort(key=lambda x: x[1], reverse=True)

        for nb, weight, nb_cat in neighbors[:30]:
            color = CATEGORY_COLORS.get(nb_cat, "#95a5a6")
            icon = CATEGORY_ICONS.get(nb_cat, "🔹")
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a2a;">'
                f'<span style="color:{color}">{icon}</span> '
                f'**{nb}** '
                f'<span style="color:#888;font-size:0.8rem">({weight} papers)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Button to navigate to that entity
            if st.button(f"→ View {nb}", key=f"nav_{nb}", use_container_width=False):
                st.session_state["selected_entity"] = nb
                st.rerun()

        if len(neighbors) > 30:
            st.caption(f"... and {len(neighbors) - 30} more connections")

    # Ask in chat button
    st.divider()
    if st.button(f"💬 Ask about '{entity}' in chat", use_container_width=True):
        st.session_state["graph_query"] = entity
        st.session_state["graph_navigate_to_chat"] = True
        st.rerun()


def render_index(G, summaries, search_query, category_filter):
    """Render the entity index / browse view."""
    st.markdown("### 📚 Entity Index")

    # Build filtered node list
    nodes = []
    for node, data in G.nodes(data=True):
        cat = data.get("category", "other")
        if category_filter and category_filter != "All" and cat != category_filter:
            continue
        if search_query and search_query.lower() not in node.lower():
            continue
        nodes.append((node, data, G.degree(node)))

    # Sort by degree (most connected first)
    nodes.sort(key=lambda x: x[2], reverse=True)

    st.caption(f"Showing {len(nodes)} entities")

    # Display as cards in a grid
    cols_per_row = 2
    for i in range(0, len(nodes), cols_per_row):
        row_nodes = nodes[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (node, data, degree) in zip(cols, row_nodes):
            with col:
                cat = data.get("category", "other")
                color = CATEGORY_COLORS.get(cat, "#95a5a6")
                icon = CATEGORY_ICONS.get(cat, "🔹")
                papers = data.get("papers", [])
                summary = summaries.get(node, "")
                summary_snippet = summary[:120] + "..." if len(summary) > 120 else summary

                st.markdown(
                    f'<div style="border:1px solid {color};border-radius:8px;'
                    f'padding:12px;margin-bottom:8px;background:#1a1a1a;">'
                    f'<div style="font-weight:700;font-size:1rem;">{icon} {node}</div>'
                    f'<div style="color:{color};font-size:0.75rem;margin:4px 0;">'
                    f'{cat.capitalize()} · {len(papers)} papers · {degree} connections</div>'
                    f'<div style="color:#aaa;font-size:0.8rem;">{summary_snippet}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("View →", key=f"view_{node}", use_container_width=True):
                    st.session_state["selected_entity"] = node
                    st.rerun()


def run():
    """Main wiki page — called from app.py navigation."""
    G = load_graph()
    if G is None:
        st.error("📂 Graph not found. Build the graph first by running:")
        st.code("python extract_entities.py && python build_graph.py && python generate_summaries.py")
        return

    summaries = load_summaries()

    # ── Top bar ────────────────────────────────────────────────────────────────
    st.markdown("### 🐟 Yakima Fisheries Knowledge Base")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        search_query = st.text_input(
            "Search entities...", 
            key="wiki_search",
            placeholder="e.g. bull trout, Yakima River, USFWS...",
            label_visibility="collapsed",
        )
    with col2:
        categories = ["All"] + sorted(set(
            d.get("category", "other") for _, d in G.nodes(data=True)
        ))
        category_filter = st.selectbox(
            "Filter by category", categories, key="wiki_cat_filter",
            label_visibility="collapsed",
        )
    with col3:
        if st.button("← Back to Index", use_container_width=True):
            st.session_state.pop("selected_entity", None)
            st.rerun()

    st.divider()

    # ── Stats row ──────────────────────────────────────────────────────────────
    if "selected_entity" not in st.session_state:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entities", G.number_of_nodes())
        c2.metric("Total Connections", G.number_of_edges())
        c3.metric("Papers Indexed", len(set(
            p for _, d in G.nodes(data=True) for p in d.get("papers", [])
        )))
        c4.metric("With Summaries", sum(1 for n in G.nodes() if summaries.get(n)))

    # ── Route: entity page or index ────────────────────────────────────────────
    # Check if search selects a single exact match
    if search_query:
        exact = [n for n in G.nodes() if n.lower() == search_query.lower()]
        if exact:
            st.session_state["selected_entity"] = exact[0]

    selected = st.session_state.get("selected_entity")
    if selected and G.has_node(selected):
        render_entity_page(selected, G, summaries)
    else:
        render_index(G, summaries, search_query, category_filter)


if __name__ == "__main__":
    run()
