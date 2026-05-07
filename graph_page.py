"""
graph_page.py — Phase 4: Graph Explorer UI for Streamlit
=========================================================
Renders the fisheries knowledge graph as an interactive PyVis network.

KNOWN LIMITATIONS:
1. PyVis click events cannot be relayed back to Python through st.components.v1.html
   because the HTML runs in a sandboxed iframe. Node selection is instead handled
   via a Streamlit selectbox — when the user picks a node from the dropdown,
   the graph re-renders showing the ego network.
2. Large graphs (hundreds of nodes) render slowly in PyVis. If total nodes > 150,
   the default view shows a filtered subset (user must pick a category first)
   rather than the full graph. This is flagged here and enforced in render_graph().

Run order for building graph data:
  python extract_entities.py && python build_graph.py && python generate_summaries.py
"""

import os
import pickle
import streamlit as st
import networkx as nx
from pyvis.network import Network

# ── Configuration ─────────────────────────────────────────────────────────────
GRAPH_PATH = "graph.pkl"
SUMMARIES_PATH = "node_summaries.pkl"
PYVIS_HEIGHT = "700"
LARGE_GRAPH_THRESHOLD = 150  # nodes; above this, default to filtered view

CATEGORY_COLORS = {
    "species": "#e74c3c",       # red
    "location": "#3498db",      # blue
    "infrastructure": "#f39c12", # orange
    "management": "#2ecc71",    # green
    "monitoring": "#9b59b6",    # purple
    "agency": "#1abc9c",        # teal
    "other": "#95a5a6",         # gray
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


def get_category_stats(G):
    """Return count of nodes per category."""
    stats = {}
    for node, data in G.nodes(data=True):
        cat = data.get("category", "other")
        stats[cat] = stats.get(cat, 0) + 1
    return stats


def get_top_nodes(G, n=30):
    """Get top N nodes by degree (most connections)."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return [node for node, deg in degrees[:n]]


def build_pyvis_html(G, node_summaries, selected_node=None, category_filter=None, node_limit=None):
    """
    Build a PyVis network HTML string.
    If selected_node is set, show only ego network (node + neighbors).
    If category_filter is set, show only nodes of that category (plus their neighbors
    if selected_node is also set).
    If node_limit is set and selected_node is None, show only top N nodes.
    """
    net = Network(height=f"{PYVIS_HEIGHT}px", width="100%", bgcolor="#1a1a2e",
                  font_color="#ffffff", directed=False)

    # Physics config (Barnes-Hut)
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250,
                   spring_strength=0.001, damping=0.09, overlap=0)

    # Determine which nodes to show
    if selected_node and G.has_node(selected_node):
        # Ego network: node + immediate neighbors
        ego_nodes = set(G.neighbors(selected_node))
        ego_nodes.add(selected_node)
        if category_filter:
            ego_nodes = {n for n in ego_nodes
                         if G.nodes[n].get("category", "other") == category_filter
                         or n == selected_node}
        nodes_to_show = ego_nodes
    elif category_filter:
        nodes_to_show = {n for n in G.nodes()
                         if G.nodes[n].get("category", "other") == category_filter}
    elif node_limit:
        nodes_to_show = set(get_top_nodes(G, node_limit))
    else:
        nodes_to_show = set(G.nodes())

    edges_to_show = []
    for u, v, data in G.edges(data=True):
        if u in nodes_to_show and v in nodes_to_show:
            edges_to_show.append((u, v, data))

    # Add nodes
    for node in nodes_to_show:
        data = G.nodes[node]
        degree = G.degree(node)
        category = data.get("category", "other")
        color = CATEGORY_COLORS.get(category, "#95a5a6")
        # Scale node size: min 10, max ~50 based on degree
        size = 10 + min(degree * 3, 40)

        # Tooltip shows summary (if available) and paper count
        summary = node_summaries.get(node, "")
        papers = data.get("papers", [])
        tooltip = f"<b>{node}</b><br>Category: {category}<br>Papers: {len(papers)}"
        if summary:
            tooltip += f"<br><br>{summary[:200]}..."

        net.add_node(node, label=node, title=tooltip, color=color,
                     size=size, font={"size": 14, "color": "#ffffff"})

    # Add edges
    for u, v, data in edges_to_show:
        weight = data.get("weight", 1)
        # Scale edge width: weight * 2, min 1, max 15
        width = min(max(weight * 2, 1), 15)
        opacity = min(0.3 + weight * 0.1, 0.9)
        net.add_edge(u, v, width=width, color=f"rgba(255,255,255,{opacity})")

    return net.generate_html()


def run():
    """Main graph explorer page — called from st.navigation."""
    G = load_graph()
    if G is None:
        st.error("📂 Graph not found. Build the graph first by running:")
        st.code("python extract_entities.py && python build_graph.py && python generate_summaries.py")
        return

    summaries = load_summaries()

    st.markdown("### 🔗 Fisheries Knowledge Graph Explorer")

    # ── Filters row ────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        all_nodes = sorted(G.nodes())
        search_query = st.text_input("🔍 Search nodes by name...", key="graph_search")
    with col2:
        cat_stats = get_category_stats(G)
        categories = ["All"] + sorted(cat_stats.keys())
        cat_filter_choice = st.selectbox("🏷️ Filter by category", categories, key="graph_cat_filter")

    category_filter = None if cat_filter_choice == "All" else cat_filter_choice

    # ── Node selector (replaces JS click events — see module docstring) ───
    # Pre-filter available nodes based on category
    available_nodes = all_nodes[:]
    if category_filter:
        available_nodes = [n for n in all_nodes
                           if G.nodes[n].get("category", "other") == category_filter]

    # If search query, further filter
    if search_query:
        available_nodes = [n for n in available_nodes
                           if search_query.lower() in n.lower()]

    selected_node = st.selectbox(
        "📌 Select a node to view its ego network",
        ["Full Graph"] + available_nodes,
        index=0,
        key="graph_node_select",
    )

    ego_node = None if selected_node == "Full Graph" else selected_node

    # Determine if full graph is too large
    total_nodes = G.number_of_nodes()
    show_all = ego_node is None and category_filter is None
    node_limit = None
    if show_all and total_nodes > LARGE_GRAPH_THRESHOLD:
        st.warning(
            f"⚠️ Full graph has {total_nodes} nodes — showing top {LARGE_GRAPH_THRESHOLD} "
            f"most-connected by default for performance. Use the category filter or search "
            f"to view specific subsets."
        )
        node_limit = LARGE_GRAPH_THRESHOLD

    # ── Render graph ──────────────────────────────────────────────────────
    html = build_pyvis_html(G, summaries, selected_node=ego_node,
                            category_filter=category_filter, node_limit=node_limit)
    st.components.v1.html(html, height=int(PYVIS_HEIGHT) + 20, scrolling=True)

    # ── Info panel (below graph) ──────────────────────────────────────────
    if ego_node and G.has_node(ego_node):
        st.divider()
        col_left, col_right = st.columns([2, 1])

        with col_left:
            node_data = G.nodes[ego_node]
            summary = summaries.get(ego_node, "")
            st.subheader(f"📋 {ego_node}")
            st.caption(f"Category: {node_data.get('category', 'other').capitalize()} | "
                       f"Degree: {G.degree(ego_node)} | "
                       f"Papers: {len(node_data.get('papers', []))}")

            if summary:
                st.markdown("#### Literature Summary")
                st.info(summary)
            else:
                st.caption("No summary available for this node.")

            # Papers list
            papers = node_data.get("papers", [])
            if papers:
                st.markdown(f"#### Papers mentioning this entity ({len(papers)})")
                for p in papers:
                    st.markdown(f"📄 `{p}`")

        with col_right:
            st.markdown("#### Neighbors")
            neighbors = sorted(G.neighbors(ego_node))
            for nb in neighbors:
                nb_cat = G.nodes[nb].get("category", "other")
                nb_color = CATEGORY_COLORS.get(nb_cat, "#95a5a6")
                weight = G[ego_node][nb].get("weight", 1)
                st.markdown(
                    f"<span style='color:{nb_color}'>●</span> "
                    f"**{nb}** (w={weight})",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            if st.button("💬 Ask about this in chat", use_container_width=True):
                st.session_state["graph_query"] = ego_node
                st.session_state["graph_navigate_to_chat"] = True
                st.rerun()


if __name__ == "__main__":
    run()
