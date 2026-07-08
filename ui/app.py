"""
FlowRevamp – Human-in-the-Loop Refinement UI
=============================================

Streamlit app that lets users:
  • View the original image with detection overlays
  • Edit OCR text per node
  • Add / remove / modify edges
  • Re-export corrected JSON

Run:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.io_helpers import list_images, load_json, save_json
from utils.visualization import draw_results

st.set_page_config(page_title="FlowRevamp – Refinement", layout="wide")
st.title("🔄 FlowRevamp – Human-in-the-Loop Refinement")

# ── Sidebar: select a processed image ───────────────────────
output_jsons = sorted(config.OUTPUT_DIR.glob("*_graph.json"))
if not output_jsons:
    st.warning("No processed results found. Run the pipeline first:\n"
               "```\npython main.py\n```")
    st.stop()

selected = st.sidebar.selectbox(
    "Select result", output_jsons,
    format_func=lambda p: p.stem.replace("_graph", ""),
)

graph = load_json(selected)
nodes = graph["nodes"]
edges = graph["edges"]

# ── Main area: visualisation ─────────────────────────────────
col_img, col_edit = st.columns([1.2, 1])

with col_img:
    st.subheader("Detection Overlay")
    show_bad = st.checkbox("Hiển thị Node lỗi (viền đỏ)", value=True)
    
    vis_nodes = nodes if show_bad else [n for n in nodes if not n.get("is_bad", False)]
    
    src_img = config.INPUT_DIR / graph.get("source_image", "")
    if src_img.exists():
        # Regenerate visualisation on the fly based on toggle
        tmp_vis = draw_results(src_img, vis_nodes, edges)
        st.image(str(tmp_vis), use_container_width=True)
    else:
        st.info("Source image not found.")

# ── Edit panel ────────────────────────────────────────────────
with col_edit:
    st.subheader("Good Nodes")
    updated_nodes = []
    _NODE_TYPES = ["Process", "Decision", "Terminal", "IO", "Connector", "Bad"]
    
    for i, node in enumerate(nodes):
        if node.get("is_bad", False): continue
        with st.expander(f'{node["id"]} – {node["type"]}', expanded=False):
            ntype = st.selectbox(
                "Type", _NODE_TYPES,
                index=_NODE_TYPES.index(node["type"])
                      if node["type"] in _NODE_TYPES else 0,
                key=f"type_{i}",
            )
            ntext = st.text_input("Text", value=node.get("text", ""),
                                   key=f"text_{i}")
            updated_nodes.append({**node, "type": ntype, "text": ntext})

    st.subheader("Bad Nodes (Overlapping)")
    for i, node in enumerate(nodes):
        if not node.get("is_bad", False): continue
        with st.expander(f'{node["id"]} – {node["type"]}', expanded=False):
            ntype = st.selectbox(
                "Type", _NODE_TYPES,
                index=_NODE_TYPES.index(node["type"])
                      if node["type"] in _NODE_TYPES else 0,
                key=f"type_{i}",
            )
            ntext = st.text_input("Text", value=node.get("text", ""),
                                   key=f"text_{i}")
            updated_nodes.append({**node, "type": ntype, "text": ntext})

    st.subheader("Edit Edges")
    node_ids = [n["id"] for n in updated_nodes]
    updated_edges = []
    for j, edge in enumerate(edges):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            src = st.selectbox("Source", node_ids,
                               index=node_ids.index(edge["source"])
                                     if edge["source"] in node_ids else 0,
                               key=f"src_{j}")
        with c2:
            tgt = st.selectbox("Target", node_ids,
                               index=node_ids.index(edge["target"])
                                     if edge["target"] in node_ids else 0,
                               key=f"tgt_{j}")
        with c3:
            lbl = st.text_input("Label", value=edge.get("label", ""),
                                 key=f"lbl_{j}")
        updated_edges.append({"source": src, "target": tgt, "label": lbl})

    # Add new edge
    st.markdown("---")
    if st.button("➕ Add Edge"):
        updated_edges.append({"source": node_ids[0], "target": node_ids[-1],
                               "label": ""})

    # Save
    if st.button("💾 Save Corrections", type="primary"):
        corrected = {
            "source_image": graph.get("source_image", ""),
            "nodes": updated_nodes,
            "edges": updated_edges,
        }
        save_json(corrected, selected)
        # Regenerate visualisation
        src_img = config.INPUT_DIR / graph.get("source_image", "")
        if src_img.exists():
            draw_results(src_img, updated_nodes, updated_edges)
        st.success("Saved! Refresh to see updated visualisation.")

# ── Raw JSON preview ──────────────────────────────────────────
with st.expander("📄 Raw JSON"):
    st.json({"nodes": updated_nodes, "edges": updated_edges})
