import html
import os
import tempfile
from typing import Any
from pyvis.network import Network


LABEL_COLORS = {
    "Disease": "#ffb3b3",
    "Symptom": "#b3d9ff",
    "Drug": "#b3ffb3",
    "Exam": "#ffffb3",
    "Indicator": "#e6b3ff",
    "Entity": "#d9d9d9",
}

RELATION_COLORS = {
    "HAS_SYMPTOM": "#ff6666",
    "RECOMMENDED_DRUG": "#66cc66",
    "TREATS": "#66cc66",
    "NEEDS_EXAM": "#6699ff",
    "HAS_INDICATOR": "#cc66ff",
    "CONTRAINDICATED_FOR": "#ff9933",
    "INFERRED": "#999999",
    "RELATED_TO": "#cccccc",
}


def build_pyvis_html(
    subgraph: dict[str, Any],
    height: str = "500px",
    width: str = "100%",
    physics: bool = True,
) -> str:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    if not nodes:
        return ""

    net = Network(
        height=height,
        width=width,
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True,
    )
    node_ids = set()

    for node in nodes:
        node_id = node.get("id")
        if node_id is None or node_id == "":
            continue
        node_ids.add(node_id)
        raw_label = str(node.get("label", node_id))
        node_type = str(node.get("type", "Entity"))
        color = LABEL_COLORS.get(node_type, LABEL_COLORS["Entity"])
        label = html.escape(raw_label, quote=True)
        safe_type = html.escape(node_type, quote=True)

        net.add_node(
            node_id,
            label=label,
            title=f"{label} ({safe_type})",
            color=color,
            size=25,
            font={"size": 14},
        )

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue
        if source not in node_ids or target not in node_ids:
            continue
        relation_raw = str(edge.get("relation", "RELATED_TO"))
        score = edge.get("score")
        color = RELATION_COLORS.get(relation_raw, RELATION_COLORS["RELATED_TO"])
        relation = html.escape(relation_raw, quote=True)

        title = relation
        if score is not None:
            try:
                score_val = float(score)
                title = f"{relation} (score: {score_val:.2f})"
            except (TypeError, ValueError):
                title = relation

        net.add_edge(
            source,
            target,
            label=relation,
            title=title,
            color=color,
            arrows="to",
            font={"size": 10, "align": "middle"},
        )

    if physics:
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "centralGravity": 0.3,
                    "springLength": 150,
                    "springConstant": 0.04
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragNodes": true
            },
            "nodes": {
                "borderWidth": 2,
                "shadow": true
            },
            "edges": {
                "smooth": {
                    "type": "continuous"
                }
            }
        }
        """)
    else:
        net.set_options("""
        {
            "physics": {"enabled": false},
            "interaction": {"hover": true}
        }
        """)

    fd, temp_path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        net.save_graph(temp_path)
        return temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return ""


def get_html_content(subgraph: dict[str, Any], **kwargs) -> str:
    html_path = build_pyvis_html(subgraph, **kwargs)
    if not html_path:
        return ""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.unlink(html_path)
        except OSError:
            pass
