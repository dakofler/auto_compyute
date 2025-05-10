"""Vizualization tools."""

from importlib import import_module
from typing import Any, Literal

from auto_compyute.autograd import Tensor, dfs

MERMAID_NODE_COLORS = {
    "const": ("#CAEDFB", "#4D93D9"),
    "leaf": ("#C6EFCE", "#4EA72E"),
    "op": ("#F2F2F2", "#808080"),
}


def _get_mermaid_node_def(node: Tensor) -> str:
    node_id = str(id(node))
    node_name = node.label
    node_data_shape = str(node.shape).replace("shape", "")
    if node.ctx is not None:
        # exclue select ops key
        node_op_kwargs = ", ".join(f"{k}={v}" for k, v in node.ctx.kwargs.items() if k != "key")
    else:
        node_op_kwargs = ""

    node_label = f"<b>{node_name}</b><br><small>"
    if node_op_kwargs:
        node_label += f"kwargs: {node_op_kwargs}<br>"
    node_label += f"shape: {node_data_shape}<br>"
    node_label += f"dtype: {node.dtype!s}</small>"
    return f'{node_id}("{node_label}")'


def _get_mermaid_node_style(n: Tensor) -> str:
    node_id = str(id(n))
    if not n.req_grad:
        fill_color, stroke_color = MERMAID_NODE_COLORS["const"]
    elif n.ctx is None:
        fill_color, stroke_color = MERMAID_NODE_COLORS["leaf"]
    else:
        fill_color, stroke_color = MERMAID_NODE_COLORS["op"]
    return f"style {node_id} fill:{fill_color},stroke:{stroke_color}"


def draw_graph(
    root_node: Tensor,
    orientation: Literal["LR", "TD"] = "LR",
    save_to_file: bool = False,
) -> Any:
    """Draws the compute graph based on a root node.

    Args:
        root_node (Tensor): Root node of the compute graph.
        orientation (Literal["LR", "TD"]): Layout of the drawn graph (LR=left-to-right,
            TD=top-to-bottom). Defaults to `LR`.
        save_to_file (bool): Whether to save the graph to an HTML-file. Defaults to `False`.

    Returns:
        Mermaid: The resulting Mermaid diagram, if `save_to_file=False`.

    Raises:
        AssertionError: If the root node is not part of a compute graph.
        ModuleNotFoundError: If `mermaid-python` is not installed.
    """
    assert root_node.req_grad, "Node not in autograd graph"

    try:
        mermaid = import_module("mermaid")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install mermaid-python to draw graphs.") from exc

    mermaid_script = f"graph {orientation}\n"
    nodes: list[Tensor] = []
    dfs(root_node, nodes, set(), include_leaf_nodes=True)

    for node in nodes:
        # add node definition and style
        mermaid_script += f"{_get_mermaid_node_def(node)}\n"
        mermaid_script += f"{_get_mermaid_node_style(node)}\n"

        # add edges from src nodes to node
        for src_node in node.src:
            src_node_id = str(id(src_node))
            node_id = str(id(node))
            mermaid_script += f"{src_node_id}-->{node_id}\n"

    mermaid_html = mermaid.Mermaid(mermaid_script)
    if save_to_file:
        with open("compute_graph.html", "w", encoding="utf-8") as f:
            f.write(mermaid_html._repr_html_())
    else:
        return mermaid_html
