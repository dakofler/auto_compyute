"""Tensor class"""

from __future__ import annotations

import base64
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Optional

from .devices import (
    Array,
    Device,
    Dim,
    Scalar,
    Shape,
    get_array_device,
    move_to_device,
    select_device,
)
from .dtypes import DType, float32, is_float
from .funcs.binary_funcs import Add, Div, Matmul, Maximum, Minimum, Mul, Sub
from .funcs.function import Context, Function
from .funcs.reduce_funcs import Max, Mean, Min, Std, Sum, Var
from .funcs.shape_funcs import Select, Transpose, View
from .funcs.unary_funcs import Exp, Pow, Tanh

__all__ = ["Tensor", "tensor", "no_grad"]


def tensor(data: Any, **factory_kwargs) -> Tensor:
    if isinstance(data, Tensor):
        return data
    device = select_device(factory_kwargs.get("device", None))
    dtype = factory_kwargs.get("dtype", None)
    requires_grad = factory_kwargs.get("requires_grad", False)
    data = device.m.array(data, dtype)
    return Tensor(data, requires_grad=requires_grad)


class Tensor:
    def __init__(
        self,
        data: Array,
        ctx: Optional[Function] = None,
        parents: Optional[tuple[Tensor, ...]] = None,
        requires_grad: bool = False,
    ) -> None:
        self.data = data
        self.ctx = ctx
        self.parents = parents
        self.requires_grad = requires_grad
        self.grad: Optional[Array] = None

    @property
    def device(self) -> Device:
        return get_array_device(self.data)

    @property
    def dtype(self) -> DType:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def T(self) -> Tensor:
        return self.transpose(-2, -1)

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __add__(self, x: Tensor | Scalar) -> Tensor:
        return self.add(x)

    def __radd__(self, x: Scalar) -> Tensor:
        return self.add(x)

    def __sub__(self, x: Tensor | Scalar) -> Tensor:
        return self.sub(x)

    def __rsub__(self, x: Scalar) -> Tensor:
        return self.sub(x, reverse=True)

    def __mul__(self, x: Tensor | Scalar) -> Tensor:
        return self.mul(x)

    def __rmul__(self, x: Scalar) -> Tensor:
        return self.mul(x)

    def __truediv__(self, x: Tensor | Scalar) -> Tensor:
        return self.truediv(x)

    def __rtruediv__(self, x: Scalar) -> Tensor:
        return self.truediv(x, reverse=True)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matmul(x)

    def __pow__(self, x: Scalar) -> Tensor:
        return self.pow(x)

    def __neg__(self) -> Tensor:
        return self.mul(-1)

    def __getitem__(self, key: Any) -> Tensor:
        if isinstance(key, tuple):
            key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
        return self.select(key)

    def __repr__(self) -> str:
        prefix = "array("
        suffix = f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
        return (
            prefix
            + self.device.m.array2string(
                self.data,
                max_line_width=80,
                precision=4,
                separator=", ",
                prefix=prefix,
                floatmode="maxprec_equal",
            )
            + suffix
        )

    # ----------------------------------------------------------------------------------
    # AUTOGRAD
    # ----------------------------------------------------------------------------------

    def apply_grad(self, grad: Array) -> None:
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, output_grad: Optional[Array] = None):
        if not self.requires_grad:
            raise ValueError("Tensor does not require gradients.")
        if self.grad is None:
            self.grad = self.device.m.ones(self.shape, dtype=float32)
        if output_grad is not None:
            assert isinstance(output_grad, Array)
            self.grad *= output_grad
        tensors = deepwalk(self, [], set())

        for t in reversed(tensors):
            assert t.ctx is not None
            assert t.parents is not None
            grads = t.ctx.backward(t.grad)
            for t, grad in zip(t.parents, grads):
                if not t.requires_grad:
                    continue
                grad = undo_broadcast(grad, t.shape)
                t.apply_grad(grad)

    # ----------------------------------------------------------------------------------
    # UNARY OPS
    # ----------------------------------------------------------------------------------

    def exp(self) -> Tensor:
        return apply_func(Exp, self)

    def pow(self, x: Scalar) -> Tensor:
        return apply_func(Pow, self, tensor(x, device=self.device))

    def tanh(self) -> Tensor:
        return apply_func(Tanh, self)

    # ----------------------------------------------------------------------------------
    # BINARY OPS
    # ----------------------------------------------------------------------------------

    def add(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Add, self, tensor(x, device=self.device))

    def sub(self, x: Tensor | Scalar, reverse: bool = False) -> Tensor:
        if reverse:
            return apply_func(Sub, tensor(x, device=self.device), self)
        return apply_func(Sub, self, tensor(x, device=self.device))

    def mul(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Mul, self, tensor(x, device=self.device))

    def truediv(self, x: Tensor | Scalar, reverse: bool = False) -> Tensor:
        if reverse:
            return apply_func(Div, tensor(x, device=self.device), self)
        return apply_func(Div, self, tensor(x, device=self.device))

    def matmul(self, x: Tensor) -> Tensor:
        return apply_func(Matmul, self, tensor(x, device=self.device))

    def maximum(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Maximum, self, tensor(x, device=self.device))

    def minimum(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Minimum, self, tensor(x, device=self.device))

    # ----------------------------------------------------------------------------------
    # REDUCE OPS
    # ----------------------------------------------------------------------------------

    def sum(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return apply_func(Sum, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return apply_func(Mean, self, dim=dim, keepdims=keepdims)

    def var(
        self, dim: Optional[Dim] = None, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
        return apply_func(Var, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def std(
        self, dim: Optional[Dim] = None, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
        return apply_func(Std, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def max(self, dim: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return apply_func(Max, self, dim=dim, keepdims=keepdims)

    def min(self, dim: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return apply_func(Min, self, dim=dim, keepdims=keepdims)

    # ----------------------------------------------------------------------------------
    # SHAPE OPS
    # ----------------------------------------------------------------------------------

    def select(self, key: Any) -> Tensor:
        if isinstance(key, Tensor):
            key = key.data
        return apply_func(Select, self, key=key)

    def split(self, split_size: int, dim: int = -1) -> list[Tensor]:
        splits = self.shape[dim] // split_size
        dim = dim % self.ndim
        return [
            self[
                (slice(None),) * dim
                + (slice(i * split_size, (i + 1) * split_size),)
                + (slice(None),) * (self.ndim - dim - 1)
            ]
            for i in range(splits)
        ]

    def transpose(self, dim1: int = -1, dim2: int = -2) -> Tensor:
        return apply_func(Transpose, self, dim1=dim1, dim2=dim2)

    def view(self, shape) -> Tensor:
        return apply_func(View, self, shape=shape)

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    def as_type(self, dtype: DType) -> Tensor:
        if self.dtype == dtype:
            return self
        data = self.data.astype(dtype)
        if is_float(dtype) and self.requires_grad:
            new_tensor = Tensor(data, self.ctx, self.parents, self.requires_grad)
            if self.grad is not None:
                new_tensor.grad = self.grad.astype(dtype)
            return new_tensor
        return Tensor(self.data.astype(dtype))

    def to(self, device: Device) -> Tensor:
        data = self.data if self.device == device else move_to_device(self.data, device)
        if self.requires_grad:
            new_tensor = Tensor(data, self.ctx, self.parents, self.requires_grad)
            if self.grad is not None:
                new_tensor.grad = move_to_device(self.grad, device)
            return new_tensor
        return Tensor(data)

    def ito(self, device: Device) -> None:
        if self.device == device:
            return
        self.data = move_to_device(self.data, device)
        if self.grad is not None:
            self.grad = move_to_device(self.grad, device)

    def item(self) -> Scalar:
        return self.data.item()


def get_shape_diff(shape1: Shape, shape2: Shape) -> Shape:
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])


def undo_broadcast(grad: Array, target_shape: Shape) -> Array:
    if grad.shape == target_shape:
        return grad
    target_ndim = len(target_shape)

    if grad.ndim == target_ndim:
        axis = get_shape_diff(grad.shape, target_shape)
        grad = grad.sum(axis, keepdims=True)
    else:
        data_shape = (1,) * (grad.ndim - target_ndim) + target_shape
        axis = get_shape_diff(grad.shape, data_shape)
        grad = grad.sum(axis=axis)

    return grad.reshape(target_shape)


def deepwalk(t: Tensor, nodes: list[Tensor], visited_node_ids: set) -> list[Tensor]:
    if id(t) not in visited_node_ids:
        visited_node_ids.add(id(t))
        if not t.parents:
            return
        for p in t.parents:
            if p.requires_grad is False:
                continue
            deepwalk(p, nodes, visited_node_ids)
        nodes.append(t)
    return nodes


def apply_func(func: type[Function], *tensors: Tensor, **kwargs: Any) -> Tensor:
    f = func(tensors[0].device)
    requires_grad = any(t.requires_grad for t in tensors)
    if autograd_active and requires_grad:
        f.ctx = Context()
        data = f.forward(*[t.data for t in tensors], **kwargs)
        return Tensor(data, f, tensors, True)
    data = f.forward(*[t.data for t in tensors], **kwargs)
    return Tensor(data)


def draw_compute_graph(node: Tensor, html: bool = False) -> None:
    colors = {
        "const": ("#CAEDFB", "#4D93D9"),
        "leaf": ("#C6EFCE", "#4EA72E"),
        "func": ("#F2F2F2", "#808080"),
    }

    def _node_label(node: Tensor) -> str:
        desc = (
            node.ctx.name
            if node.ctx is not None
            else str(node.shape)[1:-1] if node.shape else str(node.data.item())
        )

        if not node.requires_grad:
            fill_color, stroke_color = colors["const"]
        elif node.ctx is None:
            fill_color, stroke_color = colors["leaf"]
        else:
            fill_color, stroke_color = colors["func"]

        return f"{id(node)}({desc})\nstyle {str(id(node))} fill:{fill_color},stroke:{stroke_color}"

    mermaid_script = f"graph LR\n{_node_label(node)}\n"

    def _deepwalk(node: Tensor, script: str) -> str:
        if not node.parents:
            return ""
        for p in node.parents:
            p_label = _node_label(p)
            if p_label not in script:
                script += f"{p_label}\n"
            edge = f"{str(id(p))}-->{str(id(node))}\n"
            if edge not in script:
                script += edge
            if p.parents:
                script = _deepwalk(p, script)
        return script

    mermaid_script = _deepwalk(node, mermaid_script)

    graphbytes = mermaid_script.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    svg_url = f"https://mermaid.ink/svg/{base64_string}?bgColor=FFFFFF"

    if html:
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Mermaid Diagram</title>
            </head>
            <body>
                <object type="image/svg+xml" data="{svg_url}" style="width:100%; height:auto;"></object>
            </body>
            </html>
            """

        with open("graph.html", "w") as file:
            file.write(html_content)
    else:
        from IPython.display import Image, display

        display(Image(url=svg_url))


autograd_active = True


def set_autograd(active: bool) -> None:
    global autograd_active
    autograd_active = active


@contextmanager
def no_grad() -> Generator:
    set_autograd(False)
    try:
        yield
    finally:
        set_autograd(True)
