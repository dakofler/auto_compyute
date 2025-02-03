"""Autograd engine"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Optional

from .backends import (
    Array,
    Device,
    DeviceLike,
    Dim,
    Scalar,
    Shape,
    array_to_string,
    get_array_device,
    move_to_device,
)
from .dtypes import DType, float32, int32, int64, is_float
from .funcs.binary_funcs import Add, Div, Matmul, Maximum, Minimum, Mul, Sub
from .funcs.function import Cache, Function
from .funcs.reduce_funcs import Max, Mean, Min, Std, Sum, Var
from .funcs.shape_funcs import Select, Split, Squeeze, Transpose, View
from .funcs.unary_funcs import Abs, Exp, Pow, Sqrt, Tanh, Tril, Triu

__all__ = ["Tensor", "no_autograd_tracing"]


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
    def size(self) -> int:
        return self.data.size

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
        return (-self).add(x)

    def __mul__(self, x: Tensor | Scalar) -> Tensor:
        return self.mul(x)

    def __rmul__(self, x: Scalar) -> Tensor:
        return self.mul(x)

    def __truediv__(self, x: Tensor | Scalar) -> Tensor:
        return self.truediv(x)

    def __rtruediv__(self, x: Scalar) -> Tensor:
        return (self**-1).mul(x)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matmul(x)

    def __pow__(self, x: Scalar) -> Tensor:
        return self.pow(x)

    def __neg__(self) -> Tensor:
        return self.mul(-1)

    def __getitem__(self, key: Any) -> Tensor:
        return self.select(key)

    def __repr__(self) -> str:
        prefix = "array("
        suffix = f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
        return prefix + array_to_string(self.data, prefix) + suffix

    def __len__(self) -> int:
        return self.data.shape[0]

    # ----------------------------------------------------------------------------------
    # AUTOGRAD METHODS
    # ----------------------------------------------------------------------------------

    def apply_grad(self, grad: Array) -> None:
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, dy: Optional[Array] = None):
        assert self.requires_grad, "Node not in autograd graph"
        assert self.grad is None, "Cannot run backward multiple times."

        # set node grad
        if dy is None:
            self.grad = self.device.backend.ones(self.shape, dtype=self.dtype)
        else:
            assert isinstance(dy, Array)
            self.grad = dy

        # run backward through traced graph
        node_queue = build_backward_queue(self, [], set())
        for node in reversed(node_queue):
            assert node.ctx is not None, "Node has no function context"
            assert node.parents is not None, "Node has no parent nodes"
            grads = node.ctx.backward(node.grad)
            for parent, grad in zip(node.parents, grads):
                if not parent.requires_grad:
                    continue
                grad = _undo_broadcast(grad, parent.shape)
                parent.apply_grad(grad)

            # clear context of intermediate nodes
            node.grad, node.ctx, node.parents = None, None, None

    # ----------------------------------------------------------------------------------
    # UNARY OPS
    # ----------------------------------------------------------------------------------
    def abs(self) -> Tensor:
        return apply_func(Abs, self)

    def exp(self) -> Tensor:
        return apply_func(Exp, self)

    def pow(self, x: Scalar) -> Tensor:
        return apply_func(Pow, self, exp=x)

    def sqrt(self) -> Tensor:
        return apply_func(Sqrt, self)

    def tanh(self) -> Tensor:
        return apply_func(Tanh, self)

    def tril(self, diag: int = 0) -> Tensor:
        return apply_func(Tril, self, diag=diag)

    def triu(self, diag: int = 0) -> Tensor:
        return apply_func(Triu, self, diag=diag)

    # ----------------------------------------------------------------------------------
    # BINARY OPS
    # ----------------------------------------------------------------------------------

    def add(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Add, self, self.self_like(x), isinstance(x, Scalar))

    def sub(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Sub, self, self.self_like(x), isinstance(x, Scalar))

    def mul(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Mul, self, self.self_like(x), isinstance(x, Scalar))

    def truediv(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Div, self, self.self_like(x), isinstance(x, Scalar))

    def matmul(self, x: Tensor) -> Tensor:
        return apply_func(Matmul, self, x)

    def maximum(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Maximum, self, self.self_like(x), isinstance(x, Scalar))

    def minimum(self, x: Tensor | Scalar) -> Tensor:
        return apply_func(Minimum, self, self.self_like(x), isinstance(x, Scalar))

    # ----------------------------------------------------------------------------------
    # REDUCE OPS
    # ----------------------------------------------------------------------------------

    def sum(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        return apply_func(Sum, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        return apply_func(Mean, self, dim=dim, keepdims=keepdims)

    def var(
        self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
        return apply_func(Var, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def std(
        self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
        return apply_func(Std, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def max(self, dim: Optional[int] = None, *, keepdims: bool = False) -> Tensor:
        return apply_func(Max, self, dim=dim, keepdims=keepdims)

    def min(self, dim: Optional[int] = None, *, keepdims: bool = False) -> Tensor:
        return apply_func(Min, self, dim=dim, keepdims=keepdims)

    # ----------------------------------------------------------------------------------
    # SHAPE OPS
    # ----------------------------------------------------------------------------------

    def select(self, key: Any) -> Tensor:
        key = _parse_key(key)
        return apply_func(Select, self, key=key)

    def _split(self, key: Any) -> Tensor:
        key = _parse_key(key)
        return apply_func(Split, self, key=key)

    def split(self, split_size: int, *, dim: int = -1) -> list[Tensor]:
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        return [
            self._split(pre_dim_slice + (slice(i, i + split_size),) + post_dim_slice)
            for i in range(0, self.shape[dim], split_size)
        ]

    def squeeze(self) -> Tensor:
        non_singular_dims = tuple(d for d in self.shape if d > 1)
        if len(non_singular_dims) == self.ndim:
            return self
        return apply_func(Squeeze, self, shape=non_singular_dims)

    def transpose(self, dim1: int = -1, dim2: int = -2) -> Tensor:
        return apply_func(Transpose, self, dim1=dim1, dim2=dim2)

    def view(self, *dims: int) -> Tensor:
        if dims == self.shape:
            return self
        return apply_func(View, self, shape=dims)

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    def as_type(self, dtype: DType) -> Tensor:
        if self.dtype == dtype:
            return self
        data: Array = self.data.astype(dtype)
        if self.requires_grad:
            assert is_float(dtype), "Cannot change autograd node dtype to non float."
            new_tensor = Tensor(data, self.ctx, self.parents, self.requires_grad)
            if self.grad is not None:
                new_tensor.grad = self.grad.astype(dtype)
            return new_tensor
        return Tensor(data)

    def int(self) -> Tensor:
        return self.as_type(int32)

    def long(self) -> Tensor:
        return self.as_type(int64)

    def float(self) -> Tensor:
        return self.as_type(float32)

    def to(self, device: DeviceLike) -> Tensor:
        device = device if isinstance(device, Device) else Device(device)
        data = self.data if self.device == device else move_to_device(self.data, device)
        if self.requires_grad:
            new_tensor = Tensor(data, self.ctx, self.parents, self.requires_grad)
            if self.grad is not None:
                new_tensor.grad = move_to_device(self.grad, device)
            return new_tensor
        return Tensor(data)

    def ito(self, device: DeviceLike) -> None:
        device = device if isinstance(device, Device) else Device(device)
        if self.device == device:
            return
        self.data = move_to_device(self.data, device)
        if self.grad is not None:
            self.grad = move_to_device(self.grad, device)

    def item(self) -> Any:
        return self.data.item()

    def self_like(self, x: Tensor | Scalar) -> Tensor:
        if isinstance(x, Tensor):
            return x.as_type(self.dtype)
        return Tensor(self.device.backend.asarray(x, dtype=self.dtype))

    def contiguous(self) -> Tensor:
        data = self.device.backend.ascontiguousarray(self.data)
        return Tensor(data, self.ctx, self.parents, self.requires_grad)


def _get_shape_diff(shape1: Shape, shape2: Shape) -> Shape:
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])


def _undo_broadcast(grad: Array, target_shape: Shape) -> Array:
    if grad.shape == target_shape:
        return grad
    target_ndim = len(target_shape)

    if grad.ndim == target_ndim:
        axis = _get_shape_diff(grad.shape, target_shape)
        grad = grad.sum(axis, keepdims=True)
    else:
        data_shape = (1,) * (grad.ndim - target_ndim) + target_shape
        axis = _get_shape_diff(grad.shape, data_shape)
        grad = grad.sum(axis=axis)

    return grad.reshape(target_shape)


def build_backward_queue(
    node: Tensor, queue: list[Tensor], visited: set
) -> list[Tensor]:
    if node not in visited:
        visited.add(node)
        if not node.parents:
            return []
        for p in node.parents:
            if p.requires_grad is False:
                continue
            _ = build_backward_queue(p, queue, visited)
        queue.append(node)
    return queue


def apply_func(funcion: type[Function], *args: Any, **kwargs: Any) -> Tensor:
    tensor_args = tuple(a for a in args if isinstance(a, Tensor))
    function_args = tuple(a.data if isinstance(a, Tensor) else a for a in args)
    device = tensor_args[0].device
    ctx = funcion(tensor_args[0].device)

    # add autograd context to resulting node
    if autograd_tracing_active and any(a.requires_grad for a in tensor_args):
        ctx.cache = Cache()
        with device:
            data = ctx.forward(*function_args, **kwargs)
        return Tensor(data, ctx=ctx, parents=tensor_args, requires_grad=True)

    # just compute forward pass without adding context
    with device:
        data = ctx.forward(*function_args, **kwargs)
    return Tensor(data)


def draw_compute_graph(root_node: Tensor, save_to_file: bool = False) -> Any:
    assert root_node.requires_grad, "Node not in autograd graph"

    try:
        from mermaid import Mermaid  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install mermaid-python to draw graphs.") from exc

    colors = {
        "const": ("#CAEDFB", "#4D93D9"),
        "leaf": ("#C6EFCE", "#4EA72E"),
        "func": ("#F2F2F2", "#808080"),
    }

    def _get_mermaid_node_label(n: Tensor) -> str:

        # constant
        if not n.requires_grad:
            node_name = "Const"
            fill_color, stroke_color = colors["const"]

        # leaf tensor
        elif n.ctx is None:
            node_name = n.__class__.__name__
            fill_color, stroke_color = colors["leaf"]

        # function
        else:
            node_name = n.ctx.name
            fill_color, stroke_color = colors["func"]

        if len(n.shape) == 0:
            node_info = f"{n.item():.4g}"
        else:
            node_info = str(n.shape)

        label = f"{node_name}<br>{node_info}<br>{str(n.dtype)}"
        return f'{id(n)}("{label}")\nstyle {str(id(n))} fill:{fill_color},stroke:{stroke_color}'

    mermaid_script = f"graph LR\n{_get_mermaid_node_label(root_node)}\n"

    def _build_mermaid_script(node: Tensor, mermaid_script: str) -> str:
        if not node.parents:
            return ""
        for parent_node in node.parents:
            parent_label = _get_mermaid_node_label(parent_node)
            if parent_label not in mermaid_script:
                mermaid_script += f"{parent_label}\n"
            edge = f"{str(id(parent_node))}-->{str(id(node))}\n"
            if edge not in mermaid_script:
                mermaid_script += edge
            if parent_node.parents:
                mermaid_script = _build_mermaid_script(parent_node, mermaid_script)
        return mermaid_script

    mermaid_script = _build_mermaid_script(root_node, mermaid_script)
    mermaid_html = Mermaid(mermaid_script)
    if save_to_file:
        with open("compute_graph.html", "w", encoding="utf-8") as f:
            f.write(mermaid_html._repr_html_())
    else:
        return mermaid_html


def _parse_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return tuple(k.data if isinstance(k, Tensor) else k for k in key)
    if isinstance(key, Tensor):
        return key.data
    return key


autograd_tracing_active = True


def set_autograd__tracing_mode(active: bool) -> None:
    global autograd_tracing_active
    autograd_tracing_active = active


@contextmanager
def no_autograd_tracing() -> Generator:
    set_autograd__tracing_mode(False)
    try:
        yield
    finally:
        set_autograd__tracing_mode(True)
