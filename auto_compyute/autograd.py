"""Autograd node"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Optional

from .backends import Array, Backend, Shape, get_array_backend, select_backend
from .dtypes import DType, select_dtype
from .functions import (
    Add,
    Context,
    Divide,
    Function,
    Matmul,
    Mean,
    Multiply,
    Pow,
    Subtract,
    Sum,
    Tanh,
    Transpose,
)

__all__ = ["no_grad", "Node", "node", "ones", "zeros", "randn", "randu"]


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


class Node:
    def __init__(
        self,
        data: Array,
        grad_fn: Optional[Callable] = None,
        grad_fn_name: str = "",
        child_nodes: Optional[tuple[Node, ...]] = None,
        requires_grad: bool = False,
    ) -> None:
        self.data = data
        self.grad_fn = grad_fn if grad_fn is not None else lambda _: None
        self.grad_fn_name = grad_fn_name
        self.child_nodes = child_nodes if child_nodes is not None else tuple()
        self.requires_grad = requires_grad

        self.grad: Optional[Array] = None

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = f", grad_fn={self.grad_fn_name})" if self.grad_fn_name else ")"
        return (
            prefix
            + self.b.m.array2string(
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
    # PROPERTIES
    # ----------------------------------------------------------------------------------

    @property
    def b(self) -> Backend:
        return get_array_backend(self.data)

    @property
    def dtype(self) -> DType:
        return DType(self.data.dtype)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def T(self) -> Node:
        return self.transpose(-2, -1)

    # ----------------------------------------------------------------------------------
    # AUTOGRAD
    # ----------------------------------------------------------------------------------

    def apply_grad(self, grad: Array) -> None:
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, output_grad: Optional[Array] = None):
        if not self.requires_grad:
            raise ValueError("Node does not require gradients.")
        if self.grad is None:
            self.grad = self.b.m.ones(self.shape, dtype=self.dtype)
        if output_grad is not None:
            self.grad *= output_grad
        nodes: list[Node] = []
        visited_ids: set[int] = set()
        nodes = toposort(self, nodes, visited_ids)
        for n in reversed(nodes):
            n.grad_fn(n.grad)

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __add__(self, x: Node | int | float) -> Node:
        return apply_function(Add, self, x)

    def __sub__(self, x: Node | int | float) -> Node:
        return apply_function(Subtract, self, x)

    def __mul__(self, x: Node | int | float) -> Node:
        return apply_function(Multiply, self, x)

    def __truediv__(self, x: Node | int | float) -> Node:
        return apply_function(Divide, self, x)

    def __pow__(self, x: int) -> Node:
        return apply_function(Pow, self, x)

    def __matmul__(self, x: Node) -> Node:
        return apply_function(Matmul, self, x)

    # ----------------------------------------------------------------------------------
    # OPS
    # ----------------------------------------------------------------------------------

    def transpose(self, *dims: int) -> Node:
        return apply_function(Transpose, self, dims)

    def sum(
        self, dims: Optional[int | tuple[int, ...]] = None, keepdims: bool = False
    ) -> Node:
        return apply_function(Sum, self, dims, keepdims)

    def mean(
        self, dims: Optional[int | tuple[int, ...]] = None, keepdims: bool = False
    ) -> Node:
        return apply_function(Mean, self, dims, keepdims)

    def tanh(self) -> Node:
        return apply_function(Tanh, self)


def apply_function(function: type[Function], *args: Any) -> Node:
    ctx = Context()
    function_args = tuple(a.data if isinstance(a, Node) else a for a in args)
    y_data = function.forward(ctx, *function_args)

    if autograd_active:
        node_args = tuple(a for a in args if isinstance(a, Node))

        def grad_fn(output_grad) -> None:
            grads = function.backward(ctx, output_grad)
            for n, grad in zip(node_args, grads):
                if not n.requires_grad:
                    continue
                n.apply_grad(grad)

        grad_fn_name = function.__name__ + "Backward"
        return Node(y_data, grad_fn, grad_fn_name, node_args, True)

    return Node(y_data)


def toposort(n: Node, nodes: list[Node], visited_node_ids: set) -> list[Node]:
    if id(n) not in visited_node_ids:
        visited_node_ids.add(id(n))
        if not n.child_nodes:
            return
        for c in n.child_nodes:
            if c.requires_grad is False:
                continue
            toposort(c, nodes, visited_node_ids)
        nodes.append(n)
    return nodes


def get_factory_kwargs(kwargs) -> tuple[Backend, DType, bool]:
    backend = select_backend(kwargs.get("backend", None))
    dtype = select_dtype(kwargs.get("dtype", None))
    requires_grad = kwargs.get("requires_grad", False)
    return backend, dtype, requires_grad


def node(data: Any, **factory_kwargs) -> Node:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.array(data, dtype)
    return Node(data, requires_grad=requires_grad)


def ones(shape: Shape, **factory_kwargs) -> Node:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.ones(shape, dtype)
    return Node(data, requires_grad=requires_grad)


def zeros(shape: Shape, **factory_kwargs) -> Node:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.zeros(shape, dtype)
    return Node(data, requires_grad=requires_grad)


def randn(shape: Shape, mean: float = 0, var: float = 1, **factory_kwargs) -> Node:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.random.normal(mean, var, shape).astype(dtype)
    return Node(data, requires_grad=requires_grad)


def randu(shape: Shape, low: float = -1, high: float = 1, **factory_kwargs) -> Node:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.random.uniform(low, high, shape).astype(dtype)
    return Node(data, requires_grad=requires_grad)
