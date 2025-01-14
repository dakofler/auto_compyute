"""Tensor class"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Optional

from .backends import (
    Array,
    Backend,
    Dim,
    Scalar,
    Shape,
    get_array_backend,
    select_backend,
)
from .dtypes import DType, select_dtype
from .funcs.function import Context, Function
from .funcs.multiary_funcs import Add, Div, Matmul, Maximum, Mul, Pow, Sub
from .funcs.reduce_funcs import Mean, Std, Sum, Var
from .funcs.shape_funcs import Select
from .funcs.unary_funcs import Tanh, Transpose

__all__ = ["Tensor", "tensor", "ones", "zeros", "randn", "randu"]


class Tensor:
    def __init__(
        self,
        data: Array,
        grad_fn: Optional[Callable] = None,
        grad_fn_name: str = "",
        child_nodes: Optional[tuple[Tensor, ...]] = None,
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
    def T(self) -> Tensor:
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
        nodes: list[Tensor] = []
        visited_ids: set[int] = set()
        nodes = toposort(self, nodes, visited_ids)
        for n in reversed(nodes):
            n.grad_fn(n.grad)

    def __getitem__(self, key) -> Tensor:
        return apply_function(Select, self, key)

    # ----------------------------------------------------------------------------------
    # UNARY OPS
    # ----------------------------------------------------------------------------------

    def tanh(self) -> Tensor:
        return apply_function(Tanh, self)

    def transpose(self, dim1: int = -1, dim2: int = -2) -> Tensor:
        return apply_function(Transpose, self, dim1, dim2)

    # ----------------------------------------------------------------------------------
    # MULTIARY OPS
    # ----------------------------------------------------------------------------------

    def add(self, x: Tensor | Scalar) -> Tensor:
        return apply_function(Add, self, x)

    def sub(self, x: Tensor | Scalar) -> Tensor:
        return apply_function(Sub, self, x)

    def mul(self, x: Tensor | Scalar) -> Tensor:
        return apply_function(Mul, self, x)

    def truediv(self, x: Tensor | Scalar) -> Tensor:
        return apply_function(Div, self, x)

    def pow(self, x: int) -> Tensor:
        return apply_function(Pow, self, x)

    def matmul(self, x: Tensor) -> Tensor:
        return apply_function(Matmul, self, x)

    def maximum(self, x: Tensor | Scalar) -> Tensor:
        return apply_function(Maximum, self, x)

    # ----------------------------------------------------------------------------------
    # REDUCE OPS
    # ----------------------------------------------------------------------------------

    def sum(self, dims: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return apply_function(Sum, self, dims, keepdims)

    def mean(self, dims: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return apply_function(Mean, self, dims, keepdims)

    def var(self, dims: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return apply_function(Var, self, dims, keepdims)

    def std(self, dims: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return apply_function(Std, self, dims, keepdims)

    # ----------------------------------------------------------------------------------
    # SHAPE OPS
    # ----------------------------------------------------------------------------------

    def select(self, slc: Any) -> Tensor:
        return apply_function(Select, self, slc)


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


def apply_function(function: type[Function], *args: Any) -> Tensor:
    ctx = Context()
    function_args = tuple(a.data if isinstance(a, Tensor) else a for a in args)
    y_data = function.forward(ctx, *function_args)

    if autograd_active:
        node_args = tuple(a for a in args if isinstance(a, Tensor))

        def grad_fn(output_grad) -> None:
            grads = function.backward(ctx, output_grad)
            for n, grad in zip(node_args, grads):
                if not n.requires_grad:
                    continue
                n.apply_grad(grad)

        grad_fn_name = function.__name__ + "Backward"
        return Tensor(y_data, grad_fn, grad_fn_name, node_args, True)

    return Tensor(y_data)


def toposort(n: Tensor, nodes: list[Tensor], visited_node_ids: set) -> list[Tensor]:
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


def tensor(data: Any, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.array(data, dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones(shape: Shape, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.ones(shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros(shape: Shape, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.zeros(shape, dtype)
    return Tensor(data, requires_grad=requires_grad)


def randn(shape: Shape, mean: float = 0, var: float = 1, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.random.normal(mean, var, shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def randu(shape: Shape, low: float = -1, high: float = 1, **factory_kwargs) -> Tensor:
    backend, dtype, requires_grad = get_factory_kwargs(factory_kwargs)
    data = backend.m.random.uniform(low, high, shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)
