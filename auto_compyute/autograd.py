"""Tensor class"""

from __future__ import annotations

from collections.abc import Generator
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
from .dtypes import DType, float32, is_float
from .funcs.binary_funcs import Add, Div, Matmul, Maximum, Minimum, Mul, Sub
from .funcs.function import Context, Function
from .funcs.reduce_funcs import Mean, Std, Sum, Var
from .funcs.shape_funcs import Select, Transpose
from .funcs.unary_funcs import Pow, Tanh

__all__ = ["Tensor", "tensor", "no_grad"]


def tensor(data: Any, **factory_kwargs) -> Tensor:
    if isinstance(data, Tensor):
        return data
    backend = select_backend(factory_kwargs.get("backend", None))
    dtype = factory_kwargs.get("dtype", None)
    requires_grad = factory_kwargs.get("requires_grad", False)
    data = backend.m.array(data, dtype)
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

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
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
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    def as_type(self, dtype: DType) -> Tensor:
        if self.dtype == dtype:
            return self
        data = self.data.astype(dtype)
        if is_float(dtype) and self.requires_grad:
            new_tensor = Tensor(data, self.ctx, self.requires_grad)
            if self.grad is not None:
                new_tensor.grad = self.grad.astype(dtype)
            return new_tensor
        return Tensor(self.data.astype(dtype))

    def item(self) -> Scalar:
        return self.data.item()

    # ----------------------------------------------------------------------------------
    # AUTOGRAD
    # ----------------------------------------------------------------------------------

    def apply_func(
        self, func: type[Function], *tensors: Tensor, **kwargs: Any
    ) -> Tensor:
        f = func(self.b)
        requires_grad = any(t.requires_grad for t in tensors)
        if autograd_active and requires_grad:
            f.ctx = Context()
            data = f.forward(*[t.data for t in tensors], **kwargs)
            return Tensor(data, f, tensors, True)
        data = f.forward(*[t.data for t in tensors], **kwargs)
        return Tensor(data)

    def apply_grad(self, grad: Array) -> None:
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, output_grad: Optional[Array] = None):
        if not self.requires_grad:
            raise ValueError("Tensor does not require gradients.")
        if self.grad is None:
            self.grad = self.b.m.ones(self.shape, dtype=float32)
        if output_grad is not None:
            assert isinstance(output_grad, Array)
            self.grad *= output_grad
        tensors: list[Tensor] = []
        visited_ids: set[int] = set()
        tensors = topological_sort(self, tensors, visited_ids)

        for t in reversed(tensors):
            assert t.ctx is not None
            assert t.parents is not None
            grads = t.ctx.backward(t.grad)
            for t, grad in zip(t.parents, grads):
                if not t.requires_grad:
                    continue
                grad = unbroadcast(grad, t.shape)
                t.apply_grad(grad)

    def __getitem__(self, key: Any) -> Tensor:
        return self.select(key)

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

    def __pow__(self, x: Tensor | Scalar) -> Tensor:
        return self.pow(x)

    def __neg__(self) -> Tensor:
        return self.mul(-1)

    # ----------------------------------------------------------------------------------
    # UNARY OPS
    # ----------------------------------------------------------------------------------

    def tanh(self) -> Tensor:
        return self.apply_func(Tanh, self)

    # ----------------------------------------------------------------------------------
    # BINARY OPS
    # ----------------------------------------------------------------------------------

    def add(self, x: Tensor | Scalar) -> Tensor:
        return self.apply_func(Add, self, tensor(x, backend=self.b))

    def sub(self, x: Tensor | Scalar, reverse: bool = False) -> Tensor:
        if reverse:
            return self.apply_func(Sub, tensor(x, backend=self.b), self)
        return self.apply_func(Sub, self, tensor(x, backend=self.b))

    def mul(self, x: Tensor | Scalar) -> Tensor:
        return self.apply_func(Mul, self, tensor(x, backend=self.b))

    def truediv(self, x: Tensor | Scalar, reverse: bool = False) -> Tensor:
        if reverse:
            return self.apply_func(Div, tensor(x, backend=self.b), self)
        return self.apply_func(Div, self, tensor(x, backend=self.b))

    def matmul(self, x: Tensor) -> Tensor:
        return self.apply_func(Matmul, self, tensor(x, backend=self.b))

    def pow(self, x: Tensor | Scalar) -> Tensor:
        return self.apply_func(Pow, self, tensor(x, backend=self.b))

    def maximum(self, x: Tensor | Scalar) -> Tensor:
        return self.apply_func(Maximum, self, tensor(x, backend=self.b))

    def minimum(self, x: Tensor | Scalar) -> Tensor:
        return self.apply_func(Minimum, self, tensor(x, backend=self.b))

    # ----------------------------------------------------------------------------------
    # REDUCE OPS
    # ----------------------------------------------------------------------------------

    def sum(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return self.apply_func(Sum, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return self.apply_func(Mean, self, dim=dim, keepdims=keepdims)

    def var(
        self, dim: Optional[Dim] = None, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
        return self.apply_func(Var, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def std(
        self, dim: Optional[Dim] = None, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
        return self.apply_func(Std, self, dim=dim, ddof=ddof, keepdims=keepdims)

    # ----------------------------------------------------------------------------------
    # SHAPE OPS
    # ----------------------------------------------------------------------------------

    def select(self, key: Any) -> Tensor:
        if isinstance(key, Tensor):
            key = key.data
        return self.apply_func(Select, self, key=key)

    def transpose(self, dim1: int = -1, dim2: int = -2) -> Tensor:
        return self.apply_func(Transpose, self, dim1=dim1, dim2=dim2)


def get_shape_diff(shape1: Shape, shape2: Shape) -> Shape:
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])


def unbroadcast(grad: Array, target_shape: Shape) -> Array:
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


def topological_sort(
    t: Tensor, nodes: list[Tensor], visited_node_ids: set
) -> list[Tensor]:
    if id(t) not in visited_node_ids:
        visited_node_ids.add(id(t))
        if not t.parents:
            return
        for p in t.parents:
            if p.requires_grad is False:
                continue
            topological_sort(p, nodes, visited_node_ids)
        nodes.append(t)
    return nodes


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
