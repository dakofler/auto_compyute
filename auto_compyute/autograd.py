"""Autograd engine"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from itertools import chain
from typing import Any, Literal, Optional

from .backends import (
    ArrayLike,
    Device,
    DeviceLike,
    Dim,
    Scalar,
    Shape,
    ShapeLike,
    array_to_string,
    get_array_device,
    move_to_device,
    numpy,
)
from .dtypes import DType, float32, int32, int64, is_float
from .funcs import binary_funcs as BFuncs
from .funcs import reduce_funcs as RFuncs
from .funcs import shape_funcs as SFuncs
from .funcs import unary_funcs as UFuncs
from .funcs.function import Function

__all__ = ["Array", "no_autograd_tracing"]


class Array:
    """Represents a multi-dimensional array with automatic differentiation support.

    Attributes:
        data (ArrayLike): The underlying data of the array.
        ctx (Function | None, optional): The function context for automatic differentiation.
            Defaults to `None`.
        parents (tuple[Array, ...] | None, optional): The parent arrays in the computation graph.
            Defaults to `None`.
        req_grad (bool, optional): Whether gradients should be computed for this array.
            Defaults to `False`.
    """

    def __init__(
        self,
        data: ArrayLike,
        ctx: Optional[Function] = None,
        parents: Optional[tuple[Array, ...]] = None,
        req_grad: bool = False,
    ) -> None:

        self.data = data
        self.ctx = ctx
        self.parents = parents
        self.req_grad = req_grad
        self.grad: Optional[ArrayLike] = None

    @property
    def device(self) -> Device:
        """Returns the device on which the array is stored."""
        return get_array_device(self.data)

    @property
    def dtype(self) -> DType:
        """Returns the data type of the array."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the array."""
        return self.data.ndim

    @property
    def shape(self) -> Shape:
        """Returns the shape of the array."""
        return Shape(self.data.shape)

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array."""
        return self.data.size

    @property
    def T(self) -> Array:
        """Returns the transposed array."""
        return self.transpose(-2, -1)

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __add__(self, x: Array | Scalar) -> Array:
        return self.add(x)

    def __radd__(self, x: Scalar) -> Array:
        return self.add(x)

    def __sub__(self, x: Array | Scalar) -> Array:
        return self.sub(x)

    def __rsub__(self, x: Scalar) -> Array:
        return (-self).add(x)

    def __mul__(self, x: Array | Scalar) -> Array:
        return self.mul(x)

    def __rmul__(self, x: Scalar) -> Array:
        return self.mul(x)

    def __truediv__(self, x: Array | Scalar) -> Array:
        return self.truediv(x)

    def __rtruediv__(self, x: Scalar) -> Array:
        return (self**-1).mul(x)

    def __matmul__(self, x: Array) -> Array:
        return self.matmul(x)

    def __pow__(self, x: Scalar) -> Array:
        return self.pow(x)

    def __neg__(self) -> Array:
        return self.mul(-1)

    def __eq__(self, x: Array | Scalar) -> Array:  # type: ignore
        return Array(self.data == self.align(x).data)

    def __neq__(self, x: Array | Scalar) -> Array:
        return Array(self.data != self.align(x).data)

    def __lt__(self, x: Array | Scalar) -> Array:
        return Array(self.data < self.align(x).data)

    def __gt__(self, x: Array | Scalar) -> Array:
        return Array(self.data > self.align(x).data)

    def __le__(self, x: Array | Scalar) -> Array:
        return Array(self.data <= self.align(x).data)

    def __ge__(self, x: Array | Scalar) -> Array:
        return Array(self.data >= self.align(x).data)

    def __getitem__(self, key: Any) -> Array:
        return self.select(key)

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
        return prefix + array_to_string(self.data, prefix) + suffix

    def __len__(self) -> int:
        return self.data.shape[0]

    def __hash__(self) -> int:
        return id(self)

    # ----------------------------------------------------------------------------------
    # AUTOGRAD METHODS
    # ----------------------------------------------------------------------------------

    def apply_grad(self, dy: ArrayLike) -> None:
        """Accumulates the gradient for the current array.

        Args:
            dy (ArrayLike): The gradient to be applied.

        Raises:
            AssertionError: If `dy` does not have dtype `float32`.
        """
        assert dy.dtype == float32, f"Grad has invalid dtype {dy.dtype}"
        self.grad = dy if self.grad is None else self.grad + dy

    def backward(self, dy: Optional[ArrayLike] = None):
        """Performs backpropagation to compute gradients.

        - Computes and stores gradients for all nodes in the computation graph.
        - Clears intermediate node contexts after backpropagation.

        Args:
            dy (ArrayLike | None, optional): The gradient of the output with respect to
                some loss. If `None`, assumes a gradient of ones with the same shape as `self`.

        Raises:
            AssertionError: If the array is not part of the autograd graph (`req_grad=False`).
            AssertionError: If backward is called multiple times on the same node.
        """
        assert self.req_grad, "Node not in autograd graph."
        assert self.grad is None, "Cannot run backward multiple times."

        # set node grad
        if dy is None:
            self.grad = self.device.xp.ones(self.shape, dtype=self.dtype)
        else:
            assert isinstance(
                dy, ArrayLike
            ), f"Gradient must be an array, got {type(dy)}"
            self.grad = dy

        # run backward through traced graph
        node_queue = _build_backward_queue(self, [], set())
        for node in reversed(node_queue):
            assert node.ctx is not None, "Node has no function context."
            assert node.parents is not None, "Node has no parent nodes."
            grads = node.ctx.backward(node.grad)
            for parent, grad in zip(node.parents, grads):
                if not parent.req_grad:
                    continue
                grad = _undo_broadcast(grad, parent.shape)
                parent.apply_grad(grad)

            # clear context of intermediate nodes
            node.grad, node.ctx, node.parents = None, None, None

    # ----------------------------------------------------------------------------------
    # UNARY FUNCS
    # ----------------------------------------------------------------------------------
    def abs(self) -> Array:
        """Computes the element-wise absolute value of the array.

        Returns:
            Array: An array with absolute values of the elements.
        """
        return apply_func(UFuncs.Abs, self)

    def exp(self) -> Array:
        """Computes the element-wise exponential function.

        Returns:
            Array: An array where each element is `e` raised to the power of the
                corresponding element.
        """
        return apply_func(UFuncs.Exp, self)

    def pow(self, exponent: Scalar) -> Array:
        """Raises each element of the array to the given power.

        Args:
            exponent (Scalar): The exponent.

        Returns:
            Array: An array with each element raised to the power `x`.
        """
        return apply_func(UFuncs.Pow, self, exp=exponent)

    def sqrt(self) -> Array:
        """Computes the element-wise square root.

        Returns:
            Array: An array with the square root of each element.
        """
        return apply_func(UFuncs.Sqrt, self)

    def tanh(self) -> Array:
        """Computes the element-wise hyperbolic tangent.

        Returns:
            Array: An array with the tanh of each element.
        """
        return apply_func(UFuncs.Tanh, self)

    def tril(self, diag: int = 0) -> Array:
        """Returns the lower triangular part of the array.

        Args:
            diag (int, optional): The diagonal above which to zero elements. Defaults to `0`.

        Returns:
            Array: The lower triangular matrix.
        """
        return apply_func(UFuncs.Tril, self, diag=diag)

    def triu(self, diag: int = 0) -> Array:
        """Returns the upper triangular part of the array.

        Args:
            diag (int, optional): The diagonal below which to zero elements. Defaults to `0`.

        Returns:
            Array: The upper triangular matrix.
        """
        return apply_func(UFuncs.Triu, self, diag=diag)

    # ----------------------------------------------------------------------------------
    # BINARY FUNCS
    # ----------------------------------------------------------------------------------

    def add(self, x: Array | Scalar) -> Array:
        """Performs element-wise addition.

        Args:
            x (Array | Scalar): The array or scalar to add.

        Returns:
            Array: The element-wise sum.
        """
        return apply_func(BFuncs.Add, self, self.align(x))

    def sub(self, x: Array | Scalar) -> Array:
        """Performs element-wise subtraction.

        Args:
            x (Array | Scalar): The array or scalar to subtract.

        Returns:
            Array: The element-wise difference.
        """
        return apply_func(BFuncs.Sub, self, self.align(x))

    def mul(self, x: Array | Scalar) -> Array:
        """Performs element-wise multiplication.

        Args:
            x (Array | Scalar): The array or scalar to multiply.

        Returns:
            Array: The element-wise product.
        """
        return apply_func(BFuncs.Mul, self, self.align(x))

    def truediv(self, x: Array | Scalar) -> Array:
        """Performs element-wise division.

        Args:
            x (Array | Scalar): The array or scalar to divide by.

        Returns:
            Array: The element-wise quotient.
        """
        return apply_func(BFuncs.Div, self, self.align(x))

    def matmul(self, x: Array) -> Array:
        """Performs matrix multiplication.

        Args:
            x (Array): The array to multiply with.

        Returns:
            Array: The result of the matrix multiplication.
        """
        return apply_func(BFuncs.Matmul, self, x)

    def maximum(self, x: Array | Scalar) -> Array:
        """Computes the element-wise maximum.

        Args:
            x (Array | Scalar): The array or scalar to compare with.

        Returns:
            Array: The element-wise maximum values.
        """
        return apply_func(BFuncs.Maximum, self, self.align(x))

    def minimum(self, x: Array | Scalar) -> Array:
        """Computes the element-wise minimum.

        Args:
            x (Array | Scalar): The array or scalar to compare with.

        Returns:
            Array: The element-wise minimum values.
        """
        return apply_func(BFuncs.Minimum, self, self.align(x))

    # ----------------------------------------------------------------------------------
    # REDUCE FUNCS
    # ----------------------------------------------------------------------------------

    def sum(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Array:
        """Computes the sum of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, sums all elements.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Array: The sum of elements.
        """
        return apply_func(RFuncs.Sum, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Array:
        """Computes the mean of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, computes mean of all
                elements.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Array: The mean of elements.
        """
        return apply_func(RFuncs.Mean, self, dim=dim, keepdims=keepdims)

    def var(
        self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> Array:
        """Computes the variance of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, computes variance of
                all elements.
            ddof (int, optional): Delta degrees of freedom. Defaults to 1.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Array: The variance of elements.
        """
        return apply_func(RFuncs.Var, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def std(
        self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> Array:
        """Computes the standard deviation of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, computes standard
                deviation of all elements.
            ddof (int, optional): Delta degrees of freedom. Defaults to 1.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Array: The standard deviation of elements.
        """
        return apply_func(RFuncs.Std, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def max(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Array:
        """Computes the maximum value along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, finds the global
                maximum.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Array: The maximum values.
        """
        return apply_func(RFuncs.Max, self, dim=dim, keepdims=keepdims)

    def min(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Array:
        """Computes the minimum value along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, finds the global
                minimum.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Array: The minimum values.
        """
        return apply_func(RFuncs.Min, self, dim=dim, keepdims=keepdims)

    # ----------------------------------------------------------------------------------
    # SHAPE FUNCS
    # ----------------------------------------------------------------------------------

    def expand(self, *dims: int) -> Array:
        return apply_func(SFuncs.Expand, self, shape=dims)

    def select(self, key: Any) -> Array:
        key = _parse_key(key)
        return apply_func(SFuncs.Select, self, key=key)

    def _split(self, key: Any) -> Array:
        key = _parse_key(key)
        return apply_func(SFuncs.Split, self, key=key)

    def split(self, split_size: int, *, dim: int = -1) -> list[Array]:
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        return [
            self._split(pre_dim_slice + (slice(i, i + split_size),) + post_dim_slice)
            for i in range(0, self.shape[dim], split_size)
        ]

    def squeeze(self) -> Array:
        non_singular_dims = tuple(d for d in self.shape if d > 1)
        if len(non_singular_dims) == self.ndim:
            return self
        return apply_func(SFuncs.Squeeze, self, shape=non_singular_dims)

    def transpose(self, dim1: int = -1, dim2: int = -2) -> Array:
        return apply_func(SFuncs.Transpose, self, dim1=dim1, dim2=dim2)

    def view(self, *dims: int) -> Array:
        if dims == self.shape:
            return self
        return apply_func(SFuncs.View, self, shape=dims)

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    def as_type(self, dtype: DType) -> Array:
        if self.dtype == dtype:
            return self
        data: ArrayLike = self.data.astype(dtype)
        if self.req_grad:
            assert is_float(dtype), "Cannot change autograd node dtype to non float."
            arr = Array(data, self.ctx, self.parents, self.req_grad)
            if self.grad is not None:
                arr.grad = self.grad.astype(dtype)
            return arr
        return Array(data)

    def int(self) -> Array:
        return self.as_type(int32)

    def long(self) -> Array:
        return self.as_type(int64)

    def float(self) -> Array:
        return self.as_type(float32)

    def to(self, device: DeviceLike) -> Array:
        device = device if isinstance(device, Device) else Device(device)
        data = self.data if self.device == device else move_to_device(self.data, device)
        if self.req_grad:
            arr = Array(data, self.ctx, self.parents, self.req_grad)
            if self.grad is not None:
                arr.grad = move_to_device(self.grad, device)
            return arr
        return Array(data)

    def cpu(self) -> Array:
        return self.to("cpu")

    def cuda(self) -> Array:
        return self.to("cuda")

    def ito(self, device: DeviceLike) -> None:
        device = device if isinstance(device, Device) else Device(device)
        if self.device == device:
            return
        self.data = move_to_device(self.data, device)
        if self.grad is not None:
            self.grad = move_to_device(self.grad, device)

    def item(self) -> Any:
        return self.data.item()

    def contiguous(self) -> Array:
        data = self.device.xp.ascontiguousarray(self.data)
        return Array(data, self.ctx, self.parents, self.req_grad)

    def align(self, x: Array | Scalar) -> Array:
        if isinstance(x, Array):
            return x.as_type(self.dtype)
        return Array(self.device.xp.asarray(x, dtype=self.dtype))

    def to_numpy(self) -> numpy.ndarray:
        return self.cpu().data


# -------------------------------------------------------------------------------------
# AUTOGRAD FUNCTIONS
# -------------------------------------------------------------------------------------


def _undo_broadcast(grad: ArrayLike, target_shape: ShapeLike) -> ArrayLike:
    if grad.shape == target_shape:
        return grad
    target_ndim = len(target_shape)

    if grad.ndim == target_ndim:
        axis = _get_shape_diff(grad.shape, target_shape)
        grad = grad.sum(axis, keepdims=True)
    else:
        data_shape = Shape((1,) * (grad.ndim - target_ndim) + target_shape)
        axis = _get_shape_diff(grad.shape, data_shape)
        grad = grad.sum(axis=axis)

    return grad.reshape(target_shape)


def _build_backward_queue(node: Array, queue: list[Array], visited: set) -> list[Array]:
    if node not in visited:
        visited.add(node)
        if not node.parents:
            return []
        for p in node.parents:
            if p.req_grad is False:
                continue
            _ = _build_backward_queue(p, queue, visited)
        queue.append(node)
    return queue


def _all_same_device(arrays: tuple[Optional[Array], ...]) -> bool:
    return len(set(map(type, [a.data for a in arrays if a is not None]))) == 1


def apply_func(
    function: type[Function], *arrays: Optional[Array], **kwargs: Any
) -> Array:
    assert _all_same_device(arrays), "Device mismatch!"

    # create function args by extracting req_grad from arrays and handle optional arrays
    f_args = [(a.data, a.req_grad) if a is not None else (None, False) for a in arrays]
    f_args = tuple(chain(*f_args))  # type: ignore

    # get array args
    t_args = tuple(t for t in arrays if t is not None)
    device = t_args[0].device
    ctx = function(t_args[0].device)

    # return result node with autograd context
    with device:
        if autograd_tracing_active and any(a.req_grad for a in t_args):
            data = ctx.forward(*f_args, **kwargs)
            return Array(data, ctx=ctx, parents=t_args, req_grad=True)

        # return result node without autograd context
        data = ctx.forward(*f_args, **kwargs)
        return Array(data)


def draw_graph(
    root_node: Array,
    orientation: Literal["LR", "TD"] = "LR",
    save_to_file: bool = False,
) -> Any:
    assert root_node.req_grad, "Node not in autograd graph"

    try:
        from mermaid import Mermaid  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install mermaid-python to draw graphs.") from exc

    colors = {
        "const": ("#CAEDFB", "#4D93D9"),
        "leaf": ("#C6EFCE", "#4EA72E"),
        "func": ("#F2F2F2", "#808080"),
    }

    def _get_mermaid_node_label(n: Array) -> str:

        # constant
        if not n.req_grad:
            node_name = ""
            fill_color, stroke_color = colors["const"]

        # leaf node
        elif n.ctx is None:
            node_name = ""
            fill_color, stroke_color = colors["leaf"]

        # function
        else:
            node_name = n.ctx.name + "<br>"
            fill_color, stroke_color = colors["func"]

        if len(n.shape) == 0:
            node_info = f"{n.item():.4g}"
        else:
            node_info = str(n.shape).replace("shape", "")

        label = f"{node_name}{node_info}<br>{str(n.dtype)}"
        return f'{id(n)}("{label}")\nstyle {str(id(n))} fill:{fill_color},stroke:{stroke_color}'

    mermaid_script = f"graph {orientation}\n{_get_mermaid_node_label(root_node)}\n"

    def _build_mermaid_script(node: Array, mermaid_script: str) -> str:
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


# -------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------------------


def _parse_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return tuple(k.data if isinstance(k, Array) else k for k in key)
    if isinstance(key, Array):
        return key.data
    return key


def _get_shape_diff(shape1: ShapeLike, shape2: ShapeLike) -> Shape:
    return Shape(i for i in range(len(shape1)) if shape1[i] != shape2[i])
