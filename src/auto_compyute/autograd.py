"""Autograd engine."""

from __future__ import annotations

import typing
from contextlib import contextmanager
from typing import Any, Optional

from auto_compyute.backends import (
    Array,
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
    parse_device,
)
from auto_compyute.dtypes import DType, float32, int32, int64, is_float
from auto_compyute.ops import binary_ops as BOps
from auto_compyute.ops import movement_ops as MOps
from auto_compyute.ops import reduce_ops as ROps
from auto_compyute.ops import unary_ops as UOps

if typing.TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from .ops.op import Op


__all__ = ["Tensor", "no_autograd_tracking"]


class Tensor:
    """Represents a multi-dimensional tensor with automatic differentiation support.

    Args:
        data (Tensor): The underlying data of the tensor.
        ctx (Op | None, optional): The operation context for automatic differentiation.
            Defaults to `None`.
        src (tuple[Tensor | None, ...] | None, optional): Tensors used to create this
            tensor. Defaults to `None`.
        req_grad (bool, optional): Whether the gradient should be computed for this tensor.
            Defaults to `False`.
        label (str): An optional label for the tensor. Defaults to `None`.
    """

    def __init__(
        self,
        data: Array,
        ctx: Optional[Op] = None,
        src: Optional[tuple[Optional[Tensor], ...]] = None,
        req_grad: bool = False,
        label: Optional[str] = None,
    ) -> None:
        assert not req_grad or is_float(data.dtype), "Tensors that req. grad must be float."

        self.data = data
        self.ctx = ctx
        self.src = src or ()
        self.req_grad = req_grad
        self._label = label
        self.grad: Optional[Array] = None

    @property
    def label(self) -> str:
        """Returns the tensor label."""
        if self._label:
            return self._label
        if self.ctx is not None:
            return self.ctx.name
        return self.__class__.__name__

    @property
    def device(self) -> Device:
        """Returns the device on which the tensor data is stored."""
        return get_array_device(self.data)

    @property
    def dtype(self) -> DType:
        """Returns the data type of the tensor."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self.data.ndim

    @property
    def shape(self) -> Shape:
        """Returns the shape of the tensor."""
        return Shape(self.data.shape)

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self.data.size

    @property
    def T(self) -> Tensor:
        """Returns the transposed tensor."""
        return self.transpose(-2, -1)

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __add__(self, x: Tensor | Scalar) -> Tensor:
        return self.add(x)

    __radd__ = __add__

    def __sub__(self, x: Tensor | Scalar) -> Tensor:
        return self.sub(x)

    def __rsub__(self, x: Scalar) -> Tensor:
        return self.cast_to_self_dtype(x).sub(self)

    def __mul__(self, x: Tensor | Scalar) -> Tensor:
        return self.mul(x)

    __rmul__ = __mul__

    def __truediv__(self, x: Tensor | Scalar) -> Tensor:
        return self.truediv(x)

    def __rtruediv__(self, x: Scalar) -> Tensor:
        return self.cast_to_self_dtype(x).truediv(self)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.dot(x)

    def __pow__(self, x: Scalar) -> Tensor:
        return self.pow(x)

    def __neg__(self) -> Tensor:
        return self.mul(-1)

    def __eq__(self, x: Tensor | Scalar) -> Tensor:  # type: ignore
        return Tensor(self.data == self.cast_to_self_dtype(x).data)

    def __neq__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data != self.cast_to_self_dtype(x).data)

    def __lt__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data < self.cast_to_self_dtype(x).data)

    def __gt__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data > self.cast_to_self_dtype(x).data)

    def __le__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data <= self.cast_to_self_dtype(x).data)

    def __ge__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data >= self.cast_to_self_dtype(x).data)

    def __getitem__(self, key: Any) -> Tensor:
        return self.select(key)

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
        return prefix + array_to_string(self.data, prefix) + suffix

    def __len__(self) -> int:
        return 0 if len(self.shape) == 0 else self.shape[0]

    def __hash__(self) -> int:
        return id(self)

    # ----------------------------------------------------------------------------------
    # AUTOGRAD METHODS
    # ----------------------------------------------------------------------------------

    def accumulate_grad(self, dy: Array) -> None:
        """Accumulates the gradient for the current tensor.

        Args:
            dy (ArrayLike): The gradient to be applied.

        Raises:
            AssertionError: If `dy` does not have dtype `float32`.
        """
        assert dy.dtype == float32, f"Gradient has invalid dtype {dy.dtype}"
        self.grad = dy if self.grad is None else self.grad + dy

    def backward(self, dy: Optional[Array] = None):
        """Performs backpropagation to compute the gradient.

        - Computes and stores the gradient for all nodes in the computation graph.
        - Clears intermediate node contexts after backpropagation.

        Args:
            dy (ArrayLike | None, optional): The gradient of the output with respect to
                some value. If `None`, assumes a gradient of ones.

        Raises:
            AssertionError: If the tensor is not part of the autograd graph (`req_grad=False`).
            AssertionError: If backward is called multiple times on the same node.
        """
        assert self.req_grad, "Node is not part of a autograd graph."
        assert self.grad is None, "Cannot run backward multiple times."

        # set node grad
        if dy is None:
            self.grad = self.device.xp.ones(self.shape, dtype=float32)
        else:
            assert isinstance(dy, Array), "Gradient must be an array."
            self.grad = dy

        # construct a list of nodes via depth-first-search
        nodes: list[Tensor] = []
        depth_first_search(self, nodes, set())

        # run backward through the list
        for node in reversed(nodes):
            grads = node.ctx.backward(node.grad)
            for src_tensor, grad in zip(node.src, grads):
                if src_tensor is None or not src_tensor.req_grad:
                    continue
                grad = _undo_broadcast(grad, src_tensor.shape)
                src_tensor.accumulate_grad(grad)

            # clear context of intermediate nodes
            node.grad, node.ctx, node.src = None, None, None

    # ----------------------------------------------------------------------------------
    # UNARY OPS
    # ----------------------------------------------------------------------------------

    def abs(self) -> Tensor:
        """Computes the element-wise absolute value of the tensor.

        Returns:
            Tensor: A tensor with absolute values of the elements.
        """
        return apply_op(UOps.Abs, self)

    def exp(self) -> Tensor:
        """Computes the element-wise exponential function.

        Returns:
            Tensor: A tensor where each element is `e` raised to the power of the
                corresponding element.
        """
        return apply_op(UOps.Exp, self)

    def log(self) -> Tensor:
        """Computes the element-wise natural logarithm.

        Returns:
            Tensor: A tensor with the natural logarithm of the elements.
        """
        return apply_op(UOps.Log, self)

    def pow(self, exponent: Scalar) -> Tensor:
        """Raises each element of the tensor to the given power.

        Args:
            exponent (Scalar): The exponent.

        Returns:
            Tensor: A tensor with each element raised to the power `x`.
        """
        return apply_op(UOps.Pow, self, exp=exponent)

    def sqrt(self) -> Tensor:
        """Computes the element-wise square root.

        Returns:
            Tensor: A tensor with the square root of each element.
        """
        return apply_op(UOps.Sqrt, self)

    def tanh(self) -> Tensor:
        """Computes the element-wise hyperbolic tangent.

        Returns:
            Tensor: A tensor with the tanh of each element.
        """
        return apply_op(UOps.Tanh, self)

    def tril(self, diag: int = 0) -> Tensor:
        """Returns the lower triangular part of the tensor.

        Args:
            diag (int, optional): The diagonal above which to zero elements. Defaults to `0`.

        Returns:
            Tensor: The lower triangular matrix.
        """
        return apply_op(UOps.Tril, self, diag=diag)

    def triu(self, diag: int = 0) -> Tensor:
        """Returns the upper triangular part of the tensor.

        Args:
            diag (int, optional): The diagonal below which to zero elements. Defaults to `0`.

        Returns:
            Tensor: The upper triangular matrix.
        """
        return apply_op(UOps.Triu, self, diag=diag)

    # ----------------------------------------------------------------------------------
    # BINARY OPS
    # ----------------------------------------------------------------------------------

    def add(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise addition.

        Args:
            x (Tensor | Scalar): The tensor or scalar to add.

        Returns:
            Tensor: The element-wise sum.
        """
        return apply_op(BOps.Add, self, self.cast_to_self_dtype(x))

    def sub(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise subtraction.

        Args:
            x (Tensor | Scalar): The tensor or scalar to subtract.

        Returns:
            Tensor: The element-wise difference.
        """
        return apply_op(BOps.Sub, self, self.cast_to_self_dtype(x))

    def mul(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise multiplication.

        Args:
            x (Tensor | Scalar): The tensor or scalar to multiply.

        Returns:
            Tensor: The element-wise product.
        """
        return apply_op(BOps.Mul, self, self.cast_to_self_dtype(x))

    def truediv(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise division.

        Args:
            x (Tensor | Scalar): The tensor or scalar to divide by.

        Returns:
            Tensor: The element-wise quotient.
        """
        return apply_op(BOps.Div, self, self.cast_to_self_dtype(x))

    def dot(self, x: Tensor) -> Tensor:
        """Performs the dot product of the tensors. For higher dimensional tensors it performs
        parallelized dot products (eg. matrix multiplication for 2-dimesional tensors).

        Args:
            x (Tensor): The tensor to multiply with.

        Returns:
            Tensor: The result of the dot product.
        """
        return apply_op(BOps.Dot, self, x)

    def maximum(self, x: Tensor | Scalar) -> Tensor:
        """Computes the element-wise maximum.

        Args:
            x (Tensor | Scalar): The tensor or scalar to compare with.

        Returns:
            Tensor: The element-wise maximum values.
        """
        return apply_op(BOps.Maximum, self, self.cast_to_self_dtype(x))

    def minimum(self, x: Tensor | Scalar) -> Tensor:
        """Computes the element-wise minimum.

        Args:
            x (Tensor | Scalar): The tensor or scalar to compare with.

        Returns:
            Tensor: The element-wise minimum values.
        """
        return apply_op(BOps.Minimum, self, self.cast_to_self_dtype(x))

    # ----------------------------------------------------------------------------------
    # REDUCE OPS
    # ----------------------------------------------------------------------------------

    def sum(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the sum of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, sums all elements.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The sum of elements.
        """
        return apply_op(ROps.Sum, self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the mean of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, computes mean of all
                elements.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The mean of elements.
        """
        return apply_op(ROps.Mean, self, dim=dim, keepdims=keepdims)

    def var(self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False) -> Tensor:
        """Computes the variance of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, computes variance of
                all elements.
            ddof (int, optional): Delta degrees of freedom. Defaults to 1.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The variance of elements.
        """
        return apply_op(ROps.Var, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def std(self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False) -> Tensor:
        """Computes the standard deviation of elements along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, computes standard
                deviation of all elements.
            ddof (int, optional): Delta degrees of freedom. Defaults to 1.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The standard deviation of elements.
        """
        return apply_op(ROps.Std, self, dim=dim, ddof=ddof, keepdims=keepdims)

    def max(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the maximum value along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, finds the global
                maximum.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The maximum values.
        """
        return apply_op(ROps.Max, self, dim=dim, keepdims=keepdims)

    def min(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        """Computes the minimum value along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, finds the global
                minimum.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The minimum values.
        """
        return apply_op(ROps.Min, self, dim=dim, keepdims=keepdims)

    # ----------------------------------------------------------------------------------
    # MOVEMENT OPS
    # ----------------------------------------------------------------------------------

    def expand(self, *dims: int) -> Tensor:
        """Expands the tensor to the specified shape.

        Args:
            *dims (int): The target shape dimensions.

        Returns:
            Tensor: A new tensor with the expanded shape.
        """
        return apply_op(MOps.Expand, self, shape=dims)

    def select(self, key: Any) -> Tensor:
        """Selects elements from the tensor based on the given key.

        Args:
            key (Any): The selection key (e.g., indices or masks).

        Returns:
            Tensor: A new tensor containing the selected elements.
        """
        key = parse_key(key)
        return apply_op(MOps.Select, self, key=key)

    def _split(self, key: Any) -> Tensor:
        key = parse_key(key)
        return apply_op(MOps.Split, self, key=key)

    def split(self, split_size: int, *, dim: int = -1) -> list[Tensor]:
        """Splits the tensor into smaller chunks along a specified dimension.

        Args:
            split_size (int): The size of each split.
            dim (int, optional): The dimension to split along. Defaults to `-1`.

        Returns:
            list[Array]: A list of split tensors.
        """
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        return [
            self._split((*pre_dim_slice, slice(i, i + split_size), *post_dim_slice))
            for i in range(0, self.shape[dim], split_size)
        ]

    def squeeze(self) -> Tensor:
        """Removes singleton dimensions from the tensor.

        Returns:
            Tensor: A new tensor with singleton dimensions removed.
        """
        non_singular_dims = tuple(d for d in self.shape if d > 1)
        if len(non_singular_dims) == self.ndim:
            return self
        return apply_op(MOps.Squeeze, self, shape=non_singular_dims)

    def transpose(self, dim1: int = -1, dim2: int = -2) -> Tensor:
        """Swaps two dimensions of the tensor.

        Args:
            dim1 (int, optional): The first dimension to swap. Defaults to -1.
            dim2 (int, optional): The second dimension to swap. Defaults to -2.

        Returns:
            Tensor: A new tensor with the dimensions transposed.
        """
        return apply_op(MOps.Transpose, self, dim1=dim1, dim2=dim2)

    def view(self, *dims: int) -> Tensor:
        """Reshapes the tensor without changing its data.

        Args:
            *dims (int): The target shape dimensions.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        if dims == self.shape:
            return self
        return apply_op(MOps.View, self, shape=dims)

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    def as_type(self, dtype: DType) -> Tensor:
        """Casts the tensor to a specified data type.

        Args:
            dtype (DType): The target data type.

        Returns:
            Tensor: A new tensor with the specified data type.
        """
        if self.dtype == dtype:
            return self
        data: Array = self.data.astype(dtype)
        if self.req_grad:
            assert is_float(dtype), "Cannot change autograd node dtype to non float."
            arr = Tensor(data, self.ctx, self.src, self.req_grad)
            if self.grad is not None:
                arr.grad = self.grad.astype(dtype)
            return arr
        return Tensor(data)

    def int(self) -> Tensor:
        """Casts the tensor to a 32-bit integer.

        Returns:
            Tensor: A new tensor with `int32` data type.
        """
        return self.as_type(int32)

    def long(self) -> Tensor:
        """Casts the tensor to a 64-bit integer.

        Returns:
            Tensor: A new tensor with `int64` data type.
        """
        return self.as_type(int64)

    def float(self) -> Tensor:
        """Casts the tensor to a 32-bit floating point.

        Returns:
            Tensor: A new tensor with `float32` data type.
        """
        return self.as_type(float32)

    def to(self, device: DeviceLike) -> Tensor:
        """Moves the tensor to the specified device.

        Args:
            device (DeviceLike): The target device.

        Returns:
            Tensor: A new tensor on the specified device.
        """
        device = parse_device(device)
        data = self.data if device == self.device else move_to_device(self.data, device)
        if self.req_grad:
            tensor = Tensor(data, self.ctx, self.src, self.req_grad)
            if self.grad is not None:
                tensor.grad = move_to_device(self.grad, device)
            return tensor
        return Tensor(data)

    def cpu(self) -> Tensor:
        """Moves the tensor to CPU memory.

        Returns:
            Tensor: A new tensor on the CPU.
        """
        return self.to("cpu")

    def cuda(self) -> Tensor:
        """Moves the tensor to the default GPU memory.

        Returns:
            Tensor: A new tensor on the default CUDA device.
        """
        return self.to("cuda")

    def ito(self, device: DeviceLike) -> None:
        """In-place move of the tensor to a specified device.

        Args:
            device (DeviceLike): The target device.
        """
        device = parse_device(device)
        if self.device == device:
            return
        self.data = move_to_device(self.data, device)
        if self.grad is not None:
            self.grad = move_to_device(self.grad, device)

    def item(self) -> Any:
        """Returns the scalar value if the tensor contains a single element.

        Returns:
            Any: The single element contained in the tensor.
        """
        return self.data.item()

    def contiguous(self) -> Tensor:
        """Returns a contiguous (row-mayor) copy of the tensor.

        Returns:
            Tensor: A new contiguous tensor.
        """
        data = self.device.xp.ascontiguousarray(self.data)
        return Tensor(data, self.ctx, self.src, self.req_grad)

    def cast_to_self_dtype(self, x: Tensor | Scalar) -> Tensor:
        """Casts the input's data type to match self.

        Args:
            x (Tensor | Scalar): The input tensor or scalar.

        Returns:
            Tensor: A new tensor with matching data type.
        """
        if isinstance(x, Tensor):
            return x.as_type(self.dtype)
        return Tensor(self.device.xp.asarray(x, dtype=self.dtype))

    def as_numpy(self) -> numpy.ndarray:
        """Returns the tensor data as a NumPy array.

        Returns:
            numpy.ndarray: The tensor data as a NumPy array.
        """
        return self.cpu().data

    def iter_dim(self, dim: int) -> Iterator[Tensor]:
        """Iterates over a specified dimension and yields chunks of the tensor.

        Args:
            dim (int): Dimension to iterate over.

        Yields:
            Iterator[Tensor]: An iterator over tensor chunks.
        """
        dim = dim % self.ndim
        pre_dim_slice = (slice(None),) * dim
        post_dim_slice = (slice(None),) * (self.ndim - dim - 1)
        for i in range(self.shape[dim]):
            yield self[(*pre_dim_slice, i, *post_dim_slice)]

    def argmax(self, dim: Optional[Dim] = None, *, keepdims: bool = False) -> Tensor:
        """Returns the index of the maximum value along a specified dimension.

        Args:
            dim (Dim | None, optional): The dimension to reduce. If `None`, finds the global
                maximum.
            keepdims (bool, optional): Whether to retain reduced dimensions. Defaults to `False`.

        Returns:
            Tensor: The index of the maximum value.
        """
        return Tensor(self.device.xp.argmax(self.data, axis=dim, keepdims=keepdims))


# -------------------------------------------------------------------------------------
# AUTOGRAD FUNCTIONS
# -------------------------------------------------------------------------------------


def _undo_broadcast(grad: Array, target_shape: ShapeLike) -> Array:
    """Ensures that the resulting gradient of some operation matches the target shape."""
    if grad.shape == target_shape:
        return grad
    target_ndim = len(target_shape)

    if grad.ndim == target_ndim:
        dims = _get_shape_diff(grad.shape, target_shape)
        grad = grad.sum(dims, keepdims=True)
    else:
        data_shape = Shape((1,) * (grad.ndim - target_ndim) + target_shape)
        dims = _get_shape_diff(grad.shape, data_shape)
        grad = grad.sum(dims)

    return grad.reshape(target_shape)


def depth_first_search(
    root_node: Tensor,
    nodes: list[Tensor],
    visited: set[Tensor],
    *,
    include_leaf_nodes: bool = False,
) -> None:
    """Traverses the computational graph using depth-first-search and fills the list of nodes."""
    if root_node in visited:
        return
    visited.add(root_node)
    assert root_node.src is not None, (
        "Node has no src nodes. This might happen if backward was run."
    )
    for src_node in root_node.src:
        if src_node is not None:
            depth_first_search(src_node, nodes, visited, include_leaf_nodes=include_leaf_nodes)
    if include_leaf_nodes or len(root_node.src) > 0:
        nodes.append(root_node)


def apply_op(op: type[Op], *tensors: Optional[Tensor], **kwargs: Any) -> Tensor:
    """Applies an operation to one or more tensors, handling autograd if needed.

    Args:
        op (type[Op]): The operation to apply.
        *tensors (Tensor | None): Input tensors to which the operation is applied.
        **kwargs (Any): Additional keyword arguments for the operation.

    Returns:
        Tensor: The resulting tensor after calling the `forward` method.
    """
    tensor_args = [t for t in tensors if t is not None]
    device = tensor_args[0].device
    ctx = op(device.xp, kwargs)

    # compute forward pass
    fwd_args = [t.data if t is not None else None for t in tensors]
    with device:
        data = ctx.forward(*fwd_args, **kwargs)

    # return result node with autograd context
    result_req_grad = any(t.req_grad for t in tensor_args)
    if autograd_tracking_active and result_req_grad:
        return Tensor(data, ctx=ctx, src=tensors, req_grad=True)

    # return result node without autograd context
    return Tensor(data)


autograd_tracking_active = True


def _set_autograd_tracking_mode(active: bool) -> None:
    global autograd_tracking_active
    autograd_tracking_active = active


@contextmanager
def no_autograd_tracking() -> Generator:
    """Context manager for disabling autograd tracking."""
    _set_autograd_tracking_mode(False)
    try:
        yield
    finally:
        _set_autograd_tracking_mode(True)


# -------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------------------


def parse_key(key: Any) -> Any:
    """Ensures the key is processable by the backend."""
    if isinstance(key, tuple):
        return tuple(k.data if isinstance(k, Tensor) else k for k in key)
    if isinstance(key, Tensor):
        return key.data
    return key


def _get_shape_diff(shape1: ShapeLike, shape2: ShapeLike):
    """Returns the dims where two shapes do not match."""
    return tuple(i for i in range(len(shape1)) if shape1[i] != shape2[i])
