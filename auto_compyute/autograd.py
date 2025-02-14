"""Autograd engine."""

from __future__ import annotations

from collections.abc import Generator, Iterator
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
    parse_device,
)
from .dtypes import DType, float32, int32, int64, is_float
from .ops import binary_ops as BOps
from .ops import reduce_ops as ROps
from .ops import movement_ops as MOps
from .ops import unary_ops as UOps
from .ops.op import Op

__all__ = ["Tensor", "no_autograd_tracing"]


class Tensor:
    """Represents a multi-dimensional tensor with automatic differentiation support.

    Attributes:
        data (ArrayLike): The underlying data of the tensor.
        ctx (Op | None): The operation context for automatic differentiation.
        src (tuple[Tensor, ...] | None): Tensors used to create this tensor.
        req_grad (bool): Whether the tensor requires autograd tracing.
        grad (ArrayLike | None): Corresponding gradients of the tensor data.
        label (str): A label for the tensor.
    """

    def __init__(
        self,
        data: ArrayLike,
        ctx: Optional[Op] = None,
        src: Optional[tuple[Tensor, ...]] = None,
        req_grad: bool = False,
        label: Optional[str] = None,
    ) -> None:
        """Represents a multi-dimensional tensor with automatic differentiation support.

        Args:
            data (Tensor): The underlying data of the tensor.
            ctx (Op | None, optional): The operation context for automatic differentiation.
                Defaults to `None`.
            src (tuple[Tensor, ...] | None, optional): Tensors used to create this
                tensor. Defaults to `None`.
            req_grad (bool, optional): Whether the tensor requires autograd tracing.
                Defaults to `False`.
            label (str): An optional label for the tensor. Defaults to `None`.
        """
        self.data = data
        self.ctx = ctx
        self.src = src
        self.req_grad = req_grad
        self._label = label
        self.grad: Optional[ArrayLike] = None

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

    def __eq__(self, x: Tensor | Scalar) -> Tensor:  # type: ignore
        return Tensor(self.data == self.align(x).data)

    def __neq__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data != self.align(x).data)

    def __lt__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data < self.align(x).data)

    def __gt__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data > self.align(x).data)

    def __le__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data <= self.align(x).data)

    def __ge__(self, x: Tensor | Scalar) -> Tensor:
        return Tensor(self.data >= self.align(x).data)

    def __getitem__(self, key: Any) -> Tensor:
        return self.select(key)

    def __repr__(self) -> str:
        prefix = f"{self.__class__.__name__}("
        suffix = f", dtype={self.dtype}, device={self.device}"
        suffix += f", grad_fn={self.ctx.name})" if self.ctx is not None else ")"
        return prefix + array_to_string(self.data, prefix) + suffix

    def __len__(self) -> int:
        return self.data.shape[0]

    def __hash__(self) -> int:
        return id(self)

    # ----------------------------------------------------------------------------------
    # AUTOGRAD METHODS
    # ----------------------------------------------------------------------------------

    def accumulate_grad(self, dy: ArrayLike) -> None:
        """Accumulates the gradient for the current tensor.

        Args:
            dy (ArrayLike): The gradient to be applied.

        Raises:
            AssertionError: If `dy` does not have dtype `float32`.
        """
        assert dy.dtype == float32, f"Gradient has invalid dtype {dy.dtype}"
        self.grad = dy if self.grad is None else self.grad + dy

    def backward(self, dy: Optional[ArrayLike] = None):
        """Performs backpropagation to compute gradients.

        - Computes and stores gradients for all nodes in the computation graph.
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
            assert isinstance(dy, ArrayLike), "Gradient must be an array."
            self.grad = dy

        # run backward through traced graph
        node_queue = _build_backward_queue(self, [], set())
        for node in reversed(node_queue):
            assert node.ctx is not None, "Node has no function context."
            assert node.src is not None, "Node has no source nodes."
            grads = node.ctx.backward(node.grad)
            for src_tensor, grad in zip(node.src, grads):
                if src_tensor.req_grad:
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
        return apply_op(BOps.Add, self, self.align(x))

    def sub(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise subtraction.

        Args:
            x (Tensor | Scalar): The tensor or scalar to subtract.

        Returns:
            Tensor: The element-wise difference.
        """
        return apply_op(BOps.Sub, self, self.align(x))

    def mul(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise multiplication.

        Args:
            x (Tensor | Scalar): The tensor or scalar to multiply.

        Returns:
            Tensor: The element-wise product.
        """
        return apply_op(BOps.Mul, self, self.align(x))

    def truediv(self, x: Tensor | Scalar) -> Tensor:
        """Performs element-wise division.

        Args:
            x (Tensor | Scalar): The tensor or scalar to divide by.

        Returns:
            Tensor: The element-wise quotient.
        """
        return apply_op(BOps.Div, self, self.align(x))

    def matmul(self, x: Tensor) -> Tensor:
        """Performs the dot product of the tensors. For higher dimensional tensors it performs
        parallelized dot products (eg. matrix multiplication for 2-dimesional tensors).

        Args:
            x (Tensor): The tensor to multiply with.

        Returns:
            Tensor: The result of the dot product.
        """
        return apply_op(BOps.Matmul, self, x)

    def maximum(self, x: Tensor | Scalar) -> Tensor:
        """Computes the element-wise maximum.

        Args:
            x (Tensor | Scalar): The tensor or scalar to compare with.

        Returns:
            Tensor: The element-wise maximum values.
        """
        return apply_op(BOps.Maximum, self, self.align(x))

    def minimum(self, x: Tensor | Scalar) -> Tensor:
        """Computes the element-wise minimum.

        Args:
            x (Tensor | Scalar): The tensor or scalar to compare with.

        Returns:
            Tensor: The element-wise minimum values.
        """
        return apply_op(BOps.Minimum, self, self.align(x))

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

    def var(
        self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
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

    def std(
        self, dim: Optional[Dim] = None, *, ddof: int = 1, keepdims: bool = False
    ) -> Tensor:
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
        key = _parse_key(key)
        return apply_op(MOps.Select, self, key=key)

    def _split(self, key: Any) -> Tensor:
        key = _parse_key(key)
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
            self._split(pre_dim_slice + (slice(i, i + split_size),) + post_dim_slice)
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
        data: ArrayLike = self.data.astype(dtype)
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
        if device == self.device:
            return self
        data = move_to_device(self.data, device)
        if self.req_grad:
            arr = Tensor(data, self.ctx, self.src, self.req_grad)
            if self.grad is not None:
                arr.grad = move_to_device(self.grad, device)
            return arr
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

    def align(self, x: Tensor | Scalar) -> Tensor:
        """Aligns the input to match the tensor's data type.

        Args:
            x (Tensor | Scalar): The input tensor or scalar.

        Returns:
            Tensor: A new tensor with the aligned data type.
        """
        if isinstance(x, Tensor):
            return x.as_type(self.dtype)
        return Tensor(self.device.xp.asarray(x, dtype=self.dtype))

    def numpy(self) -> numpy.ndarray:
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
            yield self[pre_dim_slice + (i,) + post_dim_slice]

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


def _build_backward_queue(
    node: Tensor, queue: list[Tensor], visited: set
) -> list[Tensor]:
    if node not in visited:
        visited.add(node)
        if not node.src:
            return []
        for p in node.src:
            if p.req_grad:
                _ = _build_backward_queue(p, queue, visited)
        queue.append(node)
    return queue


def apply_op(op: type[Op], *tensors: Optional[Tensor], **kwargs: Any) -> Tensor:
    """Applies an operation to one or more tensors, handling autograd if needed.

    Args:
        op (type[Op]): The operation to apply.
        *tensors (Tensor | None): Input tensors to which the function is applied.
        **kwargs (Any): Additional keyword arguments for the function.

    Returns:
        Tensor: The resulting tensor after calling the `forward` method.
    """
    # create op args by extracting req_grad from tensors and handle optional tensors
    op_args = [(None, False) if a is None else (a.data, a.req_grad) for a in tensors]
    op_args = tuple(chain(*op_args))  # type: ignore  # flatten tuple of tuples

    # get tensor args
    t_args = tuple(t for t in tensors if t is not None)
    device = t_args[0].device
    ctx = op(device)

    with device:
        data = ctx.forward(*op_args, **kwargs)

    # return result node with autograd context
    if autograd_tracing_active and any(t.req_grad for t in t_args):
        return Tensor(data, ctx=ctx, src=t_args, req_grad=True)

    # return result node without autograd context
    return Tensor(data)


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
        from mermaid import Mermaid  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install mermaid-python to draw graphs.") from exc

    colors = {
        "const": ("#CAEDFB", "#4D93D9"),
        "leaf": ("#C6EFCE", "#4EA72E"),
        "op": ("#F2F2F2", "#808080"),
    }

    def _get_mermaid_node_label(n: Tensor) -> str:

        node_name = n.label

        if not n.req_grad:  # constant
            fill_color, stroke_color = colors["const"]
        elif n.ctx is None:  # leaf node
            fill_color, stroke_color = colors["leaf"]
        else:  # op
            fill_color, stroke_color = colors["op"]

        if len(n.shape) == 0:
            node_info = f"{n.item():.4g}"
        else:
            node_info = str(n.shape).replace("shape", "")

        label = f"{node_name}<br>{node_info}<br>{str(n.dtype)}"
        return f'{id(n)}("{label}")\nstyle {str(id(n))} fill:{fill_color},stroke:{stroke_color}'

    mermaid_script = f"graph {orientation}\n{_get_mermaid_node_label(root_node)}\n"

    def _build_mermaid_script(node: Tensor, mermaid_script: str) -> str:
        if not node.src:
            return ""
        for src_node in node.src:
            src_node_label = _get_mermaid_node_label(src_node)
            if src_node_label not in mermaid_script:
                mermaid_script += f"{src_node_label}\n"
            edge = f"{str(id(src_node))}-->{str(id(node))}\n"
            if edge not in mermaid_script:
                mermaid_script += edge
            if src_node.src:
                mermaid_script = _build_mermaid_script(src_node, mermaid_script)
        return mermaid_script

    mermaid_script = _build_mermaid_script(root_node, mermaid_script)
    mermaid_html = Mermaid(mermaid_script)
    if save_to_file:
        with open("compute_graph.html", "w", encoding="utf-8") as f:
            f.write(mermaid_html._repr_html_())
    else:
        return mermaid_html


autograd_tracing_active = True


def _set_autograd_tracing_mode(active: bool) -> None:
    global autograd_tracing_active
    autograd_tracing_active = active


@contextmanager
def no_autograd_tracing() -> Generator:
    """Context manager for disabling autograd tracing."""
    _set_autograd_tracing_mode(False)
    try:
        yield
    finally:
        _set_autograd_tracing_mode(True)


# -------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------------------


def _parse_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return tuple(k.data if isinstance(k, Tensor) else k for k in key)
    if isinstance(key, Tensor):
        return key.data
    return key


def _get_shape_diff(shape1: ShapeLike, shape2: ShapeLike) -> Shape:
    return Shape(i for i in range(len(shape1)) if shape1[i] != shape2[i])
