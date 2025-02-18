"""Neural network modules."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Optional, OrderedDict

from ..tensor_factory import ones, randn, randu, zeros
from ..tensor_functions import stack
from ..autograd import Tensor
from ..backends import Device, DeviceLike, ShapeLike
from ..dtypes import DType
from . import functional as F

__all__ = [
    "Parameter",
    "Buffer",
    "Module",
    "Modulelist",
    "Sequential",
    "GELU",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Linear",
    "Conv1D",
    "Conv2D",
    "ConvTranspose2D",
    "MaxPooling2D",
    "RNN",
    "LSTM",
    "GRU",
    "MultiHeadSelfAttention",
    "Batchnorm",
    "Layernorm",
    "Embedding",
    "Dropout",
    "Flatten",
    "Reshape",
]


class Parameter(Tensor):
    """Represents a trainable parameter tensor in a neural network.

    Args:
        data (Tensor): The underlying data of the tensor.
        label (str | None, optional): An optional label for the tensor. Defaults to `None`.
    """

    def __init__(self, data: Tensor, label: Optional[str] = None) -> None:
        super().__init__(data.data, req_grad=True, label=label)


class Buffer(Tensor):
    """Represents a non-trainable tensor in a neural network.

    Args:
        data (Tensor): The underlying data of the tensor.
        label (str | None, optional): An optional label for the tensor. Defaults to `None`.
    """

    def __init__(self, data: Tensor, label: Optional[str] = None) -> None:
        super().__init__(data.data, label=label)


class Module(ABC):
    """
    Base class for neural network modules. Implements a structure to hold trainable
    parameters, buffers, and submodules.
    """

    def __init__(self) -> None:
        self._training = True
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Buffer] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()

    @property
    def device(self) -> Device:
        """Returns the device on which the module parameters and buffers are stored."""
        try:
            return next(self.parameters()).device
        except StopIteration as e:
            raise ValueError("Module has no parameters.") from e

    @property
    def dtype(self) -> DType:
        """Returns the dtype of the module parameters and buffers."""
        try:
            return next(self.parameters()).dtype
        except StopIteration as e:
            raise ValueError("Module has no parameters.") from e

    @property
    def n_params(self) -> int:
        """Returns the number of parameters of the module."""
        return sum(p.size for p in self.parameters())

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __call__(self, *tensors: Tensor) -> Tensor:
        """Computes the forward pass of the module.

        Args:
            *tensors (Tensor): Input tensors.

        Returns:
            TensorLike: The result of the forward pass.
        """
        return self.forward(*tensors)

    def __setattr__(self, key: str, value: Any) -> None:

        # register parameters, buffers and sub-modules
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Buffer):
            self._buffers[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Modulelist):
            for i, module in enumerate(value):
                self._modules[key + "." + str(i)] = module

        return super().__setattr__(key, value)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        kwargs = [f"{k}={v}" for k, v in self.__dict__.items() if _is_repr_attr(k, v)]
        kwargs_str = ", ".join(kwargs)
        return f"{class_name}({kwargs_str})"

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    @abstractmethod
    def forward(self, *tensors: Tensor) -> Tensor:
        """Computes the forward pass of the module.

        Args:
            *tensors (Tensor): Input tensors.

        Returns:
            TensorLike: The result of the forward pass.
        """

    def modules(self, recursive: bool = True) -> Iterator[Module]:
        """Returns an iterator over submodules.

        Args:
            recursive (bool, optional): If `True`, recursively iterates through all submodules.
                Defaults to `True`.

        Yields:
            Iterator[Module]: An iterator over the submodules.
        """
        for m in self._modules.values():
            yield m
            if recursive:
                yield from m.modules()

    def parameters(self, recursive: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over parameters.

        Args:
            recursive (bool, optional): If `True`, includes parameters from all submodules.
                Defaults to `True`.

        Yields:
            Iterator[Parameter]: An iterator over the parameters.
        """
        for p in self._parameters.values():
            yield p
        if recursive:
            for m in self.modules():
                yield from m.parameters(recursive=False)

    def buffers(self, recursive: bool = True) -> Iterator[Buffer]:
        """Returns an iterator over buffers.

        Args:
            recursive (bool, optional): If `True`, includes buffers from all submodules.
                Defaults to `True`.

        Yields:
            Iterator[Buffer]: An iterator over the buffers.
        """
        for b in self._buffers.values():
            yield b
        if recursive:
            for m in self.modules():
                yield from m.buffers(recursive=False)

    def train(self) -> None:
        """
        Sets the module and its parameters to training mode. Enables gradient computation for
        parameters.
        """
        self._training = True
        for p in self.parameters(recursive=False):
            p.req_grad = True
        for m in self.modules():
            m.train()

    def eval(self) -> None:
        """
        Sets the module and its parameters to evaluation mode. Disables gradient computation for
        parameters.
        """
        self._training = False
        for p in self.parameters(recursive=False):
            p.req_grad = False
        for m in self.modules():
            m.eval()

    def to(self, device: DeviceLike) -> Module:
        """Moves the module and its contents to the specified device.

        Args:
            device (DeviceLike): The target device for computation.

        Returns:
            Module: The modules with contents on the specified device.
        """
        for t in vars(self).values():
            if isinstance(t, Tensor):
                t.ito(device)

        for module in self.modules(recursive=False):
            module.to(device)

        return self


def _is_repr_attr(key: str, value: Any) -> bool:
    return not key.startswith("_") and not isinstance(value, (Tensor, Module))


class Modulelist(list):
    """A container for storing and managing a list of modules. Inherits from the built-in list
    and is used to store submodules.

    Args:
        modules (Iterable[Module]): An iterable of modules to be stored.
    """

    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__(modules)


class Sequential(Module):
    """A sequential container of modules. Modules are applied in order.

    Args:
        *layers (Module): A sequence of modules to apply sequentially.
    """

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = Modulelist(layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        for layer in self.layers:
            x = layer(x)
        return x


class GELU(Module):
    """Applies the Gaussian Error Linear Unit (GELU) activation function."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.gelu(x)


class ReLU(Module):
    """Applies the Rectified Linear Unit (ReLU) activation function."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.relu(x)


class LeakyReLU(Module):
    """Applies the Leaky ReLU activation function.

    Args:
        alpha (float, optional): Slope for negative values. Defaults to `0.2`.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.leaky_relu(x, self.alpha)


class Sigmoid(Module):
    """Applies the sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.sigmoid(x)


class Tanh(Module):
    """Applies the hyperbolic tangent (tanh) activation function."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.tanh(x)


class Linear(Module):
    """Applies a linear transformation to the input.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        bias (bool, optional): If `True`, includes a bias term. Defaults to `True`.

    Attributes:
        w (Parameter): Weight matrix of shape (out_dim, in_dim).
        b (Parameter | None): Bias vector of shape (out_dim,) if bias is enabled, else `None`.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        k = 1 / math.sqrt(in_dim)
        self.w = Parameter(randu(out_dim, in_dim, low=-k, high=k), "Weights")
        self.b = (
            None if not bias else Parameter(randu(out_dim, low=-k, high=k), "Biases")
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.linear(x, self.w, self.b)


class Conv1D(Module):
    """Applies a 1D convolution operation.

    Args:
        in_dim (int): Input feature dimension (number of input channels).
        out_dim (int): Output feature dimension (number of kernels).
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to `3`.
        stride (int, optional): Stride of the convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to all sides. Defaults to `0`.
        dilation (int, optional): Dilation rate of the kernel. Defaults to `1`.
        bias (bool, optional): If `True`, includes a bias term. Defaults to `True`.

    Attributes:
        w (Parameter): Kernel weight tensor of shape (out_dim, in_dim, kernel_size).
        b (Parameter | None): Bias vector of shape (out_dim,) if bias is enabled, else `None`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        k = 1 / math.sqrt(in_dim * kernel_size)
        self.w = Parameter(
            randu(out_dim, in_dim, kernel_size, low=-k, high=k), "KernelWeights"
        )
        self.b = (
            None if not bias else Parameter(randu(out_dim, low=-k, high=k), "Biases")
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.conv1d(x, self.w, self.b, self.stride, self.padding, self.dilation)


class Conv2D(Module):
    """Applies a 2D convolution operation.

    Args:
        in_dim (int): Input feature dimension (number of input channels).
        out_dim (int): Output feature dimension (number of kernels).
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to `3`.
        stride (int, optional): Stride of the convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to all sides. Defaults to `0`.
        dilation (int, optional): Dilation rate of the kernel. Defaults to `1`.
        bias (bool, optional): If `True`, includes a bias term. Defaults to `True`.

    Attributes:
        w (Parameter): Kernel weight tensor of shape (out_dim, in_dim, kernel_size, kernel_size).
        b (Parameter | None): Bias vector of shape (out_dim,) if bias is enabled, else `None`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        k = 1 / math.sqrt(in_dim * kernel_size * kernel_size)
        self.w = Parameter(
            randu(out_dim, in_dim, kernel_size, kernel_size, low=-k, high=k),
            "KernelWeights",
        )
        self.b = (
            None if not bias else Parameter(randu(out_dim, low=-k, high=k), "Biases")
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation)


class ConvTranspose2D(Module):
    """Applies a transposed 2D convolution (deconvolution) operation.

    Args:
        in_dim (int): Input feature dimension (number of input channels).
        out_dim (int): Output feature dimension (number of kernels).
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to `3`.
        stride (int, optional): Stride of the convolution. Defaults to `1`.
        padding (int, optional): Zero-padding added to all sides. Defaults to `0`.
        output_padding (int, optional): Additional size added to the output. Defaults to `0`.
        dilation (int, optional): Dilation rate of the kernel. Defaults to `1`.
        bias (bool, optional): If `True`, includes a bias term. Defaults to `True`.

    Attributes:
        w (Parameter): Kernel weight tensor of shape (out_dim, in_dim, kernel_size, kernel_size).
        b (Parameter | None): Bias vector of shape (out_dim,) if bias is enabled, else `None`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.bias = bias

        k = 1 / math.sqrt(in_dim * kernel_size * kernel_size)
        self.w = Parameter(
            randu(out_dim, in_dim, kernel_size, kernel_size, low=-k, high=k),
            "KernelWeights",
        )
        self.b = (
            None if not bias else Parameter(randu(out_dim, low=-k, high=k), "Biases")
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.conv_transpose2d(
            x,
            self.w,
            self.b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )


class MaxPooling2D(Module):
    """Applies a 2D max pooling operation.

    Args:
        window_size (int, optional): Pooling window size. Defaults to `2`.
    """

    def __init__(self, window_size: int = 2) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.maxpool2d(x, self.window_size)


class RNN(Module):
    """Simple Recurrent Neural Network (RNN) module.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        return_seq (bool, optional): If `True`, returns the full sequence of hidden states.
            If `False`, returns only the last hidden state. Defaults to `False`.
    """

    def __init__(self, in_dim: int, hidden_dim: int, return_seq: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.return_seq = return_seq

        self.W_xh = Linear(in_dim, hidden_dim, bias=False)
        self.W_hh = Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        B, T, _ = x.shape
        h = []
        h_t = zeros(B, self.hidden_dim, device=x.device)

        # input to hidden projection
        xh = self.W_xh(x)

        for t in range(T):
            xh_t = xh[:, t]

            # hidden to hidden projection
            hh_t = self.W_hh(h_t)

            h_t = F.tanh(xh_t + hh_t)

            if self.return_seq:
                h.append(h_t)
        return stack(*h, dim=1) if self.return_seq else h_t


class LSTM(Module):
    """Long Short-Term Memory (LSTM) module.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        return_seq (bool, optional): If `True`, returns the full sequence of hidden states.
            If `False`, returns only the last hidden state. Defaults to `False`.
    """

    def __init__(self, in_dim: int, hidden_dim: int, return_seq: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.return_seq = return_seq

        self.W_xh = Linear(in_dim, 4 * hidden_dim, bias=False)
        self.W_hh = Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        B, T, _ = x.shape
        h = []
        h_t = zeros(B, self.hidden_dim, device=x.device)
        c_t = zeros(B, self.hidden_dim, device=x.device)

        # input to hidden projection
        xf, xi, xo, xc = self.W_xh(x).split(self.hidden_dim)

        for t in range(T):
            xf_t, xi_t, xo_t, xc_t = xf[:, t], xi[:, t], xo[:, t], xc[:, t]

            # hidden to hidden projection
            hf_t, hi_t, ho_t, hc_t = self.W_hh(h_t).split(self.hidden_dim)

            f_t = F.sigmoid(xf_t + hf_t)  # forget gate
            i_t = F.sigmoid(xi_t + hi_t)  # input gate
            o_t = F.sigmoid(xo_t + ho_t)  # output gate

            c_candidate = F.tanh(xc_t + hc_t)  # cell state candidate
            c_t = f_t * c_t + i_t * c_candidate
            h_t = o_t * F.tanh(c_t)

            if self.return_seq:
                h.append(h_t)
        return stack(*h, dim=1) if self.return_seq else h_t


class GRU(Module):
    """Gated Recurrent Unit (GRU) module.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        return_seq (bool, optional): If `True`, returns the full sequence of hidden states.
            If `False`, returns only the last hidden state. Defaults to `False`.
    """

    def __init__(self, in_dim: int, hidden_dim: int, return_seq: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.return_seq = return_seq

        self.W_xh = Linear(in_dim, 3 * hidden_dim, bias=False)
        self.W_hh = Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        B, T, _ = x.shape
        h = []
        h_t = zeros(B, self.hidden_dim, device=x.device)

        # input to hidden projection
        xr, xz, xh = self.W_xh(x).split(self.hidden_dim)

        for t in range(T):
            xr_t, xz_t, xh_t = xr[:, t], xz[:, t], xh[:, t]

            # hidden to hidden projection
            hr_t, hz_t, hh_t = self.W_hh(h_t).split(self.hidden_dim)

            r_t = F.sigmoid(xr_t + hr_t)  # reset gate
            z_t = F.sigmoid(xz_t + hz_t)  # update gate

            h_candidate = F.tanh(xh_t + r_t * hh_t)  # hidden state candidate
            h_t = z_t * h_t + (1.0 - z_t) * h_candidate

            if self.return_seq:
                h.append(h_t)
        return stack(*h, dim=1) if self.return_seq else h_t


class MultiHeadSelfAttention(Module):
    """Implements multi-head self-attention.

    Args:
        in_dim (int): Input feature dimension.
        n_heads (int): Number of attention heads.
        mask (Tensor | None, optional): Optional attention mask. Defaults to `None`.
        dropout (float, optional): Dropout probability. Defaults to `0`.
        attn_bias (bool, optional): If `True`, includes bias in attention projections. Defaults
            to `False`.
        bias (bool, optional): If `True`, includes bias in the output projection. Defaults to
            `True`.
    """

    def __init__(
        self,
        in_dim: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        dropout: float = 0,
        attn_bias: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert in_dim % n_heads == 0, "Input dim must be divisible by n_heads."

        self.in_dim = in_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_bias = attn_bias
        self.bias = bias

        self.mask = Buffer(mask, "AttnMask") if mask is not None else None
        self.qkv = Linear(in_dim, 3 * in_dim, bias=attn_bias)
        self.out = Linear(in_dim, in_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        B, S, D = x.shape
        dropout = self.dropout if self._training else 0

        # create query, key and value projections
        q, k, v = self.qkv(x).split(D)

        # split q, k, v into seperate heads
        q = q.view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, self.mask, dropout)

        # concatinate heads
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(attn)


class Batchnorm(Module):
    """Applies batch normalization.

    Args:
        in_dim (int): Number of input features.
        momentum (float, optional): Momentum for updating running stats. Defaults to `0.1`.
        eps (float, optional): Small constant added for numerical stability. Defaults to `1e-5`.

    Attributes:
        w (Parameter): Scale parameter (gamma) of shape (in_dim,).
        b (Parameter): Shift parameter (beta) of shape (in_dim,).
        rmean (Buffer): Running mean of shape (in_dim,).
        rvar (Buffer): Running variance of shape (in_dim),.
    """

    def __init__(self, in_dim: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.momentum = momentum
        self.eps = eps

        self.w = Parameter(ones(in_dim), "Gamma")
        self.b = Parameter(zeros(in_dim), "Beta")
        self.rmean = Buffer(zeros(in_dim), "Run. Mean")
        self.rvar = Buffer(ones(in_dim), "Run. Var")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.batchnorm(
            x,
            self.rmean,
            self.rvar,
            self.w,
            self.b,
            self.momentum,
            self.eps,
            self._training,
        )


class Layernorm(Module):
    """Applies layer normalization.

    Args:
        norm_shape (int | ShapeLike): Shape of the normalization dimension(s).
        eps (float, optional): Small constant added for numerical stability. Defaults to `1e-5`.

    Attributes:
        w (Parameter): Scale parameter (gamma) of shape (norm_shape).
        b (Parameter): Shift parameter (beta) of shape (norm_shape).
    """

    def __init__(self, norm_shape: int | ShapeLike, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_shape = norm_shape
        self.eps = eps

        norm_shape = (norm_shape,) if isinstance(norm_shape, int) else norm_shape
        self.w = Parameter(ones(*norm_shape), "Gamma")
        self.b = Parameter(zeros(*norm_shape), "Beta")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.layernorm(x, self.w, self.b, self.eps)


class Dropout(Module):
    """Applies dropout regularization.

    Args:
        p (float, optional): Dropout probability. Defaults to `0.5`.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.dropout(x, self.p, self._training)


class Embedding(Module):
    """Maps discrete indices to continuous embeddings.

    Args:
        n_emb (int): Number of embedding vectors.
        emb_dim (int): Dimension of each embedding vector.

    Attributes:
        w (Parameter): Embedding matrix of shape (n_emb, emb_dim).
    """

    def __init__(self, n_emb: int, emb_dim: int) -> None:
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.w = Parameter(randn(n_emb, emb_dim), "Embeddings")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return F.embedding(x, self.w)


class Flatten(Module):
    """Flattens the input tensor while preserving the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return x.view(x.shape[0], -1)


class Reshape(Module):
    """Reshapes the input tensor to the specified shape.

    Args:
        shape (ShapeLike): Target shape (excluding the batch dimension).
    """

    def __init__(self, shape: ShapeLike) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return x.view(-1, *self.shape)
