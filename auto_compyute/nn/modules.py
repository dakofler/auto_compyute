"""Neural network modules"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Optional, OrderedDict

from ..array_factory import ones, randn, randu, zeros
from ..autograd import Array
from ..backends import Device, DeviceLike, ShapeLike
from ..dtypes import DType
from . import functional as F

__all__ = [
    "Parameter",
    "Buffer",
    "Module",
    "Modulelist",
    "Sequential",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Linear",
    "Conv2D",
    "ConvTranspose2D",
    "MaxPooling2D",
    "MultiHeadSelfAttention",
    "Batchnorm",
    "Layernorm",
    "Embedding",
    "Dropout",
    "Flatten",
    "Reshape",
]


class Parameter(Array):
    def __init__(self, data: Array) -> None:
        super().__init__(data.data, req_grad=True)


class Buffer(Array):
    def __init__(self, data: Array) -> None:
        super().__init__(data.data)


class Module(ABC):
    def __init__(self) -> None:
        self._training = True
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Buffer] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()

    @property
    def device(self) -> Device:
        try:
            return next(self.parameters()).device
        except StopIteration as e:
            raise ValueError("Module has no parameters.") from e

    @property
    def dtype(self) -> DType:
        try:
            return next(self.parameters()).dtype
        except StopIteration as e:
            raise ValueError("Module has no parameters.") from e

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __call__(self, x: Array) -> Array:
        return self.forward(x)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Modulelist):
            for i, m in enumerate(value):
                self._modules[name + "." + str(i)] = m
        return super().__setattr__(name, value)

    # ----------------------------------------------------------------------------------
    # OTHER METHODS
    # ----------------------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: Array) -> Array: ...

    def modules(self, recursive: bool = True) -> Iterator[Module]:
        for m in self._modules.values():
            yield m
            if recursive:
                yield from m.modules()

    def parameters(self, recursive: bool = True) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        if recursive:
            for m in self.modules():
                yield from m.parameters(recursive=False)

    def buffers(self, recursive: bool = True) -> Iterator[Buffer]:
        for b in self._buffers.values():
            yield b
        if recursive:
            for m in self.modules():
                yield from m.buffers(recursive=False)

    def train(self) -> None:
        self._training = True
        for p in self.parameters(recursive=False):
            p.req_grad = True
        for m in self.modules():
            m.train()

    def eval(self) -> None:
        self._training = False
        for p in self.parameters(recursive=False):
            p.req_grad = False
        for m in self.modules():
            m.eval()

    def to(self, device: DeviceLike) -> None:
        for t in vars(self).values():
            if isinstance(t, Array):
                t.ito(device)

        for module in self.modules(recursive=False):
            module.to(device)


class Modulelist(list):
    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__(modules)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = Modulelist(layers)

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


class GELU(Module):
    def forward(self, x: Array) -> Array:
        return F.gelu(x)


class ReLU(Module):
    def forward(self, x: Array) -> Array:
        return F.relu(x)


class LeakyReLU(Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Array) -> Array:
        return F.leaky_relu(x, self.alpha)


class Sigmoid(Module):
    def forward(self, x: Array) -> Array:
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Array) -> Array:
        return F.tanh(x)


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        k = 1 / math.sqrt(in_dim)
        self.w = Parameter(randu(out_dim, in_dim, low=-k, high=k))
        self.b = None if not bias else Parameter(randu(out_dim, low=-k, high=k))

    def forward(self, x: Array) -> Array:
        return F.linear(x, self.w, self.b)


class Conv2D(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        k = 1 / math.sqrt(in_dim * kernel_size * kernel_size)
        self.w = Parameter(
            randu(out_dim, in_dim, kernel_size, kernel_size, low=-k, high=k)
        )
        self.b = None if not bias else Parameter(randu(out_dim, low=-k, high=k))

    def forward(self, x: Array) -> Array:
        return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation)


class ConvTranspose2D(Module):
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
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        k = 1 / math.sqrt(in_dim * kernel_size * kernel_size)
        self.w = Parameter(
            randu(out_dim, in_dim, kernel_size, kernel_size, low=-k, high=k)
        )
        self.b = None if not bias else Parameter(randu(out_dim, low=-k, high=k))

    def forward(self, x: Array) -> Array:
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
    def __init__(self, window_size: int = 2) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, x: Array) -> Array:
        return F.maxpool2d(x, self.window_size)


class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        in_dim: int,
        n_heads: int,
        mask: Optional[Array] = None,
        dropout: float = 0,
        attn_bias: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.mask = Buffer(mask) if mask is not None else None
        self.dropout = dropout
        self.qkv = Linear(in_dim, 3 * in_dim, bias=attn_bias)
        self.out = Linear(in_dim, in_dim, bias=bias)

    def forward(self, x: Array) -> Array:
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
    def __init__(self, in_dim: int, m: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.m = m
        self.eps = eps
        self.w = Parameter(ones(in_dim))
        self.b = Parameter(zeros(in_dim))
        self.rmean = Buffer(zeros(in_dim))
        self.rvar = Buffer(ones(in_dim))

    def forward(self, x: Array) -> Array:
        return F.batchnorm(
            x, self.rmean, self.rvar, self.w, self.b, self.m, self.eps, self._training
        )


class Layernorm(Module):
    def __init__(self, norm_shape: int | ShapeLike, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        norm_shape = (norm_shape,) if isinstance(norm_shape, int) else norm_shape
        self.w = Parameter(ones(*norm_shape))
        self.b = Parameter(zeros(*norm_shape))

    def forward(self, x: Array) -> Array:
        return F.layernorm(x, self.w, self.b, self.eps)


class Embedding(Module):
    def __init__(self, n_emb: int, emb_dim: int) -> None:
        super().__init__()
        self.w = Parameter(randn(n_emb, emb_dim))

    def forward(self, x: Array) -> Array:
        return F.embedding(x, self.w)


class Dropout(Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, x: Array) -> Array:
        return F.dropout(x, self.dropout, self._training)


class Flatten(Module):
    def forward(self, x: Array) -> Array:
        return x.view(x.shape[0], -1)


class Reshape(Module):
    def __init__(self, shape: ShapeLike) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Array) -> Array:
        return x.view(-1, *self.shape)
