"""Neural network modules"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Optional, OrderedDict

from ..autograd import Tensor
from ..backends import Device, Shape
from ..dtypes import DType, int64
from ..tensor_factory import ones, randn, randu, zeros
from . import functional as F

# import cupy


__all__ = [
    "Parameter",
    "Buffer",
    "Module",
    "Modulelist",
    "Sequential",
    "ReLU",
    "Linear",
    "Conv2D",
    "MaxPooling2D",
    "MultiHeadSelfAttention",
    "Batchnorm",
    "Layernorm",
    "Embedding",
    "Dropout",
    "Flatten",
]

# mem = 0


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, requires_grad=True)


class Buffer(Tensor):
    def __init__(self, data: Tensor) -> None:
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

    def __call__(self, x: Tensor) -> Tensor:
        # y = self.forward(x)
        # free, total = cupy.cuda.Device(0).mem_info
        # usage = total - free
        # global mem
        # delta = usage - mem
        # print(f"{self.__class__.__name__:30s} | {delta:_}")
        # mem = usage
        # return y
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
    def forward(self, x: Tensor) -> Tensor: ...

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
        for module in self.modules(recursive=False):
            module.train()

    def eval(self) -> None:
        self._training = False
        for module in self.modules(recursive=False):
            module.eval()

    def to(self, device: Device) -> None:
        for t in vars(self).values():
            if isinstance(t, Tensor):
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

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        k = 1 / math.sqrt(in_dim)
        self.w = Parameter(randu((out_dim, in_dim), -k, k))
        self.b = None if not bias else Parameter(randu((out_dim,), -k, k))

    def forward(self, x: Tensor) -> Tensor:
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
        self.w = Parameter(randu((out_dim, in_dim, kernel_size, kernel_size), -k, k))
        self.b = None if not bias else Parameter(randu((out_dim,), -k, k))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation)


class MaxPooling2D(Module):
    def __init__(self, window_size: int = 2) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        return F.maxpool2d(x, self.window_size)


class MultiHeadSelfAttention(Module):
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
        self.n_heads = n_heads
        self.mask = Buffer(mask) if mask is not None else None
        self.dropout = dropout
        self.qkv = Linear(in_dim, 3 * in_dim, bias=attn_bias)
        self.out = Linear(in_dim, in_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        dropout = self.dropout if self._training else 0

        qkv = self.qkv(x)
        q, k, v = qkv.split(D)
        q = q.view((B, S, self.n_heads, D // self.n_heads)).transpose(1, 2)
        k = k.view((B, S, self.n_heads, D // self.n_heads)).transpose(1, 2)
        v = v.view((B, S, self.n_heads, D // self.n_heads)).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, self.mask, dropout)
        attn = attn.transpose(1, 2).view((B, S, D))
        return self.out(attn)


class Batchnorm(Module):
    def __init__(self, in_dim: int, m: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.m = m
        self.eps = eps
        self.w = Parameter(ones((in_dim,)))
        self.b = Parameter(zeros((in_dim,)))
        self.rmean = Buffer(zeros((in_dim,)))
        self.rvar = Buffer(ones((in_dim,)))

    def forward(self, x: Tensor) -> Tensor:
        y, rmean, rvar = F.batchnorm(
            x, self.rmean, self.rvar, self.w, self.b, self.m, self.eps, self._training
        )
        self.rmean = Buffer(rmean)
        self.rvar = Buffer(rvar)
        return y


class Layernorm(Module):
    def __init__(self, norm_shape: int | Shape, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        norm_shape = (norm_shape,) if isinstance(norm_shape, int) else norm_shape
        self.w = Parameter(ones(norm_shape))
        self.b = Parameter(zeros(norm_shape))

    def forward(self, x: Tensor) -> Tensor:
        return F.layernorm(x, self.w, self.b, self.eps)


class Embedding(Module):
    def __init__(self, n_emb: int, emb_dim: int) -> None:
        super().__init__()
        self.w = Parameter(randn((n_emb, emb_dim)))

    def forward(self, x: Tensor) -> Tensor:
        assert x.dtype == int64
        return self.w[x]


class Dropout(Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.dropout, self._training)


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view((x.shape[0], -1))
