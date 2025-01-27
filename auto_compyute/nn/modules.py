from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, OrderedDict

from ..autograd import Tensor
from ..tensors import randu
from . import functional as F

__all__ = ["Parameter", "Module", "Linear", "Relu", "Sequential"]


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, requires_grad=True)


class Buffer(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data)


class Module(ABC):
    def __init__(self) -> None:
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Buffer] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        return super().__setattr__(name, value)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def parameters(self, recursive: bool = True) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        if recursive:
            for m in self.modules():
                yield from m.parameters(recursive=False)

    def modules(self, recursive: bool = True) -> Iterator[Module]:
        for m in self._modules.values():
            yield m
            if recursive:
                yield from m.modules()


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        k = 1 / math.sqrt(in_dim)
        self.w = Parameter(randu((out_dim, in_dim), -k, k))
        self.b = Parameter(randu((out_dim,), -k, k))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.w, self.b)


class Relu(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
