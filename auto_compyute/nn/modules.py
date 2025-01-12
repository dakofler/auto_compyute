from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, OrderedDict

from ..autograd import Node, randu
from .functional import linear

__all__ = ["Parameter", "Module", "Linear"]


class Parameter(Node):
    def __init__(self, data: Node) -> None:
        super().__init__(data.data, requires_grad=True)


class Module(ABC):
    def __init__(self) -> None:
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()

    def __call__(self, x: Node) -> Node:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Node) -> Node: ...

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        return super().__setattr__(name, value)

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
        self.weight = Parameter(randu((out_dim, in_dim), -k, k))
        self.bias = Parameter(randu((out_dim,), -k, k))

    def forward(self, x: Node) -> Node:
        return linear(x, self.weight, self.bias)
