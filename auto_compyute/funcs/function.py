"""Autograd function"""

from abc import ABC, abstractmethod
from typing import Any

from ..backends import Array


class Context(dict):
    __getattr__ = dict.get

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class PseudoContext(Context):
    def __setattr__(self, name: str, value: Any) -> None:
        pass


class Function(ABC):
    def __init__(self) -> None:
        self.ctx: Context = PseudoContext()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def forward(self, *args, **kwargs) -> Array: ...

    @abstractmethod
    def backward(self, output_grad) -> tuple[Array, ...]: ...
