"""Autograd function"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..backends import Array, Backend


class Context:
    def __init__(self):
        self.cache: Optional[tuple[Any, ...]] = None

    def set(self, *args: Any) -> None:
        self.cache = args

    def get(self) -> Any | tuple[Any, ...]:
        assert self.cache is not None
        values = self.cache[0] if len(self.cache) == 1 else self.cache
        self.cache = None
        return values


class PseudoContext(Context):
    def set(self, *args: Any) -> None:
        pass


class Function(ABC):
    def __init__(self, backend: Backend) -> None:
        self.m = backend.m
        self.ctx: Context = PseudoContext()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def forward(self, *args, **kwargs) -> Array: ...

    @abstractmethod
    def backward(self, output_grad) -> tuple[Array, ...]: ...
