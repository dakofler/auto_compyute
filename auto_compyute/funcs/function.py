"""Autograd function"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..devices import Array, Device


class Context:
    def __init__(self):
        self.vals: Optional[tuple[Any, ...]] = None

    def save(self, *args: Any) -> None:
        self.vals = args

    def retrieve(self) -> Any:
        assert self.vals is not None
        values = self.vals[0] if len(self.vals) == 1 else self.vals
        self.vals = None
        return values


class PseudoContext(Context):
    def save(self, *args: Any) -> None:
        pass


class Function(ABC):
    def __init__(self, device: Device) -> None:
        self.m = device.m
        self.ctx: Context = PseudoContext()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def forward(self, *args, **kwargs) -> Array: ...

    @abstractmethod
    def backward(self, output_grad) -> tuple[Array, ...]: ...
