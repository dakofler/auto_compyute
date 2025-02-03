"""Autograd function"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..backends import Array, Device


class Cache:
    def __init__(self):
        self.vals: Optional[tuple[Any, ...]] = None

    def save(self, *args: Any) -> None:
        self.vals = args

    def retrieve(self) -> Any:
        assert self.vals is not None
        values = self.vals[0] if len(self.vals) == 1 else self.vals
        self.vals = None
        return values


class DummyCache(Cache):
    def save(self, *args: Any) -> None:
        pass


class Function(ABC):
    def __init__(self, device: Device) -> None:
        self.backend = device.backend
        self.cache: Cache = DummyCache()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def forward(self, *args, **kwargs) -> Array: ...

    @abstractmethod
    def backward(self, dy: Array) -> tuple[Array, ...]: ...
