"""Autograd function"""

from abc import ABC, abstractmethod
from typing import Any

from ..backends import ArrayLike, Device


class Function(ABC):
    def __init__(self, device: Device) -> None:
        self.xp = device.xp
        self._cache: Any = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def save_to_cache(self, *args):
        self._cache = args

    def retrieve_from_cache(self) -> Any:
        assert self._cache is not None
        values = self._cache if len(self._cache) > 1 else self._cache[0]
        self._cache = None
        return values

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]: ...
