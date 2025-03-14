"""Differentiable operation base class."""

from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Optional

from ..backends import Array


class Op(ABC):
    """Base class for a differentiable operation.

    Args:
        xp (ModuleType): The backend used for computations.
    """

    def __init__(self, xp: ModuleType, kwargs: Any) -> None:
        self.xp = xp
        self.kwargs = kwargs  # for graph visualization
        self._cache: Any = None

    @property
    def name(self) -> str:
        """Returns the operation name."""
        return self.__class__.__name__

    def save_to_cache(self, *args: Any):
        """Saves values to the cache.

        Args:
            *args: Values to be cached.
        """
        self._cache = args

    def retrieve_from_cache(self) -> tuple[Any, ...]:
        """Retrieves the cached values and resets the cache afterwards.

        Returns:
            tuple[Any, ...]: The cached values.

        Raises:
            AssertionError: If no values are cached.
        """
        assert self._cache is not None
        values, self._cache = self._cache, None  # reset cache to None
        return values

    @abstractmethod
    def forward(self, *arrays: Optional[Array], **kwargs: Any) -> Array:
        """Computes the forward pass of the operation.

        Args:
            *arrays (ArrayLike | None): The input arrays.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ArrayLike: The result of the forward pass.
        """

    @abstractmethod
    def backward(self, dy: Array) -> tuple[Array, ...]:
        """Computes the backward pass (gradient) of the operation.

        Args:
            dy (ArrayLike): The gradient with respect to the output.

        Returns:
            tuple[ArrayLike, ...]: The gradient with respect to the inputs.
        """
