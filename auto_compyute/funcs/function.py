"""Autograd function base class."""

from abc import ABC, abstractmethod
from typing import Any

from ..backends import ArrayLike, Device


class Function(ABC):
    """Base class for an autograd function.

    Attributes:
        xp (ModuleType): The module used for computations.
    """

    def __init__(self, device: Device) -> None:
        """Base class for an autograd function.

        Args:
            device (Device): The device used for computations.
        """
        self.xp = device.xp
        self._cache: Any = None

    @property
    def name(self) -> str:
        """Returns the function name."""
        return self.__class__.__name__

    def save_to_cache(self, *args):
        """Saves the input arguments to the cache.

        Args:
            *args: The input arguments to be cached.
        """
        self._cache = args

    def retrieve_from_cache(self) -> Any:
        """Retrieves the cached values and resets the cache afterwards.

        Returns:
            Any: The cached values, either as a tuple or single item.

        Raises:
            AssertionError: If no values are cached.
        """
        assert self._cache is not None
        values = self._cache if len(self._cache) > 1 else self._cache[0]
        self._cache = None
        return values

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Computes the forward pass of the function.

        Args:
            *args (Any): The input values.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ArrayLike: The result of the forward pass.
        """

    @abstractmethod
    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        """Computes the backward pass (gradients) of the function.

        Args:
            dy (ArrayLike): The gradient of the loss with respect to the output.

        Returns:
            tuple[ArrayLike, ...]: The gradients with respect to the inputs.
        """
