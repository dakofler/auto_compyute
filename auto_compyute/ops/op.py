"""Differentiable operation base class."""

from abc import ABC, abstractmethod
from typing import Any

from ..backends import ArrayLike, Device


class Op(ABC):
    """Base class for a differentiable operation.

    Args:
        device (Device): The device used for computations.
    """

    def __init__(self, device: Device) -> None:
        self.xp = device.xp
        self._cache: Any = None

    @property
    def name(self) -> str:
        """Returns the operation name."""
        return self.__class__.__name__

    def save_to_cache(self, *args):
        """Saves the input arguments to the cache.

        Args:
            *args: The input arguments to be cached.
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
    def forward(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Computes the forward pass of the operation.

        Args:
            *args (Any): The input values and whether they require gradients
                (e.g. x1, x1_req_grad, x2, x2_req_grad, ...).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ArrayLike: The result of the forward pass.
        """

    @abstractmethod
    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        """Computes the backward pass (gradients) of the operation.

        Args:
            dy (ArrayLike): The gradient with respect to the output.

        Returns:
            tuple[ArrayLike, ...]: The gradients with respect to the inputs.
        """
