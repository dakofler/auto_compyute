"""Differentiable operation base class."""

from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Optional

from auto_compyute.backends import Array


class Op(ABC):
    """Base class for a differentiable operation.

    Args:
        xp (ModuleType): The backend used for computations.
    """

    def __init__(self, xp: ModuleType, kwargs: dict[str, Any]) -> None:
        self.xp = xp
        self.kwargs = kwargs  # for graph visualization
        self._stash: Optional[tuple[Any, ...]] = None

    @property
    def name(self) -> str:
        """Returns the operation name."""
        return self.__class__.__name__

    def stash(self, *args: Any):
        """Saves items for later retrieval.

        Args:
            *args: Items to be stashed.
        """
        self._stash = args

    def unstash(self) -> tuple[Any, ...]:
        """Pops stashed items.

        Returns:
            tuple[Any, ...]: Stashed items.

        Raises:
            AssertionError: If no items are stashed.
        """
        assert self._stash is not None, f"No items stashed for {self.__class__.__name__}"
        values, self._stash = self._stash, None  # reset stash
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
