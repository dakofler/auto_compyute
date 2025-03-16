"""Neural network optimizers."""

import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .modules import Parameter

if typing.TYPE_CHECKING:
    from ..backends import Array


class Optimizer(ABC):
    """Base class for all optimizers.

    Attributes:
        learning_rate (float): Step size for parameter updates.
    """

    def __init__(self, parameters: Iterable[Parameter], learning_rate: float = 1e-3) -> None:
        """Base class for all optimizers.

        Args:
            parameters (Iterable[Parameter]): Iterable of model parameters to optimize.
            learning_rate (float, optional): Learning rate for updates. Defaults to `1e-3`.
        """
        self.learning_rate = learning_rate
        self._t = 1
        self._parameters: list[Parameter] = []

        # init state
        ptrs: set[int] = set()
        for param in parameters:
            if id(param) in ptrs:
                continue
            self._parameters.append(param)
            ptrs.add(id(param))
        self._state: dict[int, dict[str, Array]] = {i: {} for i in range(len(self._parameters))}

    def reset_param_grads(self) -> None:
        """Sets gradients of all parameters to `None`."""
        for p in self._parameters:
            p.grad = None

    @abstractmethod
    def update_params(self):
        """Performs a single optimization step (update)."""


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with optional momentum.

    Attributes:
        learning_rate (float): Step size for parameter updates.
        momentum (float): Momentum factor for updates.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        momentum: float = 0,
    ) -> None:
        """Stochastic Gradient Descent (SGD) optimizer with optional momentum.

        Args:
            parameters (Iterable[Parameter]): Parameters to optimize.
            learning_rate (float, optional): Learning rate. Defaults to `1e-3`.
            momentum (float, optional): Momentum factor. Defaults to `0`.
        """
        super().__init__(parameters, learning_rate)
        self.momentum = momentum

    def update_params(self):
        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            state = self._state[i]
            grad = param.grad

            if self.momentum > 0:
                state["v"] = grad + self.momentum * state.get("v", 0)
                grad = state["v"]

            param.data -= self.learning_rate * grad

        self._t += 1


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates.

    Attributes:
        learning_rate (float): Step size for parameter updates.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.
        eps (float): Small constant added for numerical stability.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Adam optimizer with adaptive learning rates.

        Args:
            parameters (Iterable[Parameter]): Parameters to optimize.
            learning_rate (float, optional): Learning rate. Defaults to `1e-3`.
            beta1 (float, optional): Exponential decay rate for first moment estimates. Defaults to
                `0.9`.
            beta2 (float, optional): Exponential decay rate for second moment estimates. Defaults to
                `0.999`.
            eps (float, optional): Small constant added for numerical stability. Defaults to `1e-8`.
        """
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update_params(self):
        m_div = 1 - self.beta1**self._t
        v_div = 1 - self.beta2**self._t

        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            state = self._state[i]
            grad = param.grad
            state["m"] = m = state.get("m", 0) * self.beta1 + grad * (1 - self.beta1)
            state["v"] = v = state.get("v", 0) * self.beta2 + grad**2 * (1 - self.beta2)
            param.data -= self.learning_rate * (m / m_div) / (v**0.5 / v_div + self.eps)

        self._t += 1


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Attributes:
        learning_rate (float): Step size for parameter updates.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.
        eps (float): Small constant added for numerical stability.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """AdamW optimizer with decoupled weight decay.

        Args:
            parameters (Iterable[Parameter]): Parameters to optimize.
            learning_rate (float, optional): Learning rate. Defaults to `1e-3`.
            beta1 (float, optional): Exponential decay rate for first moment estimates. Defaults to
                `0.9`.
            beta2 (float, optional): Exponential decay rate for second moment estimates. Defaults to
                `0.999`.
            eps (float, optional): Small constant added for numerical stability. Defaults to `1e-8`.
            weight_decay (float, optional): Weight decay coefficient. Defaults to `1e-2`.
        """
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def update_params(self):
        m_div = 1 - self.beta1**self._t
        v_div = 1 - self.beta2**self._t
        weight_decay_modifier = 1 - self.learning_rate * self.weight_decay

        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            param.data *= weight_decay_modifier
            state = self._state[i]
            grad = param.grad
            state["m"] = m = state.get("m", 0) * self.beta1 + grad * (1 - self.beta1)
            state["v"] = v = state.get("v", 0) * self.beta2 + grad**2 * (1 - self.beta2)
            param.data -= self.learning_rate * (m / m_div) / (v**0.5 / v_div + self.eps)

        self._t += 1
