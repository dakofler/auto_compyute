"""Neural network optimizers"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..backends import Array
from .modules import Parameter


class Optimizer(ABC):
    def __init__(
        self, parameters: Iterable[Parameter], learning_rate: float = 1e-3
    ) -> None:
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
        self._state: dict[int, dict[str, Array]] = {
            i: {} for i in range(len(self._parameters))
        }

    def zero_grad(self) -> None:
        for p in self._parameters:
            p.grad = None

    @abstractmethod
    def step(self): ...


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        momentum: float = 0,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.momentum = momentum

    def step(self):
        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            if self.momentum > 0:
                v_prev = self._state[i].get("v", 0)
                v = self.momentum * v_prev + param.grad
                self._state[i]["v"] = v
                param.data -= self.learning_rate * v
            else:
                param.data -= self.learning_rate * param.grad

        self._t += 1


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self):
        m_div = 1 - self.beta1**self._t
        v_div = 1 - self.beta2**self._t

        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            # first moment estimate (exponential moving average)
            m_prev = self._state[i].get("m", 0)
            m = self.beta1 * m_prev + (1 - self.beta1) * param.grad
            self._state[i]["m"] = m

            # second moment estimate (squared gradient)
            v_prev = self._state[i].get("v", 0)
            v = self.beta2 * v_prev + (1 - self.beta2) * param.grad**2
            self._state[i]["v"] = v

            m = m / m_div
            v = v / v_div
            param.data -= self.learning_rate * m / (v**0.5 + self.eps)

        self._t += 1


class AdamW(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        m_div = 1 - self.beta1**self._t
        v_div = 1 - self.beta2**self._t

        for i, param in enumerate(self._parameters):
            if param.grad is None:
                continue

            param.data *= 1 - self.learning_rate * self.weight_decay

            # first moment estimate (exponential moving average)
            m_prev = self._state[i].get("m", 0)
            m = self.beta1 * m_prev + (1 - self.beta1) * param.grad
            self._state[i]["m"] = m

            # second moment estimate (squared gradient)
            v_prev = self._state[i].get("v", 0)
            v = self.beta2 * v_prev + (1 - self.beta2) * param.grad**2
            self._state[i]["v"] = v

            m = m / m_div
            v = v / v_div
            param.data -= self.learning_rate * m / (v**0.5 + self.eps)

        self._t += 1
