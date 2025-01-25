"""Neural network functions"""

from ..autograd import Tensor


def mse_loss(logits: Tensor, targets: Tensor):
    return logits.sub(targets).pow(2).mean()


def linear(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x.matmul(w.transpose()).add(b)


def relu(x: Tensor) -> Tensor:
    return x.maximum(0.0)


def tanh(x: Tensor) -> Tensor:
    return x.tanh()
