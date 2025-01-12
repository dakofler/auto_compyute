"""Activation functions"""

from .. import ops
from ..node import Node


def mse_loss(logits: Node, targets: Node):
    return ((logits - targets) ** 2).mean()


def linear(x: Node, w: Node, b: Node) -> Node:
    return x @ w.T + b


def relu(x: Node) -> Node:
    return ops.maximum(x, 0.0)


def tanh(x: Node) -> Node:
    return x.tanh()
