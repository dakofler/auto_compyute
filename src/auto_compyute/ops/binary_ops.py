"""Binary operations."""

from auto_compyute.backends import Array
from auto_compyute.ops.op import Op


class Add(Op):
    """Element-wise addition."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = x1 + x2
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dx1 = dy
        dx2 = dy
        return dx1, dx2


class Sub(Op):
    """Element-wise subtraction."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = x1 - x2
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        dx1 = dy
        dx2 = -dy
        return dx1, dx2


class Mul(Op):
    """Element-wise multiplication."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = x1 * x2
        self.stash(x1, x2)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.unstash()
        dx1 = dy * x2
        dx2 = dy * x1
        return dx1, dx2


class Div(Op):
    """Element-wise division."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = x1 / x2
        self.stash(x1, x2)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.unstash()
        dx1 = dy / x2
        dx2 = -(dy * x1) / (x2 * x2)
        return dx1, dx2


class Dot(Op):
    """Vector dot product."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = x1 @ x2
        self.stash(x1, x2)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        x1, x2 = self.unstash()
        dx1 = dy @ x2.swapaxes(-1, -2)  # dy @ x2.T
        dx2 = x1.swapaxes(-1, -2) @ dy  # x1.T @ dy
        return dx1, dx2


class Maximum(Op):
    """Element-wise maximum."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = self.xp.maximum(x1, x2)
        self.stash(y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.unstash()
        dx1 = dy * mask
        dx2 = dy * self.xp.invert(mask)
        return dx1, dx2


class Minimum(Op):
    """Element-wise minimum."""

    def forward(self, x1: Array, x2: Array) -> Array:
        y = self.xp.minimum(x1, x2)
        self.stash(y == x1)
        return y

    def backward(self, dy: Array) -> tuple[Array, ...]:
        mask = self.unstash()
        dx1 = dy * mask
        dx2 = dy * self.xp.invert(mask)
        return dx1, dx2
