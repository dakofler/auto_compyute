"""Binary autograd functions"""

from ..backends import ArrayLike
from .function import Function


class Add(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = x1 + x2
        if x1_req_grad or x2_req_grad:
            self.save_to_cache(x1_req_grad, x2_req_grad)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1_req_grad, x2_req_grad = self.retrieve_from_cache()
        dx1 = dy if x1_req_grad else None
        dx2 = dy if x2_req_grad else None
        return dx1, dx2


class Sub(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = x1 - x2
        if x1_req_grad or x2_req_grad:
            self.save_to_cache(x1_req_grad, x2_req_grad)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1_req_grad, x2_req_grad = self.retrieve_from_cache()
        dx1 = dy if x1_req_grad else None
        dx2 = (-dy) if x2_req_grad else None
        return dx1, dx2


class Mul(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = x1 * x2
        self.save_to_cache(
            (x1 if x2_req_grad else None),
            (x2 if x1_req_grad else None),
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = None if x2 is None else (dy * x2)
        dx2 = None if x1 is None else (dy * x1)
        return dx1, dx2


class Div(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = x1 / x2
        self.save_to_cache(
            (x1 if x2_req_grad else None),
            (x2 if x1_req_grad or x2_req_grad else None),
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = None if x2 is None else (dy / x2)
        dx2 = None if x1 is None else (-(dy * x1) / (x2 * x2))
        return (dx1, dx2)


class Matmul(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = x1 @ x2
        self.save_to_cache(
            (x1 if x2_req_grad else None),
            (x2 if x1_req_grad else None),
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = None if x2 is None else (dy @ x2.swapaxes(-1, -2))
        dx2 = None if x1 is None else (x1.swapaxes(-1, -2) @ dy)
        return dx1, dx2


class Maximum(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = self.xp.maximum(x1, x2)
        self.save_to_cache(
            x1_req_grad,
            x2_req_grad,
            ((y == x1) if x1_req_grad or x2_req_grad else None),
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1_req_grad, x2_req_grad, mask = self.retrieve_from_cache()
        dx1 = None if not x1_req_grad else (dy * mask)
        dx2 = None if not x2_req_grad else (dy * self.xp.invert(mask))
        return dx1, dx2


class Minimum(Function):
    def forward(
        self, x1: ArrayLike, x1_req_grad: bool, x2: ArrayLike, x2_req_grad: bool
    ) -> ArrayLike:
        y = self.xp.minimum(x1, x2)
        self.save_to_cache(
            x1_req_grad,
            x2_req_grad,
            ((y == x1) if x1_req_grad or x2_req_grad else None),
        )
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1_req_grad, x2_req_grad, mask = self.retrieve_from_cache()
        dx1 = None if not x1_req_grad else (dy * mask)
        dx2 = None if not x2_req_grad else (dy * self.xp.invert(mask))
        return dx1, dx2
