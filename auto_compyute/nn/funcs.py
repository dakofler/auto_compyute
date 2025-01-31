"""Neural network autograd functions"""

from types import ModuleType

from ..backends import Array, Shape
from ..funcs.function import Function

# -------------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Softmax(Function):
    def forward(self, x: Array, dim: int) -> Array:
        x = self.m.exp(x - x.max(dim, keepdims=True))
        y = x / x.sum(dim, keepdims=True)
        self.ctx.save(dim, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dim, y = self.ctx.retrieve()
        dx = y * (output_grad - (output_grad * y).sum(dim, keepdims=True))
        return (dx,)


# -------------------------------------------------------------------------------------
# LINEAR FUNCTIONS
# -------------------------------------------------------------------------------------


class Linear(Function):
    def forward(self, x: Array, w: Array, b: Array) -> Array:
        y = x @ w.swapaxes(-2, -1) + b
        self.ctx.save(x, w)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, w = self.ctx.retrieve()
        dx = output_grad @ w
        dw = (output_grad.swapaxes(-2, -1) @ x).sum(tuple(range(output_grad.ndim - 2)))
        db = output_grad.sum(tuple(range(output_grad.ndim - 1)))
        return dx, dw, db


# -------------------------------------------------------------------------------------
# CONVOLUTION FUNCTIONS
# -------------------------------------------------------------------------------------


def _pad2d_forward(m: ModuleType, x: Array, padding: int) -> Array:
    widths = tuple([(0, 0)] * (x.ndim - 2) + [(padding, padding)] * 2)
    y = m.pad(x, widths)
    return y


class Pad2D(Function):
    def forward(self, x: Array, padding: int) -> Array:
        self.ctx.save(padding)
        return _pad2d_forward(self.m, x, padding)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        padding = self.ctx.retrieve()
        dx = output_grad[..., padding:-padding, padding:-padding]
        dx = self.m.ascontiguousarray(dx)
        return (dx,)


def _dilate2d_forward(m: ModuleType, x: Array, dilation: int) -> Array:
    y_height = dilation * (x.shape[-2] - 1) + 1
    y_width = dilation * (x.shape[-1] - 1) + 1
    y = m.zeros((*x.shape[:-2], y_height, y_width), dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


class Dilate2D(Function):
    def forward(self, x: Array, dilation: int) -> Array:
        self.ctx.save(dilation)
        return _dilate2d_forward(self.m, x, dilation)

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        dilation = self.ctx.retrieve()
        dx = output_grad[..., ::dilation, ::dilation]
        return (dx,)


def _pool2d(m: ModuleType, x: Array, window_size: int, stride: int = 1) -> Array:
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)
    x_str = x.strides
    out_strides = (*x_str[:-2], x_str[-2] * stride, x_str[-1] * stride, *x_str[-2:])
    return m.lib.stride_tricks.as_strided(x, out_shape, out_strides)


def _pad_to_shape(m: ModuleType, x: Array, shape: Shape) -> Array:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return m.pad(x, padding)


class Conv2D(Function):
    def forward(self, x: Array, w: Array, stride: int) -> Array:
        self.ctx.save(x, w, stride)
        x_pooled = _pool2d(self.m, x, w.shape[-1], stride)
        y = self.m.einsum("biyxjk,oijk->boyx", x_pooled, w, order="C")
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, w, stride = self.ctx.retrieve()
        *_, input_size = x.shape
        *_, kernel_size = w.shape

        # fill elements skipped by strides with zeros
        if stride > 1:
            output_grad = _dilate2d_forward(self.m, output_grad, stride)

        # pad to match unstrided dy
        output_size = input_size - kernel_size + 1
        output_shape = (*output_grad.shape[:-2], output_size, output_size)
        output_grad = _pad_to_shape(self.m, output_grad, output_shape)

        # full pad
        output_grad = _pad2d_forward(self.m, output_grad, kernel_size - 1)

        # input grads
        output_grad_pooled = _pool2d(self.m, output_grad, kernel_size)
        w = self.m.flip(w, (-2, -1))
        dx = self.m.einsum("boyxjk,oijk->biyx", output_grad_pooled, w, order="C")

        # filter grads
        output_grad_pooled = _pool2d(self.m, output_grad, input_size)
        dw = self.m.einsum("bojkyx,biyx->oijk", output_grad_pooled, x, order="C")
        dw = self.m.flip(dw, (-2, -1))

        return dx, dw


def _repeat2d(m: ModuleType, x: Array, n_repeats: int, target_shape: Shape):
    repeat_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    repeat_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = m.lib.stride_tricks.as_strided(x.data, repeat_shape, repeat_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    if y.shape != target_shape:
        y = _pad_to_shape(m, y, target_shape)
    return y


class Maxpool2D(Function):
    def forward(self, x: Array, window_size: int) -> Array:
        y = _pool2d(self.m, x, window_size, window_size).max((-2, -1))
        self.ctx.save(x, window_size, y)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x, window_size, y = self.ctx.retrieve()
        mask = _repeat2d(self.m, y, window_size, x.shape) == x
        mask = mask.astype(output_grad.dtype)
        dx = _repeat2d(self.m, output_grad, window_size, x.shape) * mask
        return (dx,)


# -------------------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Batchnorm1D(Function):
    def forward(
        self,
        x: Array,
        rmean: Array,
        rvar: Array,
        w: Array,
        b: Array,
        m: float,
        eps: float,
        training: bool,
    ) -> Array:
        batch_dims: tuple[int, ...] = (0,) if x.ndim == 2 else (0, 2)

        if training:
            # compute mean and variance from x
            mean = x.mean(batch_dims, keepdims=True)
            std = self.m.sqrt(x.var(batch_dims, keepdims=True) + eps)
            x_norm = (x - mean) / std

            # keep running stats
            rmean = rmean * (1 - m) + mean.squeeze() * m
            rvar = rvar * (1 - m) + x.var(batch_dims, ddof=1) * m
        else:
            # use running mean and variance
            var = rvar if (x.ndim == 2) else rvar.reshape((*rvar.shape, 1))
            mean = rmean if (x.ndim == 2) else rmean.reshape((*rmean.shape, 1))
            std = self.m.sqrt(var + eps)
            x_norm = (x - mean) / std

        w = w if (x.ndim == 2) else w.reshape((*w.shape, 1))
        b = b if (x.ndim == 2) else b.reshape((*b.shape, 1))
        y = w * x_norm + b

        self.ctx.save(w, batch_dims, std, x_norm)
        return y, rmean, rvar

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        w, batch_dims, std, x_norm = self.ctx.retrieve()
        n = float(output_grad.size / output_grad.shape[1])

        # input grads
        output_grad_s = output_grad.sum(batch_dims, keepdims=True)
        output_grad_xns = (output_grad * x_norm).sum(batch_dims, keepdims=True)
        dx = (
            w / (std * n) * (n * output_grad - output_grad_s - x_norm * output_grad_xns)
        )

        # gamma grads
        dw = output_grad_xns.squeeze()

        # beta grads
        db = output_grad_s.squeeze()

        return dx, dw, db


class Batchnorm2D(Function):
    def forward(
        self,
        x: Array,
        rmean: Array,
        rvar: Array,
        w: Array,
        b: Array,
        m: float,
        eps: float,
        training: bool,
    ) -> Array:
        batch_dims = (0, 2, 3)

        if training:
            # compute mean and variance from x
            mean = x.mean(batch_dims, keepdims=True)
            std = self.m.sqrt(x.var(batch_dims, keepdims=True) + eps)
            x_norm = (x - mean) / std

            # keep running stats
            rmean = rmean * (1 - m) + mean.squeeze() * m
            rvar = rvar * (1 - m) + x.var(batch_dims, ddof=1) * m
        else:
            # use running mean and variance
            mean = rmean.view((*rmean.shape, 1, 1))
            std = self.m.sqrt(rvar.reshape((*rvar.shape, 1, 1)) + eps)
            x_norm = (x - mean) / std

        w = w.view((*w.shape, 1, 1))
        b = b.view((*b.shape, 1, 1))
        y = w * x_norm + b

        self.ctx.save(w, batch_dims, std, x_norm)
        return y, rmean, rvar

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        w, batch_dims, std, x_norm = self.ctx.retrieve()
        n = float(output_grad.size / output_grad.shape[1])

        # input grads
        output_grad_s = output_grad.sum(batch_dims, keepdims=True)
        output_grad_xns = (output_grad * x_norm).sum(batch_dims, keepdims=True)
        dx = (
            w / (std * n) * (n * output_grad - output_grad_s - x_norm * output_grad_xns)
        )

        # gamma grads
        dw = output_grad_xns.squeeze()

        # beta grads
        db = output_grad_s.squeeze()

        return dx, dw, db


# -------------------------------------------------------------------------------------
# REGULARIZATION FUNCTIONS
# -------------------------------------------------------------------------------------


class Dropout(Function):
    def forward(self, x: Array, p: float) -> Array:
        p = 1.0 - p
        dropout_mask = self.m.random.random(x.shape) < p
        y = x * dropout_mask / p
        self.ctx.save(p, dropout_mask)
        return y

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        p, dropout_mask = self.ctx.retrieve()
        dx = output_grad * dropout_mask / p
        return (dx,)


# -------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------------------------------------------


class MSELoss(Function):
    def forward(self, x: Array, y: Array) -> Array:
        diff = x - y
        loss = (diff * diff).mean()
        self.ctx.save(x.size, diff)
        return loss

    def backward(self, output_grad: Array) -> tuple[Array, ...]:
        x_size, diff = self.ctx.retrieve()
        dx = output_grad * 2.0 * diff / float(x_size)
        return (dx,)
