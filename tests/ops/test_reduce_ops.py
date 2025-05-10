"""Tests for reduce operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats
from tests.utils.test_factory import get_op_test
from tests.utils.verifications import verify_op

IN_SHAPES = ((10, 20), (10, 20, 30))
RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in IN_SHAPES)
DIMS = (None, 0, (0, 1))
MIN_MAX_DIMS = (None, 0, 1)
KEEPDIMS = (False, True)
DDOFS = (0, 1)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_sum(
    x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, keepdims: bool
) -> None:
    get_op_test("sum")((x,), dim=dim, keepdims=keepdims)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_mean(
    x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, keepdims: bool
) -> None:
    get_op_test("mean")((x,), dim=dim, keepdims=keepdims)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("ddof", DDOFS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_var(
    x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, ddof: int, keepdims: bool
) -> None:
    ac_x, torch_x = x
    ac_y = ac_x.var(dim, ddof=ddof, keepdims=keepdims)
    torch_y = torch.var(torch_x, dim, correction=ddof, keepdim=keepdims)
    verify_op((ac_x,), ac_y, (torch_x,), torch_y)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_std(
    x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, keepdims: bool
) -> None:
    get_op_test("std")((x,), dim=dim, keepdims=keepdims)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", MIN_MAX_DIMS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_max(
    x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, keepdims: bool
) -> None:
    if dim is None and keepdims:
        return
    ac_x, torch_x = x
    ac_y = ac_x.max(dim=dim, keepdims=keepdims)
    torch_y = torch_x.max() if dim is None else torch_x.max(dim=dim, keepdims=keepdims)[0]
    verify_op((ac_x,), ac_y, (torch_x,), torch_y)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dim", MIN_MAX_DIMS)
@pytest.mark.parametrize("keepdims", KEEPDIMS)
def test_min(
    x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, keepdims: bool
) -> None:
    if dim is None and keepdims:
        return
    ac_x, torch_x = x
    ac_y = ac_x.min(dim=dim, keepdims=keepdims)
    torch_y = torch_x.min() if dim is None else torch_x.min(dim=dim, keepdims=keepdims)[0]
    verify_op((ac_x,), ac_y, (torch_x,), torch_y)
