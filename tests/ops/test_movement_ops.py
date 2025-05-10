"""Tests for movement operations."""

import pytest
import torch

import auto_compyute as ac
from auto_compyute.backends import ShapeLike
from tests.utils.init import get_random_bools, get_random_floats
from tests.utils.test_factory import get_op_test
from tests.utils.verifications import verify_op

IN_SHAPES = ((10, 20), (10, 20, 30))
RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in IN_SHAPES)
KEYS = (1, [1, 1, 2], ac.tensor([1, 2]))
DIMS = (0, 1, -1)
EXPAND_IN_SHAPES = ((1, 20, 1),)
EXPAND_RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in EXPAND_IN_SHAPES)
EXPAND_SHAPES = ((10, 20, 1), (1, 20, 10), (5, 20, 4))
TRANSPOSE_DIMS = ((0, 1), (-1, -2))
VIEW_SHAPES = ((20, 10), (10, 10, 2))
SPLIT_SIZES = (2, 5)
COMB_SHAPES = ((10, 5),)
COMB_RANDOM_FLOAT_TENSORS_1 = tuple(get_random_floats(shape) for shape in COMB_SHAPES)
COMB_RANDOM_FLOAT_TENSORS_2 = tuple(get_random_floats(shape) for shape in COMB_SHAPES)
COMB_RANDOM_FLOAT_TENSORS_3 = tuple(get_random_floats(shape) for shape in COMB_SHAPES)
WHERE_RANDOM_BOOL_TENSORS = (get_random_bools(IN_SHAPES[0]),)
WHERE_RANDOM_FLOAT_TENSORS_1 = (get_random_floats(IN_SHAPES[0]), (0.0, 0.0))
WHERE_RANDOM_FLOAT_TENSORS_2 = (get_random_floats(IN_SHAPES[0]), (1.0, 1.0))


@pytest.mark.parametrize("x", EXPAND_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("shape", EXPAND_SHAPES)
def test_expand(x: tuple[ac.Tensor, torch.Tensor], shape: ShapeLike) -> None:
    get_op_test("expand")((x,), *shape)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("key", KEYS)
def test_select(x: tuple[ac.Tensor, torch.Tensor], key: int | tuple[int, ...] | ac.Tensor) -> None:
    ac_x, torch_x = x
    ac_y = ac_x[key]
    torch_y = torch_x[torch.tensor(key.data) if isinstance(key, ac.Tensor) else key]
    verify_op((ac_x,), ac_y, (torch_x,), torch_y)


@pytest.mark.parametrize("x", RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("dims", TRANSPOSE_DIMS)
def test_transpose(x: tuple[ac.Tensor, torch.Tensor], dims: tuple[int, ...]) -> None:
    get_op_test("transpose")((x,), *dims)


@pytest.mark.parametrize("x", (RANDOM_FLOAT_TENSORS[0],))
@pytest.mark.parametrize("shape", VIEW_SHAPES)
def test_view(x: tuple[ac.Tensor, torch.Tensor], shape: ShapeLike) -> None:
    get_op_test("view")((x,), *shape)


@pytest.mark.parametrize("x", (RANDOM_FLOAT_TENSORS[0],))
@pytest.mark.parametrize("split_size", SPLIT_SIZES)
@pytest.mark.parametrize("dim", DIMS)
def test_split(x: tuple[ac.Tensor, torch.Tensor], split_size: int, dim: int) -> None:
    ac_x, torch_x = x
    ac_y = ac_x.split(split_size, dim=dim)
    torch_y = torch_x.split(split_size, dim=dim)
    for ac_y, torch_y in zip(ac_y, torch_y):
        verify_op((ac_x,), ac_y, (torch_x,), torch_y)


@pytest.mark.parametrize("x_1", COMB_RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", COMB_RANDOM_FLOAT_TENSORS_2)
@pytest.mark.parametrize("x_3", COMB_RANDOM_FLOAT_TENSORS_3)
@pytest.mark.parametrize("dim", DIMS)
def test_stack(
    x_1: tuple[ac.Tensor, torch.Tensor],
    x_2: tuple[ac.Tensor, torch.Tensor],
    x_3: tuple[ac.Tensor, torch.Tensor],
    dim: int,
) -> None:
    ac_x, torch_x = tuple(zip(*[x_1, x_2, x_3]))
    ac_y = ac.stack(*ac_x, dim=dim)
    torch_y = torch.stack(torch_x, dim=dim)
    verify_op(ac_x, ac_y, torch_x, torch_y)


@pytest.mark.parametrize("x_1", COMB_RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", COMB_RANDOM_FLOAT_TENSORS_2)
@pytest.mark.parametrize("x_3", COMB_RANDOM_FLOAT_TENSORS_3)
@pytest.mark.parametrize("dim", DIMS)
def test_concat(
    x_1: tuple[ac.Tensor, torch.Tensor],
    x_2: tuple[ac.Tensor, torch.Tensor],
    x_3: tuple[ac.Tensor, torch.Tensor],
    dim: int,
) -> None:
    ac_x, torch_x = tuple(zip(*[x_1, x_2, x_3]))
    ac_y = ac.concat(*ac_x, dim=dim)
    torch_y = torch.concat(torch_x, dim=dim)
    verify_op(ac_x, ac_y, torch_x, torch_y)


@pytest.mark.parametrize("condition_tensors", WHERE_RANDOM_BOOL_TENSORS)
@pytest.mark.parametrize("x_1", WHERE_RANDOM_FLOAT_TENSORS_1)
@pytest.mark.parametrize("x_2", WHERE_RANDOM_FLOAT_TENSORS_2)
def test_where(
    condition_tensors: tuple[ac.Tensor, torch.Tensor],
    x_1: tuple[ac.Tensor, torch.Tensor],
    x_2: tuple[ac.Tensor, torch.Tensor],
) -> None:
    ac_x, torch_x = tuple(zip(*[condition_tensors, x_1, x_2]))
    ac_y = ac.where(*ac_x)
    torch_y = torch.where(*torch_x)
    verify_op(ac_x, ac_y, torch_x, torch_y)
