"""Tests for loss function operations."""

import pytest
import torch

import auto_compyute as ac
from tests.utils.init import get_random_floats, get_random_ints
from tests.utils.test_factory import get_op_test
from tests.utils.verifications import verify_op

FLOAT_IN_SHAPES = ((4, 8), (4, 8, 16), (4, 8, 16, 32))
PRED_RANDOM_FLOAT_TENSORS = tuple(get_random_floats(shape) for shape in FLOAT_IN_SHAPES)
TARGET_RANDOM_FLOAT_TENSORS = tuple(
    get_random_floats(shape, req_grad=False) for shape in FLOAT_IN_SHAPES
)
INT_IN_SHAPES = ((4,), (4, 8), (4, 8, 16))
INT_RANGES = (8, 16, 32)
TARGET_RANDOM_INT_TENSORS = tuple(
    get_random_ints(shape, 0, range) for shape, range in zip(INT_IN_SHAPES, INT_RANGES)
)
REDUCTIONS = ("mean", "sum")


@pytest.mark.parametrize("pred", PRED_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("target", TARGET_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_mse_loss(
    pred: tuple[ac.Tensor, torch.Tensor], target: tuple[ac.Tensor, torch.Tensor], reduction: str
) -> None:
    # skip tests with shape mismatch
    if pred[0].shape != target[0].shape:
        return
    get_op_test("mse_loss")((pred, target), reduction=reduction)


@pytest.mark.parametrize("pred", PRED_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("target", TARGET_RANDOM_INT_TENSORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_cross_entropy_loss(
    pred: tuple[ac.Tensor, torch.Tensor], target: tuple[ac.Tensor, torch.Tensor], reduction: str
) -> None:
    ac_pred, torch_pred = pred
    ac_target, torch_target = target

    # skip tests with shape mismatch
    if ac_pred.shape[:-1] != ac_target.shape:
        return

    ac_loss = ac.nn.functional.cross_entropy_loss(ac_pred, ac_target, reduction=reduction)

    # torch requires inputs to be  of shape (B, C, ...)
    permutation = (0, ac_pred.ndim - 1, *tuple(d for d in range(ac_pred.ndim - 1) if d > 0))
    torch_loss = torch.nn.functional.cross_entropy(
        torch_pred.permute(*permutation), torch_target, reduction=reduction
    )
    verify_op((ac_pred,), ac_loss, (torch_pred,), torch_loss)


@pytest.mark.parametrize("pred", PRED_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("target", TARGET_RANDOM_FLOAT_TENSORS)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_bce_loss(
    pred: tuple[ac.Tensor, torch.Tensor], target: tuple[ac.Tensor, torch.Tensor], reduction: str
) -> None:
    # skip tests with shape mismatch
    if pred[0].shape != target[0].shape:
        return
    get_op_test("bce_loss", "binary_cross_entropy_with_logits")((pred, target), reduction=reduction)
