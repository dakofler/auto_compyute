"""Test factory functions."""

from typing import Any, Callable

import torch

import auto_compyute as ac
from tests.utils.check import dual_input_op_check, single_input_op_check, triple_input_op_check


def get_unary_test_func(func_name: str) -> Callable[[Any], None]:
    """Returns a test function for one input."""

    def _test(x: tuple[ac.Tensor, torch.Tensor], *args, **kwargs: Any) -> None:
        ac_x, torch_x = x

        if hasattr(ac_x, func_name):
            ac_y = getattr(ac_x, func_name)(*args, **kwargs)
        elif hasattr(ac.nn.functional, func_name):
            ac_y = getattr(ac.nn.functional, func_name)(ac_x, *args, **kwargs)
        elif hasattr(ac, func_name):
            ac_y = getattr(ac, func_name)(ac_x, *args, **kwargs)
        else:
            raise AttributeError("Unknown function.")

        if hasattr(torch_x, func_name):
            torch_y = getattr(torch_x, func_name)(*args, **kwargs)
        elif hasattr(torch.nn.functional, func_name):
            torch_y = getattr(torch.nn.functional, func_name)(torch_x, *args, **kwargs)
        elif hasattr(torch, func_name):
            torch_y = getattr(torch, func_name)(torch_x, *args, **kwargs)
        else:
            raise AttributeError("Unknown function.")

        single_input_op_check(ac_x, ac_y, torch_x, torch_y)

    return _test


def get_binary_test_func(func_name: str) -> Callable[[Any], None]:
    """Returns a test function for one input that uses module functions."""

    def _test(
        x_1: tuple[ac.Tensor, torch.Tensor],
        x_2: tuple[ac.Tensor, torch.Tensor],
        *args,
        **kwargs: Any,
    ) -> None:
        (ac_x_1, torch_x_1), (ac_x_2, torch_x_2) = x_1, x_2

        if hasattr(ac_x_1, func_name):
            ac_y = getattr(ac_x_1, func_name)(ac_x_2, *args, **kwargs)
        elif hasattr(ac.nn.functional, func_name):
            ac_y = getattr(ac.nn.functional, func_name)(ac_x_1, ac_x_2, *args, **kwargs)
        elif hasattr(ac, func_name):
            ac_y = getattr(ac, func_name)(ac_x_1, ac_x_2, *args, **kwargs)
        else:
            raise AttributeError("Unknown function.")

        if hasattr(torch_x_1, func_name):
            torch_y = getattr(torch_x_1, func_name)(torch_x_2, *args, **kwargs)
        elif hasattr(torch.nn.functional, func_name):
            torch_y = getattr(torch.nn.functional, func_name)(torch_x_1, torch_x_2, *args, **kwargs)
        elif hasattr(torch, func_name):
            torch_y = getattr(torch, func_name)(torch_x_1, torch_x_2, *args, **kwargs)
        else:
            raise AttributeError("Unknown function.")

        dual_input_op_check(ac_x_1, ac_x_2, ac_y, torch_x_1, torch_x_2, torch_y)

    return _test


def get_tertiary_test_func(func_name: str) -> Callable[[Any], None]:
    """Returns a test function for three inputs that uses module functions."""

    def _test(
        x_1: tuple[ac.Tensor, torch.Tensor],
        x_2: tuple[ac.Tensor, torch.Tensor],
        x_3: tuple[ac.Tensor, torch.Tensor],
        *args,
        **kwargs: Any,
    ) -> None:
        (ac_x_1, torch_x_1), (ac_x_2, torch_x_2), (ac_x_3, torch_x_3) = x_1, x_2, x_3

        if hasattr(ac_x_1, func_name):
            ac_y = getattr(ac_x_1, func_name)(ac_x_2, ac_x_3, *args, **kwargs)
        elif hasattr(ac.nn.functional, func_name):
            ac_y = getattr(ac.nn.functional, func_name)(ac_x_1, ac_x_2, ac_x_3, *args, **kwargs)
        elif hasattr(ac, func_name):
            ac_y = getattr(ac, func_name)(ac_x_1, ac_x_2, ac_x_3, *args, **kwargs)
        else:
            raise AttributeError("Unknown function.")

        if hasattr(torch_x_1, func_name):
            torch_y = getattr(torch_x_1, func_name)(torch_x_2, torch_x_3, *args, **kwargs)
        elif hasattr(torch.nn.functional, func_name):
            torch_y = getattr(torch.nn.functional, func_name)(
                torch_x_1, torch_x_2, torch_x_3, *args, **kwargs
            )
        elif hasattr(torch, func_name):
            torch_y = getattr(torch, func_name)(torch_x_1, torch_x_2, torch_x_3, *args, **kwargs)
        else:
            raise AttributeError("Unknown function.")

        triple_input_op_check(
            ac_x_1, ac_x_2, ac_x_3, ac_y, torch_x_1, torch_x_2, torch_x_3, torch_y
        )

    return _test


def get_min_max_test_func(func_name: str) -> Callable[[Any], None]:
    def _test(
        x: tuple[ac.Tensor, torch.Tensor], dim: int | tuple[int, ...] | None, keepdims: bool
    ) -> None:
        if dim is None and keepdims:
            return
        ac_x, torch_x = x
        ac_y = getattr(ac_x, func_name)(dim=dim, keepdims=keepdims)
        if dim is None:
            torch_y = getattr(torch_x, func_name)()
        else:
            torch_y = getattr(torch_x, func_name)(dim=dim, keepdims=keepdims)[0]
        single_input_op_check(ac_x, ac_y, torch_x, torch_y)

    return _test
