"""Test factory functions."""

from types import ModuleType
from typing import Any, Callable, Optional

import torch

import auto_compyute as ac
from tests.utils.verifications import DEFAULT_TOL, verify_op


def _get_func_result(module: ModuleType, func_name: str, *args: Any, **kwargs: Any) -> Callable:
    """Selects the function or method and calls with given args and kwargs."""

    # specified function is a function of the functional module
    if hasattr(module.nn.functional, func_name):
        return getattr(module.nn.functional, func_name)(*args, **kwargs)

    # specified function is a Tensor method
    tensor = args[0]
    if hasattr(tensor, func_name):
        return getattr(tensor, func_name)(*args[1:], **kwargs)

    # specified function is a function of the module
    if hasattr(module, func_name):
        return getattr(module, func_name)(*args, **kwargs)

    raise AttributeError("Unknown function.")


def get_op_test(
    ac_func_name: str, torch_func_name: Optional[str] = None, tol: float = DEFAULT_TOL
) -> Callable[[Any], None]:
    """Returns a test function for one input tensor."""
    torch_func_name = torch_func_name or ac_func_name

    def _test(xs: tuple[tuple[ac.Tensor, torch.Tensor], ...], *args: Any, **kwargs: Any) -> None:
        ac_x, torch_x = tuple(zip(*xs))  # get tuple of ac tensors and tuple of torch tensors
        ac_y = _get_func_result(ac, ac_func_name, *ac_x, *args, **kwargs)
        torch_y = _get_func_result(torch, torch_func_name, *torch_x, *args, **kwargs)
        verify_op(ac_x, ac_y, torch_x, torch_y, tol=tol)

    return _test
