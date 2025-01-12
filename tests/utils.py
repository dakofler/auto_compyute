import numpy as np
import torch

import auto_compyute as ac


def close(ac_in, torch_in):
    return np.allclose(ac_in, torch_in.detach().numpy())


def get_data(shape):
    x = ac.randn(shape, requires_grad=True)
    torch_x = torch.tensor(x.data, requires_grad=True)
    return x, torch_x
