import torch

import custom._C as _C


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    return _C.matmul(a, b)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.is_cuda and w.is_cuda, "Inputs must be on GPU"
    assert x.is_contiguous() and w.is_contiguous(), "Inputs must be contiguous"
    return _C.rms_norm(x, w, eps)
