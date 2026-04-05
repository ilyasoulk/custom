import torch
import custom._C as _C

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    return _C.matmul(a, b)