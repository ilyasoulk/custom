import torch
import pytest
import custom.ops as ops

@pytest.mark.parametrize("m, k, d", [
    (128, 128, 128),
    (256, 512, 128),
    (10, 10, 10),
])
def test_matmul(m, k, d):
    a = torch.randn(m, k, device='cuda')
    b = torch.randn(k, d, device='cuda')

    out_custom = ops.matmul(a, b) 
    out_torch = torch.matmul(a, b)

    torch.testing.assert_close(out_custom, out_torch, atol=1e-4, rtol=1e-3)