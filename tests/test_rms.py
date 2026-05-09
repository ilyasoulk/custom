import pytest
import torch

import custom.ops as ops


def torch_rms_norm(x, w, eps):
    # x: (B, S, H)
    # w: (H,)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * w


@pytest.mark.parametrize(
    "b, s, h",
    [
        (1, 128, 128),
        (2, 256, 512),
        (4, 10, 10),
        (1, 16, 4096),
    ],
)
def test_rms_norm(b, s, h):
    eps = 1e-5

    x = torch.randn(b, s, h, device="cuda", dtype=torch.float32)
    w = torch.randn(h, device="cuda", dtype=torch.float32)

    out_custom = ops.rms_norm(x, w, eps)
    out_torch = torch_rms_norm(x, w, eps)

    torch.testing.assert_close(out_custom, out_torch, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("h", [1, 2, 7, 10, 31, 32, 33, 64, 128, 513])
def test_rms_norm_hidden_sizes(h):
    eps = 1e-5
    b, s = 2, 4

    x = torch.randn(b, s, h, device="cuda", dtype=torch.float32)
    w = torch.randn(h, device="cuda", dtype=torch.float32)

    out_custom = ops.rms_norm(x, w, eps)
    out_torch = torch_rms_norm(x, w, eps)

    torch.testing.assert_close(out_custom, out_torch, atol=1e-4, rtol=1e-4)
