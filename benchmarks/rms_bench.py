import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import custom.ops as ops


def torch_eager_rms_norm(x, w, eps):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * w


def torch_builtin_rms_norm(x, w, eps):
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=w, eps=eps)


def run_benchmark(b, s, h, eps=1e-5):
    x = torch.randn(b, s, h, device="cuda", dtype=torch.float32).contiguous()
    w = torch.randn(h, device="cuda", dtype=torch.float32).contiguous()

    for _ in range(10):
        ops.rms_norm(x, w, eps)
        torch_eager_rms_norm(x, w, eps)
        torch_builtin_rms_norm(x, w, eps)

    torch.cuda.synchronize()

    t_custom = benchmark.Timer(
        stmt="ops.rms_norm(x, w, eps)",
        globals={"x": x, "w": w, "eps": eps, "ops": ops},
        label="RMSNorm",
        sub_label=f"B={b}, S={s}, H={h}",
        description="Custom CUDA",
    )

    t_eager = benchmark.Timer(
        stmt="torch_eager_rms_norm(x, w, eps)",
        globals={
            "x": x,
            "w": w,
            "eps": eps,
            "torch_eager_rms_norm": torch_eager_rms_norm,
        },
        label="RMSNorm",
        sub_label=f"B={b}, S={s}, H={h}",
        description="PyTorch eager",
    )

    t_builtin = benchmark.Timer(
        stmt="torch_builtin_rms_norm(x, w, eps)",
        globals={
            "x": x,
            "w": w,
            "eps": eps,
            "torch_builtin_rms_norm": torch_builtin_rms_norm,
        },
        label="RMSNorm",
        sub_label=f"B={b}, S={s}, H={h}",
        description="PyTorch builtin",
    )

    print(t_custom.timeit(100))
    print(t_eager.timeit(100))
    print(t_builtin.timeit(100))


if __name__ == "__main__":
    run_benchmark(1, 1024, 768)
    run_benchmark(1, 1024, 2048)
    run_benchmark(1, 1024, 4096)
    run_benchmark(4, 2048, 4096)
