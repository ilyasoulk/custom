import torch
import torch.utils.benchmark as benchmark
import custom.ops as ops

def run_benchmark(m, k, d):
    a = torch.randn(m, k, device='cuda')
    b = torch.randn(k, d, device='cuda')

    t_custom = benchmark.Timer(
        stmt='ops.matmul(a, b)',
        globals={'a': a, 'b': b, 'ops': ops},
        label='MatMul',
        sub_label=f'{m}x{k}x{d}',
        description='Custom CUDA'
    )

    t_torch = benchmark.Timer(
        stmt='torch.matmul(a, b)',
        globals={'a': a, 'b': b},
        label='MatMul',
        sub_label=f'{m}x{k}x{d}',
        description='PyTorch (cuBLAS)'
    )

    print(t_custom.timeit(100))
    print(t_torch.timeit(100))

if __name__ == '__main__':
    run_benchmark(1024, 1024, 1024)