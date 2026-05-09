# Custom Deep Learning CUDA Kernels

This repository contains custom, highly optimized PyTorch C++/CUDA extensions implementing core operations used in modern Deep Learning architectures. The ultimate goal of this project is to build a minimal, high-performance Transformer model (LLaMA-style) completely from scratch, substituting standard PyTorch modules with these custom hardware-accelerated kernels.

## Project Structure

```text
.
├── benchmarks/              # Benchmarking scripts against native PyTorch (cuBLAS/ATen)
│   ├── matmul_bench.py
│   └── rms_bench.py
├── custom/                  # Python module wrapping the C++ extensions
│   ├── __init__.py
│   └── ops.py
├── csrc/                    # Raw C++ and CUDA source files
│   ├── matmul.cu
│   ├── rms.cu
│   └── bindings.cpp         # PyTorch pybind11 bindings
├── tests/                   # Pytest suite for correctness validation
│   ├── test_matmul.py
│   └── test_rms.py
├── pyproject.toml           # Build system config and dependency management
├── setup.py                 # C++ extension compilation script
└── README.md
```

## Getting Started

This project uses `uv` for lightning-fast dependency management and builds.

**1. Install dependencies and compile the kernels (Editable Mode)**
```bash
uv pip install -e . --no-build-isolation
```
*Note: Run this command every time you modify the `.cu` or `.cpp` files to trigger `ninja` incremental recompilation.*

**2. Run correctness tests**
```bash
pytest tests/ -v
```

**3. Run benchmarks**
```bash
python benchmarks/matmul_bench.py
```

## Current MatMul Results

Benchmark target: `1024 x 1024 x 1024` FP32 matmul on an RTX 3080 with TF32 disabled in the benchmark.

| Kernel | Time |
| --- | ---: |
| Shared-memory tiled CUDA | 1.03 ms |
| Register-tiled CUDA (`64x64x64`, `4x4` outputs/thread) | 225.86 us |
| PyTorch cuBLAS | 149.96 us |

The register-tiled kernel currently compiles with `64` registers/thread, `16 KB` shared memory, and no register spills.

## Current RMSNorm Results

Benchmark target: FP32 RMSNorm on an RTX 3080.

| Shape | Custom CUDA | PyTorch eager | PyTorch builtin |
| --- | ---: | ---: | ---: |
| `B=1, S=1024, H=768` | 47.22 us | 282.07 us | 13.94 us |
| `B=1, S=1024, H=2048` | 60.79 us | 117.22 us | 42.20 us |
| `B=1, S=1024, H=4096` | 105.50 us | 217.71 us | 84.18 us |
| `B=4, S=2048, H=4096` | 735.43 us | 1.54 ms | 663.96 us |

The custom RMSNorm kernel beats PyTorch eager across the benchmarked shapes and is approaching the optimized PyTorch builtin path for larger hidden sizes.

## Roadmap & To-Do List

### Phase 1: Matrix Multiplication Deep Dive
- [x] Naive 2D Grid MatMul
- [x] Shared Memory Tiling (VRAM latency hiding)
- [x] Register Tiling / Thread Coarsening (Instruction-Level Parallelism)
- [ ] Tensor Cores Integration (using `wmma` API for mixed-precision FP16/FP32)

### Phase 2: Fused Element-Wise Kernels (Memory-Bound Ops)
Modern transformers are heavily bottlenecked by memory bandwidth during normal operations. Fusing these reduces VRAM trips.
- [x] **Fused RMSNorm:** Standard in LLaMA/Mistral architectures.
- [ ] **Fused SwiGLU Activation:** `x * sigmoid(beta * x) * W_v`. Combining this into one kernel saves massive memory bandwidth overhead.
- [ ] **Rotary Positional Embeddings (RoPE):** Fused complex number rotations applied directly to the query and key heads in a single pass.
- [ ] **Fused Cross-Entropy Loss:** Computing Softmax and Cross-Entropy in a single kernel without materializing the full logits matrix in VRAM.

### Phase 3: Attention Mechanisms
- [ ] **Flash Attention (Forward Pass):** Migrating Flash Attention concepts (tiling queries, keys, and values in SRAM) into raw C++.
- [ ] **Flash Attention (Backward Pass):** Handling gradients without materializing the $N \times N$ attention matrix.
- [ ] **PagedAttention :** Block-level memory management for KV-caching during auto-regressive inference.

### Phase 4: Full Transformer Integration
- [ ] Write a custom `nn.Module` for a Transformer Block using purely `custom.ops`.
- [ ] Instantiate a mini-LLaMA model and load pre-trained weights.
- [ ] Run a forward pass / text generation loop comparing native PyTorch performance vs. Custom Kernel performance.


### Phase 5: Fused Optimizers (Training Efficiency)
- [ ] **Fused AdamW:** Combining the step updates (weight, gradient, momentum, variance, weight decay) into a single VRAM read/write pass to overcome memory bandwidth bottlenecks.
- [ ] **Gradient Clipping:** Fusing the global norm calculation and the gradient scaling into the optimizer step.

### Phase 6: Quantization & Inference (Deployment)
- [ ] **Weight-Only Dequantization:** A kernel to load INT4/INT8 quantized weights and scale factors from VRAM, dequantizing them to FP16 in registers on-the-fly.
- [ ] **W4A16 / W8A16 MatMul:** Fusing the dequantization step directly into the Shared Memory Tiled Matrix Multiplication for ultra-fast LLM generation.
