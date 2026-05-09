#include "ATen/ops/empty.h"
#include <cmath>
#include <torch/extension.h>
#define BLOCK_R 1
#define NUM_THREAD_PER_WARP 32


__global__ void rms_norm_kernel(const float *x, const float *w, float *out, const int H, const float eps) {
    int row_idx = blockIdx.x;
    int lane = threadIdx.x;

    float row_sum = 0.0f;

    for (int h = lane; h < H; h += NUM_THREAD_PER_WARP) {
        float xh = x[row_idx * H + h];
        row_sum += xh * xh;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }

    float scale = 0.0f;
    if (lane == 0) {
        scale = rsqrtf(row_sum/ static_cast<float>(H) + eps);
    }
    scale = __shfl_sync(0xffffffff, scale, 0);

    lane = threadIdx.x;
    for (int h = lane; h < H; h += NUM_THREAD_PER_WARP) {
        out[row_idx * H + h] = scale * x[row_idx * H + h] * w[h];
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor w, float eps) {
    int B = x.size(0);
    int S = x.size(1);
    int H = x.size(2);

    auto out = torch::empty_like(x);
    dim3 threads(32);
    dim3 blocks(B * S);

    rms_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        H,
        eps
    );

    return out;
}
