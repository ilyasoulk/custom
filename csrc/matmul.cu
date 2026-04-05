#include <torch/extension.h>


__global__ void matmul_kernel(const float *a, 
    const float *b, float *out, const int M, const int K, const int D) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx >= M || col_idx >= D) {
        return;
    }

    float value = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_element = a[row_idx * K + k];
        float b_element = b[k * D + col_idx];

        value += a_element * b_element;
    }

    out[row_idx * D + col_idx] = value;
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0);
  int K = a.size(1);
  int D = b.size(1);
  auto out = torch::empty({M, D}, a.options());
  dim3 threads(4, 4);

  int grid_x = (D + threads.x - 1) / threads.x;
  int grid_y = (M + threads.y - 1) / threads.y;

  dim3 blocks(grid_x, grid_y);
  matmul_kernel<<<blocks, threads>>>(
    a.data_ptr<float>(), 
    b.data_ptr<float>(), 
    out.data_ptr<float>(),
    M, K, D
    );

  return out;
}
