#include <torch/extension.h>
#define TILE_SIZE 32


__global__ void matmul_kernel(const float *a, 
    const float *b, float *out, const int M, const int K, const int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;

    float value = 0.0f;
    for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; ++i) {


      if (row_idx < M && (i * TILE_SIZE + threadIdx.x) < K) {
        sA[threadIdx.y][threadIdx.x] = a[row_idx * K + i * TILE_SIZE + threadIdx.x];
      } else {
        sA[threadIdx.y][threadIdx.x] = 0.0f;
      }

      if (col_idx < N && (i * TILE_SIZE + threadIdx.y) < K) {
        sB[threadIdx.y][threadIdx.x] = b[(i * TILE_SIZE + threadIdx.y) * N + col_idx];
      } else {
        sB[threadIdx.y][threadIdx.x] = 0.0f;
      }

      __syncthreads();

      for (int k = 0; k < TILE_SIZE; ++k) {
        float a_element = sA[threadIdx.y][k];
        float b_element = sB[k][threadIdx.x];

        value += a_element * b_element;
      }

      __syncthreads();
    }
    if (row_idx < M && col_idx < N) {
      out[row_idx * N + col_idx] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);
  auto out = torch::empty({M, N}, a.options());
  dim3 threads(32, 32);

  int grid_x = (N + threads.x - 1) / threads.x;
  int grid_y = (M + threads.y - 1) / threads.y;

  dim3 blocks(grid_x, grid_y);
  matmul_kernel<<<blocks, threads>>>(
    a.data_ptr<float>(), 
    b.data_ptr<float>(), 
    out.data_ptr<float>(),
    M, K, N
    );

  return out;
}
