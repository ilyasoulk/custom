#include <torch/extension.h>
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define THREAD_M 4
#define THREAD_N 4

#define THREADS_X (BLOCK_N / THREAD_N)
#define THREADS_Y (BLOCK_M / THREAD_M)

__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

__global__ void matmul_kernel(const float *a,
    const float *b, float *out, const int M, const int K, const int N) {
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    int col_idx = blockIdx.x * BLOCK_N + threadIdx.x * THREAD_N;
    int row_idx = blockIdx.y * BLOCK_M + threadIdx.y * THREAD_M;

    float acc[THREAD_M][THREAD_N] = {0.0f};
    int num_threads = blockDim.x * blockDim.y;
    int num_a_values = BLOCK_M * BLOCK_K;
    int num_b_values = BLOCK_N * BLOCK_K;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    for (int k_tile = 0; k_tile < ceil_div(K, BLOCK_K); ++k_tile) {
        for (int offset = thread_id; offset < num_a_values; offset += num_threads) {
            int smem_row = offset / BLOCK_K;
            int smem_col = offset % BLOCK_K;

            int global_row = blockIdx.y * BLOCK_M + smem_row;
            int global_col = k_tile * BLOCK_K + smem_col;

            if (global_row < M && global_col < K) {
                sA[smem_row][smem_col] = a[global_row * K + global_col];
            } else {
                sA[smem_row][smem_col] = 0.0f;
            }
        }

        for (int offset = thread_id; offset < num_b_values; offset += num_threads) {
            int smem_row = offset / BLOCK_N;
            int smem_col = offset % BLOCK_N;

            int global_row = k_tile * BLOCK_K + smem_row;
            int global_col = blockIdx.x * BLOCK_N + smem_col;
            if (global_col < N && global_row < K) {
                sB[smem_row][smem_col] = b[global_row * N + global_col];
            } else {
                sB[smem_row][smem_col] = 0.0f;
            }
        }

      __syncthreads();

      for (int k = 0; k < BLOCK_K ; ++k) {
        #pragma unroll
        for (int m = 0; m < THREAD_M; ++m) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n)
                acc[m][n] += sA[threadIdx.y * THREAD_M + m][k] * sB[k][threadIdx.x * THREAD_N + n];
        }
      }

      __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        #pragma unroll
        for (int n = 0; n < THREAD_N; ++n)
            if (row_idx + m < M && col_idx + n < N) {
                out[(row_idx + m) * N + col_idx + n] = acc[m][n];
            }
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);
  auto out = torch::empty({M, N}, a.options());

  dim3 threads(THREADS_X, THREADS_Y);
  dim3 blocks(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));
  matmul_kernel<<<blocks, threads>>>(
    a.data_ptr<float>(),
    b.data_ptr<float>(),
    out.data_ptr<float>(),
    M, K, N
    );

  return out;
}
