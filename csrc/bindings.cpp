#include <torch/extension.h>

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul_cuda, "Custom CUDA Matrix Multiplication");
}