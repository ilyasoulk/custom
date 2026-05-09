#include <torch/extension.h>

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor w, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul_cuda, "Custom CUDA Matrix Multiplication");
  m.def("rms_norm", &rms_norm_cuda, "Custom CUDA RMS Norm");
}
