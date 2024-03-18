#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor binarizePM1FI_cuda(
    torch::Tensor input,
    float f01,
    float f10,
    std::vector<std::vector<double>> index_offset,
    float block_size
  );

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor binarizePM1FI(
    torch::Tensor input,
    float f01,
    float f10,
    std::vector<std::vector<double>> index_offset,
    float block_size
  ) {
  CHECK_INPUT(input);
  return binarizePM1FI_cuda(input, f01, f10, index_offset, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binarizeFI", &binarizePM1FI, "BINARIZEPM1FI");
}
