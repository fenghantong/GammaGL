#include <torch/torch.h>
std::tuple<torch::Tensor, torch::Tensor> segment_mean_cuda_forward(torch::Tensor x, torch::Tensor index, int64_t N);