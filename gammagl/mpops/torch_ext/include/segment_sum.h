#include <torch/torch.h>

class SegmentSum : public torch::autograd::Function<SegmentSum> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor x,
                               torch::Tensor index,
                               int64_t N);
  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outs);
};