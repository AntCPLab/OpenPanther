#include "libspu/core/context.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"

namespace sanns {
// class spu::mpc::cheetah::BasicOTProtocols;
class BatchArgmaxProtocol {
 public:
  explicit BatchArgmaxProtocol(
      const std::shared_ptr<spu::KernelEvalContext> &ctx, size_t compare_radix);

  spu::NdArrayRef Compute(const spu::NdArrayRef &inp, const int64_t bitwidth,
                          const int64_t shift, const size_t batch_size,
                          const size_t max_size);

  std::vector<spu::NdArrayRef> ComputeWithIndex(const spu::NdArrayRef &inp,
                                                const spu::NdArrayRef &index,
                                                const int64_t bitwidth,
                                                const size_t batch_size,
                                                const size_t max_size);

 private:
  spu::NdArrayRef Select(spu::NdArrayRef &select_bits, spu::NdArrayRef &a);
  spu::NdArrayRef DReLU(spu::NdArrayRef &inp, int64_t bitwidth);
  spu::NdArrayRef TruncValue(spu::NdArrayRef &inp, int64_t bitwidth,
                             int64_t shift);
  size_t compare_radix_;
  bool is_sender_{false};

  std::shared_ptr<spu::KernelEvalContext> ctx_;
};
}  // namespace sanns