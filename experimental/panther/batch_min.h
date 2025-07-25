#include "libspu/core/context.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"

namespace panther {
/**
 * This protocol is used to compute the minimum value among multiple
 * values in a batch.
 *
 * The comparison is implemented based on the Millionaire protocol in Cheetah
 * Reference: "Cheetah : Lean and Fast Secure Two-Party Deep Neural
 * Network Inference" (https://eprint.iacr.org/2022/207.pdf)
 */

class BatchMinProtocol {
 public:
  explicit BatchMinProtocol(const std::shared_ptr<spu::KernelEvalContext> &ctx,
                            size_t compare_radix);

  spu::NdArrayRef Compute(const spu::NdArrayRef &inp, const int64_t bitwidth,
                          const int64_t shift, const size_t batch_size,
                          const size_t max_size);

  std::vector<spu::NdArrayRef> ComputeWithIndex(const spu::NdArrayRef &inp,
                                                const spu::NdArrayRef &index,
                                                const int64_t bitwidth,
                                                const int64_t shift,
                                                const size_t batch_size,
                                                const size_t max_size);

 private:
  spu::NdArrayRef TruncAndReduce(const spu::NdArrayRef &inp, int64_t bitwidth,
                                 int64_t shift);
  spu::NdArrayRef Select(spu::NdArrayRef &select_bits, spu::NdArrayRef &a);
  spu::NdArrayRef DReLU(spu::NdArrayRef &inp, int64_t bitwidth);
  size_t compare_radix_;
  bool is_sender_{false};

  std::shared_ptr<spu::KernelEvalContext> ctx_;
};
}  // namespace panther