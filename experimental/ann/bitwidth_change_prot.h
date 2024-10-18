#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
namespace sanns {

// class BasicOTProtocols;

// Implementation the one-bit approximate truncation
// Ref: Huang et al. "Cheetah: Lean and Fast Secure Two-Party Deep Neural
// Network Inference"
//  https://eprint.iacr.org/2022/207.pdf
//
// [(x >> s) + e]_A <- Truncate([x]_A, s) with |e| <= 1 probabilistic error
//
// Math:
//   Given x = x0 + x1 mod 2^k
//   x >> s \approx (x0 >> s) + (x1 >> s) - w * 2^{k - s} mod 2^k
//   where w = 1{x0 + x1 > 2^{k} - 1} indicates whether the sum wrap round 2^k
class BitWidthChangeProtocol {
 public:
  explicit BitWidthChangeProtocol(
      const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> &base);

  // ~BitWidthChangeProtocol();

  spu::NdArrayRef TrunReduceCompute(const spu::NdArrayRef &inp, size_t bw,
                                    size_t shift_bits);

  spu::NdArrayRef ExtendCompute(const spu::NdArrayRef &inp, size_t bw,
                                size_t extend_bw);

  spu::NdArrayRef ExtendComputeOpt(const spu::NdArrayRef &inp, size_t bw,
                                   size_t extend_bw);

 private:
  std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> basic_ot_prot_ = nullptr;
};

}  // namespace sanns