#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
namespace panther {

// Implementation the one-bit approximate truncation with reduction

/**
 * Truncation with Reduction and Extension:
 * Reference: " SIRNN : A Math Library for Secure RNN Inference"
 * (https://eprint.iacr.org/2021/459.pdf)
 *
 * When computing wrap bits in extension protocol, we use the known MSB protocol
 * from:
 * Reference: "Cheetah : Lean and Fast Secure Two-Party Deep Neural
 * Network Inference" (https://eprint.iacr.org/2022/207.pdf)
 *
 * We optimize the communication rounds in opt function:
 * More details please refer to Section 6.2 "Efficient bitwidth conversion" in
 * Panther
 *
 */

class BitwidthAdjustProtocol {
 public:
  explicit BitwidthAdjustProtocol(
      const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> &base);

  spu::NdArrayRef TrunReduceCompute(const spu::NdArrayRef &inp, size_t bw,
                                    size_t shift_bits);

  spu::NdArrayRef ExtendCompute(const spu::NdArrayRef &inp, size_t bw,
                                size_t extend_bw);

  spu::NdArrayRef ExtendComputeOpt(const spu::NdArrayRef &inp, size_t bw,
                                   size_t extend_bw);

 private:
  std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> basic_ot_prot_ = nullptr;
};

}  // namespace panther