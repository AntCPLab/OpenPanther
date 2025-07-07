#include "experimental/ann/bitwidth_change_prot.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace panther {
BitWidthChangeProtocol::BitWidthChangeProtocol(
    const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base)
    : basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
}
spu::NdArrayRef BitWidthChangeProtocol::TrunReduceCompute(
    const spu::NdArrayRef& inp, size_t bw, size_t shift_bits) {
  if (shift_bits > bw) {
    SPDLOG_INFO("Can't shift {} bits from {} bits", shift_bits, bw);
    return inp;
  }
  if (shift_bits == 0) return inp;
  const int rank = basic_ot_prot_->Rank();
  const auto field = inp.eltype().as<spu::Ring2k>()->field();
  size_t cmp_bits = std::min(shift_bits, size_t(8));
  spu::mpc::cheetah::CompareProtocol compare_prot(basic_ot_prot_, cmp_bits);

  spu::NdArrayRef wrap_bool;
  using namespace spu;

  if (rank == 0) {
    auto adjusted = spu::mpc::ring_bitmask(inp, 0, shift_bits);
    // auto adjusted = inp;
    wrap_bool = compare_prot.Compute(adjusted, true, shift_bits);
  } else {
    auto adjusted = spu::mpc::ring_neg(inp);
    DISPATCH_ALL_FIELDS(field, "wrap_adjust", [&]() {
      NdArrayView<ring2k_t> xadj(adjusted);
      ring2k_t mask = static_cast<ring2k_t>(1 << shift_bits) - 1;
      pforeach(0, inp.numel(), [&](int64_t i) {
        xadj[i] -= 1;
        xadj[i] &= mask;
      });
    });
    wrap_bool = compare_prot.Compute(adjusted, true, shift_bits);
  }

  spu::NdArrayRef wrap_a_share = basic_ot_prot_->B2ASingleBitWithSize(
      wrap_bool.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1)),
      bw - shift_bits);

  auto local_shift = spu::mpc::ring_rshift(inp, shift_bits);
  spu::mpc::ring_add_(local_shift, wrap_a_share);

  return local_shift;
}

spu::NdArrayRef BitWidthChangeProtocol::ExtendCompute(
    const spu::NdArrayRef& inp, size_t bw, size_t extend_bw) {
  // Known MSB extend
  const int rank = basic_ot_prot_->Rank();
  const auto field = inp.eltype().as<spu::Ring2k>()->field();
  const int64_t numel = inp.numel();
  // const size_t target_bw = bw + extend_bw;

  constexpr size_t N = 2;
  constexpr size_t nbits = 1;

  spu::NdArrayRef outp;
  if (0 == rank) {
    outp = spu::mpc::ring_randbit(field, inp.shape());
    std::vector<uint8_t> send(numel * N);
    using namespace spu;
    DISPATCH_ALL_FIELDS(field, "MSB0_adjush", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      NdArrayView<const u2k> xrnd(outp);
      // when msb(xA) = 0, set (r, 1^r)
      //  ow. msb(xA) = 1, set (1^r, 1^r)
      // Equals to (r^msb(xA), r^1)
      for (int64_t i = 0; i < numel; ++i) {
        send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (bw - 1)) & 1);
        send[2 * i + 1] = xrnd[i] ^ 1;
      }
    });
    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);
    using namespace spu;
    DISPATCH_ALL_FIELDS(field, "MSB0_adjust", [&]() {
      using u2k = std::make_unsigned<ring2k_t>::type;
      NdArrayView<const u2k> xinp(inp);
      for (int64_t i = 0; i < numel; ++i) {
        choices[i] = (xinp[i] >> (bw - 1)) & 1;
      }
    });
    std::vector<uint8_t> recv(numel);
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                               absl::MakeSpan(recv), nbits);

    outp = spu::mpc::ring_zeros(field, inp.shape());
    DISPATCH_ALL_FIELDS(field, "MSB0_finalize", [&]() {
      NdArrayView<ring2k_t> xoup(outp);
      pforeach(0, numel, [&](int64_t i) {
        xoup[i] = static_cast<ring2k_t>(recv[i] & 1);
      });
    });
  }
  auto wrap_bool = basic_ot_prot_->B2ASingleBitWithSize(
      outp.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1)),
      static_cast<int>(extend_bw + 1));
  spu::mpc::ring_lshift_(wrap_bool, bw);
  auto extend_result = spu::mpc::ring_sub(inp, wrap_bool);
  return extend_result;
}

spu::NdArrayRef BitWidthChangeProtocol::ExtendComputeOpt(
    const spu::NdArrayRef& inp, size_t bw, size_t extend_bw) {
  // Known MSB extend
  // If [x] > 0, the MSB is 0
  // [x] = MSB||x_l
  // the wrap bit of low bits x_l will be [msb]_0 ^ [msb]_1

  const auto field = inp.eltype().as<spu::Ring2k>()->field();

  spu::NdArrayRef outp = spu::mpc::ring_rshift(inp, bw - 1);

  auto wrap_bool = basic_ot_prot_->B2ASingleBitWithSize(
      outp.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1)),
      static_cast<int>(extend_bw + 1));
  spu::mpc::ring_lshift_(wrap_bool, bw - 1);
  auto extend_result = spu::mpc::ring_bitmask(inp, 0, bw - 1);
  spu::mpc::ring_sub_(extend_result, wrap_bool);
  return extend_result;
}

}  // namespace panther