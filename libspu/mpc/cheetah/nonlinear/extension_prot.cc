// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "libspu/mpc/cheetah/nonlinear/extension_prot.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/util.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
ExtensionProtocol::ExtensionProtocol(std::shared_ptr<BasicOTProtocols> base)
    : basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
}

ExtensionProtocol::~ExtensionProtocol() { basic_ot_prot_->Flush(); }

ArrayRef ExtensionProtocol::ZeroExtend(const ArrayRef& inp, const Meta& meta) {
  const auto src_field = inp.eltype().as<Ring2k>()->field();
  size_t src_width = SizeOf(src_field) * 8;
  size_t dst_width = SizeOf(meta.dst_field) * 8;
  SPU_ENFORCE(src_width < dst_width);

  ArrayRef wrap = ComputeWrap(inp, meta);

  ArrayRef out = ring_zeros(meta.dst_field, inp.numel());
  DISPATCH_ALL_FIELDS(src_field, "", [&]() {
    using T0 = ring2k_t;
    ArrayView<const T0> input(inp);
    DISPATCH_ALL_FIELDS(meta.dst_field, "", [&]() {
      using T1 = ring2k_t;
      ArrayView<const T1> w(wrap);
      ArrayView<T1> output(out);
      const T1 shift = static_cast<T1>(1) << src_width;
      const T1 mask = (static_cast<T1>(1) << (dst_width - src_width)) - 1;

      pforeach(0, input.numel(), [&](int64_t i) {
        output[i] = static_cast<T1>(input[i]) - shift * (w[i] & mask);
      });
    });
  });

  return out;
}

ArrayRef ExtensionProtocol::Compute(const ArrayRef& inp, const Meta& _meta) {
  const auto src_field = inp.eltype().as<Ring2k>()->field();
  size_t src_width = SizeOf(src_field) * 8;
  size_t dst_width = SizeOf(_meta.dst_field) * 8;
  SPU_ENFORCE(src_width < dst_width);

  const int rank = basic_ot_prot_->Rank();

  if (_meta.signed_arith && _meta.msb == MSB_st::unknown &&
      _meta.use_heuristic) {
    Meta meta = _meta;
    // Use heuristic optimization from SecureQ8: Add a large positive to make
    // sure the value is always positive
    meta.use_heuristic = false;
    meta.msb = MSB_st::zero;

    auto field = inp.eltype().as<Ring2k>()->field();
    size_t bit_width = SizeOf(field) * 8;

    if (rank == 0) {
      ArrayRef tmp = inp.clone();
      DISPATCH_ALL_FIELDS(field, "", [&] {
        ArrayView<ring2k_t> _inp(tmp);
        ring2k_t big_value = static_cast<ring2k_t>(1)
                             << (bit_width - kHeuristicBound);
        pforeach(0, inp.numel(),
                 [&](int64_t i) { _inp[i] = _inp[i] + big_value; });
      });

      tmp = Compute(tmp, meta);

      DISPATCH_ALL_FIELDS(meta.dst_field, "", [&] {
        ArrayView<ring2k_t> _outp(tmp);
        ring2k_t big_value = static_cast<ring2k_t>(1)
                             << (bit_width - kHeuristicBound);
        pforeach(0, inp.numel(),
                 [&](int64_t i) { _outp[i] = _outp[i] - big_value; });
      });
      return tmp;
    } else {
      return Compute(inp, meta);
    }
  }

  Meta meta = _meta;

  if (_meta.signed_arith && _meta.msb != MSB_st::unknown) {
    meta.msb = _meta.msb == MSB_st::zero ? MSB_st::one : MSB_st::zero;
  }

  ArrayRef out = DISPATCH_ALL_FIELDS(src_field, "", [&]() {
    using T0 = ring2k_t;
    ArrayView<const ring2k_t> xinp(inp);

    if (meta.signed_arith && rank == 0) {
      const T0 component = (static_cast<T0>(1) << (src_width - 1));
      // SExt(x, m, n) = ZExt(x + 2^{m-1} mod 2^m, m, n) - 2^{m-1}
      auto tmp = ring_zeros(src_field, inp.numel());
      ArrayView<T0> xtmp(tmp);
      pforeach(0, inp.numel(),
               [&](int64_t i) { xtmp[i] = xinp[i] + component; });
      return ZeroExtend(tmp, meta);
    } else {
      return ZeroExtend(inp, meta);
    }
  });

  if (meta.signed_arith && rank == 0) {
    DISPATCH_ALL_FIELDS(meta.dst_field, "", [&]() {
      const auto component = (static_cast<ring2k_t>(1) << (src_width - 1));
      // SExt(x, m, n) = ZExt(x + 2^{m-1} mod 2^m, m, n) - 2^{m-1}
      ArrayView<ring2k_t> xout(out);
      pforeach(0, inp.numel(), [&](int64_t i) { xout[i] -= component; });
    });
  }

  return out;
}

ArrayRef ExtensionProtocol::ComputeWrap(const ArrayRef& inp, const Meta& meta) {
  const int rank = basic_ot_prot_->Rank();
  const auto src_field = inp.eltype().as<Ring2k>()->field();

  switch (meta.msb) {
    case MSB_st::zero: {
      return MSB0ToWrap(inp, meta);
      break;
    }
    case MSB_st::one: {
      return MSB1ToWrap(inp, meta);
      break;
    }
    default:
      break;
  }

  CompareProtocol compare_prot(basic_ot_prot_);
  ArrayRef wrap_bool;
  // w = 1{x_A + x_B > 2^k - 1}
  //   = 1{x_A > 2^k - 1 - x_B}
  if (rank == 0) {
    wrap_bool = compare_prot.Compute(inp, true);
  } else {
    auto adjusted = ring_neg(inp);
    DISPATCH_ALL_FIELDS(src_field, "", [&]() {
      ArrayView<ring2k_t> xadj(adjusted);
      pforeach(0, inp.numel(), [&](int64_t i) { xadj[i] -= 1; });
    });
    wrap_bool = compare_prot.Compute(adjusted, true);
  }

  ArrayRef ext_wrap = ring_zeros(meta.dst_field, inp.numel());
  DISPATCH_ALL_FIELDS(src_field, "", [&]() {
    using T0 = ring2k_t;
    ArrayView<const T0> w(wrap_bool);
    DISPATCH_ALL_FIELDS(meta.dst_field, "", [&]() {
      using T1 = ring2k_t;
      ArrayView<T1> ext_w(ext_wrap);
      pforeach(0, inp.numel(),
               [&](int64_t i) { ext_w[i] = static_cast<T1>(w[i]); });
    });
  });

  return basic_ot_prot_->B2A(
      ext_wrap.as(makeType<semi2k::BShrTy>(meta.dst_field, 1)));
}

// Given msb(xA + xB mod 2^k) = 0, and xA, xB \in [0, 2^k)
// To compute w0, w1 \in [0, 2^k') such that w0 + w1 = 1{xA + xB > 2^{k} - 1}
// mod 2^k'
//
// Given msb(xA + xB mod 2^k) = 0
//   1. when xA + xB = x => w = 0
//   2. when xA + xB = x + 2^{k} => w = 1
//   For case 1: msb(xA) = msb(xB) = 0 or msb(xA) = msb(xB) = 1
//   For case 2: msb(xA) = 1 or msb(xB) = 1.
// Thus w = msb(xA) | msb(xB)
//
// 1-of-2 OT msg (r^msb(xA), r^1) on choice msb(xB)
//   - msb(xB) = 0: get (r, r^msb(xA)) => msb(xA)
//   - msb(xB) = 1: get (r, r^1) => 1
ArrayRef ExtensionProtocol::MSB0ToWrap(const ArrayRef& inp, const Meta& meta) {
  const auto src_field = inp.eltype().as<Ring2k>()->field();
  const size_t src_width = SizeOf(src_field) * 8;
  const size_t dst_width = SizeOf(meta.dst_field) * 8;
  SPU_ENFORCE(src_width < dst_width);

  const size_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();

  constexpr size_t N = 2;  // 1-of-2 OT
  constexpr size_t nbits = 1;

  ArrayRef outp;

  if (0 == rank) {
    outp = ring_randbit(meta.dst_field, numel);
    std::vector<uint8_t> send(numel * N);

    DISPATCH_ALL_FIELDS(src_field, "", [&]() {
      using T0 = std::make_unsigned<ring2k_t>::type;
      ArrayView<const T0> xinp(inp);
      DISPATCH_ALL_FIELDS(meta.dst_field, "", [&]() {
        using T1 = std::make_unsigned<ring2k_t>::type;
        ArrayView<const T1> xrnd(outp);
        // when msb(xA) = 0, set (r, 1^r)
        //  ow. msb(xA) = 1, set (1^r, 1^r)
        // Equals to (r^msb(xA), r^1)
        for (size_t i = 0; i < numel; ++i) {
          send[2 * i + 0] = xrnd[i] ^ ((xinp[i] >> (src_width - 1)) & 1);
          send[2 * i + 1] = xrnd[i] ^ 1;
        }
      });
    });

    auto sender = basic_ot_prot_->GetSenderCOT();
    sender->SendCMCC(absl::MakeSpan(send), N, nbits);
    sender->Flush();
  } else {
    std::vector<uint8_t> choices(numel, 0);

    DISPATCH_ALL_FIELDS(src_field, "", [&]() {
      using T0 = std::make_unsigned<ring2k_t>::type;
      ArrayView<const T0> xinp(inp);
      for (size_t i = 0; i < numel; ++i) {
        choices[i] = (xinp[i] >> (src_width - 1)) & 1;
      }

      std::vector<uint8_t> recv(numel);
      basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(choices), N,
                                                 absl::MakeSpan(recv), nbits);

      outp = ring_zeros(meta.dst_field, numel);

      DISPATCH_ALL_FIELDS(meta.dst_field, "", [&]() {
        using T1 = std::make_unsigned<ring2k_t>::type;
        ArrayView<T1> xoup(outp);
        pforeach(0, numel,
                 [&](int64_t i) { xoup[i] = static_cast<T1>(recv[i] & 1); });
      });
    });
  }

  return basic_ot_prot_->B2A(
      outp.as(makeType<semi2k::BShrTy>(meta.dst_field, 1)));
}

// Given msb(xA + xB mod 2^k) = 1, and xA, xB \in [0, 2^k)
// To compute w0, w1 \in [0, 2^k') such that w0 + w1 = 1{xA + xB > 2^{k} - 1}
// mod 2^k'
//            w = msb(xA) & msb(xB).
// COT msg corr=msb(xA) on choice msb(xB)
//    - msb(xB) = 0: get(-x, x) => 0
//    - msb(xB) = 1: get(-x, x + msb(xA)) => msb(xA)
ArrayRef ExtensionProtocol::MSB1ToWrap(const ArrayRef& inp, const Meta& meta) {
  const auto src_field = inp.eltype().as<Ring2k>()->field();
  const auto dst_field = meta.dst_field;
  const size_t numel = inp.numel();
  const int rank = basic_ot_prot_->Rank();
  const size_t src_width = SizeOf(src_field) * 8;

  ArrayRef cot_output = ring_zeros(dst_field, numel);

  DISPATCH_ALL_FIELDS(src_field, "", [&]() {
    using T0 = std::make_unsigned<ring2k_t>::type;
    ArrayView<const T0> xinp(inp);

    DISPATCH_ALL_FIELDS(dst_field, "", [&]() {
      using T1 = std::make_unsigned<ring2k_t>::type;
      ArrayView<T1> xout(cot_output);

      if (rank == 0) {
        std::vector<T1> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = ((xinp[i] >> (src_width - 1)) & 1);
        });

        auto sender = basic_ot_prot_->GetSenderCOT();
        sender->SendCAMCC(absl::MakeSpan(cot_input),
                          {xout.data(), (size_t)xout.numel()});
        sender->Flush();
        pforeach(0, numel, [&](int64_t i) { xout[i] = -xout[i]; });
      } else {
        std::vector<uint8_t> cot_input(numel);
        pforeach(0, numel, [&](int64_t i) {
          cot_input[i] = ((xinp[i] >> (src_width - 1)) & 1);
        });

        basic_ot_prot_->GetReceiverCOT()->RecvCAMCC(
            absl::MakeSpan(cot_input), {xout.data(), (size_t)xout.numel()});
      }
    });
  });

  return cot_output.as(makeType<semi2k::BShrTy>(dst_field, 1));
}

}  // namespace spu::mpc::cheetah