// Copyright 2022 Ant Group Co., Ltd.
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

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/semi2k/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class ExtensionProtTest : public ::testing::TestWithParam<
                              std::tuple<FieldType, bool, std::string>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, ExtensionProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(true, false),
                     testing::Values("Unknown", "Zero", "One")),
    [](const testing::TestParamInfo<ExtensionProtTest::ParamType> &p) {
      return fmt::format("{}{}MSB{}", std::get<0>(p.param),
                         std::get<1>(p.param) ? "Signed" : "Unsigned",
                         std::get<2>(p.param));
    });

template <typename T>
bool SignBit(T x) {
  using uT = typename std::make_unsigned<T>::type;
  return (static_cast<uT>(x) >> (8 * sizeof(T) - 1)) & 1;
}

TEST_P(ExtensionProtTest, Basic) {
  size_t kWorldSize = 2;
  size_t n = 100000;
  FieldType field = std::get<0>(GetParam());
  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());
  ExtensionProtocol::MSB_st msb_t;

  FieldType dst_field;
  if (field == FieldType::FM32) {
    dst_field = FieldType::FM64;
  } else {
    dst_field = FieldType::FM128;
  }

  ArrayRef input[2];
  input[0] = ring_rand(field, n);
  if (msb == "Unknown") {
    input[1] = ring_rand(field, n);
    msb_t = ExtensionProtocol::MSB_st::unknown;
  } else {
    auto msg = ring_rand(field, n);
    DISPATCH_ALL_FIELDS(field, "", [&]() {
      size_t bw = SizeOf(field) * 8;
      ArrayView<ring2k_t> xmsg(msg);

      if (msb == "Zero") {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
        pforeach(0, n, [&](int64_t i) { xmsg[i] &= mask; });
        msb_t = ExtensionProtocol::MSB_st::zero;
      } else {
        ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1));
        pforeach(0, n, [&](int64_t i) { xmsg[i] |= mask; });
        msb_t = ExtensionProtocol::MSB_st::one;
      }
    });

    input[1] = ring_sub(msg, input[0]);
  }

  ArrayRef output[2];
  size_t sent_byte = 0;
  size_t sent_act = 0;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    ExtensionProtocol ext_prot(base);
    ExtensionProtocol::Meta meta;
    meta.signed_arith = signed_arith;
    meta.msb = msb_t;
    meta.dst_field = dst_field;

    size_t bytes_0 = ctx->GetStats()->sent_bytes.load();
    size_t send_act0 = ctx->GetStats()->sent_actions.load();
    output[rank] = ext_prot.Compute(input[rank], meta);
    size_t bytes_1 = ctx->GetStats()->sent_bytes.load();
    size_t send_act1 = ctx->GetStats()->sent_actions.load();

    sent_byte = bytes_1 - bytes_0;
    sent_act = send_act1 - send_act0;
  });

  printf("n = %zd sent %fKB, %zd rounds\n", n, sent_byte / 1024., sent_act);

  auto expected = ring_add(input[0], input[1]);
  auto got = ring_add(output[0], output[1]);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using T0 = std::make_unsigned<ring2k_t>::type;
    using sT0 = std::make_signed<ring2k_t>::type;
    ArrayView<const T0> _exp(expected);
    ArrayView<const sT0> _signed_exp(expected);
    DISPATCH_ALL_FIELDS(dst_field, "", [&]() {
      using T1 = std::make_unsigned<ring2k_t>::type;
      using sT1 = std::make_signed<ring2k_t>::type;
      ArrayView<const T1> _got(got);
      ArrayView<const sT1> _signed_got(got);

      if (signed_arith) {
        for (int64_t i = 0; i < _got.numel(); ++i) {
          EXPECT_EQ(static_cast<sT1>(_signed_exp[i]), _signed_got[i]);
        }
      } else {
        for (int64_t i = 0; i < _got.numel(); ++i) {
          EXPECT_EQ(static_cast<T1>(_exp[i]), _got[i]);
        }
      }
    });
  });
}

TEST_P(ExtensionProtTest, Heuristic) {
  size_t kWorldSize = 2;
  size_t n = 100000;
  FieldType field = std::get<0>(GetParam());
  bool signed_arith = std::get<1>(GetParam());
  std::string msb = std::get<2>(GetParam());
  ExtensionProtocol::MSB_st msb_t;

  FieldType dst_field;
  if (field == FieldType::FM32) {
    dst_field = FieldType::FM64;
  } else {
    dst_field = FieldType::FM128;
  }

  ArrayRef input[2];
  input[0] = ring_rand(field, n);
  if (not signed_arith || msb != "Unknown") {
    return;
  }

  ArrayRef msg;

  int kHeuristicBound = 2;

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    // message is small enough
    auto msg = ring_rand(field, n);
    ring_rshift_(msg, kHeuristicBound);

    ArrayView<ring2k_t> xmsg(msg);
    // some message are negative
    for (size_t i = 0; i < n; i += 2) {
      xmsg[i] = -xmsg[i];
    }
    input[1] = ring_sub(msg, input[0]);
  });

  msb_t = ExtensionProtocol::MSB_st::zero;

  ArrayRef output[2];
  size_t sent_byte = 0;
  size_t sent_act = 0;
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    int rank = ctx->Rank();
    auto conn = std::make_shared<Communicator>(ctx);
    auto base = std::make_shared<BasicOTProtocols>(conn);
    ExtensionProtocol ext_prot(base);
    ExtensionProtocol::Meta meta;
    meta.signed_arith = signed_arith;
    meta.msb = msb_t;
    meta.dst_field = dst_field;

    size_t bytes_0 = ctx->GetStats()->sent_bytes.load();
    size_t send_act0 = ctx->GetStats()->sent_actions.load();
    if (rank == 0) {
      auto inp0 = input[rank].clone();
      size_t bw = SizeOf(field) * 8;
      ring_add_(inp0, ring_lshift(ring_ones(field, n), bw - kHeuristicBound));
      output[rank] = ext_prot.Compute(inp0, meta);
      ring_sub_(output[rank],
                ring_lshift(ring_ones(dst_field, n), bw - kHeuristicBound));
    } else {
      output[rank] = ext_prot.Compute(input[rank], meta);
    }
    size_t bytes_1 = ctx->GetStats()->sent_bytes.load();
    size_t send_act1 = ctx->GetStats()->sent_actions.load();

    sent_byte = bytes_1 - bytes_0;
    sent_act = send_act1 - send_act0;
  });

  auto expected = ring_add(input[0], input[1]);
  auto got = ring_add(output[0], output[1]);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using T0 = std::make_unsigned<ring2k_t>::type;
    using sT0 = std::make_signed<ring2k_t>::type;
    ArrayView<const T0> _exp(expected);
    ArrayView<const sT0> _signed_exp(expected);
    DISPATCH_ALL_FIELDS(dst_field, "", [&]() {
      using T1 = std::make_unsigned<ring2k_t>::type;
      using sT1 = std::make_signed<ring2k_t>::type;
      ArrayView<const T1> _got(got);
      ArrayView<const sT1> _signed_got(got);

      for (int64_t i = 0; i < _got.numel(); ++i) {
        EXPECT_EQ(static_cast<sT1>(_signed_exp[i]), _signed_got[i]);
      }
    });
  });
}
}  // namespace spu::mpc::cheetah