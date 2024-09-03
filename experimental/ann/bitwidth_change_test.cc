#include <random>

#include "experimental/ann/bitwidth_change_prot.h"
#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace sanns {

// numel , bw, shift_bw
// numel , bw, extend_bw
class BwChangeProtTest
    : public ::testing::TestWithParam<std::tuple<int64_t, size_t, size_t>> {};

// The parameter test for expanding and truncating togetger, so let p[3] < p[2];
INSTANTIATE_TEST_SUITE_P(topk, BwChangeProtTest,
                         testing::Values(std::make_tuple(1000, 8, 2)));

TEST_P(BwChangeProtTest, TrunReduce) {
  size_t kWorldSize = 2;
  int64_t n = std::get<0>(GetParam());
  size_t bw = std::get<1>(GetParam());
  size_t shift_bw = std::get<2>(GetParam());
  spu::NdArrayRef inp[2];
  spu::FieldType field = spu::FM32;
  inp[0] = spu::mpc::ring_rand(field, {n});
  auto msg = spu::mpc::ring_rand(field, {n});
  using namespace spu;
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xmsg = spu::NdArrayView<ring2k_t>(msg);
    ring2k_t mask = (static_cast<ring2k_t>(1) << (bw)) - 1;
    pforeach(0, msg.numel(), [&](int64_t i) { xmsg[i] &= mask; });
  });
  inp[1] = spu::mpc::ring_sub(msg, inp[0]);
  spu::NdArrayRef oup[2];
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
        int rank = ctx->Rank();
        auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
        auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
            conn, spu::CheetahOtKind::YACL_Ferret);
        [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();
        BitWidthChangeProtocol bw_prot(base);
        oup[rank] = bw_prot.TrunReduceCompute(inp[rank], bw, shift_bw);
        [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

        SPDLOG_INFO("Trun {} bits share by {} bits {} bits each #sent {}", bw,
                    shift_bw, (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
      });

  spu::mpc::ring_rshift_(msg, shift_bw);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xmsg = spu::NdArrayView<ring2k_t>(msg);
    auto xout0 = spu::NdArrayView<ring2k_t>(oup[0]);
    auto xout1 = spu::NdArrayView<ring2k_t>(oup[1]);

    ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - shift_bw)) - 1;
    for (int64_t i = 0; i < n; i++) {
      ring2k_t got = (xout0[i] + xout1[i]) & mask;
      ring2k_t expected = xmsg[i];
      EXPECT_EQ(got, expected);
    }
  });
}

TEST_P(BwChangeProtTest, Extend) {
  size_t kWorldSize = 2;
  int64_t n = std::get<0>(GetParam());
  size_t bw = std::get<1>(GetParam());
  size_t extend_bw = std::get<2>(GetParam());
  spu::NdArrayRef inp[2];
  spu::FieldType field = spu::FM32;
  inp[0] = spu::mpc::ring_rand(field, {n});
  auto msg = spu::mpc::ring_rand(field, {n});
  using namespace spu;
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xmsg = spu::NdArrayView<ring2k_t>(msg);
    ring2k_t mask = (static_cast<ring2k_t>(1) << (bw - 1)) - 1;
    pforeach(0, msg.numel(), [&](int64_t i) { xmsg[i] &= mask; });
  });
  inp[1] = spu::mpc::ring_sub(msg, inp[0]);
  spu::mpc::ring_bitmask_(inp[0], 0, bw);
  spu::mpc::ring_bitmask_(inp[1], 0, bw);
  spu::NdArrayRef oup[2];
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
        int rank = ctx->Rank();
        auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
        auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
            conn, spu::CheetahOtKind::YACL_Ferret);
        [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();
        BitWidthChangeProtocol bw_prot(base);
        oup[rank] = bw_prot.ExtendCompute(inp[rank], bw, extend_bw);
        [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

        SPDLOG_INFO("Extend {} bits share by {} bits {} bits each #sent {}", bw,
                    extend_bw, (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
      });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xmsg = spu::NdArrayView<ring2k_t>(msg);
    auto xout0 = spu::NdArrayView<ring2k_t>(oup[0]);
    auto xout1 = spu::NdArrayView<ring2k_t>(oup[1]);

    ring2k_t mask = (static_cast<ring2k_t>(1) << (bw + extend_bw)) - 1;
    for (int64_t i = 0; i < n; i++) {
      ring2k_t got = (xout0[i] + xout1[i]) & mask;
      ring2k_t expected = xmsg[i];
      EXPECT_EQ(got, expected);
    }
  });
}
}  // namespace sanns