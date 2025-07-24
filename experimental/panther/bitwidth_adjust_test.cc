#include <random>

#include "experimental/panther/bitwidth_adjust_prot.h"
#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

using DurationMillis = std::chrono::duration<double, std::milli>;
namespace panther {

class BwTrunReduceTest
    : public ::testing::TestWithParam<std::tuple<int64_t, size_t, size_t>> {};

/**
 * @brief truncation with reduction test
 * @param __0 Input data size
 * @param __1 Source bitwidth
 * @param __2 Truncation bitwidth
 * @note Constraint: parms[2] < parms[1] && parms[2] <= 8
 */
INSTANTIATE_TEST_SUITE_P(bitwidth_adjust, BwTrunReduceTest,
                         testing::Values(std::make_tuple(100000, 16, 8),
                                         std::make_tuple(100000, 23, 7)));

class BwExtendTest
    : public ::testing::TestWithParam<std::tuple<int64_t, size_t, size_t>> {};

/**
 * @brief bitwidth extension test
 * @param __0 Input data size
 * @param __1 Source bitwidth
 * @param __2 Extension bitwidth (target bits = parms[2] + parms[1])
 * @note parms[2] + parms[1] < 32 since only 32-bit values are implemented.
 */
INSTANTIATE_TEST_SUITE_P(bitwidth_adjust, BwExtendTest,
                         testing::Values(std::make_tuple(100000, 16, 10),
                                         std::make_tuple(100000, 23, 7)));

TEST_P(BwTrunReduceTest, TrunReduce) {
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
        BitwidthAdjustProtocol bw_prot(base);
        oup[rank] = bw_prot.TrunReduceCompute(inp[rank], bw, shift_bw);
        [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

        SPDLOG_INFO(
            "Rank {} Trunc {} bits share by {} bits {} bits each #sent {}",
            rank, bw, shift_bw, (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
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

TEST_P(BwExtendTest, ExtendOpt) {
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
        auto cmp_s = std::chrono::system_clock::now();
        int rank = ctx->Rank();
        auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
        auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
            conn, spu::CheetahOtKind::YACL_Ferret);
        [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();
        BitwidthAdjustProtocol bw_prot(base);
        oup[rank] = bw_prot.ExtendComputeOpt(inp[rank], bw, extend_bw);
        [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

        auto cmp_e = std::chrono::system_clock::now();
        const DurationMillis cmp_time = cmp_e - cmp_s;
        SPDLOG_INFO(
            "Rank {} Extend {} bits share by {} bits {} bits each #sent {}",
            rank, bw, extend_bw, (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
        SPDLOG_INFO("Rank {} Extend time: {} ms", rank, cmp_time.count());
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

TEST_P(BwExtendTest, Extend) {
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
        auto cmp_s = std::chrono::system_clock::now();
        int rank = ctx->Rank();
        auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
        auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
            conn, spu::CheetahOtKind::YACL_Ferret);
        [[maybe_unused]] auto b0 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = ctx->GetStats()->sent_actions.load();
        BitwidthAdjustProtocol bw_prot(base);
        oup[rank] = bw_prot.ExtendCompute(inp[rank], bw, extend_bw);
        [[maybe_unused]] auto b1 = ctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = ctx->GetStats()->sent_actions.load();

        auto cmp_e = std::chrono::system_clock::now();
        const DurationMillis cmp_time = cmp_e - cmp_s;
        SPDLOG_INFO(
            "Rank {} Extend {} bits share by {} bits {} bits each #sent {}",
            rank, bw, extend_bw, (b1 - b0) * 8. / inp[0].numel(), (s1 - s0));
        SPDLOG_INFO("Rank {} Extend time: {} ms", rank, cmp_time.count());
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

}  // namespace panther