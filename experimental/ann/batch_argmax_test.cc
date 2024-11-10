#include "batch_argmax_prot.h"
#include "emp-tool/io/io_channel.h"
#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
namespace sanns {
class BatchArgmaxTest
    : public ::testing::TestWithParam<std::pair<int64_t, int64_t>> {};

TEST_P(BatchArgmaxTest, truncate) {
  using namespace spu;
  yacl::set_num_threads(1);
  auto parms = GetParam();
  spu::FieldType field = spu::FM32;
  auto batch_size = parms.first;
  auto argmax_size = parms.second;
  spu::Shape shape = {batch_size, argmax_size};
  spu::NdArrayRef inp[2];
  spu::NdArrayRef index[2];
  inp[0] = spu::mpc::ring_rand(field, shape);
  index[0] = spu::mpc::ring_rand(field, shape);
  inp[1] = spu::mpc::ring_rand(field, shape);
  index[1] = spu::mpc::ring_rand(field, shape);

  int64_t bw = 24;
  int64_t shift = 5;
  auto input = spu::mpc::ring_rand(field, shape);
  // Input must > 0;
  spu::mpc::ring_bitmask_(input, 0, bw - 1);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    ring2k_t mask = (static_cast<ring2k_t>(1) << bw) - 1;
    ring2k_t trun_mask = (static_cast<ring2k_t>(1) << (bw - shift)) - 1;
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    pforeach(0, inp[0].numel(), [&](int64_t i) { xinp0[i] &= mask; });

    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
    auto xinp = NdArrayView<ring2k_t>(input);
    pforeach(0, inp[0].numel(), [&](int64_t i) {
      // xinp[i] = (xinp[i] + 1) & mask;
      xinp1[i] = xinp[i] - xinp0[i];
      xinp1[i] &= mask;
      xinp[i] = xinp[i] >> shift;
      xinp[i] &= trun_mask;
    });

    auto xidx0 = NdArrayView<ring2k_t>(index[0]);
    auto xidx1 = NdArrayView<ring2k_t>(index[1]);
    pforeach(0, index[0].numel(), [&](int64_t i) {
      xidx0[i] &= mask;
      xidx1[i] = (i - xidx0[i]) & mask;
    });
  });

  size_t kWorldSize = 2;
  spu::NdArrayRef cmp_oup[2];
  spu::NdArrayRef cmp_idx[2];

  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
        auto rank = lctx->Rank();
        spu::RuntimeConfig rt_config;
        rt_config.set_protocol(spu::ProtocolKind::CHEETAH);
        rt_config.set_field(field);
        rt_config.set_fxp_fraction_bits(0);
        auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
        auto *ctx = _ctx.get();
        spu::mpc::Factory::RegisterProtocol(ctx, lctx);
        auto kctx = std::make_shared<spu::KernelEvalContext>(_ctx.get());
        [[maybe_unused]] auto b0 = lctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = lctx->GetStats()->sent_actions.load();
        auto start = std::chrono::high_resolution_clock::now();
        BatchArgmaxProtocol batch_argmax(kctx, 5);
        auto _c = batch_argmax.ComputeWithIndex(inp[rank], index[rank], bw,
                                                shift, batch_size, argmax_size);

        // auto _c =
        // batch_argmax.Compute(inp[rank], bw, batch_size, argmax_size);
        auto end = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO(
            "Time {} ms",
            (std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                 .count() /
             1000));
        [[maybe_unused]] auto b1 = lctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = lctx->GetStats()->sent_actions.load();
        cmp_oup[rank] = _c[0];
        cmp_idx[rank] = _c[1];
        SPDLOG_INFO("Send actions: {}", (s1 - s0));
        SPDLOG_INFO("Send bytes: {} KB", (b1 - b0) / 1024.0);
      });
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    SPU_ENFORCE_EQ(cmp_oup[0].numel(), batch_size);
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << bw) - 1;
    const u2k trun_mask = (static_cast<u2k>(1) << (bw - shift)) - 1;
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xidx0 = NdArrayView<ring2k_t>(cmp_idx[0]);
    auto xidx1 = NdArrayView<ring2k_t>(cmp_idx[1]);
    auto xinput = NdArrayView<ring2k_t>(input);
    for (int64_t i = 0; i < cmp_oup[0].numel(); i++) {
      auto min = xinput[i * argmax_size] & trun_mask;
      for (int64_t j = 1; j < argmax_size; j++) {
        xinput[i * argmax_size + j] &= trun_mask;
        min = xinput[i * argmax_size + j] < min ? xinput[i * argmax_size + j]
                                                : min;
      }
      auto idx = (xidx0[i] + xidx1[i]) & mask;
      auto got_cmp = (xout0[i] + xout1[i]) & trun_mask;
      EXPECT_EQ(min, got_cmp);
      EXPECT_EQ(xinput[idx], min);
    }
  });
}

TEST_P(BatchArgmaxTest, localtest) {
  using namespace spu;
  yacl::set_num_threads(1);
  auto parms = GetParam();
  spu::FieldType field = spu::FM32;
  auto batch_size = parms.first;
  auto argmax_size = parms.second;
  spu::Shape shape = {batch_size, argmax_size};
  spu::NdArrayRef inp[2];
  spu::NdArrayRef index[2];
  inp[0] = spu::mpc::ring_rand(field, shape);
  index[0] = spu::mpc::ring_rand(field, shape);
  inp[1] = spu::mpc::ring_rand(field, shape);
  index[1] = spu::mpc::ring_rand(field, shape);

  int64_t bw = 24;
  int64_t shift = 0;
  auto input = spu::mpc::ring_rand(field, shape);
  // Input must > 0;
  spu::mpc::ring_bitmask_(input, 0, bw - 1);
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    ring2k_t mask = (static_cast<ring2k_t>(1) << bw) - 1;
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    pforeach(0, inp[0].numel(), [&](int64_t i) { xinp0[i] &= mask; });

    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
    auto xinp = NdArrayView<ring2k_t>(input);
    pforeach(0, inp[0].numel(), [&](int64_t i) {
      xinp1[i] = xinp[i] - xinp0[i];
      xinp1[i] &= mask;
    });

    auto xidx0 = NdArrayView<ring2k_t>(index[0]);
    auto xidx1 = NdArrayView<ring2k_t>(index[1]);
    pforeach(0, index[0].numel(), [&](int64_t i) {
      xidx0[i] &= mask;
      xidx1[i] = (i - xidx0[i]) & mask;
    });
  });

  size_t kWorldSize = 2;
  spu::NdArrayRef cmp_oup[2];
  spu::NdArrayRef cmp_idx[2];

  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
        auto rank = lctx->Rank();
        spu::RuntimeConfig rt_config;
        rt_config.set_protocol(spu::ProtocolKind::CHEETAH);
        rt_config.set_field(field);
        rt_config.set_fxp_fraction_bits(0);
        auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
        auto *ctx = _ctx.get();
        spu::mpc::Factory::RegisterProtocol(ctx, lctx);
        auto kctx = std::make_shared<spu::KernelEvalContext>(_ctx.get());
        for (int i = 0; i < 2; i++) {
          [[maybe_unused]] auto b0 = lctx->GetStats()->sent_bytes.load();
          [[maybe_unused]] auto s0 = lctx->GetStats()->sent_actions.load();
          auto start = std::chrono::high_resolution_clock::now();
          BatchArgmaxProtocol batch_argmax(kctx, 5);
          auto _c = batch_argmax.ComputeWithIndex(
              inp[rank], index[rank], bw, shift, batch_size, argmax_size);

          // auto _c =
          // batch_argmax.Compute(inp[rank], bw, batch_size, argmax_size);
          auto end = std::chrono::high_resolution_clock::now();
          SPDLOG_INFO("Time {} ms",
                      (std::chrono::duration_cast<std::chrono::microseconds>(
                           end - start)
                           .count() /
                       1000));
          [[maybe_unused]] auto b1 = lctx->GetStats()->sent_bytes.load();
          [[maybe_unused]] auto s1 = lctx->GetStats()->sent_actions.load();
          cmp_oup[rank] = _c[0];
          cmp_idx[rank] = _c[1];
          SPDLOG_INFO("Send actions: {}", (s1 - s0));
          SPDLOG_INFO("Send bytes: {} KB", (b1 - b0) / 1024.0);
        }
      });
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    SPU_ENFORCE_EQ(cmp_oup[0].numel(), batch_size);
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << bw) - 1;
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xidx0 = NdArrayView<ring2k_t>(cmp_idx[0]);
    auto xidx1 = NdArrayView<ring2k_t>(cmp_idx[1]);
    auto xinput = NdArrayView<ring2k_t>(input);
    for (int64_t i = 0; i < cmp_oup[0].numel(); i++) {
      auto min = xinput[i * argmax_size] & mask;
      for (int64_t j = 1; j < argmax_size; j++) {
        xinput[i * argmax_size + j] &= mask;
        min = xinput[i * argmax_size + j] < min ? xinput[i * argmax_size + j]
                                                : min;
      }
      auto idx = (xidx0[i] + xidx1[i]) & mask;
      auto got_cmp = (xout0[i] + xout1[i]) & mask;
      EXPECT_EQ(min, got_cmp);
      EXPECT_EQ(xinput[idx], got_cmp);
    }
  });
}

INSTANTIATE_TEST_SUITE_P(topk, BatchArgmaxTest,
                         testing::Values(std::make_pair(1000, 120)));
}  // namespace sanns