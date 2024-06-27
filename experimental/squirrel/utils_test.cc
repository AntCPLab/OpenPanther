// Copyright 2024 Ant Group Co., Ltd.
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

#include "experimental/squirrel/utils.h"

#include <random>

#include "gtest/gtest.h"
#include "xtensor/xsort.hpp"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/xt_helper.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/simulate.h"

namespace squirrel::test {
class UtilsTest : public ::testing::Test {};

template <typename T>
spu::Value infeed(spu::SPUContext* ctx, const xt::xarray<T>& ds,
                  bool need_shared = false) {
  spu::device::ColocatedIo cio(ctx);
  if (ctx->lctx()->Rank() == 0) {
    cio.hostSetVar(fmt::format("x-{}", ctx->lctx()->Rank()), ds);
  }
  cio.sync();
  auto x = cio.deviceGetVar("x-0");

  if (need_shared && not x.isSecret()) {
    x = spu::kernel::hlo::Cast(ctx, x, spu::Visibility::VIS_SECRET, x.dtype());
  }
  return x;
}

TEST_F(UtilsTest, ArgMax) {
  using namespace spu;
  using namespace spu::kernel;
  using namespace spu::mpc;
  spu::FieldType field = spu::FM32;
  spu::Shape shape = {10, 110};

  std::default_random_engine rdv(std::time(0));
  std::uniform_real_distribution<double> uniform(-1024., 1024.);

  std::vector<size_t> _shape;
  for (int i = 0; i < shape.ndim(); ++i) {
    _shape.push_back((size_t)shape[i]);
  }

  xt::xarray<double> _x(_shape);
  std::generate_n(_x.data(), _x.size(), [&]() { return 0.1 + uniform(rdv); });

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> lctx) {
    spu::RuntimeConfig rt_config;
    rt_config.set_protocol(ProtocolKind::CHEETAH);
    rt_config.set_field(field);
    rt_config.set_fxp_fraction_bits(16);

    auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
    auto* ctx = _ctx.get();
    spu::mpc::Factory::RegisterProtocol(ctx, lctx);
    auto x = infeed(ctx, _x);

    for (int axis = 0; axis < shape.ndim(); ++axis) {
      auto expected = xt::argmax(_x, axis);

      [[maybe_unused]] auto b0 = lctx->GetStats()->sent_bytes.load();
      [[maybe_unused]] auto s0 = lctx->GetStats()->sent_actions.load();

      auto start = std::chrono::high_resolution_clock::now();
      auto got = ArgMax(ctx, x, axis);

      auto end = std::chrono::high_resolution_clock::now();
      SPDLOG_INFO(
          "Time {} ms",
          (std::chrono::duration_cast<std::chrono::microseconds>(end - start)
               .count() /
           1000));
      [[maybe_unused]] auto b1 = lctx->GetStats()->sent_bytes.load();
      [[maybe_unused]] auto s1 = lctx->GetStats()->sent_actions.load();

      SPDLOG_INFO(
          "Argmax {} elements sent {} bytes, {} bits each #sent "
          "{}",
          shape[axis], (b1 - b0), (b1 - b0) * 8. / expected.size(), (s1 - s0));
      got = hlo::Reveal(ctx, got);

      ASSERT_EQ(expected.size(), (size_t)got.numel());

      if (lctx->Rank() == 0) {
        auto flatten = got.data().reshape({got.numel()});

        DISPATCH_ALL_FIELDS(field, "check", [&]() {
          NdArrayView<ring2k_t> got(flatten);
          for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_EQ(expected(i), got[i]);
          }
        });
      }
    }
  });
}

}  // namespace squirrel::test
