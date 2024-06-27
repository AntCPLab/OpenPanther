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

#include "libspu/mpc/cheetah/ot/util.h"

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah::test {

class UtilTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, UtilTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<UtilTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(UtilTest, ZipArray) {
  const size_t n = 10000;
  const auto field = GetParam();

  auto unzip = ring_zeros(field, n);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    const size_t elsze = SizeOf(field);
    for (size_t bw : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
      size_t width = elsze * 8;
      size_t pack_sze = CeilDiv(bw * n, width);
      printf("Pack size: %zu\n", pack_sze);
      auto zip = ring_zeros(field, pack_sze);
      auto array = ring_rand(field, n);
      auto inp = xt_mutable_adapt<ring2k_t>(array);
      auto mask = makeBitsMask<ring2k_t>(bw);
      inp &= mask;

      auto _zip = xt_mutable_adapt<ring2k_t>(zip);
      auto _unzip = xt_mutable_adapt<ring2k_t>(unzip);
      size_t zip_sze = ZipArrayBit<ring2k_t>({inp.data(), inp.size()}, bw,
                                             {_zip.data(), _zip.size()});
      SPU_ENFORCE(zip_sze == pack_sze);

      UnzipArrayBit<ring2k_t>({_zip.data(), zip_sze}, bw,
                              {_unzip.data(), _unzip.size()});

      for (size_t i = 0; i < n; ++i) {
        if (inp[i] != _unzip[i]) {
          std::cout << i << " " << inp[i] << " " << _unzip[i] << " "
                    << _zip[i * bw / width] << " " << _zip[i * bw / width + 1]
                    << std::endl;
        }
        EXPECT_EQ(inp[i], _unzip[i]);
      }
    }
  });
}

}  // namespace spu::mpc::cheetah::test
