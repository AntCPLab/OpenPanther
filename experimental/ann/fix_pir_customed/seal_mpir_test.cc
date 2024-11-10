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

#include "seal_mpir.h"

#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

#include "libspu/psi/cryptor/sodium_curve25519_cryptor.h"

namespace spu::seal_pir {
namespace {
struct TestParams {
  size_t batch_number;
  size_t element_number;
  size_t element_size = 288;
  size_t poly_degree = 4096;  // now we support 4096 and 8192
};

std::vector<uint8_t> GenerateDbData(TestParams params) {
  std::vector<uint8_t> db_data(params.element_number * params.element_size * 4);
  std::vector<uint32_t> db_raw_data(params.element_number *
                                    params.element_size);

  std::random_device rd;

  std::mt19937 gen(rd());

  for (uint64_t i = 0; i < params.element_number; i++) {
    for (uint64_t j = 0; j < params.element_size; j++) {
      auto val = 111 % 2048;
      val = (val << 2) + 1;
      db_raw_data[(i * params.element_size) + j] = val;
    }
  }
  memcpy(db_data.data(), db_raw_data.data(),
         params.element_number * params.element_size * 4);

  return db_data;
}

std::vector<size_t> GenerateQueryIndex(size_t batch_number,
                                       size_t element_number) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::set<size_t> query_index_set;

  while (true) {
    query_index_set.insert(gen() % element_number);

    if (query_index_set.size() == batch_number) {
      break;
    }
  }

  std::vector<size_t> query_index;
  query_index.assign(query_index_set.begin(), query_index_set.end());
  return query_index;
}

using DurationMillis = std::chrono::duration<double, std::milli>;
}  // namespace

class SealMultiPirTest : public testing::TestWithParam<TestParams> {};

TEST_P(SealMultiPirTest, WithH2A) {
  yacl::set_num_threads(32);
  auto params = GetParam();
  size_t element_number = params.element_number;
  size_t element_size = params.element_size;
  size_t batch_number = params.batch_number;

  SPDLOG_INFO(
      "N: {}, batch_number: {}, element_size: {} bytes, "
      "element_number: 2^{:.2f} = {}",
      params.poly_degree, batch_number, params.element_size,
      std::log2(params.element_number), params.element_number);

  // size_t batch_number = 256;
  double factor = 1.5;
  size_t hash_num = 3;
  spu::psi::CuckooIndex::Options cuckoo_params{batch_number, 0, hash_num,
                                               factor};

  std::vector<size_t> query_index =
      GenerateQueryIndex(batch_number, element_number);

  std::vector<uint8_t> db_bytes = GenerateDbData(params);

  auto ctxs = yacl::link::test::SetupBrpcWorld(2);

  // use dh key exchange get shared oracle seed
  psi::SodiumCurve25519Cryptor c25519_cryptor0;
  psi::SodiumCurve25519Cryptor c25519_cryptor1;

  std::future<std::vector<uint8_t>> ke_func_server =
      std::async([&] { return c25519_cryptor0.KeyExchange(ctxs[0]); });
  std::future<std::vector<uint8_t>> ke_func_client =
      std::async([&] { return c25519_cryptor1.KeyExchange(ctxs[1]); });

  std::vector<uint8_t> seed_server = ke_func_server.get();
  std::vector<uint8_t> seed_client = ke_func_client.get();

  EXPECT_EQ(seed_server, seed_client);

  ::spu::seal_pir::MultiQueryOptions options{
      {params.poly_degree, element_number, element_size}, batch_number};

  ::spu::seal_pir::MultiQueryServer mpir_server(options, cuckoo_params,
                                                seed_server);

  ::spu::seal_pir::MultiQueryClient mpir_client(options, cuckoo_params,
                                                seed_client);

  // server setup data
  mpir_server.SetDbSeperateId(db_bytes);

  std::future<void> pir_send_keys =
      std::async([&] { return mpir_client.SendGaloisKeys(ctxs[0]); });

  std::future<void> pir_recv_keys =
      std::async([&] { return mpir_server.RecvGaloisKeys(ctxs[1]); });
  pir_send_keys.get();
  pir_recv_keys.get();

  std::future<void> pir_send_pub_keys =
      std::async([&] { return mpir_client.SendPublicKey(ctxs[0]); });

  std::future<void> pir_recv_pub_keys =
      std::async([&] { return mpir_server.RecvPublicKey(ctxs[1]); });
  pir_send_pub_keys.get();
  pir_recv_pub_keys.get();

  // do pir query/answer
  const auto pir_start_time = std::chrono::system_clock::now();

  std::future<std::vector<std::vector<uint32_t>>> pir_service_func =
      std::async([&] { return mpir_server.DoMultiPirAnswer(ctxs[0], true); });
  std::future<std::vector<std::vector<uint32_t>>> pir_client_func = std::async(
      [&] { return mpir_client.DoMultiPirQuery(ctxs[1], query_index, true); });

  // const auto pir_client_stop_time = std::chrono::system_clock::now();

  std::vector<std::vector<uint32_t>> random_mask = pir_service_func.get();
  std::vector<std::vector<uint32_t>> query_reply_bytes = pir_client_func.get();

  const auto pir_end_time = std::chrono::system_clock::now();
  const DurationMillis pir_time = pir_end_time - pir_start_time;

  SPDLOG_INFO("pir time(online) : {} ms", pir_time.count());
  auto logt = 13;
  uint32_t mask = (1ULL << logt) - 1;
  for (size_t idx = 0; idx < query_reply_bytes.size(); ++idx) {
    if (mpir_client.test_query[idx].db_index == 0) continue;
    auto query_index = mpir_client.test_query[idx].db_index;
    std::vector<uint32_t> query_db_bytes(params.element_size);

    std::memcpy(query_db_bytes.data(),
                &db_bytes[query_index * params.element_size * 4],
                params.element_size * 4);

    for (size_t item = 0; item < query_reply_bytes[idx].size(); item++) {
      [[maybe_unused]] auto h2a =
          mask & (random_mask[idx][item] + query_reply_bytes[idx][item]);
      EXPECT_EQ(query_db_bytes[item] >> 2, h2a >> 2);
      // EXPECT_EQ(query_db_bytes[item], h2a);
      // SPDLOG_INFO("{} {}", query_db_bytes[item], h2a);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Works_Instances, SealMultiPirTest,
                         testing::Values(TestParams{100, 100000, 4096,
                                                    4096})  //
);

}  // namespace spu::seal_pir
