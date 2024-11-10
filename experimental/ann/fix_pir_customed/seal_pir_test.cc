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

#include "experimental/ann/fix_pir_customed/seal_pir.h"

#include <chrono>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "yacl/link/test_util.h"

namespace spu::seal_pir {
namespace {
struct TestParams {
  size_t N = 4096;
  size_t element_number;
  size_t element_size = 288;
  size_t query_size = 0;
};

std::vector<uint8_t> GenerateDbData(TestParams params) {
  std::vector<uint8_t> db_data(params.element_number * params.element_size * 4);
  std::vector<uint32_t> db_raw_data(params.element_number *
                                    params.element_size);

  std::random_device rd;

  std::mt19937 gen(rd());

  for (uint64_t i = 0; i < params.element_number; i++) {
    for (uint64_t j = 0; j < params.element_size; j++) {
      auto val = gen() % 512;
      db_raw_data[(i * params.element_size) + j] = val;
    }
  }
  memcpy(db_data.data(), db_raw_data.data(),
         params.element_number * params.element_size * 4);

  return db_data;
}

using DurationMillis = std::chrono::duration<double, std::milli>;
}  // namespace

class SealPirTest : public testing::TestWithParam<TestParams> {};

TEST_P(SealPirTest, Works) {
  auto params = GetParam();
  size_t n = params.N;
  auto ctxs = yacl::link::test::SetupBrpcWorld(2);
  yacl::set_num_threads(1);
  // yacl::ThreadPool(72);

  SPDLOG_INFO(
      "N: {}, element size: {} coeffs , element_number: 2^{:.2f} = {}, "
      "query_size(Indistinguishable degree) {}, logt: {}",
      n, params.element_size, std::log2(params.element_number),
      params.element_number, params.query_size, 24);

  std::vector<uint8_t> db_data = GenerateDbData(params);

  spu::seal_pir::SealPirOptions options{n, params.element_number,
                                        params.element_size, params.query_size};
  spu::seal_pir::SealPirClient client(options);

  std::shared_ptr<IDbPlaintextStore> plaintext_store =
      std::make_shared<MemoryDbPlaintextStore>();
#ifdef DEC_DEBUG_
  spu::seal_pir::SealPirServer server(options, client, plaintext_store);
#else
  spu::seal_pir::SealPirServer server(options, plaintext_store);
#endif

  // === server setup
  // using uint32_t to save coeff

  std::shared_ptr<IDbElementProvider> db_provider =
      std::make_shared<MemoryDbElementProvider>(db_data,
                                                params.element_size * 4);
  SPDLOG_INFO("Server start database encode!");
  server.SetDatabase(db_provider);

  SPDLOG_INFO("Server fininshed SetDatabase");

  /* galkey data 28MB, cause pipeline unittest timeout
  // client send galois keys to server
  std::future<void> client_galkey_func =
      std::async([&] { return client.SendGaloisKeys(ctxs[0]); });
  std::future<void> server_galkey_func =
      std::async([&] { return server.RecvGaloisKeys(ctxs[1]); });

  client_galkey_func.get();
  server_galkey_func.get();
  */
  // use offline
  seal::GaloisKeys galkey = client.GenerateGaloisKeys(0);
  server.SetGaloisKeys(galkey, 0);

  galkey = client.GenerateGaloisKeys(1);
  server.SetGaloisKeys(galkey, 1);

  std::random_device rd;

  std::mt19937 gen(rd());
  // size_t index = 40;
  size_t index = gen() % params.element_number;

  // do pir query/answer
  const auto pir_start_time = std::chrono::system_clock::now();
  const auto s0 = ctxs[0]->GetStats()->sent_bytes.load();
  const auto r0 = ctxs[1]->GetStats()->sent_bytes.load();
  std::future<std::vector<uint32_t>> pir_client_func =
      std::async([&] { return client.DoPirQuery(ctxs[0], index); });
  std::future<void> pir_service_func =
      std::async([&] { return server.DoPirAnswer(ctxs[1]); });

  pir_service_func.get();
  std::vector<uint32_t> query_reply_bytes = pir_client_func.get();

  const auto s1 = ctxs[0]->GetStats()->sent_bytes.load();
  const auto r1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Client -> Server {} KB, Server -> Client {} KB",
              (s1 - s0) / 1024.0, (r1 - r0) / 1024.0);

  const auto pir_end_time = std::chrono::system_clock::now();
  const DurationMillis pir_time = pir_end_time - pir_start_time;

  SPDLOG_INFO("one pir online query, total time : {} ms", pir_time.count());

  EXPECT_EQ(std::memcmp(query_reply_bytes.data(),
                        &db_data[index * params.element_size * 4],
                        params.element_size * 4),
            0);
}

// INSTANTIATE_TEST_SUITE_P(
//     Works_Instances, SealPirTest,
//     testing::Values(TestParams{4096, 1 << 19}       // element size default
//     288B

//                     ));

INSTANTIATE_TEST_SUITE_P(Works_Instances, SealPirTest,
                         testing::Values(TestParams{4096, 4096, 20 * 128}));
// small element_size to avoid out of memory

}  // namespace spu::seal_pir
