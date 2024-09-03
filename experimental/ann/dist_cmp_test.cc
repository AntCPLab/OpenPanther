#include "dist_cmp.h"

#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/utils/simulate.h"
using DurationMillis = std::chrono::duration<double, std::milli>;
namespace sanns {
class DistanceCmpTest
    : public testing::TestWithParam<std::pair<uint32_t, uint32_t>> {};
// TEST_P(DistanceCmpTest, localtest) {
//   auto parms = GetParam();
//   size_t num_points = parms.first;
//   size_t points_dim = parms.second;

//   size_t N = 4096;
//   size_t logt = 24;

//   auto ctxs = yacl::link::test::SetupWorld(2);
//   DisClient client(N, logt, ctxs[0]);
//   DisServer server(N, logt, ctxs[1]);
//   // TODO:use public key send and recv
//   server.SetPublicKey(client.GetPublicKey());

//   std::vector<uint32_t> q(points_dim);
//   std::vector<std::vector<uint32_t>> ps(num_points,
//                                         std::vector<uint32_t>(points_dim,
//                                         0));
//   for (size_t i = 0; i < points_dim; i++) {
//     q[i] = rand() % 256;
//   }
//   for (size_t i = 0; i < num_points; i++) {
//     for (size_t point_i = 0; point_i < points_dim; point_i++) {
//       ps[i][point_i] = rand() % 256;
//     }
//   }
//   auto c0 = ctxs[0]->GetStats()->sent_bytes.load();

//   client.GenerateQuery(q);
//   auto query = server.RecvQuery(points_dim);
//   auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
//   SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);

//   auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
//   auto response = server.DoDistanceCmp(ps, query);
//   // TODO: H2A
//   auto vec_reply = client.RecvReply(num_points);

//   auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
//   SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);
//   const uint32_t MASK = (1 << logt) - 1;
//   for (size_t i = 0; i < num_points; i++) {
//     uint32_t exp = 0;
//     for (size_t point_i = 0; point_i < points_dim; point_i++) {
//       exp += q[point_i] * ps[i][point_i];
//     }
//     auto get = (response[i] + vec_reply[i]) & MASK;
//     EXPECT_NEAR(get, exp, 1);
//   }
// }

TEST_P(DistanceCmpTest, test_distance) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;
  std::cout << num_points << " " << points_dim << std::endl;
  yacl::set_num_threads(64);
  size_t N = 2048;
  size_t logt = 24;

  // auto ctxs = yacl::link::test::SetupBrpcWorld(2);
  // TODO:use public key send and recv

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim, 0));
  for (size_t i = 0; i < points_dim; i++) {
    q[i] = rand() % 256;
  }
  for (size_t i = 0; i < num_points; i++) {
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      ps[i][point_i] = rand() % 256;
    }
  }
  size_t kWorldSize = 2;
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctxs) {
        auto rank = ctxs->Rank();
        ctxs->SetThrottleWindowSize(1000);

        if (rank == 0) {
          DisClient client(N, logt, ctxs);
          client.SendPublicKey();
          auto dc_s = std::chrono::system_clock::now();
          auto cs0 = ctxs->GetStats()->sent_bytes.load();
          client.GenerateQuery(q);
          auto vec_reply = client.RecvReply({100, 1000}, num_points);
          // TODO: H2A
          auto cs1 = ctxs->GetStats()->sent_bytes.load();
          auto dc_e = std::chrono::system_clock::now();
          const DurationMillis dc_time = dc_e - dc_s;
          SPDLOG_INFO("Client Time: {} ms", dc_time.count());

          SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);

        } else {
          DisServer server(N, logt, ctxs);
          server.RecvPublicKey();
          auto dc_s = std::chrono::system_clock::now();
          auto c0 = ctxs->GetStats()->sent_bytes.load();
          auto s0 = ctxs->GetStats()->sent_actions.load();

          auto query = server.RecvQuery(points_dim);
          auto response = server.DoDistanceCmp(ps, query, {100, 1000});

          auto c1 = ctxs->GetStats()->sent_bytes.load();
          auto s1 = ctxs->GetStats()->sent_actions.load();
          SPDLOG_INFO("Comm: {} MB Rounds: {}", (c1 - c0) / 1024.0 / 1024.0,
                      s1 - s0);
          auto dc_e = std::chrono::system_clock::now();
          const DurationMillis dc_time = dc_e - dc_s;
          SPDLOG_INFO("Server Time: {} ms", dc_time.count());
        }
      });

  // const uint32_t MASK = (1 << logt) - 1;
  // for (size_t i = 0; i < num_points; i++) {
  //   uint32_t exp = 0;
  //   uint32_t distance = 0;
  //   uint32_t q_2 = 0;
  //   uint32_t p_2 = 0;
  //   for (size_t point_i = 0; point_i < points_dim; point_i++) {
  //     exp += q[point_i] * ps[i][point_i];
  //     distance += (q[point_i] - ps[i][point_i]) * (q[point_i] -
  //     ps[i][point_i]); p_2 += ps[i][point_i] * ps[i][point_i]; q_2 +=
  //     q[point_i] * q[point_i];
  //   }
  //   auto get = (response[i] + vec_reply[i]) & MASK;
  //   auto cmp_dis = p_2 + q_2 - 2 * get;
  //   EXPECT_NEAR(get, exp, 1);
  //   EXPECT_NEAR(cmp_dis, distance, 2);
  // }
}

// TEST_P(DistanceCmpTest, test_shared_distance) {
//   auto parms = GetParam();
//   size_t num_points = parms.first;
//   size_t points_dim = parms.second;

//   size_t N = 4096;
//   size_t logt = 24;
//   size_t mask = (1ULL << logt) - 1;
//   auto ctxs = yacl::link::test::SetupWorld(2);
//   DisClient client(N, logt, ctxs[0]);
//   DisServer server(N, logt, ctxs[1]);
//   // TODO:use public key send and recv
//   server.SetPublicKey(client.GetPublicKey());

//   std::vector<uint32_t> q(points_dim);
//   std::vector<std::vector<uint32_t>> ps(num_points,
//                                         std::vector<uint32_t>(points_dim,
//                                         0));
//   std::vector<std::vector<uint32_t>> rp0(num_points,
//                                          std::vector<uint32_t>(points_dim,
//                                          0));
//   std::vector<std::vector<uint32_t>> rp1(num_points,
//                                          std::vector<uint32_t>(points_dim,
//                                          0));

//   for (size_t i = 0; i < points_dim; i++) {
//     q[i] = rand() % 256;
//   }
//   for (size_t i = 0; i < num_points; i++) {
//     for (size_t point_i = 0; point_i < points_dim; point_i++) {
//       ps[i][point_i] = rand() % 256;
//       rp0[i][point_i] = rand() % 256;
//       rp1[i][point_i] = (ps[i][point_i] - rp0[i][point_i]) & mask;
//       SPU_ENFORCE(((rp0[i][point_i] + rp1[i][point_i]) & mask) ==
//                   ps[i][point_i]);
//     }
//   }

//   auto c0 = ctxs[0]->GetStats()->sent_bytes.load();

//   client.GenerateQuery(q);
//   auto query = server.RecvQuery(points_dim);
//   auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
//   SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);

//   auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
//   auto response = server.DoDistanceCmp(rp1, query);
//   // TODO: H2A
//   auto vec_reply = client.RecvReply(num_points);

//   auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
//   SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);
//   const uint32_t MASK = (1 << logt) - 1;
//   for (size_t i = 0; i < num_points; i++) {
//     uint32_t exp = 0;
//     uint32_t distance = 0;
//     uint32_t q_2 = 0;
//     uint32_t p_2 = 0;
//     uint32_t qpr0 = 0;
//     for (size_t point_i = 0; point_i < points_dim; point_i++) {
//       exp += q[point_i] * rp1[i][point_i];
//       distance += (q[point_i] - ps[i][point_i]) * (q[point_i] -
//       ps[i][point_i]); p_2 += ps[i][point_i] * ps[i][point_i]; q_2 +=
//       q[point_i] * q[point_i]; qpr0 += q[point_i] * rp0[i][point_i];
//     }
//     auto get = (response[i] + vec_reply[i]) & MASK;
//     qpr0 &= MASK;
//     exp &= MASK;
//     auto cmp_dis = (p_2 + q_2 - 2 * get - 2 * qpr0) & MASK;
//     EXPECT_NEAR(get, exp, 1);
//     EXPECT_NEAR(cmp_dis, distance, 2);
//   }
// }
// TEST_P()
INSTANTIATE_TEST_SUITE_P(distance, DistanceCmpTest,
                         testing::Values(std::make_pair(100000, 128)));
}  // namespace sanns
