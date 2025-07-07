#include "dist_cmp.h"

#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/utils/simulate.h"
using DurationMillis = std::chrono::duration<double, std::milli>;
namespace panther {
class DistanceCmpTest
    : public testing::TestWithParam<std::pair<uint32_t, uint32_t>> {};

INSTANTIATE_TEST_SUITE_P(distance, DistanceCmpTest,
                         testing::Values(std::make_pair(3540, 128)));

// In this setup:
// - The client holds the query vector.
// - The server holds the dataset (points).
// All data is quantized to a fixed bitwidth (e.g. 8).
TEST_P(DistanceCmpTest, distance_pp) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  // FHE parms:
  size_t N = 2048;
  size_t logt = 24;

  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, logt, ctxs[0]);
  DisServer server(N, logt, ctxs[1]);

  // For correctness test only
  server.SetPublicKey(client.GetPublicKey());

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim));

  // Generate random points
  uint32_t item_bitwidth = 8;
  uint32_t item_mask = (1 << item_bitwidth) - 1;
  for (auto &item : q) {
    item = rand() & item_mask;
  }
  for (auto &p : ps) {
    for (auto &item : p) {
      item = rand() & item_mask;
    }
  }

  // Send Query
  auto c0 = ctxs[0]->GetStats()->sent_bytes.load();
  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Send Query Comm: {:.2f} MB", (c1 - c0) / 1024.0 / 1024.0);

  // Recv Response: distances between p and each q_i
  auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
  server.DoDistanceCmp(ps, query);
  auto vec_reply = client.RecvReply(num_points);
  auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Response Comm: {:.2f} MB", (cs1 - cs0) / 1024.0 / 1024.0);

  // Correctness
  const uint32_t MASK = (1 << logt) - 1;
  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      exp += q[point_i] * ps[i][point_i];
    }
    [[maybe_unused]] auto get = (response[i] + vec_reply[i]) & MASK;
    EXPECT_NEAR(get, exp, 1);
  }
}

// In this setup:
// - The client holds the query vector.
// - The server and the client holds the shared dataset (points).
// The query is quantized to a fixed bitwidth (e.g. 8).
// All data is shared in Z_{2^k}, where k = logt. This leads to large FHE
// parameters (poly_degree = 4096).
// An optimized version using smaller parameters is available in
// dist_cmp_ss_test.cc.
TEST_P(DistanceCmpTest, distanc_ps) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  size_t N = 4096;
  size_t logt = 24;
  size_t mask = (1ULL << logt) - 1;
  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, logt, ctxs[0]);
  DisServer server(N, logt, ctxs[1]);

  // Only for local test
  server.SetPublicKey(client.GetPublicKey());

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> rp0(num_points,
                                         std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> rp1(num_points,
                                         std::vector<uint32_t>(points_dim, 0));

  for (size_t i = 0; i < points_dim; i++) {
    q[i] = 1;
  }
  for (size_t i = 0; i < num_points; i++) {
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      ps[i][point_i] = rand() % 256;
      rp0[i][point_i] = rand() & mask;
      rp1[i][point_i] = (ps[i][point_i] - rp0[i][point_i]) & mask;
      SPU_ENFORCE(((rp0[i][point_i] + rp1[i][point_i]) & mask) ==
                  ps[i][point_i]);
    }
  }

  auto c0 = ctxs[0]->GetStats()->sent_bytes.load();

  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);

  auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
  auto response = server.DoDistanceCmp(rp1, query);
  // TODO: H2A
  auto vec_reply = client.RecvReply(num_points);

  auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);
  const uint32_t MASK = (1 << logt) - 1;
  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    uint32_t distance = 0;
    uint32_t q_2 = 0;
    uint32_t p_2 = 0;
    uint32_t qpr0 = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      exp += q[point_i] * rp1[i][point_i];
      distance += (q[point_i] - ps[i][point_i]) * (q[point_i] - ps[i][point_i]);
      p_2 += ps[i][point_i] * ps[i][point_i];
      q_2 += q[point_i] * q[point_i];
      qpr0 += q[point_i] * rp0[i][point_i];
    }
    auto get = (response[i] + vec_reply[i]) & MASK;
    qpr0 &= MASK;
    exp &= MASK;
    auto cmp_dis = (p_2 + q_2 - 2 * get - 2 * qpr0) & MASK;
    EXPECT_NEAR(get, exp, 1);
    EXPECT_NEAR(cmp_dis, distance, 2);
  }
}

// In this setup:
// - The client holds the query vector.
// - The server holds the dataset (points).
// All data is quantized to a fixed bitwidth (e.g. 8).
TEST_P(DistanceCmpTest, distance_pp_with_H2A) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  // FHE parms:
  size_t N = 2048;
  size_t logt = 24;

  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, logt, ctxs[0]);
  DisServer server(N, logt, ctxs[1]);

  // For correctness test only
  server.SetPublicKey(client.GetPublicKey());

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim));

  // Generate random points
  uint32_t item_bitwidth = 8;
  uint32_t item_mask = (1 << item_bitwidth) - 1;
  for (auto &item : q) {
    item = rand() & item_mask;
  }
  for (auto &p : ps) {
    for (auto &item : p) {
      item = rand() & item_mask;
    }
  }

  // Send Query
  auto c0 = ctxs[0]->GetStats()->sent_bytes.load();
  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Send Query Comm: {:.2f} MB", (c1 - c0) / 1024.0 / 1024.0);

  // Recv Response: distances between p and each q_i
  auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
  server.DoDistanceCmp(ps, query);
  auto vec_reply = client.RecvReply(num_points);
  auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Response Comm: {:.2f} MB", (cs1 - cs0) / 1024.0 / 1024.0);

  // Correctness
  const uint32_t MASK = (1 << logt) - 1;
  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      exp += q[point_i] * ps[i][point_i];
    }
    [[maybe_unused]] auto get = (response[i] + vec_reply[i]) & MASK;
    EXPECT_NEAR(get, exp, 1);
  }
}

// TEST_P()
TEST_P(DistanceCmpTest, distanc_ps_with_H2A) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  size_t N = 4096;
  size_t logt = 24;
  size_t mask = (1ULL << logt) - 1;
  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, logt, ctxs[0]);
  DisServer server(N, logt, ctxs[1]);

  // Only for local test
  server.SetPublicKey(client.GetPublicKey());

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> rp0(num_points,
                                         std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> rp1(num_points,
                                         std::vector<uint32_t>(points_dim, 0));

  for (size_t i = 0; i < points_dim; i++) {
    q[i] = 1;
  }
  for (size_t i = 0; i < num_points; i++) {
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      ps[i][point_i] = rand() % 256;
      rp0[i][point_i] = rand() & mask;
      rp1[i][point_i] = (ps[i][point_i] - rp0[i][point_i]) & mask;
      SPU_ENFORCE(((rp0[i][point_i] + rp1[i][point_i]) & mask) ==
                  ps[i][point_i]);
    }
  }

  auto c0 = ctxs[0]->GetStats()->sent_bytes.load();

  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);

  auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
  auto response = server.DoDistanceCmp(rp1, query);
  // TODO: H2A
  auto vec_reply = client.RecvReply(num_points);

  auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);
  const uint32_t MASK = (1 << logt) - 1;
  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    uint32_t distance = 0;
    uint32_t q_2 = 0;
    uint32_t p_2 = 0;
    uint32_t qpr0 = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      exp += q[point_i] * rp1[i][point_i];
      distance += (q[point_i] - ps[i][point_i]) * (q[point_i] - ps[i][point_i]);
      p_2 += ps[i][point_i] * ps[i][point_i];
      q_2 += q[point_i] * q[point_i];
      qpr0 += q[point_i] * rp0[i][point_i];
    }
    auto get = (response[i] + vec_reply[i]) & MASK;
    qpr0 &= MASK;
    exp &= MASK;
    auto cmp_dis = (p_2 + q_2 - 2 * get - 2 * qpr0) & MASK;
    EXPECT_NEAR(get, exp, 1);
    EXPECT_NEAR(cmp_dis, distance, 2);
  }
}
}  // namespace panther
