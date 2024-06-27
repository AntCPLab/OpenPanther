#include "dist_cmp.h"

#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

namespace sanns {
class DistanceCmpTest
    : public testing::TestWithParam<std::pair<uint32_t, uint32_t>> {};
TEST_P(DistanceCmpTest, localtest) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  int N = 4096;

  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, num_points, points_dim, ctxs[0]);
  DisServer server(N, ctxs[1]);
  // TODO:use public key send and recv
  server.set_pub_key(client.get_pub_key());

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

  auto query = client.GenerateQuery(q);
  auto response = server.DoDistanceCmp(ps, query);
  // TODO: H2A
  auto vec_reply = client.RecvReply();
  const uint32_t MASK = (1 << logt) - 1;
  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      exp += q[point_i] * ps[i][point_i];
    }
    auto get = (response[i / N][i % N] + vec_reply[i]) & MASK;
    EXPECT_NEAR(get, exp, 1);
  }
}
// TEST_P()
INSTANTIATE_TEST_SUITE_P(distance, DistanceCmpTest,
                         testing::Values(std::make_pair(100000, 128)));
}  // namespace sanns
