#include "dist_cmp.h"
#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
using DurationMillis = std::chrono::duration<double, std::milli>;
namespace panther {
class SSDistanceCmpTest
    : public testing::TestWithParam<std::pair<uint32_t, uint32_t>> {};
INSTANTIATE_TEST_SUITE_P(distance, SSDistanceCmpTest,
                         testing::Values(std::make_pair(100000, 128)));

// Optimized version of distance computation for secret-shared data with smaller
// parameters.
//
// Let all points are quantized to 8 bits and the points value are shared in
// 9-bit form, ensuring the MSB is zero.
//   - It reduces the cost of computing wrap bits.
//   - The random plaintext is only 9 bits and does not require large FHE
//   parameters.
//
// This function is only used to test correctness under these optimized
// conditions.
TEST_P(SSDistanceCmpTest, test_shared_distance) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  size_t N = 2048;
  size_t logt = 24;
  size_t bitwidth = 8;
  size_t item_mask = (1 << bitwidth) - 1;
  size_t random_mask = (1 << (bitwidth + 1)) - 1;

  // local test
  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, logt, ctxs[0]);
  DisServer server(N, logt, ctxs[1]);

  // Only for local correctness test
  server.SetPublicKey(client.GetPublicKey());

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> p0(num_points,
                                        std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> p1(num_points,
                                        std::vector<uint32_t>(points_dim, 0));

  // Generate random query
  for (size_t i = 0; i < points_dim; i++) {
    q[i] = rand() & item_mask;
  }

  std::vector<uint32_t> v_msb0(num_points * points_dim);
  std::vector<uint32_t> v_msb1(num_points * points_dim);

  std::vector<std::vector<uint32_t>> wrap(num_points,
                                          std::vector<uint32_t>(points_dim, 0));

  for (size_t i = 0; i < num_points; i++) {
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      // SS the points
      ps[i][point_i] = rand() & item_mask;
      p0[i][point_i] = rand() & random_mask;
      p1[i][point_i] = (ps[i][point_i] - p0[i][point_i]) & random_mask;
      SPU_ENFORCE(((p0[i][point_i] + p1[i][point_i]) & random_mask) ==
                  ps[i][point_i]);

      // MSB = 0, wrap = msb([p]_0) \xor msb[[p]_1]
      wrap[i][point_i] = (p0[i][point_i] ^ p1[i][point_i]) >> 8;
      v_msb0[i * points_dim + point_i] = p0[i][point_i] >> 8;
      v_msb1[i * points_dim + point_i] = p1[i][point_i] >> 8;
      SPU_ENFORCE((v_msb0[i * points_dim + point_i] ^
                   v_msb1[i * points_dim + point_i]) == wrap[i][point_i]);

      // split [p]_i to msb(p_i)|| x and let [p]_i = x
      p0[i][point_i] = p0[i][point_i] & item_mask;
      p1[i][point_i] = p1[i][point_i] & item_mask;
    }
  }

  // Using FHE to compute L2 Norm
  auto c0 = ctxs[0]->GetStats()->sent_bytes.load();
  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);
  auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
  auto response = server.DoDistanceCmpWithH2A(p1, query);
  auto vec_reply = client.RecvReplySS(num_points);
  auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);

  // Using OT to compute the wrap bit
  spu::FieldType field = spu::FM32;
  size_t kWorldSize = 2;
  spu::NdArrayRef msg[2];
  auto msb0 = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  auto msb1 = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  memcpy(&msb0.at(0), &(v_msb0[0]), num_points * points_dim * sizeof(uint32_t));
  memcpy(&msb1.at(0), &(v_msb1[0]), num_points * points_dim * sizeof(uint32_t));
  msg[0] = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  msg[1] = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});

  // p_i = msb_i || x_i
  // msg_i = x_i * msb_i
  // q * [p] = q * x_0 + q * x_1 - [wrap] * q
  // [wrap] * q = [wrap]_0 * q + [wrap]_1 * q - 2[wrap]_0 * [wrap]_1 * q
  for (size_t i = 0; i < num_points; i++) {
    memcpy(&msg[0].at(i * points_dim), &(q[0]), points_dim * sizeof(uint32_t));
  }
  auto bx = spu::mpc::ring_mul(msg[0], msb0);
  spu::mpc::ring_sub_(msg[0], bx);
  spu::mpc::ring_sub_(msg[0], bx);
  spu::NdArrayRef cmp_oup[2];
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
        auto rank = lctx->Rank();
        auto conn = std::make_shared<spu::mpc::Communicator>(lctx);
        auto base_ot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
            conn, spu::CheetahOtKind::YACL_Ferret);
        [[maybe_unused]] auto b0 = lctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = lctx->GetStats()->sent_actions.load();
        if (rank == 0) {
          cmp_oup[rank] = base_ot->PrivateMulxSend(msg[rank]);
        } else {
          cmp_oup[rank] = base_ot->PrivateMulxRecv(
              msg[rank],
              msb1.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1)));
        }
        [[maybe_unused]] auto b1 = lctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s1 = lctx->GetStats()->sent_actions.load();
      });

  const uint32_t MASK = (1 << logt) - 1;
  using namespace spu;

  // test correctness of protocol to compute $q * wrap$
  for (size_t i = 0; i < num_points; i++) {
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      auto cmp = bx.at<uint32_t>(i * points_dim + point_i) +
                 cmp_oup[0].at<uint32_t>(i * points_dim + point_i) +
                 cmp_oup[1].at<uint32_t>(i * points_dim + point_i);
      auto exp = q[point_i] * wrap[i][point_i];
      EXPECT_EQ(cmp, exp);
    }
  }

  // test the correctness of total result
  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    uint32_t distance = 0;
    uint32_t q_2 = 0;
    uint32_t p_2 = 0;
    uint32_t qp0 = 0;
    uint32_t sub_value = 0;

    // compute distance in plain

    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      // exp: q * p_1
      exp += q[point_i] * p1[i][point_i];

      // distance :L2 Norm
      distance += (q[point_i] - ps[i][point_i]) * (q[point_i] - ps[i][point_i]);
      p_2 += ps[i][point_i] * ps[i][point_i];
      q_2 += q[point_i] * q[point_i];
      qp0 += q[point_i] * p0[i][point_i];
      if (wrap[i][point_i] == 1) {
        // add wrap * q
        sub_value += q[point_i] << bitwidth;
      }
    }
    auto get = (response[i] + vec_reply[i]) & MASK;
    qp0 &= MASK;
    exp &= MASK;
    // cmp = q^2 + p^2 - 2qp
    // qp = q * p_0 + q * p_1 -  wrap * q
    auto cmp_dis = (p_2 + q_2 - 2 * get - 2 * qp0 + 2 * sub_value) & MASK;
    EXPECT_NEAR(get, exp, 1);
    EXPECT_NEAR(cmp_dis, distance, 2);
  }
}

}  // namespace panther