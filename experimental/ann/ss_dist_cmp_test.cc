#include "dist_cmp.h"
#include "gtest/gtest.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
using DurationMillis = std::chrono::duration<double, std::milli>;
namespace sanns {
class SSDistanceCmpTest
    : public testing::TestWithParam<std::pair<uint32_t, uint32_t>> {};

TEST_P(SSDistanceCmpTest, test_shared_distance) {
  auto parms = GetParam();
  size_t num_points = parms.first;
  size_t points_dim = parms.second;

  size_t N = 2048;
  size_t logt = 24;
  auto ctxs = yacl::link::test::SetupWorld(2);
  DisClient client(N, logt, ctxs[0]);
  DisServer server(N, logt, ctxs[1]);
  // TODO:use public key send and recv
  server.SetPublicKey(client.GetPublicKey());

  std::vector<uint32_t> q(points_dim);
  std::vector<std::vector<uint32_t>> ps(num_points,
                                        std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> rp0(num_points,
                                         std::vector<uint32_t>(points_dim, 0));
  std::vector<std::vector<uint32_t>> rp1(num_points,
                                         std::vector<uint32_t>(points_dim, 0));

  for (size_t i = 0; i < points_dim; i++) {
    q[i] = i % 256;
  }

  std::vector<uint32_t> v_msb0;
  std::vector<uint32_t> v_msb1;
  v_msb0.resize(num_points * points_dim);
  v_msb1.resize(num_points * points_dim);

  std::vector<std::vector<uint32_t>> wrap(num_points,
                                          std::vector<uint32_t>(points_dim, 0));
  for (size_t i = 0; i < num_points; i++) {
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      ps[i][point_i] = rand() % 256;
      rp0[i][point_i] = rand() % 512;
      rp1[i][point_i] = (ps[i][point_i] - rp0[i][point_i]) % 512;
      v_msb0[i * points_dim + point_i] = rp0[i][point_i] >> 8;
      v_msb1[i * points_dim + point_i] = rp1[i][point_i] >> 8;
      SPU_ENFORCE(((rp0[i][point_i] + rp1[i][point_i]) % 512) ==
                  ps[i][point_i]);
      wrap[i][point_i] = (rp0[i][point_i] ^ rp1[i][point_i]) >> 8;

      SPU_ENFORCE((v_msb0[i * points_dim + point_i] ^
                   v_msb1[i * points_dim + point_i]) == wrap[i][point_i]);
      rp0[i][point_i] = rp0[i][point_i] % 256;
      rp1[i][point_i] = rp1[i][point_i] % 256;
    }
  }

  auto c0 = ctxs[0]->GetStats()->sent_bytes.load();

  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto c1 = ctxs[0]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);

  auto cs0 = ctxs[1]->GetStats()->sent_bytes.load();
  auto response = server.DoDistanceCmp(rp1, query);
  auto vec_reply = client.RecvReply(num_points);
  auto cs1 = ctxs[1]->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);

  // Select p;
  spu::FieldType field = spu::FM32;
  size_t kWorldSize = 2;
  spu::NdArrayRef msb0;
  spu::NdArrayRef msb1;
  spu::NdArrayRef msg[2];
  msb0 = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  msb1 = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  memcpy(&msb0.at(0), &(v_msb0[0]), num_points * points_dim * 4);
  memcpy(&msb1.at(0), &(v_msb1[0]), num_points * points_dim * 4);
  msg[0] = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  msg[1] = spu::mpc::ring_zeros(field, {int64_t(num_points * points_dim)});
  for (size_t i = 0; i < num_points; i++) {
    memcpy(&msg[0].at(i * points_dim), &(q[0]), points_dim * 4);
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
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    // using u2k = std::make_unsigned<ring2k_t>::type;

    auto xinp = NdArrayView<ring2k_t>(bx);
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    for (size_t i = 0; i < num_points; i++) {
      for (size_t point_i = 0; point_i < points_dim; point_i++) {
        auto cmp = xinp[i * points_dim + point_i] +
                   xout0[i * points_dim + point_i] +
                   xout1[i * points_dim + point_i];
        auto exp = q[point_i] * wrap[i][point_i];
        EXPECT_EQ(cmp, exp);
      }
    }
  });

  for (size_t i = 0; i < num_points; i++) {
    uint32_t exp = 0;
    uint32_t distance = 0;
    uint32_t q_2 = 0;
    uint32_t p_2 = 0;
    uint32_t qpr0 = 0;
    uint32_t sub_value = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      exp += q[point_i] * rp1[i][point_i];
      distance += (q[point_i] - ps[i][point_i]) * (q[point_i] - ps[i][point_i]);
      p_2 += ps[i][point_i] * ps[i][point_i];
      q_2 += q[point_i] * q[point_i];
      qpr0 += q[point_i] * rp0[i][point_i];
      if (wrap[i][point_i] == 1) {
        sub_value += 256 * q[point_i];
      }
    }
    auto get = (response[i] + vec_reply[i]) & MASK;
    qpr0 &= MASK;
    exp &= MASK;
    auto cmp_dis = (p_2 + q_2 - 2 * get - 2 * qpr0 + 2 * sub_value) & MASK;
    EXPECT_NEAR(get, exp, 1);
    EXPECT_NEAR(cmp_dis, distance, 2);
  }
}
// TEST_P()
INSTANTIATE_TEST_SUITE_P(distance, SSDistanceCmpTest,
                         testing::Values(std::make_pair(100000, 128)));
}  // namespace sanns