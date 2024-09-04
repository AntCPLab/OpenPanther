#include "panther_util.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/utils/simulate.h"

using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace sanns;
using namespace std;
using namespace spu;
const size_t pir_logt = 2;
const size_t pir_fixt = 2;
const size_t logt = 24;
const size_t cluster_shift = 5;
const uint32_t dims = 128;
const size_t N = 4096;
const size_t dis_N = 2048;
const size_t compare_radix = 5;
const size_t max_cluster_points = 20;
const std::vector<int64_t> k_c = {50810, 25603, 9968, 3227, 29326};
const std::vector<int64_t> group_bin_number = {458, 270, 178, 84, 262};
const std::vector<int64_t> group_k_number = {50, 31, 19, 13, 10};
const size_t total_points_num = 1000000;
const size_t topk_k = 10;
const size_t pointer_dc_bits = 8;
const size_t cluster_dc_bits = 5;
const size_t message_size = 3;
auto test_data = read_data(1, 128, "dataset/test.txt");
auto cluster_data = read_data(118934, 128, "dataset/centrios.txt");
auto stash = read_data(29326, 1, "dataset/stash.txt");
auto ps = read_data(1000000, 128, "dataset/dataset.txt");
auto ptoc = read_data(89608, 20, "dataset/ptoc.txt");

void discard(vector<uint32_t>& v, size_t begin, size_t len, uint32_t bw) {
  for (size_t i = 0; i < len; i++) {
    v[begin + i] = v[begin + i] >> bw;
  }
}
vector<pair<uint32_t, uint32_t>> approximate_topk(vector<uint32_t>& input,
                                                  size_t begin, size_t len,
                                                  size_t l) {
  vector<pair<uint32_t, uint32_t>> vid(len);
  for (size_t i = 0; i < len; i++) {
    vid[i].first = input[begin + i];
    vid[i].second = begin + i;
  }
  size_t bin_size = ceil((float)len / l);
  vector<pair<uint32_t, uint32_t>> indexs(l);
  for (size_t i = 0; i < l; i++) {
    uint32_t min_index = 0;
    uint32_t min_value = 1 << 24;
    if (len < i * bin_size) {
      indexs.at(i).first = (min_value - 1) >> 1;
      indexs.at(i).second = min_index;
    } else {
      for (size_t j = 0; j < min(bin_size, len - i * bin_size); j++) {
        size_t now = i * bin_size + j;
        if (min_value > vid.at(now).first) {
          min_index = vid.at(now).second;
          min_value = vid.at(now).first;
        }
      }
      indexs.at(i).first = min_value;
      indexs.at(i).second = min_index;
    }
  }
  return indexs;
}
size_t kWorldSize = 2;

int main() {
  yacl::set_num_threads(64);
  int64_t total_bin_number = 0;
  int64_t max_bin_size = 0;
  size_t batch_size = 0;
  size_t cluster_num = cluster_data.size() - stash.size();

  for (size_t i = 0; i < group_k_number.size(); i++) {
    total_bin_number += group_bin_number[i];
    auto bin_size =
        std::ceil(static_cast<double>(k_c[i]) / group_bin_number[i]);
    max_bin_size = max_bin_size > bin_size ? max_bin_size : bin_size;
  }
  SPDLOG_INFO("Total bin number (batch size of argmin): {} , Max bin size: {}",
              total_bin_number, max_bin_size);
  for (size_t i = 0; i < group_k_number.size() - 1; i++) {
    batch_size += group_k_number[i];
  }
  size_t ele_size = (dims + 2 * message_size) * max_cluster_points;
  SPDLOG_INFO(
      "Batch query size: {}, Cluster size: {}, Element size (coeff size): {}",
      batch_size, cluster_num, ele_size);

  vector<uint32_t> vec_reply;
  vector<uint32_t> response;
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
        auto rank = lctx->Rank();
        if (rank == 0) {
          DisClient client(dis_N, logt, lctx);
          client.SendPublicKey();
          auto c0 = lctx->GetStats()->sent_bytes.load();
          // send query
          client.GenerateQuery(test_data[0]);
          vec_reply = client.RecvReply(cluster_data.size());
          auto c1 = lctx->GetStats()->sent_bytes.load();
          SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);
        } else {
          DisServer server(dis_N, logt, lctx);
          server.RecvPublicKey();

          auto cs0 = lctx->GetStats()->sent_bytes.load();
          auto query = server.RecvQuery(dims);
          response = server.DoDistanceCmp(cluster_data, query);
          auto cs1 = lctx->GetStats()->sent_bytes.load();
          SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);
        }
      });
  // compute distance
  const uint32_t MASK = (1 << logt) - 1;
  std::vector<uint32_t> dis(cluster_data.size());
  for (size_t i = 0; i < cluster_data.size(); i++) {
    uint32_t exp = 0;
    uint32_t p_2 = 0;
    uint32_t q_2 = 0;
    for (size_t point_i = 0; point_i < dims; point_i++) {
      exp += test_data[0][point_i] * cluster_data[i][point_i];
      p_2 += test_data[0][point_i] * test_data[0][point_i];
      q_2 += cluster_data[i][point_i] * cluster_data[i][point_i];
    }
    auto get = (response[i] + vec_reply[i]) & MASK;
    response[i] = p_2 - 2 * response[i];
    vec_reply[i] = q_2 - 2 * vec_reply[i];
    SPU_ENFORCE(get <= (exp + 1) && (exp <= (get + 1)));
    dis[i] = p_2 + q_2 - 2 * exp;
  }

  vector<uint32_t> id(cluster_data.size());
  for (size_t i = 0; i < cluster_num; i++) {
    id[i] = i;
  };
  for (size_t i = cluster_num; i < cluster_data.size(); i++) {
    id[i] = stash[i - cluster_num][0];
  }

  spu::NdArrayRef inp[2];
  spu::NdArrayRef index[2];

  spu::NdArrayRef cmp_oup[2];
  spu::NdArrayRef cmp_idx[2];

  index[0] = spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});
  index[1] = spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});
  //   inp[0] = spu::mpc::ring_zeros(spu::FM32, {total_bin_number,
  //   max_bin_size});

  inp[1] = spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});
  inp[0] = spu::mpc::ring_ones(spu::FM32, {total_bin_number, max_bin_size});
  inp[0] = spu::mpc::ring_sub(inp[1], inp[0]);
  spu::mpc::ring_bitmask_(inp[0], 0, logt - 1);

  spu::pforeach(0, total_bin_number, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      int64_t sum = 0;
      size_t t_i = 0;
      size_t point_sum = 0;
      while (t_i < k_c.size()) {
        if (i < sum) {
          break;
        }
        sum += group_bin_number[t_i];
        point_sum += k_c[t_i];
        t_i++;
      }
      t_i = t_i - 1;
      point_sum -= k_c[t_i];
      size_t bin_size = std::ceil((float)k_c[t_i] / group_bin_number[t_i]);
      size_t tmp = i - (sum - group_bin_number[t_i]);
      if (int64_t(bin_size * tmp) > k_c[t_i]) break;
      auto now_bin_size = min(bin_size, k_c[t_i] - tmp * bin_size);
      using namespace spu;
      DISPATCH_ALL_FIELDS(spu::FM32, "trans_to_topk", [&]() {
        auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
        auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
        auto xidx = NdArrayView<ring2k_t>(index[1]);
        mempcpy(&xidx[i * max_bin_size], &id[point_sum + tmp * bin_size],
                now_bin_size * 4);
        mempcpy(&xinp1[i * max_bin_size], &response[point_sum + tmp * bin_size],
                now_bin_size * 4);
        mempcpy(&xinp0[i * max_bin_size],
                &vec_reply[point_sum + tmp * bin_size], now_bin_size * 4);
      });
    }
  });

  spu::FieldType field = spu::FM32;
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
        auto rank = lctx->Rank();
        spu::RuntimeConfig rt_config;
        rt_config.set_protocol(spu::ProtocolKind::CHEETAH);
        rt_config.set_field(field);
        rt_config.set_fxp_fraction_bits(0);
        auto _ctx = std::make_unique<spu::SPUContext>(rt_config, lctx);
        auto* ctx = _ctx.get();
        spu::mpc::Factory::RegisterProtocol(ctx, lctx);
        auto kctx = std::make_shared<spu::KernelEvalContext>(_ctx.get());
        [[maybe_unused]] auto b0 = lctx->GetStats()->sent_bytes.load();
        [[maybe_unused]] auto s0 = lctx->GetStats()->sent_actions.load();
        auto start = std::chrono::high_resolution_clock::now();
        BatchArgmaxProtocol batch_argmax(kctx, 5);
        auto _c = batch_argmax.ComputeWithIndex(inp[rank], index[rank], logt,
                                                cluster_dc_bits,
                                                total_bin_number, max_bin_size);

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
  size_t sum = 0;
  size_t sum_k = 0;
  size_t sum_bin = 0;
  vector<uint32_t> candidate(total_bin_number);
  vector<uint64_t> min_k(batch_size);
  for (size_t i = 0; i < k_c.size() - 1; i++) {
    size_t begin = sum;
    size_t len = k_c[i];
    size_t l = group_bin_number[i];
    size_t k = group_k_number[i];
    discard(dis, begin, len, 5);
    auto index = approximate_topk(dis, begin, len, l);
    for (size_t j = 0; j < l; j++) {
      candidate[sum_bin + j] = index[j].second;
    }
    std::partial_sort(index.begin(), index.begin() + k, index.end(),
                      less<pair<uint32_t, uint32_t>>());
    for (size_t j = 0; j < k; j++) {
      min_k[sum_k + j] = uint64_t(index[j].second);
    }
    sum += k_c[i];
    sum_bin += l;
    sum_k += k;
  }
  auto input =
      spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});
  DISPATCH_ALL_FIELDS(field, "", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    u2k mask = (static_cast<u2k>(1) << logt) - 1;
    u2k trun_mask = (static_cast<u2k>(1) << (logt - cluster_dc_bits)) - 1;
    auto xout0 = NdArrayView<ring2k_t>(cmp_oup[0]);
    auto xout1 = NdArrayView<ring2k_t>(cmp_oup[1]);
    auto xidx0 = NdArrayView<ring2k_t>(cmp_idx[0]);
    auto xidx1 = NdArrayView<ring2k_t>(cmp_idx[1]);
    for (int64_t i = 0;
         i < total_bin_number - group_bin_number[group_bin_number.size() - 1];
         i++) {
      auto idx = (xidx0[i] + xidx1[i]) & mask;
      auto got_cmp = (xout0[i] + xout1[i]) & trun_mask;
      if (got_cmp != static_cast<u2k>(262143)) {
        SPU_ENFORCE_EQ(dis[idx], got_cmp);
        SPU_ENFORCE_EQ(dis[candidate[i]], got_cmp);
      };
    }
  });
  vector<uint32_t> pir_res_s;
  vector<uint32_t> pir_res_c;

  auto encoded_db = PirData(cluster_num, ele_size, ps, ptoc, pir_logt,
                            max_cluster_points, pir_fixt);
  auto ctxs = yacl::link::test::SetupBrpcWorld(2);

  // use dh key exchange get shared oracle seed
  psi::SodiumCurve25519Cryptor c25519_cryptor0;
  psi::SodiumCurve25519Cryptor c25519_cryptor1;

  std::future<std::vector<uint8_t>> ke_func_server =
      std::async([&] { return c25519_cryptor0.KeyExchange(ctxs[1]); });
  std::future<std::vector<uint8_t>> ke_func_client =
      std::async([&] { return c25519_cryptor1.KeyExchange(ctxs[0]); });
  std::vector<uint8_t> seed_server = ke_func_server.get();
  std::vector<uint8_t> seed_client = ke_func_client.get();

  spu::seal_pir::MultiQueryOptions options{
      {N, cluster_num, ele_size, 0, logt}, batch_size, 3};

  spu::psi::CuckooIndex::Options cuckoo_params{batch_size, 0, 3, 1.5};

  spu::seal_pir::MultiQueryServer mpir_server(options, cuckoo_params,
                                              seed_server);

  spu::seal_pir::MultiQueryClient mpir_client(options, cuckoo_params,
                                              seed_client);

  std::future<void> pir_send_keys =
      std::async([&] { return mpir_client.SendGaloisKeys(ctxs[1]); });

  std::future<void> pir_recv_keys =
      std::async([&] { return mpir_server.RecvGaloisKeys(ctxs[0]); });
  pir_send_keys.get();
  pir_recv_keys.get();

  std::future<void> pir_send_pub_keys =
      std::async([&] { return mpir_client.SendPublicKey(ctxs[1]); });

  std::future<void> pir_recv_pub_keys =
      std::async([&] { return mpir_server.RecvPublicKey(ctxs[0]); });
  pir_send_pub_keys.get();
  pir_recv_pub_keys.get();
  // do pir query/answer
  const auto pir_start_time = std::chrono::system_clock::now();

  mpir_server.SetDatabase(encoded_db);

  std::future<std::vector<std::vector<uint32_t>>> pir_service_func =
      std::async([&] { return mpir_server.DoMultiPirAnswer(ctxs[1], true); });
  std::future<std::vector<std::vector<uint32_t>>> pir_client_func = std::async(
      [&] { return mpir_client.DoMultiPirQuery(ctxs[0], min_k, true); });

  // const auto pir_client_stop_time = std::chrono::system_clock::now();

  std::vector<std::vector<uint32_t>> random_mask = pir_service_func.get();
  std::vector<std::vector<uint32_t>> query_reply_bytes = pir_client_func.get();

  const auto pir_end_time = std::chrono::system_clock::now();
  const DurationMillis pir_time = pir_end_time - pir_start_time;
  SPDLOG_INFO("pir time(online) : {} ms", pir_time.count());
}
