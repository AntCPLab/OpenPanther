#include "common.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/utils/simulate.h"

// This file only used for debug

using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace panther;
using namespace std;
using namespace spu;
const size_t pir_logt = 12;
const size_t pir_fixt = 2;
const size_t logt = 24;
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
auto neighbors = read_data(1, 10, "dataset/neighbors.txt");

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
  // parameters compute
  yacl::set_num_threads(32);
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

  // Compute inner product
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

  // Compute distance
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
  SPDLOG_INFO("Distance computate correct!");

  // Prepare data input for batchargmin
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
  index[1] = PrepareBatchArgmin(id, k_c, group_bin_number,
                                {total_bin_number, max_bin_size}, 0);
  inp[1] = PrepareBatchArgmin(response, k_c, group_bin_number,
                              {total_bin_number, max_bin_size}, MASK >> 1);
  inp[0] = PrepareBatchArgmin(vec_reply, k_c, group_bin_number,
                              {total_bin_number, max_bin_size}, 0);

  // Batch argmin:
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
        BatchMinProtocol batch_argmax(kctx, 5);
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

  // Do plain argmin to check the result's correctness
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

      // if (i == 0) {
      // std::cout << min_k[j] << ":" << index[j].first << std::endl;
      // }
    }
    sum += k_c[i];
    sum_bin += l;
    sum_k += k;
  }

  // Check batch argmin result
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

  // Prepare the pir data (all clients need to run only once);
  auto encoded_db = PirData(cluster_num, ele_size, ps, ptoc, pir_logt,
                            max_cluster_points, pir_fixt);
  auto ctxs = yacl::link::test::SetupWorld(2);
  // use dh key exchange get shared oracle seed
  psi::SodiumCurve25519Cryptor c25519_cryptor0;
  psi::SodiumCurve25519Cryptor c25519_cryptor1;

  std::future<std::vector<uint8_t>> ke_func_server =
      std::async([&] { return c25519_cryptor0.KeyExchange(ctxs[0]); });
  std::future<std::vector<uint8_t>> ke_func_client =
      std::async([&] { return c25519_cryptor1.KeyExchange(ctxs[1]); });
  std::vector<uint8_t> seed_server = ke_func_server.get();
  std::vector<uint8_t> seed_client = ke_func_client.get();

  // Pir parameters setting
  spu::seal_pir::MultiQueryOptions options{{N, cluster_num, ele_size},
                                           batch_size};
  spu::psi::CuckooIndex::Options cuckoo_params{batch_size, 0, 3, 1.5};

  spu::seal_pir::MultiQueryServer mpir_server(options, cuckoo_params,
                                              seed_server);
  spu::seal_pir::MultiQueryClient mpir_client(options, cuckoo_params,
                                              seed_client);

  // Public keys and galois keys send and recv
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

  // Encode pir database
  mpir_server.SetDbSeperateId(encoded_db);

  // Doing pir:
  const auto pir_start_time = std::chrono::system_clock::now();
  std::future<std::vector<std::vector<uint32_t>>> pir_service_func =
      std::async([&] { return mpir_server.DoMultiPirAnswer(ctxs[0], true); });
  std::future<std::vector<std::vector<uint32_t>>> pir_client_func = std::async(
      [&] { return mpir_client.DoMultiPirQuery(ctxs[1], min_k, true); });
  std::vector<std::vector<uint32_t>> pir_s = pir_service_func.get();
  std::vector<std::vector<uint32_t>> pir_c = pir_client_func.get();
  const auto pir_end_time = std::chrono::system_clock::now();
  const DurationMillis pir_time = pir_end_time - pir_start_time;
  SPDLOG_INFO("pir time(online) : {} ms", pir_time.count());

  // Do truncation and extend to logt bits;
  std::vector<std::vector<uint32_t>> fixres_0(
      pir_c.size() * max_cluster_points,
      std::vector<uint32_t>(dims + 2 * message_size));

  std::vector<std::vector<uint32_t>> fixres_1(
      pir_c.size() * max_cluster_points,
      std::vector<uint32_t>(dims + 2 * message_size));
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
        if (rank == 0) {
          fixres_0 = FixPirResult(pir_c, pir_logt, pir_fixt, logt,
                                  pir_c.size() * max_cluster_points,
                                  dims + 2 * message_size, kctx);
        } else {
          fixres_1 = FixPirResult(pir_s, pir_logt, pir_fixt, logt,
                                  pir_s.size() * max_cluster_points,
                                  dims + 2 * message_size, kctx);
        }
      });
  std::vector<std::vector<uint32_t>> fix_pir_res(
      fixres_1.size(), std::vector<uint32_t>(fixres_1[0].size()));

  for (size_t idx = 0; idx < pir_c.size(); idx++) {
    auto query_index = mpir_client.test_query[idx].db_index;
    if (query_index == 0) continue;
    std::vector<uint32_t> query_db_bytes(ele_size);
    std::memcpy(query_db_bytes.data(), &encoded_db[query_index * ele_size * 4],
                ele_size * 4);
    for (size_t item = 0; item < ele_size; item++) {
      uint32_t mask = (1 << pir_logt) - 1;
      [[maybe_unused]] auto h2a = mask & (pir_c[idx][item] + pir_s[idx][item]);
      SPU_ENFORCE_EQ(query_db_bytes[item] >> 2, h2a >> 2);
    }

    for (size_t point_idx = 0; point_idx < max_cluster_points; point_idx++) {
      uint32_t mask = (1 << logt) - 1;
      for (size_t dim_i = 0; dim_i < dims; dim_i++) {
        auto pirres = fixres_0[idx * max_cluster_points + point_idx][dim_i] +
                      fixres_1[idx * max_cluster_points + point_idx][dim_i];

        pirres &= mask;
        SPU_ENFORCE_EQ(
            query_db_bytes[point_idx * (dims + 2 * message_size) + dim_i] >> 2,
            pirres);
        fix_pir_res[idx * max_cluster_points + point_idx][dim_i] = pirres;
      }
      for (size_t dim_i = dims; dim_i < fixres_0[0].size(); dim_i++) {
        auto pirres = fixres_0[idx * max_cluster_points + point_idx][dim_i] +
                      fixres_1[idx * max_cluster_points + point_idx][dim_i];
        pirres &= mask;
        fix_pir_res[idx * max_cluster_points + point_idx][dim_i] = pirres;
      }
    }
  }
  size_t pirres_size = fix_pir_res.size();
  std::vector<std::vector<uint32_t>> p_0(pirres_size,
                                         std::vector<uint32_t>(dims));

  std::vector<std::vector<uint32_t>> p_1(pirres_size,
                                         std::vector<uint32_t>(dims));
  std::vector<uint32_t> pid_0(pirres_size);
  std::vector<uint32_t> pid_1(pirres_size);
  std::vector<uint32_t> p_2_0(pirres_size);
  std::vector<uint32_t> p_2_1(pirres_size);

  PirResultForm(fixres_0, p_0, pid_0, p_2_0, dims, message_size);
  PirResultForm(fixres_1, p_1, pid_1, p_2_1, dims, message_size);
  std::vector<uint32_t> dis_0;
  std::vector<uint32_t> dis_1;
  spu::mpc::utils::simulate(
      kWorldSize, [&](std::shared_ptr<yacl::link::Context> lctx) {
        auto rank = lctx->Rank();
        if (rank == 0) {
          DisClient client(2048, logt, lctx);
          client.SendPublicKey();
          auto c0 = lctx->GetStats()->sent_bytes.load();

          // send query
          client.GenerateQuery(test_data[0]);

          dis_0 = client.RecvReply(p_0.size());
          auto c1 = lctx->GetStats()->sent_bytes.load();
          SPDLOG_INFO("Comm: {} MB", (c1 - c0) / 1024.0 / 1024.0);
        } else {
          DisServer server(2048, logt, lctx);
          server.RecvPublicKey();
          auto cs0 = lctx->GetStats()->sent_bytes.load();
          auto query = server.RecvQuery(dims);

          dis_1 = server.DoDistanceCmp(p_1, query);
          auto cs1 = lctx->GetStats()->sent_bytes.load();
          SPDLOG_INFO("Response Comm: {} MB", (cs1 - cs0) / 1024.0 / 1024.0);
        }
      });

  // Compute distance
  std::vector<uint32_t> end_dis(p_0.size());
  uint32_t q_2 = 0;
  for (size_t i = 0; i < dims; i++) {
    q_2 += test_data[0][i] * test_data[0][i];
  }
  for (size_t i = 0; i < p_0.size(); i++) {
    uint32_t exp = 0;
    uint32_t cmp = 0;
    uint32_t ip0 = 0;
    uint32_t pid = (pid_0[i] + pid_1[i]) & MASK;
    [[maybe_unused]] uint32_t p_2 = (p_2_0[i] + p_2_1[i]) & MASK;
    for (size_t point_i = 0; point_i < dims; point_i++) {
      ip0 += test_data[0][point_i] * p_0[i][point_i];
      if (pid != 10447816) {
        cmp += (test_data[0][point_i] - ps[pid][point_i]) *
               (test_data[0][point_i] - ps[pid][point_i]);
      } else {
        cmp = 1 << 18;
      }
      exp += (test_data[0][point_i] * (p_1[i][point_i] & MASK)) & MASK;
    }
    exp = exp & MASK;
    uint32_t ip1 = (dis_0[i] + dis_1[i]) & MASK;

    dis_0[i] = (q_2 + p_2_0[i] - 2 * ip0 - 2 * dis_0[i]) & MASK;
    dis_1[i] = (p_2_1[i] - 2 * dis_1[i]) & MASK;
    uint32_t get = (q_2 + p_2 - 2 * ip1 - 2 * ip0) & MASK;
    uint32_t dis_now = (dis_0[i] + dis_1[i]) & MASK;
    end_dis[i] = get;
    SPU_ENFORCE(get == dis_now);
    SPU_ENFORCE((ip1 <= exp) || (exp <= ip1));
  }
  // SPDLOG_INFO("Distance compute twice correctly!");

  // Approximate top-k for stash:

  size_t stash_start = 0;
  for (size_t i = 0; i < k_c.size() - 1; i++) {
    stash_start += k_c[i];
  }
  discard(dis, stash_start, k_c[k_c.size() - 1], 5);
  size_t l = group_bin_number[k_c.size() - 1];
  auto s_index = approximate_topk(dis, stash_start, k_c[k_c.size() - 1], l);
  std::partial_sort(s_index.begin(), s_index.begin() + topk_k, s_index.end(),
                    less<pair<uint32_t, uint32_t>>());

  discard(end_dis, 0, p_0.size(), 5);
  std::vector<pair<uint32_t, uint32_t>> end_pair(p_0.size() + topk_k);
  for (size_t i = 0; i < p_0.size(); i++) {
    end_pair[i].first = end_dis[i];
    end_pair[i].second = (pid_0[i] + pid_1[i]) & MASK;
    // std::cout << "Distance: " << end_pair[i].first
    //           << " Id: " << end_pair[i].second << std::endl;
  }
  for (size_t i = p_0.size(), j = 0; j < topk_k; i++, j++) {
    end_pair[i].first = s_index[j].first;
    end_pair[i].second = s_index[j].second;
  }

  // End topk:
  std::partial_sort(end_pair.begin(), end_pair.begin() + topk_k, end_pair.end(),
                    less<pair<uint32_t, uint32_t>>());
  uint32_t correct = 0;
  for (size_t i = 0; i < topk_k; i++) {
    // std::cout << end_pair[i].second << ":" << end_pair[i].first << std::endl;
    if (std::find(neighbors[0].begin(), neighbors[0].end(),
                  end_pair[i].second) != neighbors[0].end())
      correct++;
  }

  for (size_t i = 0; i < topk_k; i++) {
    uint32_t distance = 0;
    uint32_t id = neighbors[0][i];
    for (size_t d = 0; d < dims; d++) {
      distance += (test_data[0][d] - ps[id][d]) * (test_data[0][d] - ps[id][d]);
    }
  }

  SPDLOG_INFO("Accuracy: {}/{} = {}", correct, topk_k,
              (double)correct / topk_k);
}
