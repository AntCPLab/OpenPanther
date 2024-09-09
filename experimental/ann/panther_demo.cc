#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "batch_argmax_prot.h"
#include "dist_cmp.h"
#include "experimental/ann/bitwidth_change_prot.h"
#include "experimental/ann/fix_pir_customed/seal_mpir.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
#include "topk.h"
#include "yacl/base/buffer.h"
#include "yacl/link/link.h"
#include "yacl/link/test_util.h"

#include "libspu/core/context.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/core/value.h"
#include "libspu/device/io.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/psi/cryptor/sodium_curve25519_cryptor.h"
using namespace sanns;
using DurationMillis = std::chrono::duration<double, std::milli>;

const size_t pir_logt = 12;
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

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

llvm::cl::opt<uint32_t> EmpPort("emp_port", llvm::cl::init(7111),
                                llvm::cl::desc("emp port"));

std::shared_ptr<yacl::link::Context> MakeLink(const std::string& parties,
                                              size_t rank) {
  yacl::link::ContextDesc lctx_desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  for (size_t rank = 0; rank < hosts.size(); rank++) {
    const auto id = fmt::format("party{}", rank);
    lctx_desc.parties.push_back({id, hosts[rank]});
  }
  auto lctx = yacl::link::FactoryBrpc().CreateContext(lctx_desc, rank);
  lctx->ConnectToMesh();
  return lctx;
}

std::unique_ptr<spu::SPUContext> MakeSPUContext() {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());
  spu::RuntimeConfig config;
  config.set_protocol(spu::ProtocolKind::CHEETAH);
  config.set_field(spu::FM32);
  auto hctx = std::make_unique<spu::SPUContext>(config, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}

std::vector<std::vector<uint32_t>> ReadClusterPoint(size_t point_number,
                                                    size_t dim) {
  std::vector<std::vector<uint32_t>> points(point_number,
                                            std::vector<uint32_t>(dim, 0));
  for (size_t i = 0; i < point_number; i++) {
    for (size_t j = 0; j < dim; j++) {
      // Generate random cluster pointer
      points[i][j] = 1 % 256;
    }
  }
  return points;
};

std::vector<uint32_t> ReadQuery(size_t num_dims) {
  // TODO(ljy):
  std::vector<uint32_t> q(num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    q[i] = rand() % 256;
  }
  return q;
}

std::vector<uint32_t> GcTopk(spu::NdArrayRef& value, spu::NdArrayRef& index,
                             const std::vector<int64_t>& g_bin_num,
                             const std::vector<int64_t>& g_k_number,
                             size_t bw_value, size_t bw_index,
                             emp::NetIO* gc_io) {
  int64_t sum_k = 0;
  for (auto k : g_k_number) {
    sum_k += k;
  }

  std::vector<uint32_t> res(sum_k);

  SPU_ENFORCE_EQ(g_bin_num.size(), g_k_number.size());

  using namespace spu;
  for (size_t begin = 0; begin < g_bin_num.size(); begin++) {
    DISPATCH_ALL_FIELDS(spu::FM32, "trans_to_topk", [&]() {
      auto xval = NdArrayView<ring2k_t>(value);
      auto xidx = NdArrayView<ring2k_t>(index);

      size_t now_k = 0;
      size_t now_bin = 0;
      for (size_t i = 0; i < begin; i++) {
        now_k += g_k_number[i];
        now_bin += g_bin_num[i];
      }

      auto real_bin = g_bin_num[begin];
      auto k = g_k_number[begin];
      auto bin = (real_bin / k) * k;
      std::vector<uint32_t> input_value(bin);
      std::vector<uint32_t> input_index(bin);
      memcpy(&input_value[0], &xval[now_bin], bin * sizeof(uint32_t));
      memcpy(&input_index[0], &xidx[now_bin], bin * sizeof(uint32_t));
      auto start = std::chrono::system_clock::now();
      // std::cout << "bin: " << bin << " k: " << k << " bw_v: " << bw_value
      // << "bw_id: " << bw_index << std::endl;
      auto topk_id =
          sanns::gc::TopK(bin, k, bw_value, bw_index, input_value, input_index);

      auto end = std::chrono::system_clock::now();
      const DurationMillis topk_time = end - start;

      memcpy(&res[now_k], topk_id.data(), k * sizeof(uint32_t));
    });
  }
  return res;
};

spu::seal_pir::MultiQueryClient PrepareMpirClient(
    size_t batch_number, uint32_t ele_number, uint32_t ele_size,
    std::shared_ptr<yacl::link::Context>& lctx) {
  double factor = 1.5;
  size_t hash_num = 3;
  spu::psi::CuckooIndex::Options cuckoo_params{batch_number, 0, hash_num,
                                               factor};
  // cuckoo hash parms
  spu::psi::SodiumCurve25519Cryptor c25519_cryptor;

  std::vector<uint8_t> seed_client = c25519_cryptor.KeyExchange(lctx);
  spu::seal_pir::MultiQueryOptions options{{N, ele_number, ele_size, 0, logt},
                                           batch_number};
  spu::seal_pir::MultiQueryClient mpir_client(options, cuckoo_params,
                                              seed_client);
  mpir_client.SendGaloisKeys(lctx);
  mpir_client.SendPublicKey(lctx);
  return mpir_client;
}

std::vector<uint8_t> GenerateDbData(size_t element_number,
                                    size_t element_size) {
  SPDLOG_INFO("DB: element number:{} element size:{}", element_number,
              element_size);
  std::vector<uint8_t> db_data(element_number * element_size * 4);
  std::vector<uint32_t> db_raw_data(element_number * element_size);

  std::random_device rd;

  std::mt19937 gen(rd());

  for (uint64_t i = 0; i < element_number; i++) {
    for (uint64_t j = 0; j < element_size; j++) {
      auto val = 1 % 512;
      db_raw_data[(i * element_size) + j] = val == 0 ? 1 : val;
    }
  }
  memcpy(db_data.data(), db_raw_data.data(), element_number * element_size * 4);
  return db_data;
}

spu::seal_pir::MultiQueryServer PrepareMpirServer(
    size_t batch_number, size_t ele_number, size_t ele_size,
    std::shared_ptr<yacl::link::Context>& lctx) {
  auto db_bytes = GenerateDbData(ele_number, ele_size);
  double factor = 1.5;
  size_t hash_num = 3;
  spu::psi::CuckooIndex::Options cuckoo_params{batch_number, 0, hash_num,
                                               factor};

  spu::psi::SodiumCurve25519Cryptor c25519_cryptor;

  std::vector<uint8_t> seed_server = c25519_cryptor.KeyExchange(lctx);
  spu::seal_pir::MultiQueryOptions options{{N, ele_number, ele_size, 0, logt},
                                           batch_number};
  spu::seal_pir::MultiQueryServer mpir_server(options, cuckoo_params,
                                              seed_server);

  mpir_server.SetDbSeperateId(db_bytes);
  mpir_server.RecvGaloisKeys(lctx);
  mpir_server.RecvPublicKey(lctx);
  return mpir_server;
}

std::vector<std::vector<uint32_t>> FixPirResult(
    std::vector<std::vector<uint32_t>>& pir_result, size_t logt,
    size_t shift_bits, size_t target_bits, int64_t num_points,
    int64_t points_dim, const std::shared_ptr<spu::KernelEvalContext>& ct) {
  std::vector<std::vector<uint32_t>> result(num_points,
                                            std::vector<uint32_t>(points_dim));
  int64_t query_size = pir_result.size();
  int64_t num_slot = pir_result[0].size();
  // spu::NdArrayRef result;
  auto nd_inp = spu::mpc::ring_zeros(spu::FM32, {num_points * points_dim});
  std::memcpy(&nd_inp.at(0), pir_result.data(),
              query_size * num_slot * sizeof(uint32_t));
  auto out = spu::mpc::cheetah::TiledDispatchOTFunc(
      ct.get(), nd_inp,
      [&](const spu::NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        BitWidthChangeProtocol prot(base_ot);
        auto trun_value = prot.TrunReduceCompute(input, logt, shift_bits);

        return prot.ExtendCompute(trun_value, logt - shift_bits,
                                  target_bits - logt + shift_bits);
      });
  spu::pforeach(0, num_points, [&](int64_t i) {
    std::memcpy(result[i].data(), &out.at(i * points_dim),
                points_dim * out.elsize());
  });
  return result;
}
std::vector<uint32_t> Truncate(
    std::vector<uint32_t>& pir_result, size_t logt, size_t shift_bits,
    const std::shared_ptr<spu::KernelEvalContext>& ct) {
  int64_t num_points = pir_result.size();
  std::vector<uint32_t> result(num_points);
  // spu::NdArrayRef result;
  auto nd_inp = spu::mpc::ring_zeros(spu::FM32, {num_points});
  std::memcpy(&nd_inp.at(0), pir_result.data(), num_points * sizeof(uint32_t));
  auto out = spu::mpc::cheetah::TiledDispatchOTFunc(
      ct.get(), nd_inp,
      [&](const spu::NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        BitWidthChangeProtocol prot(base_ot);
        return prot.TrunReduceCompute(input, logt, shift_bits);
      });

  std::memcpy(result.data(), &out.at(0), num_points * out.elsize());

  return result;
}

int main(int argc, char** argv) {
  // TODO(ljy): add command line operations

  llvm::cl::ParseCommandLineOptions(argc, argv);
  yacl::set_num_threads(64);

  // Argmin:
  int64_t total_bin_number = 0;
  int64_t max_bin_size = 0;
  for (size_t i = 0; i < group_k_number.size(); i++) {
    total_bin_number += group_bin_number[i];
    auto bin_size =
        std::ceil(static_cast<double>(k_c[i]) / group_bin_number[i]);
    max_bin_size = max_bin_size > bin_size ? max_bin_size : bin_size;
  }
  SPDLOG_INFO("Total bin number (batch size of argmin): {} , Max bin size: {}",
              total_bin_number, max_bin_size);

  size_t batch_size = 0;
  size_t cluster_num = 0;
  for (size_t i = 0; i < group_k_number.size() - 1; i++) {
    batch_size += group_k_number[i];
    cluster_num += k_c[i];
  }

  // p, ID(p), p^2
  size_t message = 3;
  size_t ele_size = (dims + 2 * message) * max_cluster_points;
  size_t all_number = cluster_num + k_c[k_c.size() - 1];

  SPDLOG_INFO(
      "Batch query size: {}, Cluster size: {}, Element size (coeff size): {}",
      batch_size, cluster_num, ele_size);

  size_t cluster_id_bw = std::ceil(std::log2(cluster_num));

  size_t id_bw = std::ceil(std::log2(total_points_num));

  // context init
  auto hctx = MakeSPUContext();
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = Rank.getValue();

  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  auto nworkers = ot_state->maximum_instances();
  for (size_t i = 0; i < nworkers; i++) {
    ot_state->LazyInit(comm, i);
  };

  // Bootstrap time can be amortized to many queries
  // We let boot strap there, because we want the boot strap will produce too
  // many OTs, which can't be used in once computation.

  const auto boot_strap_s = std::chrono::system_clock::now();
  yacl::parallel_for(0, nworkers, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      ot_state->get(i)->GetSenderCOT()->Bootstrap();
      ot_state->get(i)->GetReceiverCOT()->Bootstrap();
    }
  });

  const auto boot_strap_e = std::chrono::system_clock::now();
  const DurationMillis boot_strap_time = boot_strap_e - boot_strap_s;
  // std::cout << boot_strap_time.count() << " ms" << std::endl;

  if (rank == 0) {
    // prepare mpir client
    auto mpir_client =
        PrepareMpirClient(batch_size, cluster_num, ele_size, lctx);

    DisClient dis_client(dis_N, logt, lctx);
    dis_client.SendPublicKey();

    // Computation begin
    emp::NetIO* gc_io = new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1",
                                       EmpPort.getValue());
    gc_io->sync();

    // Start counter:
    auto e2e_gc = gc_io->counter;
    auto e2e_lx = lctx->GetStats()->sent_bytes.load();
    auto total_time_s = std::chrono::system_clock::now();

    // Distance Compute
    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();
    auto q = ReadQuery(dims);
    dis_client.GenerateQuery(q);
    auto dis = dis_client.RecvReply(all_number);
    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();
    auto distance_cmp_e = std::chrono::system_clock::now();
    const DurationMillis dis_cmp_time = distance_cmp_e - total_time_s;
    SPDLOG_INFO("Distance cmp time: {} ms", dis_cmp_time.count());
    SPDLOG_INFO("Distance client sent actions: {}, Distance comm: {} MB",
                r1 - r0, (c1 - c0) / 1024.0 / 1024.0);

    spu::NdArrayRef distance =
        spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});
    spu::NdArrayRef index =
        spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});

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
          auto xinp0 = NdArrayView<ring2k_t>(distance);
          mempcpy(&xinp0[i * max_bin_size], &dis[point_sum + tmp * bin_size],
                  now_bin_size * 4);
        });
      }
    });

    auto argmax_r0 = lctx->GetStats()->sent_actions.load();
    auto argmax_c0 = lctx->GetStats()->sent_bytes.load();

    BatchArgmaxProtocol batch_argmax(kctx, compare_radix);

    // Index and value uint32_t
    auto _out = batch_argmax.ComputeWithIndex(
        distance, index, logt, cluster_dc_bits, total_bin_number, max_bin_size);

    auto argmax_e = std::chrono::system_clock::now();
    const DurationMillis argmax_time = argmax_e - distance_cmp_e;
    SPDLOG_INFO("Argmin cmp time: {} ms, {} {}", argmax_time.count(),
                total_bin_number, max_bin_size);
    auto argmax_r1 = lctx->GetStats()->sent_actions.load();
    auto argmax_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Argmin client sent actions: {}, Argmin comm: {} MB",
                argmax_r1 - argmax_r0,
                (argmax_c1 - argmax_c0) / 1024.0 / 1024.0);

    SPDLOG_INFO("Batch number: {}, element number: {}", total_bin_number,
                max_bin_size);
    auto max_value = _out[0];
    auto max_index = _out[1];

    emp::setup_semi_honest(gc_io, 2 - rank);
    gc_io->flush();
    size_t initial_counter = gc_io->counter;
    auto topk_id =
        GcTopk(max_value, max_index, group_bin_number, group_k_number,
               logt - cluster_dc_bits, cluster_id_bw, gc_io);
    gc_io->flush();
    auto gc_topk_e = std::chrono::system_clock::now();
    const DurationMillis gc_topk_time = gc_topk_e - argmax_e;
    SPDLOG_INFO("GC_naive_topk cmp time: {} ms", gc_topk_time.count());

    size_t naive_topk_comm = gc_io->counter - initial_counter;
    SPDLOG_INFO("GC_naive_topk comm: {} MB", naive_topk_comm / 1024.0 / 1024.0);

    std::vector<uint64_t> pir_query(batch_size, 0);
    for (size_t i = 0; i < batch_size; i++) {
      pir_query[i] = topk_id[i];
      SPU_ENFORCE(pir_query[i] < cluster_num);
    }

    size_t pir_c0 = lctx->GetStats()->sent_bytes.load();
    auto pir_res = mpir_client.DoMultiPirQuery(lctx, pir_query, true);
    size_t pir_c1 = lctx->GetStats()->sent_bytes.load();

    SPDLOG_INFO("PIR client query comm: {} MB",
                (pir_c1 - pir_c0) / 1024.0 / 1024.0);
    auto num_pir_points = pir_res.size() * max_cluster_points + topk_k;

    num_pir_points = (std::ceil(num_pir_points / topk_k) * topk_k);

    size_t fix_c0 = lctx->GetStats()->sent_bytes.load();
    auto fix_pir = FixPirResult(pir_res, pir_logt - 2, pir_fixt, logt,
                                pir_res.size() * max_cluster_points,
                                dims + 2 * message, kctx);
    size_t fix_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Fix pir comm: {} MB", (fix_c1 - fix_c0) / 1024.0 / 1024.0);
    auto pir_e = std::chrono::system_clock::now();
    const DurationMillis pir_time = pir_e - gc_topk_e;
    SPDLOG_INFO("PIR cmp time: {} ms", pir_time.count());

    auto d2_start = lctx->GetStats()->sent_bytes.load();
    auto point_dis = dis_client.RecvReply(num_pir_points);
    std::vector<uint32_t> pir_point_ids(num_pir_points);
    auto d2_end = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Distance Compute comm: {} MB",
                (d2_end - d2_start) / 1024.0 / 1024.0);

    auto dis_e = std::chrono::system_clock::now();
    const DurationMillis dis2_time = dis_e - pir_e;
    SPDLOG_INFO("Distance Compute cmp time: {} ms", dis2_time.count());

    gc_io->flush();
    gc_io->sync();

    size_t trun_c0 = lctx->GetStats()->sent_bytes.load();
    point_dis = Truncate(point_dis, logt, pointer_dc_bits, kctx);
    size_t trun_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Trunc comm: {} MB", (trun_c1 - trun_c0) / 1024.0 / 1024.0);

    auto end_topk0 = gc_io->counter;
    auto end_topk_s = std::chrono::system_clock::now();
    gc::TopK(num_pir_points, topk_k, logt - pointer_dc_bits, id_bw, point_dis,
             pir_point_ids);
    gc_io->flush();
    auto end_topk1 = gc_io->counter;

    auto end_topk_e = std::chrono::system_clock::now();
    const DurationMillis end_topk_time = end_topk_e - end_topk_s;
    SPDLOG_INFO("End_topk_{}-{} cmp time: {} ms", num_pir_points, topk_k,
                end_topk_time.count());
    SPDLOG_INFO("End_topk_{}-{} comm: {} MB", num_pir_points, topk_k,
                (end_topk1 - end_topk0) / 1024.0 / 1024.0);
    emp::finalize_semi_honest();

    auto total_time_e = std::chrono::system_clock::now();

    const DurationMillis total_time = total_time_e - total_time_s;
    SPDLOG_INFO("Total time: {} ms", total_time.count());

    auto e2e_gc_end = gc_io->counter;
    auto e2e_lx_end = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Total comm: {} MB",
                (e2e_gc_end - e2e_gc + e2e_lx_end - e2e_lx) / 1024.0 / 1024.0);
  } else {
    // server
    auto mpir_server =
        PrepareMpirServer(batch_size, cluster_num, ele_size, lctx);

    DisServer dis_server(dis_N, logt, lctx);
    dis_server.RecvPublicKey();

    auto ps = ReadClusterPoint(all_number, dims);
    auto total_time_s = std::chrono::system_clock::now();

    emp::NetIO* gc_io = new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1",
                                       EmpPort.getValue());
    gc_io->sync();
    auto e2e_gc = gc_io->counter;
    auto e2e_lx = lctx->GetStats()->sent_bytes.load();

    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();
    auto q = dis_server.RecvQuery(dims);
    auto dis = dis_server.DoDistanceCmp(ps, q);
    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();
    auto distance_cmp_e = std::chrono::system_clock::now();
    const DurationMillis dis_cmp_time = distance_cmp_e - total_time_s;
    SPDLOG_INFO("Distance cmp time: {} ms", dis_cmp_time.count());
    SPDLOG_INFO("Distance server sent actions: {}, Distance comm: {} MB",
                r1 - r0, (c1 - c0) / 1024.0 / 1024.0);
    auto distance =
        spu::mpc::ring_ones(spu::FM32, {total_bin_number, max_bin_size});

    auto zero =
        spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});
    distance = spu::mpc::ring_sub(zero, distance);
    spu::mpc::ring_bitmask_(distance, 0, logt - 1);

    auto index =
        spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});

    vector<uint32_t> id(all_number);
    // std::cout << all_number << std::endl;
    for (size_t i = 0; i < cluster_num; i++) {
      id[i] = i;
    };
    for (size_t i = cluster_num; i < all_number; i++) {
      id[i] = i - cluster_num;
    }

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
          auto xinp0 = NdArrayView<ring2k_t>(distance);
          auto xidx = NdArrayView<ring2k_t>(index);

          mempcpy(&xidx[i * max_bin_size], &id[point_sum + tmp * bin_size],
                  now_bin_size * 4);
          mempcpy(&xinp0[i * max_bin_size], &dis[point_sum + tmp * bin_size],
                  now_bin_size * 4);
        });
      }
    });
    // auto index = spu::mpc::ring_rand_range(
    // spu::FM32, {total_bin_number, max_bin_size}, 1, cluster_num);

    auto argmax_r0 = lctx->GetStats()->sent_actions.load();
    auto argmax_c0 = lctx->GetStats()->sent_bytes.load();

    BatchArgmaxProtocol batch_argmax(kctx, compare_radix);
    auto _out = batch_argmax.ComputeWithIndex(
        distance, index, logt, cluster_dc_bits, total_bin_number, max_bin_size);

    auto argmax_r1 = lctx->GetStats()->sent_actions.load();
    auto argmax_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO(
        "Batch number: {}, element number: {}, Argmax server sent actions: "
        "{}, "
        "Argmin comm: {} MB",
        total_bin_number, max_bin_size, argmax_r1 - argmax_r0,
        (argmax_c1 - argmax_c0) / 1024.0 / 1024.0);

    auto argmax_e = std::chrono::system_clock::now();
    const DurationMillis argmax_time = argmax_e - distance_cmp_e;
    SPDLOG_INFO("Argmin cmp time: {} ms", argmax_time.count());

    auto max_value = _out[0];
    auto max_index = _out[1];

    emp::setup_semi_honest(gc_io, 2 - rank);
    size_t initial_counter = gc_io->counter;
    gc_io->flush();
    auto topk_id =
        GcTopk(max_value, max_index, group_bin_number, group_k_number,
               logt - cluster_dc_bits, cluster_id_bw, gc_io);
    gc_io->flush();

    auto gc_topk_e = std::chrono::system_clock::now();
    const DurationMillis gc_topk_time = gc_topk_e - argmax_e;
    SPDLOG_INFO("Gc_Naive_topk cmp time: {} ms", gc_topk_time.count());

    size_t naive_topk_comm = gc_io->counter - initial_counter;
    SPDLOG_INFO("Gc_Naive_topk cmp comm: {} MB",
                naive_topk_comm / 1024.0 / 1024.0);

    size_t pir_c0 = lctx->GetStats()->sent_bytes.load();
    auto pir_res = mpir_server.DoMultiPirAnswer(lctx, true);
    // std::cout << pir_res.size() << pir_res[0].size() << endl;
    size_t pir_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("PIR server response comm: {} MB",
                (pir_c1 - pir_c0) / 1024.0 / 1024.0);
    auto num_pir_points = (max_cluster_points)*pir_res.size() + topk_k;
    num_pir_points = (std::ceil(num_pir_points / topk_k) * topk_k);
    size_t fix_c0 = lctx->GetStats()->sent_bytes.load();
    auto fix_pir = FixPirResult(pir_res, pir_logt - 2, pir_fixt, logt,
                                pir_res.size() * max_cluster_points,
                                dims + 2 * message, kctx);

    size_t fix_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Fix pir comm: {} MB", (fix_c1 - fix_c0) / 1024.0 / 1024.0);

    auto pir_e = std::chrono::system_clock::now();
    const DurationMillis pir_time = pir_e - gc_topk_e;
    SPDLOG_INFO("PIR cmp time: {} ms", pir_time.count());

    std::vector<std::vector<uint32_t>> r(num_pir_points,
                                         std::vector<uint32_t>(dims, 1));

    auto d2_start = lctx->GetStats()->sent_bytes.load();
    auto point_dis = dis_server.DoDistanceCmp(r, q);
    auto d2_end = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Point_distance comm: {} MB",
                (d2_end - d2_start) / 1024.0 / 1024.0);
    auto dis_e = std::chrono::system_clock::now();
    const DurationMillis dis2_time = dis_e - pir_e;
    SPDLOG_INFO("Point_distance time: {} ms", dis2_time.count());

    std::vector<uint32_t> pir_point_ids(num_pir_points);

    gc_io->flush();
    gc_io->sync();
    size_t trun_c0 = lctx->GetStats()->sent_bytes.load();
    point_dis = Truncate(point_dis, logt, pointer_dc_bits, kctx);
    size_t trun_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Trunc comm: {} MB", (trun_c1 - trun_c0) / 1024.0 / 1024.0);

    auto end_topk_s = std::chrono::system_clock::now();
    auto end_topk0 = gc_io->counter;
    gc::TopK(num_pir_points, topk_k, logt - pointer_dc_bits, id_bw, point_dis,
             pir_point_ids);
    auto end_topk_e = std::chrono::system_clock::now();
    gc_io->flush();

    const DurationMillis end_topk_time = end_topk_e - end_topk_s;
    SPDLOG_INFO("End_topk_{}-{} time: {} ms", num_pir_points, topk_k,
                end_topk_time.count());

    auto end_topk1 = gc_io->counter;
    SPDLOG_INFO("End_topk_{}-{} comm: {} MB", num_pir_points, topk_k,
                (end_topk1 - end_topk0) / 1024.0 / 1024.0);

    emp::finalize_semi_honest();
    auto total_time_e = std::chrono::system_clock::now();

    const DurationMillis total_time = total_time_e - total_time_s;
    SPDLOG_INFO("Total time: {} ms", total_time.count());
    auto e2e_gc_end = gc_io->counter;
    auto e2e_lx_end = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Total comm: {} MB",
                (e2e_gc_end - e2e_gc + e2e_lx_end - e2e_lx) / 1024.0 / 1024.0);
  }

  return 0;
}
