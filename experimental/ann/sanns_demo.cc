#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "batch_argmax_prot.h"
#include "dist_cmp.h"
#include "experimental/ann/fix_pir/seal_mpir.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
#include "topk.h"
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

const size_t logt = 20;
const uint32_t dims = 128;
const size_t N = 4096;
const size_t compare_radix = 5;
const size_t max_cluster_points = 20;
const std::vector<int64_t> k_c = {50810, 25603, 9968, 4227, 31412};
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
      points[i][j] = rand() % 256;
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
                             size_t bw_value, size_t bw_discard,
                             size_t bw_index) {
  int64_t sum = 0;
  int64_t sum_k = 0;
  for (auto& i : g_bin_num) {
    sum += i;
  }
  for (auto& i : g_k_number) {
    sum_k += i;
  }
  std::vector<uint32_t> res(sum_k);

  SPU_ENFORCE_EQ(value.numel(), sum);
  SPU_ENFORCE_EQ(g_bin_num.size(), g_k_number.size());

  using namespace spu;
  DISPATCH_ALL_FIELDS(spu::FM32, "trans_to_topk", [&]() {
    auto xval = NdArrayView<ring2k_t>(value);
    auto xidx = NdArrayView<ring2k_t>(index);
    size_t now_k = 0;
    size_t now_bin = 0;
    size_t g_num = g_bin_num.size();

    for (size_t g_i = 0; g_i < g_num; g_i++) {
      auto bin = g_bin_num[g_i];
      auto k = g_k_number[g_i];
      std::vector<uint32_t> input_value(bin);
      std::vector<uint32_t> input_index(bin);
      memcpy(&input_value[0], &xval[now_bin], bin * sizeof(uint32_t));
      memcpy(&input_index[0], &xidx[now_bin], bin * sizeof(uint32_t));
      auto start = std::chrono::system_clock::now();
      auto topk_id = sanns::gc::NaiveTopK(bin, k, bw_value, bw_discard,
                                          bw_index, input_value, input_index);

      auto end = std::chrono::system_clock::now();
      const DurationMillis topk_time = end - start;
      // std::cout << "Time: " << topk_time.count() << std::endl;

      memcpy(&res[now_k], topk_id.data(), k * sizeof(uint32_t));
      now_bin += bin;
      now_k += k;
    }
  });
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
      auto val = rand() % 512;
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

  mpir_server.SetDatabase(db_bytes);
  mpir_server.RecvGaloisKeys(lctx);
  mpir_server.RecvPublicKey(lctx);
  return mpir_server;
}

int main(int argc, char** argv) {
  // TODO(ljy): add command line operations

  llvm::cl::ParseCommandLineOptions(argc, argv);
  yacl::set_num_threads(72);

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

  // pir parms calculate:
  // pir: query batch size
  size_t batch_size = 0;
  // pir: total number number of elements
  size_t cluster_num = 0;
  for (size_t i = 0; i < group_k_number.size() - 1; i++) {
    batch_size += group_k_number[i];
    cluster_num += k_c[i];
  }

  // p, ID(p), p^2
  size_t ele_size = (dims + 2) * max_cluster_points;

  SPDLOG_INFO(
      "Batch query size: {}, Cluster size: , Element size (coeff size): {}",
      batch_size, cluster_num, ele_size);

  size_t cluster_id_bw = std::ceil(std::log2(cluster_num));

  size_t id_bw = std::ceil(std::log2(total_points_num));

  // context init
  auto hctx = MakeSPUContext();
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = Rank.getValue();

  // prepare ferret-ot
  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  ot_state->LazyInit(comm, 0);

  // client
  if (rank == 0) {
    // prepare mpir client
    auto mpir_client =
        PrepareMpirClient(batch_size, cluster_num, ele_size, lctx);

    // prepare distance compute client
    // (HE parameters for distance calculation are independent of the PIR)
    DisClient dis_client(N, logt, lctx);
    dis_client.SendPublicKey();

    emp::NetIO* gc_io = new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1",
                                       EmpPort.getValue());
    auto e2e_gc = gc_io->counter;
    auto e2e_lx = lctx->GetStats()->sent_bytes.load();

    auto total_time_s = std::chrono::system_clock::now();
    // Distance Compute
    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();
    auto q = ReadQuery(dims);
    dis_client.GenerateQuery(q);
    auto dis =
        dis_client.RecvReply({total_bin_number, max_bin_size}, cluster_num);
    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();
    auto distance_cmp_e = std::chrono::system_clock::now();
    const DurationMillis dis_cmp_time = distance_cmp_e - total_time_s;
    SPDLOG_INFO("Distance cmp time: {} ms", dis_cmp_time.count());
    SPDLOG_INFO("Distance client sent actions: {}, comm: {} KB", r1 - r0,
                (c1 - c0) / 1024.0);

    auto index =
        spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});

    auto argmax_r0 = lctx->GetStats()->sent_actions.load();
    auto argmax_c0 = lctx->GetStats()->sent_bytes.load();

    BatchArgmaxProtocol batch_argmax(kctx, compare_radix);
    // Index and value uint32_t
    auto _out = batch_argmax.ComputeWithIndex(dis, index, logt,
                                              total_bin_number, max_bin_size);

    auto argmax_e = std::chrono::system_clock::now();
    const DurationMillis argmax_time = argmax_e - distance_cmp_e;
    SPDLOG_INFO("Argmax cmp time: {} ms", argmax_time.count());

    auto argmax_r1 = lctx->GetStats()->sent_actions.load();
    auto argmax_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Argmax client sent actions: {}, comm: {} KB",
                argmax_r1 - argmax_r0, (argmax_c1 - argmax_c0) / 1024.0);

    auto max_value = _out[0];
    auto max_index = _out[1];

    emp::setup_semi_honest(gc_io, 2 - rank);
    gc_io->flush();
    size_t initial_counter = gc_io->counter;
    auto topk_id = GcTopk(max_value, max_index, group_bin_number,
                          group_k_number, logt, cluster_dc_bits, cluster_id_bw);

    auto gc_topk_e = std::chrono::system_clock::now();
    const DurationMillis gc_topk_time = gc_topk_e - argmax_e;
    SPDLOG_INFO("Gc topk cmp time: {} ms", gc_topk_time.count());

    size_t naive_topk_comm = gc_io->counter - initial_counter;
    SPDLOG_INFO("Communication for cluster topk: {} KB",
                naive_topk_comm / 1024.0);

    std::vector<uint64_t> pir_query(batch_size, 0);
    for (size_t i = 0; i < batch_size; i++) {
      pir_query[i] = topk_id[i];
      SPU_ENFORCE(pir_query[i] < cluster_num);
    }

    size_t pir_c0 = lctx->GetStats()->sent_bytes.load();
    auto pir_res = mpir_client.DoMultiPirQuery(lctx, pir_query, true);
    size_t pir_c1 = lctx->GetStats()->sent_bytes.load();

    auto pir_e = std::chrono::system_clock::now();
    const DurationMillis pir_time = pir_e - gc_topk_e;
    SPDLOG_INFO("PIR cmp time: {} ms", pir_time.count());
    SPDLOG_INFO("PIR client query comm: {} KB", (pir_c1 - pir_c0) / 1024.0);
    auto num_pir_points = pir_res.size() * max_cluster_points + 10;

    auto d2_start = lctx->GetStats()->sent_bytes.load();
    auto point_dis = dis_client.RecvReply(num_pir_points);
    std::vector<uint32_t> pir_point_ids(num_pir_points);
    auto d2_end = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Point distance comm: {} KB", (d2_end - d2_start) / 1024.0);

    auto dis_e = std::chrono::system_clock::now();
    const DurationMillis dis2_time = dis_e - pir_e;
    SPDLOG_INFO("Dis 2 cmp time: {} ms", dis2_time.count());

    gc_io->flush();
    auto end_topk0 = gc_io->counter;
    gc::NaiveTopK(num_pir_points, topk_k, logt, pointer_dc_bits, id_bw,
                  point_dis, pir_point_ids);
    auto end_topk1 = gc_io->counter;

    auto end_topk_e = std::chrono::system_clock::now();
    const DurationMillis end_topk_time = end_topk_e - pir_e;
    SPDLOG_INFO("Topk cmp time: {} ms", end_topk_time.count());

    SPDLOG_INFO("End topk {}-{} comm: {} KB", num_pir_points, topk_k,
                (end_topk1 - end_topk0) / 1024.0);
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

    DisServer dis_server(N, logt, lctx);
    dis_server.RecvPublicKey();

    auto total_time_s = std::chrono::system_clock::now();
    emp::NetIO* gc_io = new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1",
                                       EmpPort.getValue());
    auto e2e_gc = gc_io->counter;
    auto e2e_lx = lctx->GetStats()->sent_bytes.load();

    // TODO(ljy): shuffle the points?
    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();
    auto ps = ReadClusterPoint(cluster_num, dims);
    auto q = dis_server.RecvQuery(dims);
    auto dis =
        dis_server.DoDistanceCmp(ps, q, {total_bin_number, max_bin_size});

    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();
    auto distance_cmp_e = std::chrono::system_clock::now();
    const DurationMillis dis_cmp_time = distance_cmp_e - total_time_s;
    SPDLOG_INFO("Distance cmp time: {} ms", dis_cmp_time.count());
    SPDLOG_INFO("Distance server sent actions: {}, comm: {} KB", r1 - r0,
                (c1 - c0) / 1024.0);

    auto index = spu::mpc::ring_rand_range(
        spu::FM32, {total_bin_number, max_bin_size}, 1, cluster_num);

    auto argmax_r0 = lctx->GetStats()->sent_actions.load();
    auto argmax_c0 = lctx->GetStats()->sent_bytes.load();

    BatchArgmaxProtocol batch_argmax(kctx, compare_radix);
    auto _out = batch_argmax.ComputeWithIndex(dis, index, logt,
                                              total_bin_number, max_bin_size);

    auto argmax_r1 = lctx->GetStats()->sent_actions.load();
    auto argmax_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Argmax server sent actions: {}, comm cost: {} KB",
                argmax_r1 - argmax_r0, (argmax_c1 - argmax_c0) / 1024.0);

    auto argmax_e = std::chrono::system_clock::now();
    const DurationMillis argmax_time = argmax_e - distance_cmp_e;
    SPDLOG_INFO("Argmax cmp time: {} ms", argmax_time.count());

    auto max_value = _out[0];
    auto max_index = _out[1];

    emp::setup_semi_honest(gc_io, 2 - rank);
    size_t initial_counter = gc_io->counter;
    gc_io->flush();
    auto topk_id = GcTopk(max_value, max_index, group_bin_number,
                          group_k_number, logt, cluster_dc_bits, cluster_id_bw);

    auto gc_topk_e = std::chrono::system_clock::now();
    const DurationMillis gc_topk_time = gc_topk_e - argmax_e;
    SPDLOG_INFO("Gc topk cmp time: {} ms", gc_topk_time.count());

    size_t naive_topk_comm = gc_io->counter - initial_counter;
    SPDLOG_INFO("Communication for cluster topk: {} KB",
                naive_topk_comm / 1024.0);

    gc_io->flush();
    size_t pir_c0 = lctx->GetStats()->sent_bytes.load();
    auto pir_res = mpir_server.DoMultiPirAnswer(lctx, true);
    size_t pir_c1 = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("PIR server response comm: {} KB", (pir_c1 - pir_c0) / 1024.0);
    auto pir_e = std::chrono::system_clock::now();
    const DurationMillis pir_time = pir_e - gc_topk_e;
    SPDLOG_INFO("PIR cmp time: {} ms", pir_time.count());

    auto num_pir_points = max_cluster_points * pir_res.size() + 10;
    std::vector<std::vector<uint32_t>> r(num_pir_points,
                                         std::vector<uint32_t>(dims, 0));
    auto d2_start = lctx->GetStats()->sent_bytes.load();
    auto point_dis = dis_server.DoDistanceCmp(r, q);
    auto d2_end = lctx->GetStats()->sent_bytes.load();
    SPDLOG_INFO("Point distance comm: {} KB", (d2_end - d2_start) / 1024.0);
    auto dis_e = std::chrono::system_clock::now();
    const DurationMillis dis2_time = dis_e - pir_e;
    SPDLOG_INFO("Dis 2 cmp time: {} ms", dis2_time.count());

    std::vector<uint32_t> pir_point_ids(num_pir_points);
    auto end_topk0 = gc_io->counter;
    gc_io->flush();
    gc::NaiveTopK(num_pir_points, topk_k, logt, pointer_dc_bits, id_bw,
                  point_dis, pir_point_ids);
    auto end_topk_e = std::chrono::system_clock::now();
    const DurationMillis end_topk_time = end_topk_e - pir_e;
    SPDLOG_INFO("Topk cmp time: {} ms", end_topk_time.count());

    auto end_topk1 = gc_io->counter;
    SPDLOG_INFO("End topk {}-{} comm: {} KB", num_pir_points, topk_k,
                (end_topk1 - end_topk0) / 1024.0);

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
