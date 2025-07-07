#include "common.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/utils/simulate.h"

using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace panther;
using namespace std;
using namespace spu;

// hyparameters:

const size_t pir_logt = 13;
const size_t pir_fixt = 2;
const size_t logt = 23;
const size_t compare_radix = 5;
const uint32_t dims = 96;
const size_t N = 4096;
const size_t dis_N = 2048;
const size_t max_cluster_points = 48;
const size_t fake_cluster_points = 40;
const std::vector<int64_t> k_c = {209727, 107417, 39132, 14424,
                                  5796,   2394,   50649};
const std::vector<int64_t> group_bin_number = {924, 458, 178, 93, 84, 84, 423};
const std::vector<int64_t> group_k_number = {88, 46, 25, 13, 7, 7, 10};
const size_t total_points_num = 10000000;
const size_t topk_k = 10;
const size_t pointer_dc_bits = 8;
const size_t cluster_dc_bits = 5;
const size_t message_size = 3;
const size_t fake_size = (dims + 2 * message_size) * fake_cluster_points;
const size_t ele_size = (dims + 2 * message_size) * max_cluster_points;
const uint32_t MASK = (1 << logt) - 1;

auto test_data = RandData(1, dims);
// Only for check the correctness of result:
auto neighbors = RandData(1, 10);

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));
llvm::cl::opt<uint32_t> PantherRank("rank", llvm::cl::init(0),
                                    llvm::cl::desc("self rank"));

llvm::cl::opt<uint32_t> EmpPort("emp_port", llvm::cl::init(7111),
                                llvm::cl::desc("emp port"));

int main(int argc, char** argv) {
  yacl::set_num_threads(32);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  auto hctx = MakeSPUContext(Parties.getValue(), PantherRank.getValue());
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = PantherRank.getValue();
  // Init Ferret-OT setup:
  // Ferret-OT produce too many OTs for once computation.
  // Bootstrap time can be amortized to many queries
  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  auto nworkers = ot_state->maximum_instances();
  for (size_t i = 0; i < nworkers; i++) {
    ot_state->LazyInit(comm, i);
  };
  const auto boot_strap_s = std::chrono::system_clock::now();
  yacl::parallel_for(0, nworkers, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      ot_state->get(i)->GetSenderCOT()->Bootstrap();
      ot_state->get(i)->GetReceiverCOT()->Bootstrap();
    }
  });
  const auto boot_strap_e = std::chrono::system_clock::now();
  const DurationMillis boot_strap_time = boot_strap_e - boot_strap_s;

  // Init PIR HE key:
  size_t batch_size = 0;
  size_t cluster_num = 0;
  for (size_t i = 0; i < group_k_number.size() - 1; i++) {
    batch_size += group_k_number[i];
    cluster_num += k_c[i];
  }
  size_t stash_size = k_c[k_c.size() - 1];
  size_t first_ps_size = cluster_num + stash_size;
  auto mpir_client =
      PrepareMpirClient(batch_size, cluster_num, fake_size, lctx, N, pir_logt);

  // Init distance HE key:
  DisClient dis_client(dis_N, logt, lctx);
  dis_client.SendPublicKey();

  // Init gc NetIO:
  [[maybe_unused]] emp::NetIO* gc_io =
      new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1", EmpPort.getValue());

  // Step(1): Distance Compute:
  SPDLOG_INFO("--------------- Start query -----------");

  gc_io->sync();
  auto total_time_s = std::chrono::system_clock::now();
  auto e2e_gc = gc_io->counter;
  auto e2e_lx = lctx->GetStats()->sent_bytes.load();

  uint32_t q_2 = 0;
  for (size_t i = 0; i < dims; i++) {
    q_2 += test_data[0][i] * test_data[0][i];
  }
  auto dis_cmp_r0 = lctx->GetStats()->sent_actions.load();
  auto dis_cmp_c0 = lctx->GetStats()->sent_bytes.load();
  auto distance_cmp_s = std::chrono::system_clock::now();
  dis_client.GenerateQuery(test_data[0]);

  auto dis_reply = dis_client.RecvReply(first_ps_size);

  auto dis_cmp_r1 = lctx->GetStats()->sent_actions.load();
  auto dis_cmp_c1 = lctx->GetStats()->sent_bytes.load();
  for (size_t i = 0; i < dis_reply.size(); i++) {
    dis_reply[i] = (q_2 - 2 * dis_reply[i]) & MASK;
  }
  auto distance_cmp_e = std::chrono::system_clock::now();
  const DurationMillis dis_cmp_time = distance_cmp_e - distance_cmp_s;
  SPDLOG_INFO("Distance cmp time: {} ms", dis_cmp_time.count());
  SPDLOG_INFO("Distance client sent actions: {}, Distance comm: {} MB",
              dis_cmp_r1 - dis_cmp_r0,
              (dis_cmp_c1 - dis_cmp_c0) / 1024.0 / 1024.0);

  // Step(2): Argmin in each bin:
  int64_t total_bin_number = 0;
  int64_t max_bin_size = 0;
  for (size_t i = 0; i < group_k_number.size(); i++) {
    total_bin_number += group_bin_number[i];
    auto bin_size =
        std::ceil(static_cast<double>(k_c[i]) / group_bin_number[i]);
    max_bin_size = max_bin_size > bin_size ? max_bin_size : bin_size;
  }
  auto value = PrepareBatchArgmin(dis_reply, k_c, group_bin_number,
                                  {total_bin_number, max_bin_size}, 0);
  auto index =
      spu::mpc::ring_zeros(spu::FM32, {total_bin_number, max_bin_size});

  // Start computation:
  auto argmax_r0 = lctx->GetStats()->sent_actions.load();
  auto argmax_c0 = lctx->GetStats()->sent_bytes.load();
  BatchArgmaxProtocol batch_argmax(kctx, compare_radix);
  // Index and value uint32_t
  auto argmin_res = batch_argmax.ComputeWithIndex(
      value, index, logt, cluster_dc_bits, total_bin_number, max_bin_size);

  auto argmax_e = std::chrono::system_clock::now();
  const DurationMillis argmax_time = argmax_e - distance_cmp_e;
  SPDLOG_INFO("Argmin cmp time: {} ms, ({}, {})", argmax_time.count(),
              total_bin_number, max_bin_size);
  auto argmax_r1 = lctx->GetStats()->sent_actions.load();
  auto argmax_c1 = lctx->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Argmin client sent actions: {}, Argmin comm: {} MB",
              argmax_r1 - argmax_r0, (argmax_c1 - argmax_c0) / 1024.0 / 1024.0);

  // Step(3): GC compute top-k cluster and stash result
  emp::setup_semi_honest(gc_io, 2 - rank);
  gc_io->flush();
  size_t initial_counter = gc_io->counter;
  auto min_value = argmin_res[0];
  auto min_index = argmin_res[1];
  size_t cluster_id_bw = std::ceil(std::log2(cluster_num));
  auto topk_id_c =
      GcTopkCluster(min_value, min_index, group_bin_number, group_k_number,
                    logt - cluster_dc_bits, cluster_id_bw, gc_io);
  auto gc_topk_e = std::chrono::system_clock::now();
  const DurationMillis gc_topk_time = gc_topk_e - argmax_e;
  SPDLOG_INFO("GC_naive_topk cmp time: {} ms", gc_topk_time.count());
  size_t naive_topk_comm = gc_io->counter - initial_counter;
  SPDLOG_INFO("GC_naive_topk client sent comm: {} MB",
              naive_topk_comm / 1024.0 / 1024.0);

  // Step(4): Multi-query PIR
  size_t pir_c0 = lctx->GetStats()->sent_bytes.load();
  auto pir_c = mpir_client.DoMultiPirQuery(lctx, topk_id_c, true);
  auto pir_size = pir_c.size();
  size_t pir_c1 = lctx->GetStats()->sent_bytes.load();

  SPDLOG_INFO("PIR client query comm: {} MB",
              (pir_c1 - pir_c0) / 1024.0 / 1024.0);

  // Step(4.2 (optional)): Fix PIR point result
  size_t fix_c0 = lctx->GetStats()->sent_bytes.load();

  FixPirResult(pir_c, pir_logt, pir_fixt, logt,
               pir_c.size() * fake_cluster_points, dims + 2 * message_size,
               kctx);
  auto fix_pir_c = FixPirResultOpt(
      pir_c, pir_logt, pir_fixt, logt, pir_c.size() * fake_cluster_points,
      dims + 2 * message_size, dims, rank, kctx, test_data[0]);
  // fix_pir_c = FixPirResultOpt(
  // pir_c, pir_logt, pir_fixt, logt, pir_c.size() * fake_cluster_points,
  // dims + 2 * message_size, dims, rank, kctx, test_data[0]);

  size_t fix_c1 = lctx->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Fix pir comm: {} MB", (fix_c1 - fix_c0) / 1024.0 / 1024.0);
  auto pir_e = std::chrono::system_clock::now();
  const DurationMillis pir_time = pir_e - gc_topk_e;
  SPDLOG_INFO("PIR cmp time: {} ms", pir_time.count());
  size_t pirres_size = pir_c.size();
  std::vector<std::vector<uint32_t>> p(pirres_size,
                                       std::vector<uint32_t>(dims));

  std::vector<uint32_t> pid(pirres_size);
  std::vector<uint32_t> p_2(pirres_size);
  PirResultForm(pir_c, p, pid, p_2, dims, message_size);

  // Step(5):Compute distance with points
  auto d2_start = lctx->GetStats()->sent_bytes.load();
  auto dis_c = dis_client.RecvReply(pirres_size);
  auto d2_end = lctx->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Point_distance comm: {} MB",
              (d2_end - d2_start) / 1024.0 / 1024.0);
  std::vector<uint32_t> dis(p.size());
  for (size_t i = 0; i < p.size(); i++) {
    uint32_t ip0 = 0;
    for (size_t point_i = 0; point_i < dims; point_i++) {
      ip0 += test_data[0][point_i] * p[i][point_i];
    }
    dis[i] =
        (q_2 + p_2[i] - 2 * ip0 - 2 * dis_c[i] + 2 * fix_pir_c[i][0]) & MASK;
  }
  auto dis_e = std::chrono::system_clock::now();
  const DurationMillis dis2_time = dis_e - pir_e;
  SPDLOG_INFO("Point_distance time: {} ms", dis2_time.count());

  // Step(6): End topk computation
  auto end_topk_s = std::chrono::system_clock::now();
  size_t trun_c0 = lctx->GetStats()->sent_bytes.load();
  auto trun_dis = Truncate(dis, logt, pointer_dc_bits, kctx);
  size_t trun_c1 = lctx->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Trunc comm: {} MB", (trun_c1 - trun_c0) / 1024.0 / 1024.0);

  uint32_t sum_bin = 0;
  for (size_t i = 0; i < group_bin_number.size() - 1; i++) {
    sum_bin += group_bin_number[i];
  }
  size_t stash_bw = logt - cluster_dc_bits;
  size_t discard_bw = pointer_dc_bits - cluster_dc_bits;
  size_t id_bw = std::ceil(std::log2(total_points_num));

  auto end_topk0 = gc_io->counter;
  gc_io->flush();
  gc_io->sync();

  std::vector<uint32_t> random_dis(max_cluster_points * pir_size, 1);
  std::vector<uint32_t> random_id(max_cluster_points * pir_size, 1);
  auto id =
      GcEndTopk(random_dis, random_id, min_value, min_index,
                logt - pointer_dc_bits, id_bw, stash_bw, discard_bw, sum_bin,
                group_bin_number[group_bin_number.size() - 1], topk_k, gc_io);
  auto end_topk_e = std::chrono::system_clock::now();
  gc_io->flush();
  const DurationMillis end_topk_time = end_topk_e - end_topk_s;
  SPDLOG_INFO("End_topk_{}-{} time: {} ms", dis.size(), topk_k,
              end_topk_time.count());
  auto end_topk1 = gc_io->counter;
  SPDLOG_INFO("End_topk_{}-{} comm: {} MB", dis.size(), topk_k,
              (end_topk1 - end_topk0) / 1024.0 / 1024.0);
  emp::finalize_semi_honest();

  auto total_time_e = std::chrono::system_clock::now();
  const DurationMillis total_time = total_time_e - total_time_s;
  SPDLOG_INFO("Total time: {} ms", total_time.count());
  auto e2e_gc_end = gc_io->counter;
  auto e2e_lx_end = lctx->GetStats()->sent_bytes.load();
  auto topk_time =
      end_topk_time.count() + argmax_time.count() + gc_topk_time.count();

  auto distance_time = dis_cmp_time.count() + dis2_time.count();
  SPDLOG_INFO("Distance time: {} ms", distance_time);
  SPDLOG_INFO("Topk time: {} ms", topk_time);
  SPDLOG_INFO("Pir time: {} ms", pir_time.count());

  auto distance_com = dis_cmp_c1 - dis_cmp_c0 + d2_end - d2_start;
  auto pir_com = fix_c1 - fix_c0 + pir_c1 - pir_c0;
  auto topk_com =
      e2e_gc_end - e2e_gc + argmax_c1 - argmax_c0 + trun_c1 - trun_c0;

  SPDLOG_INFO("Total comm: {} MB",
              (e2e_gc_end - e2e_gc + e2e_lx_end - e2e_lx) / 1024.0 / 1024.0);
  SPDLOG_INFO("Distance comm: {} MB", distance_com / 1024.0 / 1024.0);
  SPDLOG_INFO("TopK comm: {} MB", topk_com / 1024.0 / 1024.0);
  SPDLOG_INFO("Pir comm: {} MB", pir_com / 1024.0 / 1024.0);
  // Print result ID
  // uint32_t correct = 0;
  // SPDLOG_INFO("{}-NNs IDs: ", topk_k);
  // std::cout << "(";
  // for (size_t i = 0; i < topk_k; i++) {
  //   std::cout << id[i] << " ";
  //   if (std::find(neighbors[0].begin(), neighbors[0].end(), id[i]) !=
  //       neighbors[0].end())
  //     correct++;
  // }
  // std::cout << ")" << std::endl;
  // SPDLOG_INFO("Accuracy: {}/{} = {}", correct, topk_k,
  //             (double)correct / topk_k);
}
