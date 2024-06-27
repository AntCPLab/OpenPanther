#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "batch_argmax_prot.h"
#include "dist_cmp.h"
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
#include "libspu/fix_pir/seal_mpir.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/psi/cryptor/sodium_curve25519_cryptor.h"
using namespace sanns;

const int32_t bw = 20;
const uint32_t cluster_number = 100000;

const uint32_t max_c_id_num = 20;
const uint32_t dims = 128;
const size_t pit_poly_degree = 4096;

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

llvm::cl::opt<uint32_t> EmpPort("emp_port", llvm::cl::init(9111),
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

std::vector<int32_t> naive_topk(int n, int k, int item_bits, int discard_bits,
                                int id_bits, std::vector<uint32_t>& input,
                                std::vector<uint32_t>& index) {
  std::vector<int32_t> gc_id(k);
  int32_t item_mask = (1 << item_bits) - 1;
  std::unique_ptr<emp::Integer[]> A = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> B = std::make_unique<emp::Integer[]>(n);

  std::unique_ptr<emp::Integer[]> A_idx = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> B_idx = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> INPUT = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> INDEX = std::make_unique<emp::Integer[]>(n);

  std::unique_ptr<emp::Integer[]> MIN_TOPK =
      std::make_unique<emp::Integer[]>(k);
  std::unique_ptr<emp::Integer[]> MIN_ID = std::make_unique<emp::Integer[]>(k);

  // Use for test

  for (int i = 0; i < n; ++i) {
    input[i] &= item_mask;
    index[i] &= id_bits;

    A[i] = Integer(item_bits, input[i], ALICE);
    B[i] = Integer(item_bits, input[i], BOB);

    A_idx[i] = Integer(id_bits, index[i], ALICE);
    B_idx[i] = Integer(id_bits, index[i], BOB);

    INDEX[i] = A_idx[i] + B_idx[i];
    INPUT[i] = A[i] + B[i];
  }

  sanns::gc::Discard(INPUT.get(), n, discard_bits);
  sanns::gc::Naive_topk(INPUT.get(), INDEX.get(), n, k, MIN_TOPK.get(),
                        MIN_ID.get());

  for (int i = 0; i < k; i++) {
    // gc_res[i] = MIN_TOPK[i].reveal<int32_t>(PUBLIC);
    gc_id[i] = MIN_ID[i].reveal<int32_t>(ALICE);
  }
  return gc_id;
}

std::vector<int32_t> gc_topk(spu::NdArrayRef& value, spu::NdArrayRef& index,
                             std::vector<int64_t>& g_bin_num,
                             std::vector<int64_t>& g_k_number) {
  int64_t sum = 0;
  int64_t sum_k = 0;
  for (auto& i : g_bin_num) {
    sum += i;
  }
  for (auto& i : g_k_number) {
    sum_k += i;
  }
  std::vector<int32_t> res(sum_k);

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
      auto topk_id = naive_topk(bin, k, bw, 0, bw, input_value, input_index);
      memcpy(&res[now_k], topk_id.data(), k * sizeof(uint32_t));
      now_bin += bin;
      now_k += k;
    }
  });
  return res;
};

void pir_client(std::vector<uint32_t> querys, uint32_t ele_number,
                yacl::link::context lctx) {
  auto batch_number = querys.size();
  double factor = 1.5;
  size_t hash_num = 3;
  spu::psi::CuckooIndex::Options cuckoo_params{batch_number, 0, hash_num,
                                               factor};
  // cuckoo hash parms
  // NOTE: only for test
  for (auto q& : query) {
    q = q % ele_number;
    SPU_ENFORCE_LE(q, ele_number);
  }
  spu::psi::SodiumCurve25519Cryptor c25519_cryptor;
  std::future<std::vector<uint8_t>> ke_func_client =
      std::async([&] { return c25519_cryptor0.KeyExchange(lctx); });
  std::vector<uint8_t> seed_client = ke_func_client.get();
  spu::seal_pir::MultiQuery options{{
                                        pir_poly_degree,
                                        ele_number,
                                        8000,
                                    },
                                    batch_number};
}

void pir_server() {
  auto batch_number = querys.size();
  double factor = 1.5;
  size_t hash_num = 3;
  spu::psi::CuckooIndex::Options cuckoo_params{batch_number, 0, hash_num,
                                               factor};

  // TODO(ljy):preprocessing
  std::vector<std::vector<uint32_t>> point_data(
      cluster_number, std::vector<uint32_t>(max_c_id_num * dims, 1));

  std::future<std::vector<uint8_t>> ke_func_server =
      std::async([&] { return c25519_cryptor0.KeyExchange(lctx); });
  std::vector<uint8_t> seed_client = ke_func_server.get();
}

int main(int argc, char** argv) {
  // TODO(ljy): add command line operations
  llvm::cl::ParseCommandLineOptions(argc, argv);
  const size_t N = 4096;
  const size_t compare_radix = 4;
  uint32_t num_points = 130000;
  int64_t bins_number = 1252;
  int64_t bins_item = 120;
  std::vector<int64_t> group_bin_number = {458, 270, 178, 84, 262};
  std::vector<int64_t> group_k_number = {50, 31, 19, 13, 10};

  auto hctx = MakeSPUContext();
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = Rank.getValue();

  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();
  ot_state->LazyInit(comm, 0);
  // spu::mpc::cheetah::InitOTState(kctx.get(), 2048);

  SPDLOG_INFO("Prepare done!");

  spu::NdArrayRef dis;
  std::cout << "test" << std::endl;
  spu::NdArrayRef index;
  if (rank == 0) {
    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();
    auto q = ReadQuery(dims);
    DisClient dis_client(N, num_points, dims, lctx);
    dis_client.GenerateQuery(q);
    dis = dis_client.RecvReply({bins_number, bins_item});
    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();

    SPDLOG_INFO("Distance : sent actions: {}, comm cost: {} KB", r1 - r0,
                (c1 - c0) / 1024.0);
    index = spu::mpc::ring_zeros(spu::FM32, {bins_number, bins_item});
  } else {
    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();

    DisServer dis_server(N, lctx);
    // TODO(ljy): shuffle the points?
    auto ps = ReadClusterPoint(num_points, dims);
    auto q = dis_server.RecvQuery(dims);
    dis = dis_server.DoDistanceCmp(ps, q, {bins_number, bins_item});

    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();

    SPDLOG_INFO("Distance : sent actions: {}, comm cost: {} KB", r1 - r0,
                (c1 - c0) / 1024.0);

    index = spu::mpc::ring_randbit(spu::FM32, {bins_number, bins_item});
  }
  auto r0 = lctx->GetStats()->sent_actions.load();
  auto c0 = lctx->GetStats()->sent_bytes.load();

  BatchArgmaxProtocol batch_argmax(kctx, compare_radix);
  auto _out =
      batch_argmax.ComputeWithIndex(dis, index, bw, bins_number, bins_item);

  auto r1 = lctx->GetStats()->sent_actions.load();
  auto c1 = lctx->GetStats()->sent_bytes.load();
  SPDLOG_INFO("Argmax : sent actions: {}, comm cost: {} KB", r1 - r0,
              (c1 - c0) / 1024.0);

  auto max_value = _out[0];
  auto max_index = _out[1];

  emp::NetIO* gc_io =
      new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1", EmpPort.getValue());
  emp::setup_semi_honest(gc_io, rank + 1);

  size_t initial_counter = gc_io->counter;
  auto topk_id =
      gc_topk(max_value, max_index, group_bin_number, group_k_number);

  size_t naive_topk_comm = gc_io->counter - initial_counter;
  SPDLOG_INFO("Communication for test_naive_topk: {} KB",
              naive_topk_comm / 1024.0);
  emp::finalize_semi_honest();
  // DISPATCH_ALL_FIELDS(spu::FM32, " ", [&]() {
  //
  // });

  return 0;
}
