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

using namespace panther;
using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9530,127.0.0.1:9531"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

llvm::cl::opt<uint32_t> EmpPort("emp_port", llvm::cl::init(7111),
                                llvm::cl::desc("emp port"));

llvm::cl::opt<float> Delta("delta", llvm::cl::init(0.01),
                           llvm::cl::desc("delta"));

llvm::cl::opt<uint32_t> Topk("k", llvm::cl::init(10), llvm::cl::desc("delta"));

llvm::cl::opt<uint32_t> NumT("t", llvm::cl::init(1),
                             llvm::cl::desc("number of threads"));

const int64_t n = 1000000;
const size_t bw = 24;
const size_t compare_radix = 4;

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

std::unique_ptr<spu::SPUContext> MakeSPUContext(size_t num_thread) {
  auto lctx = MakeLink(Parties.getValue(), Rank.getValue());
  spu::RuntimeConfig config;
  config.set_protocol(spu::ProtocolKind::CHEETAH);
  config.set_field(spu::FM32);
  config.set_max_concurrency(num_thread);
  auto hctx = std::make_unique<spu::SPUContext>(config, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  size_t num_thread = NumT.getValue();
  yacl::set_num_threads(num_thread);

  uint32_t k = Topk.getValue();
  float delta = Delta.getValue();
  int64_t l = static_cast<size_t>(k / delta);
  int64_t bin_size = n / l;
  SPDLOG_INFO("n:{} k:{} l:{} delta:{} bin_size:{}", n, k, l, delta, bin_size);

  auto values = spu::mpc::ring_rand(spu::FM32, {l, bin_size});
  auto idx = spu::mpc::ring_rand(spu::FM32, {l, bin_size});
  size_t id_bw = std::ceil(std::log2(n));

  auto hctx = MakeSPUContext(num_thread);
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = Rank.getValue();
  emp::NetIO* gc_io =
      new emp::NetIO(rank == ALICE ? nullptr : "127.0.0.1", EmpPort.getValue());
  gc_io->sync();

  auto* comm = kctx->getState<spu::mpc::Communicator>();
  auto* ot_state = kctx->getState<spu::mpc::cheetah::CheetahOTState>();

  auto nworkers = ot_state->maximum_instances();
  // size_t nworkers = 32;

  for (size_t i = 0; i < nworkers; i++) {
    ot_state->LazyInit(comm, i);
  }

  auto r0 = lctx->GetStats()->sent_actions.load();
  auto c0 = lctx->GetStats()->sent_bytes.load();
  const auto topk_s = std::chrono::system_clock::now();
  BatchArgmaxProtocol batch_argmax(kctx, compare_radix);
  auto out = batch_argmax.ComputeWithIndex(values, idx, bw, 0, l, bin_size);
  auto r1 = lctx->GetStats()->sent_actions.load();
  auto c1 = lctx->GetStats()->sent_bytes.load();
  SPDLOG_INFO(
      "Batch number: {}, element number: {}, Argmax server sent actions: "
      "{}, "
      "Argmin comm: {} MB",
      l, bin_size, r1 - r0, (c1 - c0) / 1024.0 / 1024.0);

  std::vector<uint32_t> l_value(l);
  std::vector<uint32_t> l_index(l);
  emp::setup_semi_honest(gc_io, rank);

  auto start_topk = gc_io->counter;
  gc_io->flush();
  gc::TopK(l, k, bw, id_bw, l_value, l_index);
  gc_io->flush();
  auto end_topk = gc_io->counter;
  SPDLOG_INFO("GC comm: {} MB", (end_topk - start_topk) / 1024.0 / 1024.0);
  SPDLOG_INFO("All comm: {} MB",
              (c1 - c0 + end_topk - start_topk) / 1024.0 / 1024.0);
  emp::finalize_semi_honest();
  const auto topk_e = std::chrono::system_clock::now();
  const DurationMillis Topk_time = topk_e - topk_s;
  SPDLOG_INFO("Distance cmp time: {} ms", Topk_time.count());
  // SPDLOG_INFO("Bootstrap time: {} ms", ot_state->get(0));
}