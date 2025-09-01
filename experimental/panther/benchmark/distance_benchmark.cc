#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "../protocol/batch_min.h"
#include "../protocol/dist_cmp.h"
#include "../protocol/bitwidth_adjust.h"
#include "../protocol/customize_pir/seal_mpir.h"
#include "../protocol/gc_topk.h"
#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
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
using namespace panther;
using DurationMillis = std::chrono::duration<double, std::milli>;

llvm::cl::opt<std::string> Parties(
    "parties", llvm::cl::init("127.0.0.1:9522,127.0.0.1:9523"),
    llvm::cl::desc("server list, format: host1:port1[,host2:port2, ...]"));

llvm::cl::opt<uint32_t> Rank("rank", llvm::cl::init(0),
                             llvm::cl::desc("self rank"));

llvm::cl::opt<uint32_t> EmpPort("emp_port", llvm::cl::init(7111),
                                llvm::cl::desc("emp port"));

const size_t repeat_times = 50;

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
      points[i][j] = rand() % 256;
    }
  }
  return points;
};

std::vector<uint32_t> ReadQuery(size_t num_dims) {
  std::vector<uint32_t> q(num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    q[i] = rand() % 256;
  }
  return q;
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  yacl::set_num_threads(32);
  srand(0);
  const size_t logt = 24;
  // FHE poly degree
  const size_t dis_N = 2048;
  const size_t dims = 128;
  const size_t cluster_num = 100000;

  // context init
  auto hctx = MakeSPUContext();
  auto kctx = std::make_shared<spu::KernelEvalContext>(hctx.get());
  auto lctx = hctx->lctx();
  auto rank = Rank.getValue();

  lctx->SetThrottleWindowSize(0);
  if (rank == 0) {
    // prepare mpir client
    // auto to = lctx->NextRank();
    DisClient dis_client(dis_N, logt, lctx);
    dis_client.SendPublicKey();

    // prepare distance compute client
    // (HE parameters for distance calculation are independent of the PIR)

    auto q = ReadQuery(dims);
    auto total_time_s = std::chrono::system_clock::now();
    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();

    for (size_t i = 0; i < repeat_times; i++) {
      dis_client.GenerateQuery(q);
      auto dis = dis_client.RecvReplySS(cluster_num);
    }

    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();

    auto distance_cmp_e = std::chrono::system_clock::now();

    const DurationMillis dis_cmp_time_2 = distance_cmp_e - total_time_s;
    SPDLOG_INFO("Distance cmp time: {} ms",
                dis_cmp_time_2.count() / repeat_times);
    SPDLOG_INFO("Distance client sent actions: {}, Distance comm: {} MB",
                (r1 - r0) / repeat_times,
                (c1 - c0) / 1024.0 / 1024.0 / repeat_times);
  } else {
    DisServer dis_server(dis_N, logt, lctx);
    dis_server.RecvPublicKey();
    auto total_time_s = std::chrono::system_clock::now();

    auto ps = ReadClusterPoint(cluster_num, dims);

    auto r0 = lctx->GetStats()->sent_actions.load();
    auto c0 = lctx->GetStats()->sent_bytes.load();

    for (size_t i = 0; i < repeat_times; i++) {
      auto q = dis_server.RecvQuery(dims);
      auto dis = dis_server.DoDistanceCmpWithH2A(ps, q);
    }

    auto r1 = lctx->GetStats()->sent_actions.load();
    auto c1 = lctx->GetStats()->sent_bytes.load();
    auto distance_cmp_e = std::chrono::system_clock::now();
    const DurationMillis dis_cmp_time = distance_cmp_e - total_time_s;

    SPDLOG_INFO("Distance cmp time: {} ms",
                dis_cmp_time.count() / repeat_times);
    SPDLOG_INFO("Distance server sent actions: {}, Distance comm: {} MB",
                (r1 - r0) / repeat_times,
                (c1 - c0) / 1024.0 / 1024.0 / repeat_times);
  }

  lctx->WaitLinkTaskFinish();
  return 0;
}
