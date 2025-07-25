#include <chrono>
#include <random>
#include <set>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "batch_min.h"
#include "dist_cmp.h"
#include "experimental/panther/bitwidth_adjust.h"
#include "experimental/panther/customize_pir/seal_mpir.h"
#include "gc_topk.h"
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

namespace panther {

std::vector<std::vector<uint32_t>> RandData(size_t n, size_t dim);

std::vector<std::vector<uint32_t>> RandIdData(size_t n, size_t dims,
                                              size_t range);
std::vector<std::vector<uint32_t>> read_data(size_t n, size_t dim,
                                             string filename);

std::vector<std::vector<uint32_t>> RandClusterPoint(size_t point_number,
                                                    size_t dim);

std::vector<std::vector<uint32_t>> FixPirResult(
    std::vector<std::vector<uint32_t>>& pir_result, size_t logt,
    size_t shift_bits, size_t target_bits, int64_t num_points,
    int64_t points_dim, const std::shared_ptr<spu::KernelEvalContext>& ct);
std::vector<size_t> GcTopkCluster(spu::NdArrayRef& value,
                                  spu::NdArrayRef& index,
                                  const std::vector<int64_t>& g_bin_num,
                                  const std::vector<int64_t>& g_k_number,
                                  size_t bw_value, size_t bw_index,
                                  emp::NetIO* gc_io);

spu::seal_pir::MultiQueryClient PrepareMpirClient(
    size_t batch_number, uint32_t ele_number, uint32_t ele_size,
    std::shared_ptr<yacl::link::Context>& lctx, size_t N, size_t logt);

std::vector<uint8_t> PirData(size_t element_number, size_t element_size,
                             std::vector<std::vector<uint32_t>>& ps,
                             std::vector<std::vector<uint32_t>>& ptoc,
                             size_t pir_logt, uint32_t max_c_ps,
                             size_t pir_fixt);

spu::seal_pir::MultiQueryServer PrepareMpirServer(
    size_t batch_number, size_t ele_number, size_t ele_size,
    std::shared_ptr<yacl::link::Context>& lctx, size_t N, size_t logt,
    std::vector<uint8_t>& db_bytes);

std::vector<std::vector<uint32_t>> FixPirResult(
    std::vector<std::vector<uint32_t>>& pir_result, size_t logt,
    size_t shift_bits, size_t target_bits, int64_t num_points,
    int64_t points_dim, const std::shared_ptr<spu::KernelEvalContext>& ct);
spu::NdArrayRef PrepareBatchArgmin(std::vector<uint32_t>& input,
                                   const std::vector<int64_t>& num_center,
                                   const std::vector<int64_t>& num_bin,
                                   spu::Shape shape, uint32_t init_v);
void PirResultForm(const std::vector<std::vector<uint32_t>>& input,
                   std::vector<std::vector<uint32_t>>& p,
                   std::vector<uint32_t>& id, std::vector<uint32_t>& p_2,
                   size_t dims, size_t message);
std::unique_ptr<spu::SPUContext> MakeSPUContext(const std::string& parties,
                                                size_t rank);
std::vector<uint32_t> Truncate(
    std::vector<uint32_t>& pir_result, size_t logt, size_t shift_bits,
    const std::shared_ptr<spu::KernelEvalContext>& ct);

std::vector<int32_t> GcEndTopk(std::vector<uint32_t>& value,
                               std::vector<uint32_t>& id,
                               spu::NdArrayRef& stash_v,
                               spu::NdArrayRef& stash_id, size_t bw_value,
                               size_t bw_id, size_t bw_stash,
                               size_t discard_stash, size_t start_point,
                               size_t n_stash, size_t k, emp::NetIO* gc_io);

std::vector<std::vector<uint32_t>> FixPirResultOpt(
    std::vector<std::vector<uint32_t>>& pir_result, size_t logt,
    size_t shift_bits, size_t target_bits, int64_t num_points,
    int64_t total_size, int64_t points_dim, size_t rank,
    const std::shared_ptr<spu::KernelEvalContext>& ct,
    const std::vector<uint32_t>& q = std::vector<uint32_t>());
}  // namespace panther