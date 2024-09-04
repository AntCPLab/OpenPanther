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

namespace sanns {

std::vector<std::vector<uint32_t>> read_data(size_t n, size_t dim,
                                             string filename);

std::vector<std::vector<uint32_t>> RandClusterPoint(size_t point_number,
                                                    size_t dim);

std::vector<uint32_t> RandQuery(size_t num_dims);

std::vector<uint32_t> GcTopk(spu::NdArrayRef& value, spu::NdArrayRef& index,
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
}  // namespace sanns