#include "panther_util.h"
using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace spu;

namespace sanns {

// This function is used to transform input to parellized form
spu::NdArrayRef PrepareBatchArgmin(std::vector<uint32_t>& input,
                                   const std::vector<int64_t>& num_center,
                                   const std::vector<int64_t>& num_bin,
                                   spu::Shape shape, uint32_t init_v) {
  spu::FieldType field = spu::FM32;
  int64_t sum_bin = shape[0];
  int64_t max_bin_size = shape[1];

  size_t group_num = num_center.size();
  SPU_ENFORCE(num_bin.size() == group_num);

  NdArrayRef res(makeType<RingTy>(spu::FM32), shape);
  auto numel = res.numel();
  DISPATCH_ALL_FIELDS(field, "Init", [&]() {
    NdArrayView<ring2k_t> _res(res);
    pforeach(0, numel, [&](int64_t idx) { _res[idx] = ring2k_t(init_v); });
  });

  spu::pforeach(0, sum_bin, [&](int64_t begin, int64_t end) {
    for (int64_t bin_index = begin; bin_index < end; bin_index++) {
      int64_t sum = num_bin[0];
      size_t point_sum = 0;
      // in which group
      size_t group_i = 0;
      while (group_i < group_num) {
        if (bin_index < sum) {
          break;
        }
        sum += num_bin[group_i + 1];
        point_sum += num_center[group_i];
        group_i++;
      }
      sum -= num_bin[group_i];
      int64_t bin_size =
          std::ceil((float)num_center[group_i] / num_bin[group_i]);

      int64_t index_in_group = bin_index - sum;
      if (bin_size * index_in_group < num_center[group_i]) {
        auto now_bin_size =
            min(bin_size, num_center[group_i] - index_in_group * bin_size);
        DISPATCH_ALL_FIELDS(spu::FM32, "trans_to_topk", [&]() {
          auto xres = NdArrayView<ring2k_t>(res);
          mempcpy(&xres[bin_index * max_bin_size],
                  &input[point_sum + index_in_group * bin_size],
                  now_bin_size * 4);
        });
      }
    }
  });
  return res;
}
std::vector<std::vector<uint32_t>> read_data(size_t n, size_t dim,
                                             string filename) {
  std::ifstream inputFile("./experimental/ann/" + filename);
  if (!inputFile.is_open()) {
    std::cerr << "Can't open it!" << std::endl;
  }
  std::vector<std::vector<uint32_t>> numbers(n, vector<uint32_t>(dim));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      if (!(inputFile >> numbers[i][j])) {
        std::cerr << "Read Error!" << std::endl;
        std::cerr << filename << std::endl;
      }
    }
  }
  inputFile.close();
  std::cout << "input data: (" << numbers.size() << ", " << numbers[0].size()
            << ")" << std::endl;
  return numbers;
}

std::vector<std::vector<uint32_t>> RandClusterPoint(size_t point_number,
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

std::vector<uint32_t> RandQuery(size_t num_dims) {
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

      // gc_io->flush();

      // std::cout << gc_io->is_server << std::endl;
      // std::cout << "test" << std::endl;
      auto topk_id =
          sanns::gc::TopK(bin, k, bw_value, bw_index, input_value, input_index);
      // gc_io->flush();
      auto end = std::chrono::system_clock::now();
      const DurationMillis topk_time = end - start;

      memcpy(&res[now_k], topk_id.data(), k * sizeof(uint32_t));
    });
  }
  return res;
};

spu::seal_pir::MultiQueryClient PrepareMpirClient(
    size_t batch_number, uint32_t ele_number, uint32_t ele_size,
    std::shared_ptr<yacl::link::Context>& lctx, size_t N, size_t logt) {
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

std::vector<uint8_t> PirData(size_t element_number, size_t element_size,
                             std::vector<std::vector<uint32_t>>& ps,
                             std::vector<std::vector<uint32_t>>& ptoc,
                             size_t pir_logt, uint32_t max_c_ps,
                             size_t pir_fixt) {
  size_t dims = ps[0].size();
  SPDLOG_INFO("DB: element number:{} element size:{}", element_number,
              element_size);
  std::vector<uint8_t> db_data(element_number * element_size * 4);
  std::vector<uint32_t> db_raw_data(element_number * element_size);
  size_t num_points = 0;
  uint32_t id_point = 0;
  size_t count = 0;
  size_t p_2 = 0;
  size_t index = 0;
  size_t mask = (1 << pir_logt) - 1;
  // std::cout << element_number << " " << element_size << std::endl;
  for (uint64_t i = 0; i < element_number; i++) {
    for (uint64_t j = 0; j < element_size; j++) {
      if (num_points == dims) {
        if (count < 3) {
          db_raw_data[i * element_size + j] = (index >> (count * 8)) & 127;
        } else {
          db_raw_data[i * element_size + j] = (p_2 >> ((count - 3) * 8)) & 127;
        }
        count++;
        if (count == 6) {
          count = 0;
          id_point++;
          if (id_point == max_c_ps) {
            id_point = 0;
          }
          num_points = 0;
        }
      } else {
        index = ptoc[i][id_point];
        if (index == 111111112) {
          db_raw_data[i * element_size + j] = 0;
          p_2 = mask;
        } else {
          db_raw_data[i * element_size + j] = ps[index][num_points];
          p_2 += ps[index][num_points] * ps[index][num_points];
        }
        num_points++;
      }
      // std::cout << index << std::endl;
      db_raw_data[i * element_size + j] =
          (db_raw_data[i * element_size + j] << pir_fixt) + 1;
    }
  }
  memcpy(db_data.data(), db_raw_data.data(), element_number * element_size * 4);

  return db_data;
}

spu::seal_pir::MultiQueryServer PrepareMpirServer(
    size_t batch_number, size_t ele_number, size_t ele_size,
    std::shared_ptr<yacl::link::Context>& lctx, size_t N, size_t logt,
    std::vector<uint8_t>& db_bytes) {
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

void PirResultForm(const std::vector<std::vector<uint32_t>>& input,
                   std::vector<std::vector<uint32_t>>& p,
                   std::vector<std::vector<uint32_t>>& id,
                   std::vector<std::vector<uint32_t>>& p_2, size_t dims,
                   size_t message) {
  SPU_ENFORCE(input[0].size() == dims + 2 * message);
  int64_t numel = input.size();
  pforeach(0, numel, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      memcpy(&p[i], &input[i], dims * 4);
      id[i] = 0;
      p_2[i] = 0;
      for (size_t m_i = dims; m_i < dims + message; m_i++) {
        id[i] = id[i] << size_of(uint8_t);
        p_2[i] = p_2[i] << size_of(uint8_t);
        id[i] += input[m_i];
        p_2[i] = input[m_i + message];
      }
    });
};
}  // namespace sanns
