#include "common.h"

#include "libspu/mpc/cheetah/type.h"
using DurationMillis = std::chrono::duration<double, std::milli>;
using namespace spu;

namespace panther {

std::vector<std::vector<uint32_t>> RandData(size_t n, size_t dims) {
  SPDLOG_INFO("Generate random data: element number:{} element size:{}", n,
              dims);
  std::vector<std::vector<uint32_t>> db_data(n, std::vector<uint32_t>(dims));

  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < dims; j++) {
      db_data[i][j] = rand() % 256;
    }
  }
  return db_data;
}

std::vector<std::vector<uint32_t>> RandIdData(size_t n, size_t dims,
                                              size_t range) {
  SPDLOG_INFO("Generate fake ID: element number:{} element size:{}", n, dims);
  std::vector<std::vector<uint32_t>> db_data(n, std::vector<uint32_t>(dims));

  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < dims; j++) {
      db_data[i][j] = (i * dims + j) % range;
    }
  }
  return db_data;
}

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

std::unique_ptr<spu::SPUContext> MakeSPUContext(const std::string& parties,
                                                size_t rank) {
  auto lctx = MakeLink(parties, rank);
  spu::RuntimeConfig config;
  config.set_protocol(spu::ProtocolKind::CHEETAH);
  config.set_field(spu::FM32);
  auto hctx = std::make_unique<spu::SPUContext>(config, lctx);
  spu::mpc::Factory::RegisterProtocol(hctx.get(), lctx);
  return hctx;
}
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

std::vector<size_t> GcTopkCluster(spu::NdArrayRef& value,
                                  spu::NdArrayRef& index,
                                  const std::vector<int64_t>& g_bin_num,
                                  const std::vector<int64_t>& g_k_number,
                                  size_t bw_value, size_t bw_index,
                                  emp::NetIO* gc_io) {
  SPU_ENFORCE_EQ(g_bin_num.size(), g_k_number.size());
  int64_t sum_k = 0;
  for (size_t i = 0; i < g_k_number.size() - 1; i++) {
    sum_k += g_k_number[i];
  }
  std::vector<uint64_t> res(sum_k);
  std::vector<uint32_t> tmp_res(sum_k);

  using namespace spu;
  for (size_t begin = 0; begin < g_bin_num.size() - 1; begin++) {
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
      auto bin = uint32_t(std::ceil((double)real_bin / k)) * k;
      std::vector<uint32_t> input_value(bin, 10000000);
      std::vector<uint32_t> input_index(bin);
      memcpy(&input_value[0], &xval[now_bin], real_bin * sizeof(uint32_t));
      memcpy(&input_index[0], &xidx[now_bin], real_bin * sizeof(uint32_t));
      auto start = std::chrono::system_clock::now();

      gc_io->flush();
      auto topk_id = panther::gc::TopK(bin, k, bw_value, bw_index, input_value,
                                       input_index);
      gc_io->flush();
      auto end = std::chrono::system_clock::now();
      const DurationMillis topk_time = end - start;

      memcpy(&tmp_res[now_k], topk_id.data(), k * sizeof(uint32_t));
    });
  }
  for (int64_t i = 0; i < sum_k; i++) {
    res[i] = tmp_res[i];
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
  size_t mask = (1 << (3 * 8)) - 1;
  // std::cout << element_number << " " << element_size << std::endl;
  for (uint64_t i = 0; i < element_number; i++) {
    for (uint64_t j = 0; j < element_size; j++) {
      if (num_points == dims) {
        if (count < 3) {
          db_raw_data[i * element_size + j] =
              (index >> ((2 - count) * 8)) & 255;
        } else {
          db_raw_data[i * element_size + j] = (p_2 >> ((5 - count) * 8)) & 255;
        }
        count++;
        if (count == 6) {
          count = 0;
          id_point++;
          if (id_point == max_c_ps) {
            id_point = 0;
          }
          num_points = 0;
          p_2 = 0;
        }
      } else {
        index = ptoc[i][id_point];
        if (index == 111111112) {
          db_raw_data[i * element_size + j] = 0;
          p_2 = mask >> 2;
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
    std::vector<uint8_t>& encoded_db) {
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
  mpir_server.RecvGaloisKeys(lctx);
  mpir_server.RecvPublicKey(lctx);
  mpir_server.SetDbSeperateId(encoded_db);
  return mpir_server;
}

std::vector<std::vector<uint32_t>> FixPirResultOpt(
    std::vector<std::vector<uint32_t>>& pir_result, size_t logt,
    size_t shift_bits, size_t target_bits, int64_t num_points,
    int64_t total_size, int64_t points_dim, size_t rank,
    const std::shared_ptr<spu::KernelEvalContext>& ct,
    const std::vector<uint32_t>& q) {
  std::vector<std::vector<uint32_t>> res(
      num_points, std::vector<uint32_t>(total_size - points_dim + 1, 0));
  int64_t query_size = pir_result.size();
  int64_t num_slot = pir_result[0].size();

  auto nd_inp = spu::mpc::ring_zeros(spu::FM32, {num_points * total_size});

  spu::pforeach(0, query_size, [&](int64_t i) {
    std::memcpy(&nd_inp.at(i * num_slot), pir_result[i].data(), num_slot * 4);
  });

  // spu::NdArrayRef result;
  auto trun_out = spu::mpc::cheetah::TiledDispatchOTFunc(
      ct.get(), nd_inp,
      [&](const spu::NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        BitwidthAdjustProtocol prot(base_ot);
        auto o = prot.TrunReduceCompute(input, logt, shift_bits);

        spu::mpc::ring_bitmask_(o, 0, logt - shift_bits);
        return o;
        // return prot.ExtendComputeOpt(o, logt - shift_bits,
        //  target_bits - logt + shift_bits);
      });

  spu::NdArrayRef msb = spu::mpc::ring_rshift(trun_out, logt - shift_bits - 1);

  spu::NdArrayRef cmp_out;
  const auto field = trun_out.eltype().as<spu::Ring2k>()->field();
  if (rank == 0) {
    auto msg = spu::mpc::ring_ones(field, {int64_t(num_points * total_size)});
    for (int64_t i = 0; i < num_points; i++) {
      memcpy(&msg.at(i * total_size), &(q[0]), points_dim * 4);
      // memset(&msg.at(i * total_size + points_dim), uint32_t(1),
      //  (total_size - points_dim) * 4);
    }
    auto bx = spu::mpc::ring_mul(msg, msb);
    spu::mpc::ring_sub_(msg, bx);
    spu::mpc::ring_sub_(msg, bx);

    cmp_out = spu::mpc::cheetah::TiledDispatchOTFunc(
        ct.get(), msg,
        [&](const spu::NdArrayRef& msg,
            const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>&
                base_ot) { return base_ot->PrivateMulxSend(msg); });

    spu::mpc::ring_add_(cmp_out, bx);
  } else {
    auto msg = spu::mpc::ring_zeros(field, {int64_t(num_points * total_size)});
    cmp_out = spu::mpc::cheetah::TiledDispatchOTFunc(
        ct.get(), msg, msb,
        [&](const spu::NdArrayRef& msg, const spu::NdArrayRef& select,
            const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>&
                base_ot) {
          return base_ot->PrivateMulxRecv(
              msg,
              select.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1)));
        });
  }

  pir_result.resize(num_points);
  for (auto& r : pir_result) {
    r.resize(total_size);
  }

  spu::mpc::ring_bitmask_(trun_out, 0, logt - shift_bits - 1);
  spu::pforeach(0, num_points, [&](int64_t i) {
    std::memcpy(pir_result[i].data(), &trun_out.at(i * total_size),
                total_size * 4);
  });

  using namespace spu;
  spu::mpc::ring_lshift_(cmp_out, logt - shift_bits - 1);
  uint32_t MASK = (1 << target_bits) - 1;
  DISPATCH_ALL_FIELDS(FM32, "deal_with_sum", [&]() {
    auto xout = spu::NdArrayView<ring2k_t>(cmp_out);

    spu::pforeach(0, num_points, [&](int64_t i) {
      uint32_t sum = 0;
      for (int64_t point_i = 0; point_i < points_dim; point_i++) {
        sum += xout[i * total_size + point_i];
      }
      res[i][0] = sum & MASK;
      for (int64_t m_i = points_dim; m_i < total_size; m_i++) {
        pir_result[i][m_i] =
            (pir_result[i][m_i] - xout[i * total_size + m_i]) & MASK;
      }
    });
  });
  return res;
}

void PirResultForm(const std::vector<std::vector<uint32_t>>& input,
                   std::vector<std::vector<uint32_t>>& p,
                   std::vector<uint32_t>& id, std::vector<uint32_t>& p_2,
                   size_t dims, size_t message) {
  SPU_ENFORCE(input[0].size() == dims + 2 * message);
  int64_t numel = input.size();
  pforeach(0, numel, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      memcpy(&p[i][0], &input[i][0], dims * 4);
      id[i] = 0;
      p_2[i] = 0;
      for (size_t m_i = dims; m_i < dims + message; m_i++) {
        id[i] = id[i] << 8;
        p_2[i] = p_2[i] << 8;
        id[i] += input[i][m_i];
        p_2[i] += input[i][m_i + message];
      }
    }
  });
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
        BitwidthAdjustProtocol prot(base_ot);
        return prot.TrunReduceCompute(input, logt, shift_bits);
      });

  std::memcpy(result.data(), &out.at(0), num_points * out.elsize());

  return result;
}

std::vector<int32_t> GcEndTopk(std::vector<uint32_t>& value,
                               std::vector<uint32_t>& id,
                               spu::NdArrayRef& stash_v,
                               spu::NdArrayRef& stash_id, size_t bw_value,
                               size_t bw_id, size_t bw_stash,
                               size_t discard_stash, size_t start_point,
                               size_t n_stash, size_t k, emp::NetIO* gc_io) {
  SPU_ENFORCE_EQ(stash_v.elsize(), stash_id.elsize());
  SPU_ENFORCE_EQ(value.size(), id.size());

  auto pad_s = uint32_t(std::ceil((double)n_stash / k)) * k;
  std::vector<uint32_t> s_value(pad_s, 10000);
  std::vector<uint32_t> s_id(pad_s);
  using namespace spu;
  DISPATCH_ALL_FIELDS(spu::FM32, "end_topk", [&]() {
    auto xval = NdArrayView<ring2k_t>(stash_v);
    auto xidx = NdArrayView<ring2k_t>(stash_id);

    memcpy(&s_value[0], &xval[start_point], n_stash * sizeof(uint32_t));
    memcpy(&s_id[0], &xidx[start_point], n_stash * sizeof(uint32_t));
  });

  size_t n = value.size();
  gc_io->flush();
  auto topk_id = panther::gc::EndTopK(n, k, bw_value, bw_id, pad_s, bw_stash,
                                      discard_stash, value, id, s_value, s_id);
  gc_io->flush();
  SPU_ENFORCE_EQ(topk_id.size(), k);
  return topk_id;
};

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
  for (int64_t i = 0; i < query_size; i++) {
    std::memcpy(&nd_inp.at(i * num_slot), pir_result[i].data(), num_slot * 4);
  }
  auto out = spu::mpc::cheetah::TiledDispatchOTFunc(
      ct.get(), nd_inp,
      [&](const spu::NdArrayRef& input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base_ot) {
        BitwidthAdjustProtocol prot(base_ot);
        auto trun_value = prot.TrunReduceCompute(input, logt, shift_bits);
        spu::mpc::ring_bitmask_(trun_value, 0, logt - shift_bits);
        return prot.ExtendComputeOpt(trun_value, logt - shift_bits,
                                     target_bits - logt + shift_bits);
      });

  spu::mpc::ring_bitmask_(out, 0, target_bits);
  spu::pforeach(0, num_points, [&](int64_t i) {
    std::memcpy(result[i].data(), &out.at(i * points_dim), points_dim * 4);
  });
  return result;
}
}  // namespace panther
