#include "dist_cmp.h"

#include "seal/util/rlwe.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace sanns {

using DurationMillis = std::chrono::duration<double, std::milli>;
DisClient::DisClient(size_t degree, size_t logt,
                     const std::shared_ptr<yacl::link::Context> &conn) {
  degree_ = degree;
  seal_params_ =
      std::make_unique<seal::EncryptionParameters>(seal::scheme_type::bfv);
  seal_params_->set_poly_modulus_degree(degree);
  if (8192 == degree_)
    seal_params_->set_coeff_modulus(
        seal::CoeffModulus::Create(degree, {45, 45, 45, 45}));
  else {
    seal_params_->set_coeff_modulus(seal::CoeffModulus::BFVDefault(degree));
  }
  seal_params_->set_plain_modulus(1ULL << logt);
  context_ = std::make_unique<seal::SEALContext>(*(seal_params_));
  yacl::set_num_threads(1);
  keygen_ = std::make_unique<seal::KeyGenerator>(*context_);
  seal::SecretKey secret_key = keygen_->secret_key();

  keygen_->create_public_key(public_key_);
  encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_);
  encryptor_->set_secret_key(secret_key);
  decryptor_ = std::make_unique<seal::Decryptor>(*context_, secret_key);
  conn_ = conn;
}

void DisClient::SendPublicKey() {
  seal::PublicKey pubkey = GetPublicKey();

  auto pubkey_buffer =
      spu::mpc::cheetah::EncodeSEALObject<seal::PublicKey>(pubkey);

  conn_->Send(conn_->NextRank(), pubkey_buffer,
              fmt::format("send public key to rank-{}", conn_->Rank()));
};

std::vector<seal::Ciphertext> DisClient::GenerateQuery(
    std::vector<uint32_t> &q) {
  size_t q_size = q.size();
  const auto query_s = std::chrono::system_clock::now();
  std::vector<seal::Ciphertext> enc_q(q_size);
  std::vector<seal::Plaintext> plain_q(
      q_size, seal::Plaintext(seal_params_->poly_modulus_degree()));
  yacl::parallel_for(0, q_size, [&](size_t begin, size_t end) {
    // std::cout << begin << " " << end << std::endl;
    for (size_t i = begin; i < end; i++) {
      plain_q[i][0] = q[i];
      encryptor_->encrypt_symmetric(plain_q[i], enc_q[i]);
    }
  });
  std::vector<yacl::Buffer> q_buffer(q_size);

  yacl::parallel_for(0, q_size, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      q_buffer[i] = spu::mpc::cheetah::EncodeSEALObject(enc_q[i]);
    }
  });

  const auto query_e = std::chrono::system_clock::now();
  const DurationMillis query_time = query_e - query_s;
  // SPDLOG_INFO("Query time: {} ms", query_time.count());
  int next = conn_->NextRank();

  auto tag = "query";

  for (size_t i = 0; i < q_size; i++) {
    conn_->SendAsync(next, q_buffer[i], tag);
  }
  // conn_->Flush();
  // conn_->WaitLinkTaskFinish();
  return enc_q;
}

spu::NdArrayRef DisClient::RecvReply(spu::Shape r_padding_shape,
                                     size_t num_points) {
  size_t num_rlwes = std::ceil(static_cast<double>(num_points) / degree_);
  std::vector<seal::Ciphertext> reply(num_rlwes);
  auto next = conn_->NextRank();
  std::vector<yacl::Buffer> recv(num_rlwes);

  auto recv_s = std::chrono::system_clock::now();
  for (size_t i = 0; i < num_rlwes; i++) {
    recv[i] = conn_->Recv(next, "distance");
  }

  auto recv_e = std::chrono::system_clock::now();
  DurationMillis recv_time = recv_e - recv_s;

  // SPDLOG_INFO("Recv time: {} ms", recv_time.count());
  for (size_t i = 0; i < num_rlwes; i++) {
    spu::mpc::cheetah::DecodeSEALObject(recv[i], *context_, &reply[i]);
  }
  return DecodeReply(reply, r_padding_shape, num_points);
}

std::vector<uint32_t> DisClient::RecvReply(size_t num_points) {
  size_t num_rlwes = std::ceil(static_cast<double>(num_points) / degree_);
  std::vector<seal::Ciphertext> reply(num_rlwes);
  auto next = conn_->NextRank();
  auto recv_s = std::chrono::system_clock::now();
  for (size_t i = 0; i < num_rlwes; i++) {
    auto recv = conn_->Recv(next, "distance");
    spu::mpc::cheetah::DecodeSEALObject(recv, *context_, &reply[i]);
  }
  auto recv_e = std::chrono::system_clock::now();
  DurationMillis recv_time = recv_e - recv_s;
  // SPDLOG_INFO("Recv time: {} ms", recv_time.count());
  return DecodeReply(reply, num_points);
}

spu::NdArrayRef DisClient::DecodeReply(std::vector<seal::Ciphertext> &reply,
                                       spu::Shape r_padding_shape,
                                       size_t num_points) {
  std::vector<seal::Plaintext> plain_reply(reply.size());

  auto vec_reply = spu::mpc::ring_zeros(spu::FM32, r_padding_shape);
  using namespace spu;
  DISPATCH_ALL_FIELDS(spu::FM32, "", [&]() {
    auto dim2 = vec_reply.dim(1);
    auto dim1 = vec_reply.dim(0);
    auto xvec_reply = NdArrayView<ring2k_t>(vec_reply);
    for (size_t i = 0; i < reply.size(); i++) {
      decryptor_->decrypt(reply[i], plain_reply[i]);
      for (size_t j = 0; j < plain_reply[i].coeff_count(); j++) {
        // TODO(ljy): row and col need to change to the really bin
        auto index = i * degree_ + j;

        if (index < dim2 * dim1) {
          auto row = index / dim2;
          auto col = index % dim2;

          xvec_reply[row * dim2 + col] = plain_reply[i][j];
        }
      }
    }
  });
  return vec_reply;
}
std::vector<uint32_t> DisClient::DecodeReply(
    std::vector<seal::Ciphertext> &reply, size_t num_points) {
  std::vector<seal::Plaintext> plain_reply(reply.size());
  std::vector<uint32_t> vec_reply(num_points);
  uint32_t count = 0;
  for (size_t i = 0; i < reply.size() && count < num_points; i++) {
    decryptor_->decrypt(reply[i], plain_reply[i]);
    for (size_t j = 0; j < plain_reply[i].coeff_count() && count < num_points;
         j++) {
      vec_reply[count] = plain_reply[i][j];
      count++;
    }
  }
  return vec_reply;
}

//----------------------------------------------------------------------------------------------
// Server Impl
// TODO(ljy): using CPRNG to generate the random polynomial
struct DisServer::Impl : public spu::mpc::cheetah::EnableCPRNG {
 public:
  Impl(){};

 private:
  // std::vector<seal::Ciphertext> H2A(std::vector<seal::Ciphertext> &ct);
};  /// Server compute distance

DisServer::DisServer(size_t degree, size_t logt,
                     const std::shared_ptr<yacl::link::Context> &conn) {
  degree_ = degree;
  seal_params_ =
      std::make_unique<seal::EncryptionParameters>(seal::scheme_type::bfv);
  seal_params_->set_poly_modulus_degree(degree);
  if (8192 == degree_)
    seal_params_->set_coeff_modulus(
        seal::CoeffModulus::Create(degree, {45, 45, 45, 45}));
  else {
    seal_params_->set_coeff_modulus(seal::CoeffModulus::BFVDefault(degree));
  }
  auto m = seal_params_->coeff_modulus();
  // for (auto i : m) {
  //   std::cout << i.bit_count() << " ";
  // }
  // std::cout << std::endl;

  seal_params_->set_plain_modulus(1ULL << logt);
  context_ = std::make_unique<seal::SEALContext>(*(seal_params_));
  evaluator_ = std::make_unique<seal::Evaluator>(*context_);

  impl_ = std::make_shared<Impl>();

  // TODO(ljy): change the logic to be more clean;
  std::vector<seal::Modulus> raw_modulus = seal_params_->coeff_modulus();
  std::vector<seal::Modulus> modulus = seal_params_->coeff_modulus();
  if (degree != 2048) {
    modulus.pop_back();
  }
  // modulus.pop_back();
  seal_params_->set_coeff_modulus(modulus);
  seal::SEALContext test_context(*seal_params_, false,
                                 seal::sec_level_type::none);
  seal_params_->set_coeff_modulus(raw_modulus);
  msh_ = std::make_unique<spu::mpc::cheetah::ModulusSwitchHelper>(test_context,
                                                                  logt);
  conn_ = conn;
}

std::vector<seal::Ciphertext> DisServer::RecvQuery(size_t query_size) {
  std::vector<seal::Ciphertext> q(query_size);
  auto next = conn_->NextRank();
  const auto recv_s = std::chrono::system_clock::now();
  std::vector<yacl::Buffer> recv(query_size);
  for (size_t i = 0; i < query_size; i++) {
    recv[i] = conn_->Recv(next, "query");
  }

  const auto recv_e = std::chrono::system_clock::now();
  const DurationMillis recv_time = recv_e - recv_s;
  // SPDLOG_INFO("Recv time: {} ms", recv_time.count());

  for (size_t i = 0; i < query_size; i++) {
    spu::mpc::cheetah::DecodeSEALObject(recv[i], *context_, &q[i]);
  }
  return q;
}

std::vector<uint32_t> DisServer::DoDistanceCmp(
    std::vector<std::vector<uint32_t>> &points,
    std::vector<seal::Ciphertext> &q) {
  SPU_ENFORCE_NE(points.size(), static_cast<size_t>(0));
  const auto distance_cmp_s = std::chrono::system_clock::now();
  size_t num_rlwes = std::ceil(static_cast<double>(points.size()) / degree_);
  std::vector<seal::Plaintext> pre_points = PrePoints(points);
  std::vector<seal::Ciphertext> response(num_rlwes);
  size_t point_dim = q.size();

  yacl::parallel_for(0, point_dim, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      if (q[i].is_ntt_form() == false)
        evaluator_->transform_to_ntt_inplace(q[i]);
    }
  });
  const auto distance_ntt_e = std::chrono::system_clock::now();
  const DurationMillis dis_ntt_time = distance_ntt_e - distance_cmp_s;
  // SPDLOG_INFO("Distance NTT time: {} ms", dis_ntt_time.count());

  yacl::parallel_for(0, num_rlwes, [&](size_t begin, size_t end) {
    // std::cout << begin << " " << end << std::endl;
    for (size_t bfv_index = begin; bfv_index < end; bfv_index++) {
      seal::Ciphertext tmp;
      for (size_t i = 0; i < point_dim; i++) {
        if (i == 0) {
          evaluator_->multiply_plain(
              q[i], pre_points[bfv_index * point_dim + i], response[bfv_index]);
        } else {
          evaluator_->multiply_plain(
              q[i], pre_points[bfv_index * point_dim + i], tmp);
          evaluator_->add_inplace(response[bfv_index], tmp);
        }
      }
      evaluator_->transform_from_ntt_inplace(response[bfv_index]);
    }
  });

  // After this operation: the response will be secret shared
  const auto H2A_s = std::chrono::system_clock::now();
  auto rand_msk = H2A(response, points.size());
  const auto H2A_e = std::chrono::system_clock::now();
  const DurationMillis H2A_time = H2A_e - H2A_s;
  // SPDLOG_INFO("H2A Time: {} ms", H2A_time.count());

  std::vector<yacl::Buffer> ciphers(num_rlwes);
  for (size_t i = 0; i < num_rlwes; i++) {
    ciphers[i] = spu::mpc::cheetah::EncodeSEALObject(response[i]);
  }

  const auto distance_cmp_e = std::chrono::system_clock::now();
  const DurationMillis distance_cmp_time = distance_cmp_e - distance_cmp_s;
  // SPDLOG_INFO("Distance Compute Time : {} ms", distance_cmp_time.count());
  // TODO(ljy): H2A unit test
  int next = conn_->NextRank();
  for (size_t i = 0; i < num_rlwes; i++) {
    auto tag = "distance";
    conn_->SendAsync(next, ciphers[i], tag);
  }
  // auto tag = "distance";
  // conn_->Send(next, ciphers[num_rlwes - 1], tag);
  return rand_msk;
}

spu::NdArrayRef DisServer::DoDistanceCmp(
    std::vector<std::vector<uint32_t>> &points,
    std::vector<seal::Ciphertext> &q, spu::Shape shape) {
  SPU_ENFORCE_NE(points.size(), static_cast<size_t>(0));

  const auto distance_cmp_s = std::chrono::system_clock::now();
  size_t num_rlwes = std::ceil(static_cast<double>(points.size()) / degree_);
  std::vector<seal::Plaintext> pre_points = PrePoints(points);
  const auto prepoint_e = std::chrono::system_clock::now();
  const DurationMillis prepoint_time = prepoint_e - distance_cmp_s;
  // SPDLOG_INFO("Prepoint time: {}", prepoint_time.count());
  std::vector<seal::Ciphertext> response(num_rlwes);
  size_t point_dim = q.size();
  yacl::parallel_for(0, point_dim, [&](size_t begin, size_t end) {
    // std::cout << begin << " " << end << std::endl;
    for (size_t i = begin; i < end; i++) {
      if (q[i].is_ntt_form() == false)
        evaluator_->transform_to_ntt_inplace(q[i]);
    }
  });

  const auto distance_ntt_e = std::chrono::system_clock::now();
  const DurationMillis dis_ntt_time = distance_ntt_e - distance_cmp_s;
  // SPDLOG_INFO("Distance NTT time: {} ms", dis_ntt_time.count());

  yacl::parallel_for(0, num_rlwes, [&](size_t begin, size_t end) {
    for (size_t bfv_index = begin; bfv_index < end; bfv_index++) {
      seal::Ciphertext tmp;
      for (size_t i = 0; i < point_dim; i++) {
        if (i == 0) {
          evaluator_->multiply_plain(
              q[i], pre_points[bfv_index * point_dim + i], response[bfv_index]);
        } else {
          evaluator_->multiply_plain(
              q[i], pre_points[bfv_index * point_dim + i], tmp);
          evaluator_->add_inplace(response[bfv_index], tmp);
        }
      }
      evaluator_->transform_from_ntt_inplace(response[bfv_index]);
    }
  });

  // After this operation: the response will be secret shared

  const auto H2A_s = std::chrono::system_clock::now();
  auto rand_msk = H2A(response, shape);
  const auto H2A_e = std::chrono::system_clock::now();
  const DurationMillis H2A_time = H2A_e - H2A_s;
  // SPDLOG_INFO("H2A Time: {} ms", H2A_time.count());

  std::vector<yacl::Buffer> ciphers(num_rlwes);
  for (size_t i = 0; i < num_rlwes; i++) {
    ciphers[i] = spu::mpc::cheetah::EncodeSEALObject(response[i]);
  }
  const auto distance_cmp_e = std::chrono::system_clock::now();
  const DurationMillis distance_cmp_time = distance_cmp_e - distance_cmp_s;
  // SPDLOG_INFO("Distance Compute Time : {} ms", distance_cmp_time.count());

  // TODO(ljy): H2A unit test
  int next = conn_->NextRank();

  // conn_->Send(next, ciphers[0], tag);
  for (size_t i = 0; i < num_rlwes; i++) {
    auto tag = "distance";
    conn_->SendAsync(next, ciphers[i], tag);
  }

  // auto tag = "distance";
  // conn_->Send(next, ciphers[num_rlwes - 1], tag);

  return rand_msk;
}

void DisServer::DecodePolyToVector(const seal::Plaintext &poly,
                                   std::vector<uint32_t> &out) {
  auto poly_wrap = absl::MakeConstSpan(poly.data(), poly.coeff_count());
  auto out_wrap = absl::MakeSpan(out.data(), degree_);
  msh_->ModulusDownRNS(poly_wrap, out_wrap);
}

const auto field = spu::FM32;

std::vector<uint32_t> DisServer::H2A(std::vector<seal::Ciphertext> &ct,
                                     uint32_t points_num) {
  seal::Plaintext rand;
  seal::Ciphertext zero_ct;
  std::vector<uint32_t> res(points_num);
  std::vector<uint32_t> out(degree_, 0);
  size_t count = 0;
  for (size_t idx = 0; idx < ct.size(); idx++) {
    // TODO(ljy): preprocess generate more encrypted_zero
    seal::util::encrypt_zero_asymmetric(public_key_, *context_,
                                        ct[idx].parms_id(),
                                        ct[idx].is_ntt_form(), zero_ct);
    evaluator_->add_inplace(ct[idx], zero_ct);

    impl_->UniformPoly(*context_, &rand, ct[idx].parms_id());
    spu::mpc::cheetah::SubPlainInplace(ct[idx], rand, *context_);

    spu::mpc::cheetah::ModulusSwtichInplace(ct[idx], 1, *context_);
    DecodePolyToVector(rand, out);
    for (size_t coeff_i = 0; coeff_i < degree_ && count < points_num;
         coeff_i++) {
      res[count] = out[coeff_i];
      count++;
    }
  }
  return res;
}

spu::NdArrayRef DisServer::H2A(std::vector<seal::Ciphertext> &ct,
                               spu::Shape shape) {
  seal::Plaintext rand;
  seal::Ciphertext zero_ct;
  auto res = spu::mpc::ring_zeros(spu::FM32, shape);
  std::vector<uint32_t> out(degree_, 0);
  for (size_t idx = 0; idx < ct.size(); idx++) {
    // TODO(ljy): preprocess generate more encrypted_zero
    seal::util::encrypt_zero_asymmetric(public_key_, *context_,
                                        ct[idx].parms_id(),
                                        ct[idx].is_ntt_form(), zero_ct);
    evaluator_->add_inplace(ct[idx], zero_ct);

    impl_->UniformPoly(*context_, &rand, ct[idx].parms_id());
    spu::mpc::cheetah::SubPlainInplace(ct[idx], rand, *context_);

    spu::mpc::cheetah::ModulusSwtichInplace(ct[idx], 1, *context_);
    DecodePolyToVector(rand, out);
    using namespace spu;
    DISPATCH_ALL_FIELDS(field, " ", [&]() {
      auto xres = NdArrayView<ring2k_t>(res);
      auto dim1 = res.dim(0);
      auto dim2 = res.dim(1);
      size_t total = dim1 * dim2;
      for (size_t coeff_i = 0; coeff_i < degree_; coeff_i++) {
        auto index = idx * degree_ + coeff_i;
        if (index < total) {
          auto row = index / dim2;
          auto col = index % dim2;
          xres[row * dim2 + col] = out[coeff_i];
        }
      }
    });
  }
  return res;
}

void DisServer::RecvPublicKey() {
  yacl::Buffer pubkey_buffer =
      conn_->Recv(conn_->NextRank(),
                  fmt::format("recv public key from rank-{}", conn_->Rank()));
  seal::PublicKey pubkey;
  spu::mpc::cheetah::DecodeSEALObject(pubkey_buffer, *context_, &pubkey);
  SetPublicKey(pubkey);
}

std::vector<seal::Plaintext> DisServer::PrePoints(
    std::vector<std::vector<uint32_t>> &points) {
  auto num_points = points.size();
  auto point_dim = points[0].size();
  size_t num_bfv =
      point_dim * std::ceil(static_cast<double>(num_points) / degree_);
  std::vector<seal::Plaintext> plain_p(num_bfv, seal::Plaintext(degree_));
  for (size_t j = 0; j < point_dim; j++) {
    for (size_t i = 0; i < num_points; i++) {
      size_t bfv_index = i / degree_;
      size_t coeff_index = i % degree_;
      plain_p[j + bfv_index * point_dim][coeff_index] = points[i][j];
    }
  }
  // yacl::set_num_threads(1);
  yacl::parallel_for(0, num_bfv, [&](size_t begin, size_t end) {
    for (size_t idx = begin; idx < end; idx++) {
      evaluator_->transform_to_ntt_inplace(plain_p[idx],
                                           context_->first_parms_id());
    }
  });

  return plain_p;
}

}  // namespace sanns