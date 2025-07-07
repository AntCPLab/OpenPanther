#include "dist_cmp.h"

#include "seal/util/rlwe.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace panther {

using DurationMillis = std::chrono::duration<double, std::milli>;

DisClient::DisClient(size_t degree, size_t logt,
                     const std::shared_ptr<yacl::link::Context> &conn) {
  degree_ = degree;
  conn_ = conn;

  seal_params_ =
      std::make_unique<seal::EncryptionParameters>(seal::scheme_type::bfv);
  seal_params_->set_poly_modulus_degree(degree);
  seal_params_->set_coeff_modulus(seal::CoeffModulus::BFVDefault(degree));
  seal_params_->set_plain_modulus(1ULL << logt);

  context_ = std::make_unique<seal::SEALContext>(*(seal_params_));
  keygen_ = std::make_unique<seal::KeyGenerator>(*context_);
  seal::SecretKey secret_key = keygen_->secret_key();

  keygen_->create_public_key(public_key_);
  encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_);
  encryptor_->set_secret_key(secret_key);
  decryptor_ = std::make_unique<seal::Decryptor>(*context_, secret_key);
}

void DisClient::SendPublicKey() {
  auto pubkey = GetPublicKey();
  auto pubkey_buffer =
      spu::mpc::cheetah::EncodeSEALObject<seal::PublicKey>(pubkey);
  conn_->Send(conn_->NextRank(), pubkey_buffer,
              fmt::format("send public key to rank-{}", conn_->Rank()));
};

// Return Enc(q[0]), Enc(q[1]),..., Enc(q[q_dim-1]);
std::vector<seal::Ciphertext> DisClient::GenerateQuery(
    std::vector<uint32_t> &q) {
  size_t q_dim = q.size();

  std::vector<seal::Ciphertext> enc_q(q_dim);
  std::vector<seal::Plaintext> plain_q(q_dim, seal::Plaintext(degree_));
  yacl::parallel_for(0, q_dim, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      plain_q[i][0] = q[i];
      encryptor_->encrypt_symmetric(plain_q[i], enc_q[i]);
    }
  });

  std::vector<yacl::Buffer> q_buffer(q_dim);
  yacl::parallel_for(0, q_dim, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      q_buffer[i] = spu::mpc::cheetah::EncodeSEALObject(enc_q[i]);
    }
  });

  int next = conn_->NextRank();
  for (size_t i = 0; i < q_dim; i++) {
    conn_->SendAsync(next, q_buffer[i], "query");
  }
  return enc_q;
}

std::vector<uint32_t> DisClient::DecodeReply(
    std::vector<seal::Ciphertext> &reply, size_t num_points, bool is_ss) {
  std::vector<seal::Plaintext> plain_reply(reply.size());
  std::vector<uint32_t> vec_reply(num_points);
  for (size_t i = 0, count = 0; i < reply.size() && count < num_points; i++) {
    decryptor_->decrypt(reply[i], plain_reply[i]);
    // When using H2A, the noise budget is not applicable.
    if (is_ss == 0) {
      SPDLOG_INFO("Noise budget in response: {} bits",
                  decryptor_->invariant_noise_budget(reply[i]));
    }

    for (size_t j = 0; j < degree_ && count < num_points; j++, count++) {
      vec_reply[count] = plain_reply[i][j];
    }
  }
  return vec_reply;
}

std::vector<uint32_t> DisClient::RecvReply(size_t num_points) {
  size_t num_rlwes = std::ceil(static_cast<double>(num_points) / degree_);
  std::vector<seal::Ciphertext> reply(num_rlwes);
  auto next = conn_->NextRank();
  for (size_t i = 0; i < num_rlwes; i++) {
    auto recv = conn_->Recv(next, "distance");
    spu::mpc::cheetah::DecodeSEALObject(recv, *context_, &reply[i]);
  }
  return DecodeReply(reply, num_points, false);
}

std::vector<uint32_t> DisClient::RecvReplySS(size_t num_points) {
  size_t num_rlwes = std::ceil(static_cast<double>(num_points) / degree_);
  std::vector<seal::Ciphertext> reply(num_rlwes);
  auto next = conn_->NextRank();
  for (size_t i = 0; i < num_rlwes; i++) {
    auto recv = conn_->Recv(next, "distance");
    spu::mpc::cheetah::DecodeSEALObject(recv, *context_, &reply[i]);
  }
  return DecodeReply(reply, num_points, true);
}

spu::NdArrayRef DisClient::RecvReplySS(spu::Shape r_padding_shape,
                                       size_t num_points) {
  std::vector<uint32_t> response = RecvReplySS(num_points);
  return ReshapeReply(response, r_padding_shape);
}

spu::NdArrayRef DisClient::ReshapeReply(std::vector<uint32_t> &response,
                                        spu::Shape padding_shape) {
  auto vec_reply = spu::mpc::ring_zeros(spu::FM32, padding_shape);
  auto num_points = response.size();
  auto dim0 = padding_shape[0];
  auto dim1 = padding_shape[1];
  SPU_ENFORCE(dim0 * dim1 > static_cast<int64_t>(num_points));
  auto array_reply = spu::mpc::ring_zeros(spu::FM32, {(dim0 * dim1)});
  memcpy(&array_reply.at(0), &response[0], num_points * sizeof(uint32_t));
  return array_reply.reshape(padding_shape);
};

//----------------------------------------------------------------------------------------------
// Server

std::vector<uint32_t> DisServer::H2A(std::vector<seal::Ciphertext> &ct,
                                     uint32_t num_points) {
  seal::Plaintext rand;
  seal::Ciphertext zero_ct;
  std::vector<uint32_t> res(num_points);
  std::vector<uint32_t> out(degree_, 0);
  for (size_t idx = 0; idx < ct.size(); idx++) {
    seal::util::encrypt_zero_asymmetric(public_key_, *context_,
                                        ct[idx].parms_id(),
                                        ct[idx].is_ntt_form(), zero_ct);
    evaluator_->add_inplace(ct[idx], zero_ct);

    spu::mpc::cheetah::ModulusSwtichInplace(ct[idx], 1, *context_);

    // Optimization: Halving the communication
    // Instead of returning (b - r), we return r to the client.
    // Since r can be generated from a seed, this reduces communication and
    // maintains security.
    // Inspired by garbled circuits optimization strategy,
    // the client can be viewed as maintaining the zero polynomial.
    rand.parms_id() = seal::parms_id_zero;
    rand.resize(degree_);

    memcpy(rand.data(), ct[idx].data(0), degree_ * 8);
    rand.parms_id() = ct[idx].parms_id();

    spu::mpc::cheetah::SubPlainInplace(ct[idx], rand, *context_);

    DecodePolyToVector(rand, out);
    memcpy(&res[idx * degree_], &out[0],
           std::min(degree_, num_points - degree_ * idx) * sizeof(uint32_t));
  }
  return res;
}
DisServer::DisServer(size_t degree, size_t logt,
                     const std::shared_ptr<yacl::link::Context> &conn)
    : degree_(degree), logt_(logt), conn_(conn) {
  SPU_ENFORCE((degree == 2048) || (degree == 4096));

  seal_params_ =
      std::make_unique<seal::EncryptionParameters>(seal::scheme_type::bfv);
  seal_params_->set_poly_modulus_degree(degree);
  seal_params_->set_coeff_modulus(seal::CoeffModulus::BFVDefault(degree));
  seal_params_->set_plain_modulus(1ULL << logt);

  context_ = std::make_unique<seal::SEALContext>(*(seal_params_));
  evaluator_ = std::make_unique<seal::Evaluator>(*context_);

  // Prepare modulus switch helper for H2A
  std::vector<seal::Modulus> raw_modulus = seal_params_->coeff_modulus();
  std::vector<seal::Modulus> modulus = seal_params_->coeff_modulus();

  while (modulus.size() > 1) {
    modulus.pop_back();
  }

  seal_params_->set_coeff_modulus(modulus);
  seal::SEALContext ms_context(*seal_params_, false,
                               seal::sec_level_type::none);

  msh_ = std::make_unique<spu::mpc::cheetah::ModulusSwitchHelper>(ms_context,
                                                                  logt);
  seal_params_->set_coeff_modulus(raw_modulus);
}

std::vector<seal::Ciphertext> DisServer::RecvQuery(size_t query_size) {
  std::vector<seal::Ciphertext> q(query_size);
  auto next = conn_->NextRank();

  std::vector<yacl::Buffer> recv(query_size);
  for (size_t i = 0; i < query_size; i++) {
    recv[i] = conn_->Recv(next, "query");
  }

  for (size_t i = 0; i < query_size; i++) {
    spu::mpc::cheetah::DecodeSEALObject(recv[i], *context_, &q[i]);
  }
  return q;
}

void DisServer::DoDistanceCmp(std::vector<std::vector<uint32_t>> &points,
                              std::vector<seal::Ciphertext> &q) {
  SPU_ENFORCE_NE(points.size(), static_cast<size_t>(0));

  size_t num_rlwes = std::ceil(static_cast<double>(points.size()) / degree_);
  std::vector<seal::Plaintext> pre_points = EncodePointsToPoly(points);
  std::vector<seal::Ciphertext> response(num_rlwes);
  size_t point_dim = q.size();
  yacl::parallel_for(0, point_dim, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      if (q[i].is_ntt_form() == false)
        evaluator_->transform_to_ntt_inplace(q[i]);
    }
  });

  yacl::parallel_for(0, num_rlwes, [&](size_t begin, size_t end) {
    for (size_t bfv_index = begin; bfv_index < end; bfv_index++) {
      seal::Ciphertext tmp;
      evaluator_->multiply_plain(q[0], pre_points[bfv_index * point_dim],
                                 response[bfv_index]);
      for (size_t i = 1; i < point_dim; i++) {
        evaluator_->multiply_plain(q[i], pre_points[bfv_index * point_dim + i],
                                   tmp);
        evaluator_->add_inplace(response[bfv_index], tmp);
      }
      evaluator_->transform_from_ntt_inplace(response[bfv_index]);
    }
  });

  // Return the computed distance result
  std::vector<yacl::Buffer> ciphers(num_rlwes);
  for (size_t i = 0; i < num_rlwes; i++) {
    ciphers[i] = spu::mpc::cheetah::EncodeSEALObject(response[i]);
  }

  int next = conn_->NextRank();
  for (size_t i = 0; i < num_rlwes; i++) {
    auto tag = "distance";
    conn_->SendAsync(next, ciphers[i], tag);
  }
}

std::vector<uint32_t> DisServer::DoDistanceCmpWithH2A(
    std::vector<std::vector<uint32_t>> &points,
    std::vector<seal::Ciphertext> &q) {
  SPU_ENFORCE_NE(points.size(), static_cast<size_t>(0));

  size_t num_rlwes = std::ceil(static_cast<double>(points.size()) / degree_);
  std::vector<seal::Plaintext> pre_points = EncodePointsToPoly(points);
  std::vector<seal::Ciphertext> response(num_rlwes);
  size_t point_dim = q.size();
  yacl::parallel_for(0, point_dim, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      if (q[i].is_ntt_form() == false)
        evaluator_->transform_to_ntt_inplace(q[i]);
    }
  });

  yacl::parallel_for(0, num_rlwes, [&](size_t begin, size_t end) {
    for (size_t bfv_index = begin; bfv_index < end; bfv_index++) {
      seal::Ciphertext tmp;
      evaluator_->multiply_plain(q[0], pre_points[bfv_index * point_dim],
                                 response[bfv_index]);
      for (size_t i = 1; i < point_dim; i++) {
        evaluator_->multiply_plain(q[i], pre_points[bfv_index * point_dim + i],
                                   tmp);
        evaluator_->add_inplace(response[bfv_index], tmp);
      }
      evaluator_->transform_from_ntt_inplace(response[bfv_index]);
    }
  });
  auto rand_msk = H2A(response, points.size());
  // Return the computed distance result
  std::vector<yacl::Buffer> ciphers(num_rlwes);
  for (size_t i = 0; i < num_rlwes; i++) {
    ciphers[i] = spu::mpc::cheetah::EncodeSEALObject(response[i]);
  }

  int next = conn_->NextRank();
  for (size_t i = 0; i < num_rlwes; i++) {
    auto tag = "distance";
    conn_->SendAsync(next, ciphers[i], tag);
  }
  return rand_msk;
}
spu::NdArrayRef DisServer::DoDistanceCmpWithH2A(
    spu::Shape shape, std::vector<std::vector<uint32_t>> &points,
    std::vector<seal::Ciphertext> &q) {
  SPU_ENFORCE_NE(points.size(), static_cast<size_t>(0));

  size_t num_rlwes = std::ceil(static_cast<double>(points.size()) / degree_);

  std::vector<seal::Plaintext> pre_points = EncodePointsToPoly(points);
  std::vector<seal::Ciphertext> response(num_rlwes);

  size_t point_dim = q.size();
  yacl::parallel_for(0, point_dim, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; i++) {
      if (q[i].is_ntt_form() == false)
        evaluator_->transform_to_ntt_inplace(q[i]);
    }
  });

  yacl::parallel_for(0, num_rlwes, [&](size_t begin, size_t end) {
    for (size_t bfv_index = begin; bfv_index < end; bfv_index++) {
      seal::Ciphertext tmp;
      evaluator_->multiply_plain(q[0], pre_points[bfv_index * point_dim],
                                 response[bfv_index]);
      for (size_t i = 1; i < point_dim; i++) {
        evaluator_->multiply_plain(q[i], pre_points[bfv_index * point_dim + i],
                                   tmp);
        evaluator_->add_inplace(response[bfv_index], tmp);
      }
      evaluator_->transform_from_ntt_inplace(response[bfv_index]);
    }
  });

  // After this operation: the response will be secret shared
  // Return a random mask with the specified shape
  auto rand_vec = H2A(response, points.size());
  auto rand_array = ReshapeVector(rand_vec, shape);

  std::vector<yacl::Buffer> ciphers(num_rlwes);
  for (size_t i = 0; i < num_rlwes; i++) {
    ciphers[i] = spu::mpc::cheetah::EncodeSEALObject(response[i]);
  }

  int next = conn_->NextRank();

  for (size_t i = 0; i < num_rlwes; i++) {
    auto tag = "distance";
    conn_->SendAsync(next, ciphers[i], tag);
  }

  return rand_array;
}

void DisServer::DecodePolyToVector(const seal::Plaintext &poly,
                                   std::vector<uint32_t> &out) {
  auto poly_wrap = absl::MakeConstSpan(poly.data(), poly.coeff_count());
  auto out_wrap = absl::MakeSpan(out.data(), degree_);
  msh_->ModulusDownRNS(poly_wrap, out_wrap);
}

const auto field = spu::FM32;

void DisServer::RecvPublicKey() {
  yacl::Buffer pubkey_buffer =
      conn_->Recv(conn_->NextRank(),
                  fmt::format("recv public key from rank-{}", conn_->Rank()));
  seal::PublicKey pubkey;
  spu::mpc::cheetah::DecodeSEALObject(pubkey_buffer, *context_, &pubkey);
  SetPublicKey(pubkey);
}

std::vector<seal::Plaintext> DisServer::EncodePointsToPoly(
    std::vector<std::vector<uint32_t>> &points) {
  auto num_points = points.size();
  auto point_dim = points[0].size();

  int64_t num_bfv =
      point_dim * std::ceil(static_cast<double>(num_points) / degree_);
  std::vector<seal::Plaintext> plain_p(num_bfv, seal::Plaintext(degree_));

  uint32_t MASK = (1 << logt_) - 1;

  for (size_t i = 0; i < num_points; i++) {
    for (size_t j = 0; j < point_dim; j++) {
      size_t bfv_index = i / degree_;
      size_t coeff_index = i % degree_;
      plain_p[j + bfv_index * point_dim][coeff_index] = points[i][j] & MASK;
    }
  }

  std::vector<seal::Plaintext> pp_ntt(num_bfv, seal::Plaintext(degree_));
  yacl::parallel_for(0, num_bfv, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; idx++) {
      evaluator_->transform_to_ntt_inplace(plain_p[idx],
                                           context_->first_parms_id());
    }
  });
  return plain_p;
}

spu::NdArrayRef DisServer::ReshapeVector(std::vector<uint32_t> &response,
                                         spu::Shape padding_shape) {
  auto vec_reply = spu::mpc::ring_zeros(spu::FM32, padding_shape);
  auto num_points = response.size();
  auto dim0 = padding_shape[0];
  auto dim1 = padding_shape[1];
  SPU_ENFORCE(dim0 * dim1 > static_cast<int64_t>(num_points));
  auto array_reply = spu::mpc::ring_zeros(spu::FM32, {(dim0 * dim1)});
  memcpy(&array_reply.at(0), &response[0], num_points * sizeof(uint32_t));
  return array_reply.reshape(padding_shape);
}

}  // namespace panther