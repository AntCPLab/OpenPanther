#pragma once

#include "seal/seal.h"
#include "yacl/link/context.h"

#include "libspu/mpc/cheetah/arith/vector_encoder.h"
namespace sanns {

const uint32_t logt = 24;
class DisClient {
 public:
  DisClient(size_t degree, uint32_t num_points, uint32_t points_dims,
            const std::shared_ptr<yacl::link::Context> &conn);

  std::vector<seal::Ciphertext> GenerateQuery(std::vector<uint32_t> &q);

  spu::NdArrayRef RecvReply(spu::Shape r_padding_shape);
  spu::NdArrayRef DecodeReply(std::vector<seal::Ciphertext> &reply,
                              spu::Shape r_padding_shape);

  // Only for test;
  inline seal::PublicKey get_pub_key() { return public_key_; };

 private:
  uint32_t num_points_;
  uint32_t points_dims_;
  std::shared_ptr<yacl::link::Context> conn_;
  size_t degree_;
  seal::PublicKey public_key_;
  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Evaluator> evaluator_;
  std::unique_ptr<seal::EncryptionParameters> seal_params_;
  std::unique_ptr<seal::KeyGenerator> keygen_;
  std::unique_ptr<seal::Encryptor> encryptor_;
  std::unique_ptr<seal::Decryptor> decryptor_;
};

class DisServer {
 public:
  DisServer(size_t degree, const std::shared_ptr<yacl::link::Context> &conn);
  spu::NdArrayRef DoDistanceCmp(std::vector<std::vector<uint32_t>> &points,
                                std::vector<seal::Ciphertext> &q,
                                spu::Shape shape);
  std::vector<seal::Plaintext> PrePoints(
      std::vector<std::vector<uint32_t>> &points);

  spu::NdArrayRef H2A(std::vector<seal::Ciphertext> &ct, spu::Shape shape);
  inline void set_pub_key(seal::PublicKey pub_key) { public_key_ = pub_key; };

  std::vector<seal::Ciphertext> RecvQuery(size_t query_size);

 private:
  void DecodePolyToVector(const seal::Plaintext &poly,
                          std::vector<uint32_t> &out);

  std::shared_ptr<yacl::link::Context> conn_;
  size_t degree_;
  struct Impl;
  std::shared_ptr<Impl> impl_;
  seal::PublicKey public_key_;
  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Evaluator> evaluator_;
  std::unique_ptr<seal::EncryptionParameters> seal_params_;
  std::unique_ptr<spu::mpc::cheetah::ModulusSwitchHelper> msh_;
};
}  // namespace sanns