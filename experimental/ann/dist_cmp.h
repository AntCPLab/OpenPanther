#pragma once

#include "seal/seal.h"
#include "yacl/link/context.h"

#include "libspu/mpc/cheetah/arith/vector_encoder.h"
namespace sanns {

class DisClient {
 public:
  DisClient(size_t degree, size_t logt,
            const std::shared_ptr<yacl::link::Context> &conn);

  std::vector<seal::Ciphertext> GenerateQuery(std::vector<uint32_t> &q);

  spu::NdArrayRef RecvReply(spu::Shape r_padding_shape, size_t num_points);
  std::vector<uint32_t> RecvReply(size_t num_points);
  spu::NdArrayRef DecodeReply(std::vector<seal::Ciphertext> &reply,
                              spu::Shape r_padding_shape, size_t num_points);

  std::vector<uint32_t> DecodeReply(std::vector<seal::Ciphertext> &reply,
                                    size_t num_points);

  // Only for test;
  inline seal::PublicKey GetPublicKey() { return public_key_; };
  void SendPublicKey();

 private:
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
  DisServer(size_t degree, size_t logt,
            const std::shared_ptr<yacl::link::Context> &conn);

  std::vector<uint32_t> DoDistanceCmp(
      std::vector<std::vector<uint32_t>> &points,
      std::vector<seal::Ciphertext> &q);

  spu::NdArrayRef DoDistanceCmp(std::vector<std::vector<uint32_t>> &points,
                                std::vector<seal::Ciphertext> &q,
                                spu::Shape shape);
  std::vector<seal::Plaintext> PrePoints(
      std::vector<std::vector<uint32_t>> &points);

  spu::NdArrayRef H2A(std::vector<seal::Ciphertext> &ct, spu::Shape shape);

  std::vector<uint32_t> H2A(std::vector<seal::Ciphertext> &ct,
                            uint32_t points_num);
  inline void SetPublicKey(seal::PublicKey pub_key) { public_key_ = pub_key; };

  void RecvPublicKey();

  std::vector<seal::Ciphertext> RecvQuery(size_t query_size);

 private:
  void DecodePolyToVector(const seal::Plaintext &poly,
                          std::vector<uint32_t> &out);
  uint32_t logt_;
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