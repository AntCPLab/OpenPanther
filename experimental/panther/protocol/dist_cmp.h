#pragma once

#include "seal/seal.h"
#include "yacl/link/context.h"

#include "libspu/mpc/cheetah/arith/vector_encoder.h"
namespace panther {

class DisClient {
 public:
  DisClient(size_t degree, size_t logt,
            const std::shared_ptr<yacl::link::Context> &conn);

  std::vector<seal::Ciphertext> GenerateQuery(std::vector<uint32_t> &q);

  // Get distance result
  std::vector<uint32_t> RecvReply(size_t num_points);

  // Get shared distance result
  std::vector<uint32_t> RecvReplySS(size_t num_points);

  spu::NdArrayRef RecvReplySS(spu::Shape r_padding_shape, size_t num_points);

  inline seal::PublicKey GetPublicKey() { return public_key_; };
  void SendPublicKey();

 private:
  std::vector<uint32_t> DecodeReply(std::vector<seal::Ciphertext> &reply,
                                    size_t num_points, bool is_ss);

  spu::NdArrayRef ReshapeReply(std::vector<uint32_t> &response,
                               spu::Shape padding_shape);
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

  std::vector<seal::Ciphertext> RecvQuery(size_t query_size);

  // Return the result of the distance computation
  void DoDistanceCmp(std::vector<std::vector<uint32_t>> &points,
                     std::vector<seal::Ciphertext> &q);

  // Return the secret-shared result of the distance computation using H2A
  std::vector<uint32_t> DoDistanceCmpWithH2A(
      std::vector<std::vector<uint32_t>> &points,
      std::vector<seal::Ciphertext> &q);

  // Return the secret-shared result of the distance computation using H2A with
  // specific shape
  spu::NdArrayRef DoDistanceCmpWithH2A(
      spu::Shape padding_shape, std::vector<std::vector<uint32_t>> &points,
      std::vector<seal::Ciphertext> &q);

  // Only for local test
  inline void SetPublicKey(seal::PublicKey pub_key) { public_key_ = pub_key; };

  void RecvPublicKey();

 private:
  void DecodePolyToVector(const seal::Plaintext &poly,
                          std::vector<uint32_t> &out);
  std::vector<seal::Plaintext> EncodePointsToPoly(
      std::vector<std::vector<uint32_t>> &points);

  std::vector<uint32_t> H2A(std::vector<seal::Ciphertext> &ct,
                            uint32_t num_points);

  spu::NdArrayRef ReshapeVector(std::vector<uint32_t> &response,
                                spu::Shape padding_shape);
  size_t degree_;
  uint32_t logt_;
  std::shared_ptr<yacl::link::Context> conn_;
  seal::PublicKey public_key_;
  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Evaluator> evaluator_;
  std::unique_ptr<seal::EncryptionParameters> seal_params_;
  std::unique_ptr<spu::mpc::cheetah::ModulusSwitchHelper> msh_;
};
}  // namespace panther