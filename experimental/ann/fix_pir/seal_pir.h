// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "seal/seal.h"
#include "seal_pir_utils.h"
#include "yacl/base/byte_container_view.h"
#include "yacl/link/link.h"

#include "libspu/mpc/cheetah/arith/vector_encoder.h"

#include "experimental/ann/fix_pir/serializable.pb.h"

namespace spu::seal_pir {

//
// SealPIR paper:
//   PIR with compressed queries and amortized query processing
//   https://eprint.iacr.org/2017/1142.pdf
// code reference microsoft opensource implement:
// https://github.com/microsoft/SealPIR

struct SealPirOptions {
  // RLWE polynomial degree
  size_t poly_modulus_degree;
  // db element number
  size_t element_number;
  // byte size of per element
  size_t element_size;
  // number of real query data
  // 这个 query_size 其实是不可区分度，例如，db 有 100条数据，query_size = 10
  // 那整个数据库分成了10份，Client端在查询的时候，就只在其中的某一份进行查询
  size_t query_size = 0;
  // log2 of plaintext modulus
  size_t logt = 24;
};

struct PirParams {
  bool enable_symmetric;
  bool enable_mswitching;
  std::uint64_t num_of_plaintexts;
  std::uint64_t ele_num;
  std::uint64_t ele_size;
  std::uint64_t elements_per_plaintext;
  std::uint32_t d;  // number of dimensions for the database (1 or 2)
  std::uint32_t expansion_ratio;    // ratio of ciphertext to plaintext
  std::vector<std::uint64_t> nvec;  // size of each of the d dimensions
  std::uint32_t slot_count;
};

class SealPir {
 public:
  explicit SealPir(const SealPirOptions &options) : options_(options) {
    SetPolyModulusDegree(options.poly_modulus_degree);

    evaluator_ = std::make_unique<seal::Evaluator>(*context_);

    if (options.query_size > 0) {
      SetPirParams(options.query_size, options.element_size);
    } else {
      SetPirParams(options.element_number, options.element_size);
    }
  }

  /**
   * @brief Set the seal parameter Poly Modulus Degree
   *
   * @param degree seal Poly degree 2048/4096/8192
   */
  void SetPolyModulusDegree(size_t degree);

  /**
   * @brief Set the Pir Params object
   *
   * @param element_number  db element_number
   * @param element_size   db element bytes
   */
  void SetPirParams(size_t element_number, size_t element_size);

  template <typename T>
  std::string SerializeSealObject(const T &object) {
    std::ostringstream output;
    object.save(output);
    return output.str();
  }

  template <typename T>
  T DeSerializeSealObject(const std::string &object_bytes,
                          bool safe_load = false) {
    T seal_object;
    std::istringstream object_input(object_bytes);
    if (safe_load) {
      seal_object.load(*context_, object_input);
    } else {
      seal_object.unsafe_load(*context_, object_input);
    }
    return seal_object;
  }

  std::string SerializePlaintexts(const std::vector<seal::Plaintext> &plains);

  std::vector<seal::Plaintext> DeSerializePlaintexts(
      const std::string &plaintext_bytes, bool safe_load = false);

  yacl::Buffer SerializeCiphertexts(
      const std::vector<seal::Ciphertext> &ciphers);

  std::vector<seal::Ciphertext> DeSerializeCiphertexts(
      const CiphertextsProto &ciphers_proto, bool safe_load = false);

  std::vector<seal::Ciphertext> DeSerializeCiphertexts(
      const yacl::Buffer &ciphers_buffer, bool safe_load = false);

  yacl::Buffer SerializeQuery(
      SealPirQueryProto *query_proto,
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers);

  yacl::Buffer SerializeQuery(
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers);

  std::vector<std::vector<seal::Ciphertext>> DeSerializeQuery(
      const yacl::Buffer &query_buffer, bool safe_load = false);

  std::vector<std::vector<seal::Ciphertext>> DeSerializeQuery(
      const SealPirQueryProto &query_proto, bool safe_load = false);

 protected:
  SealPirOptions options_;
  PirParams pir_params_;
  std::unique_ptr<seal::EncryptionParameters> enc_params_;

  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Evaluator> evaluator_;
};

//
// general single server PIR protocol
//
//     PirClient           PirServer
//                          SetupDB          offline
// ======================================
//       query                               online
//             ------------>
//                            answer
//             <------------
//   extract result
//

//
// SealPIR protocol
//
//   SealPirClient         SealPirServer
//                          SetupDB          offline
// ==================================
//   SendGaloisKeys                          online
//             ------------>  SetGaloisKeys
// ----------------------------------
//    DoPirQuery               DoPirAnswer
//   GenerateQuery
//             ------------>  ExpandQuery
//                            GenerateReply
//             <------------
//   DecodeReply
//

class SealPirClient;

class SealPirServer : public SealPir {
 public:
#ifdef DEC_DEBUG_
  SealPirServer(const SealPirOptions &options, SealPirClient &client);

#else
  SealPirServer(const SealPirOptions &options,
                std::shared_ptr<IDbPlaintextStore> plaintext_store);
#endif

  // read db data, convert to Seal::Plaintext
  void SetDatabase(const std::shared_ptr<IDbElementProvider> &db_provider);
  void SetDatabase(const std::vector<yacl::ByteContainerView> &db_vec);

  // set client GaloisKeys
  void SetGaloisKeys(const seal::GaloisKeys &galkey) { galois_key_ = galkey; }

  void SetPublicKey(const seal::PublicKey &pubkey) { public_key_ = pubkey; }

  // expand one query Seal:Ciphertext
  std::vector<seal::Ciphertext> ExpandQuery(const seal::Ciphertext &encrypted,
                                            std::uint32_t m);

  // GenerateReply for query_ciphers
  std::vector<seal::Ciphertext> GenerateReply(
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers,
      size_t start_pos = 0);

  std::vector<seal::Ciphertext> GenerateReply(
      const std::vector<std::vector<seal::Ciphertext>> &query_ciphers,
      std::vector<uint32_t> &random);

  std::string SerializeDbPlaintext(int db_index = 0);
  void DeSerializeDbPlaintext(const std::string &db_serialize_bytes,
                              int db_index = 0);

  // receive, deserialize, and set client GaloisKeys
  void RecvGaloisKeys(const std::shared_ptr<yacl::link::Context> &link_ctx);
  void RecvPublicKey(const std::shared_ptr<yacl::link::Context> &link_ctx);

  // receive client query, and answer
  void DoPirAnswer(const std::shared_ptr<yacl::link::Context> &link_ctx);

  void H2A(std::vector<seal::Ciphertext> &ct,
           std::vector<uint32_t> &random_mask);

 private:
  void DecodePolyToVector(const seal::Plaintext &poly,
                          std::vector<uint32_t> &out);
  struct Impl;
  std::shared_ptr<Impl> impl_;
  std::unique_ptr<spu::mpc::cheetah::ModulusSwitchHelper> msh_;
  // for debug use, get noise budget
#ifdef DEC_DEBUG_
  SealPirClient &client_;
#endif

  std::vector<std::unique_ptr<std::vector<seal::Plaintext>>> db_vec_;

  std::shared_ptr<IDbPlaintextStore> plaintext_store_;

  bool is_db_preprocessed_;

  seal::GaloisKeys galois_key_;
  seal::PublicKey public_key_;

  void DecomposeToPlaintextsPtr(const seal::Ciphertext &encrypted,
                                seal::Plaintext *plain_ptr, int logt);

  std::vector<seal::Plaintext> DecomposeToPlaintexts(
      const seal::Ciphertext &encrypted);
  std::vector<seal::Plaintext> DecomposeToPlaintexts(
      seal::EncryptionParameters params, const seal::Ciphertext &ct);

  // compute encrypted * x^{-2^j}
  // ref:
  // https://github.com/microsoft/SealPIR/blob/ee1a5a3922fc9250f9bb4e2416ff5d02bfef7e52/src/pir_server.cpp#L397
  void multiply_power_of_X(const seal::Ciphertext &encrypted,
                           seal::Ciphertext &destination, uint32_t index);
};

class SealPirClient : public SealPir {
 public:
  explicit SealPirClient(const SealPirOptions &options);

  // db_index to  seal::Plaintexts index and offset
  uint64_t GetQueryIndex(uint64_t element_idx);
  uint64_t GetQueryOffset(uint64_t element_idx);

  // get Seal::Ciphertext
  std::vector<std::vector<seal::Ciphertext>> GenerateQuery(size_t index);

  // decode server's answer reply
  seal::Plaintext DecodeReply(const std::vector<seal::Ciphertext> &reply);

  // send GaloisKeys to server
  void SendGaloisKeys(const std::shared_ptr<yacl::link::Context> &link_ctx);
  void SendPublicKey(const std::shared_ptr<yacl::link::Context> &link_ctx);

  // generate GaloisKeys
  seal::GaloisKeys GenerateGaloisKeys();
  seal::PublicKey GetPublicKey() { return public_key_; };

  void ComputeInverseScales();

  // when Dimension > 1
  // Compose plaintexts to ciphertext
  // seal::Ciphertext ComposeToCiphertext(
  //     const std::vector<seal::Plaintext> &plains);

  std::vector<uint8_t> ExtractBytes(seal::Plaintext pt, uint64_t offset);
  std::vector<uint32_t> ExtractBytes(seal::Plaintext pt);

  void ComposeToCiphertext(seal::EncryptionParameters params,
                           std::vector<seal::Plaintext>::const_iterator pt_iter,
                           const size_t ct_poly_count, seal::Ciphertext &ct);

  void ComposeToCiphertext(seal::EncryptionParameters params,
                           const std::vector<seal::Plaintext> &pts,
                           seal::Ciphertext &ct);

  // plaintext coefficient to bytes
  std::vector<uint32_t> PlaintextToBytes(const seal::Plaintext &plain,
                                         uint32_t ele_size);

  // PirQuery
  std::vector<uint32_t> DoPirQuery(
      const std::shared_ptr<yacl::link::Context> &link_ctx, size_t db_index);

 private:
  std::unique_ptr<seal::KeyGenerator> keygen_;
  seal::PublicKey public_key_;

  std::unique_ptr<seal::Encryptor> encryptor_;
  std::unique_ptr<seal::Decryptor> decryptor_;

  std::vector<uint64_t> indices_;  // the indices for retrieval.

  // set friend class
  friend class SealPirServer;
};

}  // namespace spu::seal_pir
