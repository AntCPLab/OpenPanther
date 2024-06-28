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

#include "seal_pir.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>

#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "spdlog/spdlog.h"
#include "yacl/base/exception.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
using namespace std;

namespace spu::seal_pir {

using DurationMillis = std::chrono::duration<double, std::milli>;
namespace {
// Number of coefficients needed to represent a database element
uint64_t CoefficientsPerElement(uint32_t logt, uint64_t ele_size) {
  // return std::ceil(8 * ele_size / static_cast<double>(logt));
  return ele_size;
}

// Number of database elements that can fit in a single FV plaintext
uint64_t ElementsPerPtxt(uint32_t logt, uint64_t N, uint64_t ele_size) {
  // adjust ele_size = coef_per_ele;
  // uint64_t coeff_per_ele = CoefficientsPerElement(logt, ele_size);
  uint64_t ele_per_ptxt = 1;
  YACL_ENFORCE(ele_per_ptxt > 0);
  return ele_per_ptxt;
}

// Number of FV plaintexts needed to represent the database
uint64_t PlaintextsPerDb(uint32_t logtp, uint64_t N, uint64_t ele_num,
                         uint64_t ele_size) {
  uint64_t ele_per_ptxt = ElementsPerPtxt(logtp, N, ele_size);
  return std::ceil(static_cast<double>(ele_num) / ele_per_ptxt);
}

std::vector<uint64_t> EleToCoeff(uint32_t limit, const uint32_t *bytes,
                                 uint64_t size) {
  uint64_t size_out = CoefficientsPerElement(limit, size);
  std::vector<uint64_t> output(size_out);
  uint32_t mask = (1ULL << limit) - 1;
  for (uint64_t i = 0; i < size_out; i++) {
    uint32_t coeff = bytes[i] & mask;
    output[i] = coeff;
  }
  return output;
}
// std::vector<uint64_t> BytesToCoeffs(uint32_t limit, const uint8_t *bytes,
//                                     uint64_t size) {
//   uint64_t size_out = CoefficientsPerElement(limit, size);
//   std::vector<uint64_t> output(size_out);

//   uint32_t room = limit;
//   uint64_t *target = output.data();

//   for (uint32_t i = 0; i < size; i++) {
//     uint8_t src = bytes[i];
//     uint32_t rest = 8;
//     while (rest != 0) {
//       if (room == 0) {
//         target++;
//         room = limit;
//       }
//       uint32_t shift = rest;
//       if (room < rest) {
//         shift = room;
//       }
//       *target = *target << shift;
//       *target = *target | (src >> (8 - shift));
//       src = src << shift;
//       room -= shift;
//       rest -= shift;
//     }
//   }

//   *target = *target << room;
//   return output;
// }

void CoeffsToBytes(uint32_t limit, const seal::Plaintext &coeffs,
                   uint32_t *output, uint32_t ele_size) {
  uint64_t mask = (1ULL << limit) - 1;
  for (uint32_t i = 0; i < coeffs.coeff_count() && i < ele_size; i++) {
    output[i] = static_cast<uint32_t>(coeffs[i] & mask);
    // std::cout << output[i] << std::endl;
  }
}

void CoeffsToBytes(uint32_t limit, const seal::Plaintext &coeffs,
                   uint8_t *output, uint32_t size_out, uint32_t ele_size) {
  uint32_t room = 8;
  uint32_t j = 0;
  uint8_t *target = output;
  uint32_t bits_left = ele_size * 8;
  for (uint32_t i = 0; i < coeffs.coeff_count(); i++) {
    if (bits_left == 0) {
      bits_left = ele_size * 8;
    }
    uint64_t src = coeffs[i];
    uint32_t rest = min(limit, bits_left);
    while ((rest != 0) && j < size_out) {
      uint32_t shift = rest;
      if (room < rest) {
        shift = room;
      }

      target[j] = target[j] << shift;
      target[j] = target[j] | (src >> (limit - shift));
      src = src << shift;
      room -= shift;
      rest -= shift;
      bits_left -= shift;
      if (room == 0) {
        j++;
        room = 8;
      }
    }
  }
}

// void CoeffsToBytes(uint32_t limit, const seal::Plaintext &coeffs,
//                    uint8_t *output, uint32_t size_out) {
//   uint32_t room = 8;
//   uint32_t j = 0;
//   uint8_t *target = output;

//   for (uint32_t i = 0; i < coeffs.coeff_count(); i++) {
//     uint64_t src = coeffs[i];
//     uint32_t rest = limit;
//     while ((rest != 0) && j < size_out) {
//       uint32_t shift = rest;
//       if (room < rest) {
//         shift = room;
//       }
//       target[j] = target[j] << shift;
//       target[j] = target[j] | (src >> (limit - shift));
//       src = src << shift;
//       room -= shift;
//       rest -= shift;
//       if (room == 0) {
//         j++;
//         room = 8;
//       }
//     }
//   }
// }

void VectorToPlaintext(const std::vector<uint64_t> &coeffs,
                       seal::Plaintext *plain) {
  uint32_t coeff_count = coeffs.size();
  plain->resize(coeff_count);
  seal::util::set_uint(coeffs.data(), coeff_count, plain->data());
}

std::vector<uint64_t> GetDimensions(uint64_t num_of_plaintexts, uint32_t d) {
  YACL_ENFORCE(d > 0);
  YACL_ENFORCE(num_of_plaintexts > 0);

  std::uint64_t root =
      max(static_cast<uint32_t>(2),
          static_cast<uint32_t>(floor(pow(num_of_plaintexts, 1.0 / d))));

  std::vector<std::uint64_t> dimensions(d, root);

  for (uint32_t i = 0; i < d; i++) {
    if (static_cast<uint64_t>(accumulate(dimensions.begin(), dimensions.end(),
                                         1, multiplies<uint64_t>())) >
        num_of_plaintexts) {
      break;
    }
    dimensions[i] += 1;
  }

  std::uint32_t prod = accumulate(dimensions.begin(), dimensions.end(), 1,
                                  multiplies<uint64_t>());
  YACL_ENFORCE(prod >= num_of_plaintexts);
  return dimensions;
}

std::vector<uint64_t> ComputeIndices(uint64_t query_index,
                                     std::vector<uint64_t> nvec) {
  uint32_t num = nvec.size();
  uint64_t product = 1;

  for (uint32_t i = 0; i < num; i++) {
    product *= nvec[i];
  }

  uint64_t j = query_index;
  std::vector<uint64_t> result;

  for (uint32_t i = 0; i < num; i++) {
    product /= nvec[i];
    uint64_t ji = j / product;

    result.push_back(ji);
    j -= ji * product;
  }

  return result;
}

}  // namespace

void SealPir::SetPolyModulusDegree(size_t degree) {
  // degree 至少 4096
  if (degree < 4096) {
    YACL_THROW("poly_modulus_degree {} is not support.", degree);
  }
  enc_params_ =
      std::make_unique<seal::EncryptionParameters>(seal::scheme_type::bfv);
  enc_params_->set_poly_modulus_degree(degree);
  enc_params_->set_plain_modulus((1ULL << options_.logt));
  // enc_params_->set_plain_modulus((1ULL << options_.logt));
  // enc_params_->set_plain_modulus(
  // seal::PlainModulus::Batching(degree, options_.logt + 1));
  // enc_params_->set_coeff_modulus(seal::CoeffModulus::BFVDefault(degree));
  enc_params_->set_coeff_modulus(
      seal::CoeffModulus::Create(degree, {36, 36, 37}));

  context_ = std::make_unique<seal::SEALContext>(*(enc_params_));
}

void SealPir::SetPirParams(size_t element_number, size_t element_size) {
  uint32_t logt = std::floor(std::log2(enc_params_->plain_modulus().value()));
  uint32_t N = enc_params_->poly_modulus_degree();

  // one FV plaintext can hold how many elements
  uint64_t elements_per_plaintext = ElementsPerPtxt(logt, N, element_size);

  // number of FV plaintexts needed to represent all elements
  uint64_t num_of_plaintexts =
      PlaintextsPerDb(logt, N, element_number, element_size);

  // TODO(ljy): let d be 2 and which can be faster
  size_t d = 2;
  // if (element_number > 8192) {
  // d = 2;
  // }

  std::vector<uint64_t> nvec = GetDimensions(num_of_plaintexts, d);

  // TODO:expansion ratio modulus switch
  uint32_t expansion_ratio = 0;
  for (const auto &modulus : enc_params_->coeff_modulus()) {
    double logqi = std::log2(modulus.value());
    expansion_ratio += ceil(logqi / logt);
  }

  // dimension
  pir_params_.enable_symmetric = true;
  pir_params_.enable_mswitching = true;
  pir_params_.ele_num = element_number;
  pir_params_.ele_size = element_size;
  pir_params_.elements_per_plaintext = elements_per_plaintext;
  pir_params_.num_of_plaintexts = num_of_plaintexts;
  pir_params_.d = d;
  pir_params_.nvec = nvec;
  pir_params_.slot_count = N;
  // because one ciphertext = two polys
  pir_params_.expansion_ratio = expansion_ratio << 1;
}

std::string SealPir::SerializePlaintexts(
    const std::vector<seal::Plaintext> &plains) {
  spu::seal_pir::PlaintextsProto plains_proto;

  for (const auto &plain : plains) {
    std::string plain_bytes = SerializeSealObject<seal::Plaintext>(plain);

    plains_proto.add_data(plain_bytes.data(), plain_bytes.length());
  }
  return plains_proto.SerializeAsString();
}

std::vector<seal::Plaintext> SealPir::DeSerializePlaintexts(
    const std::string &plaintext_bytes, bool safe_load) {
  spu::seal_pir::PlaintextsProto plains_proto;
  plains_proto.ParseFromArray(plaintext_bytes.data(), plaintext_bytes.length());

  std::vector<seal::Plaintext> plains(plains_proto.data_size());

  yacl::parallel_for(0, plains_proto.data_size(),
                     [&](int64_t begin, int64_t end) {
                       for (int i = begin; i < end; ++i) {
                         plains[i] = DeSerializeSealObject<seal::Plaintext>(
                             plains_proto.data(i), safe_load);
                       }
                     });
  return plains;
}

yacl::Buffer SealPir::SerializeCiphertexts(
    const std::vector<seal::Ciphertext> &ciphers) {
  spu::seal_pir::CiphertextsProto ciphers_proto;

  for (const auto &cipher : ciphers) {
    std::string cipher_bytes = SerializeSealObject<seal::Ciphertext>(cipher);

    ciphers_proto.add_ciphers(cipher_bytes.data(), cipher_bytes.length());
  }

  yacl::Buffer b(ciphers_proto.ByteSizeLong());
  ciphers_proto.SerializePartialToArray(b.data(), b.size());
  return b;
}

std::vector<seal::Ciphertext> SealPir::DeSerializeCiphertexts(
    const CiphertextsProto &ciphers_proto, bool safe_load) {
  std::vector<seal::Ciphertext> ciphers(ciphers_proto.ciphers_size());

  yacl::parallel_for(0, ciphers_proto.ciphers_size(),
                     [&](int64_t begin, int64_t end) {
                       for (int i = begin; i < end; ++i) {
                         ciphers[i] = DeSerializeSealObject<seal::Ciphertext>(
                             ciphers_proto.ciphers(i), safe_load);
                       }
                     });
  return ciphers;
}

std::vector<seal::Ciphertext> SealPir::DeSerializeCiphertexts(
    const yacl::Buffer &ciphers_buffer, bool safe_load) {
  CiphertextsProto ciphers_proto;
  ciphers_proto.ParseFromArray(ciphers_buffer.data(), ciphers_buffer.size());

  return DeSerializeCiphertexts(ciphers_proto, safe_load);
}

yacl::Buffer SealPir::SerializeQuery(
    SealPirQueryProto *query_proto,
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers) {
  for (const auto &query_cipher : query_ciphers) {
    CiphertextsProto *ciphers_proto = query_proto->add_query_cipher();
    for (const auto &ciphertext : query_cipher) {
      std::string cipher_bytes =
          SerializeSealObject<seal::Ciphertext>(ciphertext);

      ciphers_proto->add_ciphers(cipher_bytes.data(), cipher_bytes.length());
    }
  }

  yacl::Buffer b(query_proto->ByteSizeLong());
  query_proto->SerializePartialToArray(b.data(), b.size());
  return b;
}

yacl::Buffer SealPir::SerializeQuery(
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers) {
  SealPirQueryProto query_proto;

  query_proto.set_query_size(0);
  query_proto.set_start_pos(0);

  return SerializeQuery(&query_proto, query_ciphers);
}

std::vector<std::vector<seal::Ciphertext>> SealPir::DeSerializeQuery(
    const SealPirQueryProto &query_proto, bool safe_load) {
  std::vector<std::vector<seal::Ciphertext>> pir_query(
      query_proto.query_cipher_size());

  yacl::parallel_for(
      0, query_proto.query_cipher_size(), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const auto &ciphers = query_proto.query_cipher(i);

          pir_query[i].resize(ciphers.ciphers_size());
          for (int j = 0; j < ciphers.ciphers_size(); ++j) {
            pir_query[i][j] = DeSerializeSealObject<seal::Ciphertext>(
                ciphers.ciphers(j), safe_load);
          }
        }
      });
  return pir_query;
}

std::vector<std::vector<seal::Ciphertext>> SealPir::DeSerializeQuery(
    const yacl::Buffer &query_buffer, bool safe_load) {
  SealPirQueryProto query_proto;
  query_proto.ParseFromArray(query_buffer.data(), query_buffer.size());

  return DeSerializeQuery(query_proto, safe_load);
}

#ifdef DEC_DEBUG_
// use client decrypt key get noise budget
SealPirServer::SealPirServer(
    const SealPirOptions &options,
    std::shared_ptr<IDbPlaintextStore> &plaintext_store, SealPirClient &client)
    : SealPir(options), plaintext_store_(plaintext_store), client_(client) {}
#else
SealPirServer::SealPirServer(const SealPirOptions &options,
                             std::shared_ptr<IDbPlaintextStore> plaintext_store)
    : SealPir(options),
      plaintext_store_(std::move(plaintext_store)),
      is_db_preprocessed_(false) {
  yacl::set_num_threads(1);
  impl_ = std::make_shared<Impl>();

  std::vector<seal::Modulus> raw_modulus = enc_params_->coeff_modulus();
  std::vector<seal::Modulus> modulus = enc_params_->coeff_modulus();
  modulus.pop_back();
  modulus.pop_back();
  enc_params_->set_coeff_modulus(modulus);
  seal::SEALContext test_context(*enc_params_, false,
                                 seal::sec_level_type::none);
  enc_params_->set_coeff_modulus(raw_modulus);
  msh_ = std::make_unique<spu::mpc::cheetah::ModulusSwitchHelper>(test_context,
                                                                  options.logt);
}
#endif

// set database

void SealPirServer::SetDatabase(
    const std::shared_ptr<IDbElementProvider> &db_provider) {
  // byte size of whole db
  uint64_t db_size = options_.element_number * options_.element_size * 4;
  // SPDLOG_INFO("DB ByteSize:{}", db_provider->GetDbByteSize());
  YACL_ENFORCE(db_provider->GetDbByteSize() == db_size);

  uint32_t logt = std::floor(std::log2(enc_params_->plain_modulus().value()));
  uint32_t N = enc_params_->poly_modulus_degree();

  // number of FV plaintexts needed to represent all elements
  uint64_t num_of_plaintexts;
  uint64_t db_num;
  if (options_.query_size == 0) {
    num_of_plaintexts = PlaintextsPerDb(logt, N, options_.element_number,
                                        options_.element_size);
    // std::cout << "Num of Plaintext: " << num_of_plaintexts << std::endl;
    db_num = 1;
  } else {
    // 单个数据库的输入大小是 query_size
    num_of_plaintexts =
        PlaintextsPerDb(logt, N, options_.query_size, options_.element_size);
    // 以 query_size（不可区分度） 为单位长度，向上取整
    db_num = (options_.element_number + options_.query_size - 1) /
             options_.query_size;
  }

  //  number of FV plaintexts needed to create the d-dimensional matrix
  uint64_t prod = 1;
  for (auto n : pir_params_.nvec) {
    prod *= n;
  }
  uint64_t matrix_plaintexts = prod;
  YACL_ENFORCE(num_of_plaintexts <= matrix_plaintexts);

  uint64_t ele_per_ptxt = pir_params_.elements_per_plaintext;
  uint64_t bytes_per_ptxt = ele_per_ptxt * options_.element_size * 4;
  uint64_t coeff_per_ptxt =
      ele_per_ptxt * CoefficientsPerElement(logt, options_.element_size);

  // SPDLOG_INFO("Elements per plaintext:{}, coeff per plaintext:{}",
  // ele_per_ptxt, coeff_per_ptxt);
  YACL_ENFORCE(coeff_per_ptxt <= N);

  plaintext_store_->SetSubDbNumber(db_num);

  // 逐 sub db 的构造 Plaintext
  for (size_t idx = 0; idx < db_num; ++idx) {
    // 每一个 sub db
    std::vector<seal::Plaintext> db_vec;
    db_vec.reserve(matrix_plaintexts);

    // byte offset
    uint32_t offset = idx * options_.query_size * options_.element_size * 4;

    for (uint64_t i = 0; i < num_of_plaintexts; i++) {
      uint64_t process_bytes = 0;

      if (db_size <= offset) {
        break;
      } else if (db_size < offset + bytes_per_ptxt) {
        process_bytes = db_size - offset;
      } else {
        process_bytes = bytes_per_ptxt;
      }
      // process_bytes 是当前需要处理的 bytes 数量
      YACL_ENFORCE(process_bytes % options_.element_size == 0);

      // Get the coefficients of the elements that will be packed in plaintext i
      std::vector<uint8_t> element_bytes =
          db_provider->ReadElement(offset, process_bytes);
      // how many elements
      uint64_t ele_in_chunk = process_bytes / (options_.element_size * 4);
      // SPDLOG_INFO("Ele in chunk:{}", ele_in_chunk);

      vector<uint64_t> coefficients(coeff_per_ptxt);

      for (uint64_t ele = 0; ele < ele_in_chunk; ele++) {
        // 这里对 element_bytes 的读取，无需再加 offset
        vector<uint64_t> element_coeffs =
            EleToCoeff(logt,
                       reinterpret_cast<uint32_t *>(element_bytes.data()) +
                           (options_.element_size * ele),
                       options_.element_size);
        std::copy(
            element_coeffs.begin(), element_coeffs.end(),
            coefficients.begin() +
                (CoefficientsPerElement(logt, options_.element_size) * ele));
      }

      offset += process_bytes;

      uint64_t used = coefficients.size();

      YACL_ENFORCE(used <= coeff_per_ptxt);

      // Pad the rest with 1s
      for (uint64_t j = 0; j < (N - used); j++) {
        coefficients.push_back(1);
      }

      seal::Plaintext plain;
      VectorToPlaintext(coefficients, &plain);
      db_vec.push_back(std::move(plain));
    }

    // Add padding to make database a matrix
    uint64_t current_plaintexts = db_vec.size();
    YACL_ENFORCE(current_plaintexts <= num_of_plaintexts);

#ifdef DEC_DEBUG_
    SPDLOG_INFO(
        "adding: {} FV plaintexts of padding (equivalent to: {} elements",
        (matrix_plaintexts - current_plaintexts),
        (matrix_plaintexts - current_plaintexts) *
            elements_per_ptxt(logt, N, options_.element_size));
#endif

    std::vector<uint64_t> padding(N, 1);

    for (uint64_t i = 0; i < (matrix_plaintexts - current_plaintexts); i++) {
      seal::Plaintext plain;
      VectorToPlaintext(padding, &plain);
      db_vec.push_back(plain);
    }

    // pre process db
    yacl::parallel_for(0, db_vec.size(), [&](int64_t begin, int64_t end) {
      for (uint32_t i = begin; i < end; i++) {
        evaluator_->transform_to_ntt_inplace(db_vec[i],
                                             context_->first_parms_id());
      }
    });
    plaintext_store_->SavePlaintexts(db_vec, idx);
  }
  // all db has been transform_to_ntt
  is_db_preprocessed_ = true;
}

void SealPirServer::SetDatabase(
    const std::vector<yacl::ByteContainerView> &db_vec) {
  std::vector<uint8_t> db_flatten_bytes(db_vec.size() * options_.element_size *
                                        4);
  for (size_t idx = 0; idx < db_vec.size(); ++idx) {
    std::memcpy(&db_flatten_bytes[idx * options_.element_size * 4],
                db_vec[idx].data(), db_vec[idx].size());
  }

  std::shared_ptr<IDbElementProvider> db_provider =
      std::make_shared<MemoryDbElementProvider>(db_flatten_bytes,
                                                options_.element_size * 4);

  return SetDatabase(db_provider);
}

std::vector<seal::Ciphertext> SealPirServer::ExpandQuery(
    const seal::Ciphertext &encrypted, std::uint32_t m) {
  seal::GaloisKeys &galkey = galois_key_;

  // Assume that m is a power of 2. If not, round it to the next power of 2.
  uint32_t logm = std::ceil(std::log2(m));
  seal::Plaintext two("2");

  std::vector<int> galois_elts;
  auto n = enc_params_->poly_modulus_degree();
  YACL_ENFORCE(logm <= std::ceil(std::log2(n)), "m > n is not allowed.");

  galois_elts.reserve(std::ceil(std::log2(n)));
  for (int i = 0; i < std::ceil(std::log2(n)); i++) {
    galois_elts.push_back((n + seal::util::exponentiate_uint(2, i)) /
                          seal::util::exponentiate_uint(2, i));
  }

  vector<seal::Ciphertext> temp;
  temp.push_back(encrypted);
  seal::Ciphertext tempctxt;
  seal::Ciphertext tempctxt_rotated;
  seal::Ciphertext tempctxt_shifted;
  seal::Ciphertext tempctxt_rotatedshifted;

  for (uint32_t i = 0; i < logm - 1; i++) {
    vector<seal::Ciphertext> newtemp(temp.size() << 1);
    // temp[a] = (j0 = a (mod 2**i) ? ) : Enc(x^{j0 - a}) else Enc(0).  With
    // some scaling....
    // & (n-1) to avoid debug mode Runtime error
    int index_raw = (n << 1) - (1 << i);
    int index = (index_raw * galois_elts[i]) % (n << 1);

    for (uint32_t a = 0; a < temp.size(); a++) {
      evaluator_->apply_galois(temp[a], galois_elts[i], galkey,
                               tempctxt_rotated);
      evaluator_->add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_X(temp[a], tempctxt_shifted, index_raw);

      multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index);

      evaluator_->add(tempctxt_shifted, tempctxt_rotatedshifted,
                      newtemp[a + temp.size()]);
    }
    temp = newtemp;
  }
  // Last step of the loop
  vector<seal::Ciphertext> newtemp(temp.size() << 1);
  int index_raw = (n << 1) - (1 << (logm - 1));
  int index = (index_raw * galois_elts[logm - 1]) % (n << 1);
  for (uint32_t a = 0; a < temp.size(); a++) {
    if (a >= (m - (1 << (logm - 1)))) {  // corner case.
      evaluator_->multiply_plain(temp[a], two,
                                 newtemp[a]);  // plain multiplication by 2.
    } else {
      evaluator_->apply_galois(temp[a], galois_elts[logm - 1], galkey,
                               tempctxt_rotated);
      evaluator_->add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_X(temp[a], tempctxt_shifted, index_raw);
      multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index);
      evaluator_->add(tempctxt_shifted, tempctxt_rotatedshifted,
                      newtemp[a + temp.size()]);
    }
  }

  auto first = newtemp.begin();
  auto last = newtemp.begin() + m;
  vector<seal::Ciphertext> newVec(first, last);

  return newVec;
}

void SealPirServer::multiply_power_of_X(const seal::Ciphertext &encrypted,
                                        seal::Ciphertext &destination,
                                        uint32_t index) {
  auto coeff_mod_count = enc_params_->coeff_modulus().size() - 1;
  auto coeff_count = enc_params_->poly_modulus_degree();
  auto encrypted_count = encrypted.size();

  destination = encrypted;
  for (size_t i = 0; i < encrypted_count; i++) {
    for (size_t j = 0; j < coeff_mod_count; j++) {
      seal::util::negacyclic_shift_poly_coeffmod(
          encrypted.data(i) + (j * coeff_count), coeff_count, index,
          enc_params_->coeff_modulus()[j],
          destination.data(i) + (j * coeff_count));
    }
  }
}

std::vector<seal::Ciphertext> SealPirServer::GenerateReply(
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers,
    std::vector<uint32_t> &random) {
  const auto gr_s = std::chrono::system_clock::now();
  std::vector<uint64_t> nvec = pir_params_.nvec;
  uint64_t product = 1;

  for (auto n : nvec) {
    product *= n;
  }
  // std::cout << "Product : " << product << std::endl;

  // auto coeff_count = enc_params_->poly_modulus_degree();

  size_t sub_db_index = 0;
  YACL_ENFORCE(options_.query_size == 0);

  std::vector<seal::Plaintext> db_plaintexts =
      plaintext_store_->ReadPlaintexts(sub_db_index);
  std::vector<seal::Plaintext> *cur = &db_plaintexts;

  std::vector<seal::Plaintext> intermediate_plain;  // decompose....

  auto pool = seal::MemoryManager::GetPool();

  int N = enc_params_->poly_modulus_degree();

  // int logt = std::floor(std::log2(enc_params_->plain_modulus().value()));

  const auto expand_s = std::chrono::system_clock::now();
  for (uint32_t i = 0; i < nvec.size(); i++) {
    std::vector<seal::Ciphertext> expanded_query;

    uint64_t n_i = nvec[i];

    for (uint32_t j = 0; j < query_ciphers[i].size(); j++) {
      uint64_t total = N;
      if (j == query_ciphers[i].size() - 1) {
        total = n_i % N;
      }
      // SPDLOG_INFO("dim:{} n:{}", i, n_i);
      std::vector<seal::Ciphertext> expanded_query_part =
          ExpandQuery(query_ciphers[i][j], total);

      expanded_query.insert(
          expanded_query.end(),
          std::make_move_iterator(expanded_query_part.begin()),
          std::make_move_iterator(expanded_query_part.end()));
      expanded_query_part.clear();
    }

    YACL_ENFORCE(expanded_query.size() == n_i, "size mismatch!!! {}-{}",
                 expanded_query.size(), n_i);

    const auto expand_e = std::chrono::system_clock::now();
    const DurationMillis expand_time = expand_e - expand_s;
    SPDLOG_INFO("Expand time: {} ms", expand_time.count());

    yacl::parallel_for(
        0, expanded_query.size(), [&](int64_t begin, int64_t end) {
          for (uint32_t jj = begin; jj < end; jj++) {
            evaluator_->transform_to_ntt_inplace(expanded_query[jj]);
          }
        });

    // Transform plaintext to NTT. If database is pre-processed, can skip
    if ((!is_db_preprocessed_) || i > 0) {
      yacl::parallel_for(0, cur->size(), [&](int64_t begin, int64_t end) {
        for (uint32_t jj = begin; jj < end; jj++) {
          evaluator_->transform_to_ntt_inplace((*cur)[jj],
                                               context_->first_parms_id());
        }
      });
    }

    product /= n_i;
    std::vector<seal::Ciphertext> intermediateCtxts(product);

    yacl::parallel_for(0, product, [&](int64_t begin, int64_t end) {
      for (int k = begin; k < end; k++) {
        uint64_t j = 0;
        while ((*cur)[k + j * product].is_zero()) {
          j++;
        }
        evaluator_->multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                   intermediateCtxts[k]);

        seal::Ciphertext temp;
        for (j += 1; j < n_i; j++) {
          if ((*cur)[k + j * product].is_zero()) {
            continue;
          }
          evaluator_->multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                     temp);
          evaluator_->add_inplace(intermediateCtxts[k],
                                  temp);  // Adds to first component.
        }
      }
    });

    yacl::parallel_for(
        0, intermediateCtxts.size(), [&](int64_t begin, int64_t end) {
          for (uint32_t jj = begin; jj < end; jj++) {
            evaluator_->transform_from_ntt_inplace(intermediateCtxts[jj]);
            if (pir_params_.enable_mswitching) {
              evaluator_->mod_switch_to_inplace(intermediateCtxts[jj],
                                                context_->last_parms_id());
            } else {
            }
          }
        });

    if (i == nvec.size() - 1) {
      const auto gr_e = std::chrono::system_clock::now();
      const DurationMillis g_reply_time = gr_e - gr_s;
      // SPDLOG_INFO("Generate reply time: {}", g_reply_time.count());
      return intermediateCtxts;
    } else {
      intermediate_plain.clear();
      intermediate_plain.reserve(pir_params_.expansion_ratio * product);
      cur = &intermediate_plain;

      for (uint64_t rr = 0; rr < product; rr++) {
        seal::EncryptionParameters parms;
        if (pir_params_.enable_mswitching) {
          parms = context_->last_context_data()->parms();
        } else {
          parms = context_->first_context_data()->parms();
        }

        std::vector<seal::Plaintext> plains =
            DecomposeToPlaintexts(parms, intermediateCtxts[rr]);

        for (uint32_t jj = 0; jj < plains.size(); jj++) {
          intermediate_plain.emplace_back(plains[jj]);
        }
      }
      product = intermediate_plain.size();  // multiply by expansion rate.
    }
  }

  std::vector<seal::Ciphertext> fail(1);
  return fail;
}

std::vector<seal::Ciphertext> SealPirServer::GenerateReply(
    const std::vector<std::vector<seal::Ciphertext>> &query_ciphers,
    size_t start_pos) {
  const auto gr_s = std::chrono::system_clock::now();
  std::vector<uint64_t> nvec = pir_params_.nvec;
  uint64_t product = 1;

  for (auto n : nvec) {
    product *= n;
  }
  // std::cout << "Product : " << product << std::endl;

  // auto coeff_count = enc_params_->poly_modulus_degree();

  size_t sub_db_index = 0;
  if (options_.query_size > 0) {
    YACL_ENFORCE(start_pos % options_.query_size == 0);
    sub_db_index = start_pos / options_.query_size;
  }

  std::vector<seal::Plaintext> db_plaintexts =
      plaintext_store_->ReadPlaintexts(sub_db_index);
  std::vector<seal::Plaintext> *cur = &db_plaintexts;

  std::vector<seal::Plaintext> intermediate_plain;  // decompose....

  auto pool = seal::MemoryManager::GetPool();

  int N = enc_params_->poly_modulus_degree();

  // int logt = std::floor(std::log2(enc_params_->plain_modulus().value()));

  for (uint32_t i = 0; i < nvec.size(); i++) {
    std::vector<seal::Ciphertext> expanded_query;

    uint64_t n_i = nvec[i];

    for (uint32_t j = 0; j < query_ciphers[i].size(); j++) {
      uint64_t total = N;
      if (j == query_ciphers[i].size() - 1) {
        total = n_i % N;
      }
      // SPDLOG_INFO("dim:{} n:{}", i, n_i);
      const auto expand_s = std::chrono::system_clock::now();
      std::vector<seal::Ciphertext> expanded_query_part =
          ExpandQuery(query_ciphers[i][j], total);

      const auto expand_e = std::chrono::system_clock::now();
      const DurationMillis expand_time = expand_e - expand_s;
      // SPDLOG_INFO("Expand time: {} ms", expand_time.count());
      expanded_query.insert(
          expanded_query.end(),
          std::make_move_iterator(expanded_query_part.begin()),
          std::make_move_iterator(expanded_query_part.end()));
      expanded_query_part.clear();
    }

    YACL_ENFORCE(expanded_query.size() == n_i, "size mismatch!!! {}-{}",
                 expanded_query.size(), n_i);

    // // decrypt expand_query to verift
    // std::cout << "nvec i: " << i << "\n";

    //  for(size_t ii = 0; ii < expanded_query.size(); ii++) {
    //   seal::Plaintext expandPlain;
    //   decryptor_->decrypt(expanded_query[ii], expandPlain);
    //   if (expandPlain.is_zero()) {
    //     std::cout << "ii: " << ii << " plaintext is Zero" << "\n";
    //   }else {
    //     std::cout << "ii: " << ii << ", plaintext: " << expandPlain.data()[0]
    //     << "\n";
    //   }
    //  }

#ifdef DEC_DEBUG_
    SPDLOG_INFO("Checking expanded query, size = {}", expanded_query.size());

    seal::Plaintext tempPt;
    for (size_t h = 0; h < expanded_query.size(); h++) {
      client_.decryptor_->decrypt(expanded_query[h], tempPt);

      SPDLOG_INFO("h:{} noise budget = {}, tempPt: {}", h,
                  client_.decryptor_->invariant_noise_budget(expanded_query[h]),
                  tempPt.to_string());
    }
#endif

    // Transform expanded query to NTT, and ...
    yacl::parallel_for(
        0, expanded_query.size(), [&](int64_t begin, int64_t end) {
          for (uint32_t jj = begin; jj < end; jj++) {
            evaluator_->transform_to_ntt_inplace(expanded_query[jj]);
          }
        });

    // Transform plaintext to NTT. If database is pre-processed, can skip
    if ((!is_db_preprocessed_) || i > 0) {
      yacl::parallel_for(0, cur->size(), [&](int64_t begin, int64_t end) {
        for (uint32_t jj = begin; jj < end; jj++) {
          evaluator_->transform_to_ntt_inplace((*cur)[jj],
                                               context_->first_parms_id());
        }
      });
    }

#ifdef DEC_DEBUG_
    for (uint64_t k = 0; k < product; k++) {
      if ((*cur)[k].is_zero()) {
        SPDLOG_INFO("k: {}/{}-th ptxt = 0 ", (k + 1), product);
      }
    }
#endif

    product /= n_i;
    std::vector<seal::Ciphertext> intermediateCtxts(product);

    yacl::parallel_for(0, product, [&](int64_t begin, int64_t end) {
      for (int k = begin; k < end; k++) {
        uint64_t j = 0;
        while ((*cur)[k + j * product].is_zero()) {
          j++;
        }
        evaluator_->multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                   intermediateCtxts[k]);

        seal::Ciphertext temp;
        for (j += 1; j < n_i; j++) {
          if ((*cur)[k + j * product].is_zero()) {
            // SPDLOG_INFO("cur[{}] is zero, k:{}, j:{}", (k + j * product), k,
            // j);
            continue;
          }
          evaluator_->multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                     temp);
          evaluator_->add_inplace(intermediateCtxts[k],
                                  temp);  // Adds to first component.
        }
      }
    });

    yacl::parallel_for(
        0, intermediateCtxts.size(), [&](int64_t begin, int64_t end) {
          for (uint32_t jj = begin; jj < end; jj++) {
            evaluator_->transform_from_ntt_inplace(intermediateCtxts[jj]);
            if (pir_params_.enable_mswitching) {
              evaluator_->mod_switch_to_inplace(intermediateCtxts[jj],
                                                context_->last_parms_id());
            } else {
            }
          }
        });

    if (i == nvec.size() - 1) {
      const auto gr_e = std::chrono::system_clock::now();

      const DurationMillis g_reply_time = gr_e - gr_s;
      // SPDLOG_INFO("Generate reply time: {}", g_reply_time.count());
      return intermediateCtxts;
    } else {
      intermediate_plain.clear();
      intermediate_plain.reserve(pir_params_.expansion_ratio * product);
      cur = &intermediate_plain;

      for (uint64_t rr = 0; rr < product; rr++) {
        seal::EncryptionParameters parms;
        if (pir_params_.enable_mswitching) {
          parms = context_->last_context_data()->parms();
          evaluator_->mod_switch_to_inplace(intermediateCtxts[rr],
                                            context_->last_parms_id());
        } else {
          parms = context_->first_context_data()->parms();
        }

        std::vector<seal::Plaintext> plains =
            DecomposeToPlaintexts(parms, intermediateCtxts[rr]);

        for (uint32_t jj = 0; jj < plains.size(); jj++) {
          intermediate_plain.emplace_back(plains[jj]);
        }
      }
      product = intermediate_plain.size();  // multiply by expansion rate.
    }
  }

  std::vector<seal::Ciphertext> fail(1);
  return fail;
}

vector<seal::Plaintext> SealPirServer::DecomposeToPlaintexts(
    seal::EncryptionParameters params, const seal::Ciphertext &ct) {
  const uint32_t pt_bits_per_coeff = log2(params.plain_modulus().value());
  const auto coeff_count = params.poly_modulus_degree();
  const auto coeff_mod_count = params.coeff_modulus().size();
  const uint64_t pt_bitmask = (1 << pt_bits_per_coeff) - 1;

  vector<seal::Plaintext> result(ComputeExpansionRatio(params) * ct.size());
  auto pt_iter = result.begin();
  for (size_t poly_index = 0; poly_index < ct.size(); ++poly_index) {
    for (size_t coeff_mod_index = 0; coeff_mod_index < coeff_mod_count;
         ++coeff_mod_index) {
      const double coeff_bit_size =
          log2(params.coeff_modulus()[coeff_mod_index].value());
      const size_t local_expansion_ratio =
          ceil(coeff_bit_size / pt_bits_per_coeff);
      size_t shift = 0;
      for (size_t i = 0; i < local_expansion_ratio; ++i) {
        pt_iter->resize(coeff_count);
        for (size_t c = 0; c < coeff_count; ++c) {
          (*pt_iter)[c] =
              (ct.data(poly_index)[coeff_mod_index * coeff_count + c] >>
               shift) &
              pt_bitmask;
        }
        ++pt_iter;
        shift += pt_bits_per_coeff;
      }
    }
  }
  return result;
}

std::string SealPirServer::SerializeDbPlaintext(int db_index) {
  return SerializePlaintexts(*db_vec_[db_index].get());
}

void SealPirServer::DeSerializeDbPlaintext(
    const std::string &db_serialize_bytes, int db_index) {
  std::vector<seal::Plaintext> plaintext_vec =
      DeSerializePlaintexts(db_serialize_bytes);

  db_vec_[db_index] =
      std::make_unique<std::vector<seal::Plaintext>>(plaintext_vec);
}

void SealPirServer::RecvPublicKey(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer pubkey_buffer = link_ctx->Recv(
      link_ctx->NextRank(),
      fmt::format("recv public key from rank-{}", link_ctx->Rank()));

  std::string pubkey_str(pubkey_buffer.size(), '\0');
  std::memcpy(pubkey_str.data(), pubkey_buffer.data(), pubkey_buffer.size());

  auto pubkey = DeSerializeSealObject<seal::PublicKey>(pubkey_str);
  SetPublicKey(pubkey);
}

void SealPirServer::RecvGaloisKeys(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer galkey_buffer = link_ctx->Recv(
      link_ctx->NextRank(),
      fmt::format("recv galios key from rank-{}", link_ctx->Rank()));

  std::string galkey_str(galkey_buffer.size(), '\0');
  std::memcpy(galkey_str.data(), galkey_buffer.data(), galkey_buffer.size());

  auto galkey = DeSerializeSealObject<seal::GaloisKeys>(galkey_str);
  SetGaloisKeys(galkey);
}

void SealPirServer::DoPirAnswer(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  yacl::Buffer query_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv query ciphers"));

  SealPirQueryProto query_proto;
  query_proto.ParseFromArray(query_buffer.data(), query_buffer.size());

  std::vector<std::vector<seal::Ciphertext>> query_ciphers =
      DeSerializeQuery(query_proto);
  SPDLOG_INFO("Server received query and finished deserialize query");

  std::vector<seal::Ciphertext> reply_ciphers =
      GenerateReply(query_ciphers, query_proto.start_pos());

  yacl::Buffer reply_buffer = SerializeCiphertexts(reply_ciphers);
  link_ctx->SendAsync(
      link_ctx->NextRank(), reply_buffer,
      fmt::format("send query reply size:{}", reply_buffer.size()));
  SPDLOG_INFO("Server finished GenerateReply and send to Client");
}

// TODO(ljy): clean H2A
struct SealPirServer::Impl : public spu::mpc::cheetah::EnableCPRNG {
 public:
  Impl(){};
};

void SealPirServer::DecodePolyToVector(const seal::Plaintext &poly,
                                       std::vector<uint32_t> &out) {
  auto poly_wrap = absl::MakeConstSpan(poly.data(), poly.coeff_count());
  auto out_wrap =
      absl::MakeSpan(out.data(), enc_params_->poly_modulus_degree());
  msh_->ModulusDownRNS(poly_wrap, out_wrap);
}

void SealPirServer::H2A(std::vector<seal::Ciphertext> &ct,
                        std::vector<uint32_t> &random_mask) {
  seal::Plaintext rand;
  seal::Ciphertext zero_ct;
  std::vector<uint32_t> out(enc_params_->poly_modulus_degree(), 0);

  impl_->UniformPoly(*context_, &rand, ct[0].parms_id());
  DecodePolyToVector(rand, out);
  memcpy(random_mask.data(), out.data(), random_mask.size() * 4);

  for (size_t idx = 0; idx < ct.size(); idx++) {
    // TODO(ljy): preprocess generate more encrypted_zero
    spu::mpc::cheetah::ModulusSwtichInplace(ct[idx], 1, *context_);
    seal::util::encrypt_zero_asymmetric(public_key_, *context_,
                                        ct[idx].parms_id(),
                                        ct[idx].is_ntt_form(), zero_ct);
    evaluator_->add_inplace(ct[idx], zero_ct);
    spu::mpc::cheetah::SubPlainInplace(ct[idx], rand, *context_);
  }
}

// Client Defination
// SealPirClient
SealPirClient::SealPirClient(const SealPirOptions &options) : SealPir(options) {
  keygen_ = std::make_unique<seal::KeyGenerator>(*context_);

  seal::SecretKey secret_key = keygen_->secret_key();
  keygen_->create_public_key(public_key_);

  encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_);
  encryptor_->set_secret_key(secret_key);

  decryptor_ = std::make_unique<seal::Decryptor>(*context_, secret_key);
  yacl::set_num_threads(1);
}

seal::GaloisKeys SealPirClient::GenerateGaloisKeys() {
  // Generate the Galois keys needed for coeff_select.
  std::vector<uint32_t> galois_elts;
  int N = enc_params_->poly_modulus_degree();
  int logN = seal::util::get_power_of_two(N);

  galois_elts.reserve(logN);
  for (int i = 0; i < logN; i++) {
    galois_elts.push_back((N + seal::util::exponentiate_uint(2, i)) /
                          seal::util::exponentiate_uint(2, i));
  }

  seal::GaloisKeys galois_keys;
  keygen_->create_galois_keys(galois_elts, galois_keys);

  return galois_keys;
}

std::vector<std::vector<seal::Ciphertext>> SealPirClient::GenerateQuery(
    size_t index) {
  size_t query_indx = GetQueryIndex(index);

  indices_ = ComputeIndices(query_indx, pir_params_.nvec);

  // ComputeInverseScales();

  std::vector<std::vector<seal::Ciphertext>> result(pir_params_.d);
  int N = enc_params_->poly_modulus_degree();

  seal::Plaintext pt(enc_params_->poly_modulus_degree());
  for (uint32_t i = 0; i < indices_.size(); i++) {
    uint32_t num_ptxts = ceil((pir_params_.nvec[i] + 0.0) / N);
    uint64_t log_total;
    for (uint32_t j = 0; j < num_ptxts; j++) {
      pt.set_zero();

      if (indices_[i] >= N * j && indices_[i] <= N * (j + 1)) {
        uint64_t real_index = indices_[i] - N * j;
        int64_t n_i = pir_params_.nvec[i];
        uint64_t total = N;
        if (j == num_ptxts - 1) {
          total = n_i % N;
        }
        log_total = ceil(log2(total));
        pt[real_index] = 1;
      }
      seal::Ciphertext dest;
      if (pir_params_.enable_symmetric) {
        encryptor_->encrypt_symmetric(pt, dest);
      } else {
        encryptor_->encrypt(pt, dest);
      }

      const auto &modulus =
          context_->get_context_data(dest.parms_id())->parms().coeff_modulus();
      size_t num_modulus = dest.coeff_modulus_size();
      size_t num_coeff = dest.poly_modulus_degree();

      // std::cout << dest.size() << std::endl;
      for (size_t k = 0; k < dest.size(); k++) {
        uint64_t *dst_ptr = dest.data(k);
        for (size_t l = 0; l < num_modulus; ++l) {
          uint64_t inv_;
          seal::util::MultiplyUIntModOperand a;
          seal::util::try_invert_uint_mod(pow(2, log_total), modulus.at(l),
                                          inv_);
          a.set(inv_, modulus[l]);
          seal::util::multiply_poly_scalar_coeffmod(dst_ptr, num_coeff, a,
                                                    modulus.at(l), dst_ptr);
          dst_ptr += num_coeff;
        }
      }

      result[i].push_back(dest);
    }
  }

  return result;
}

seal::Plaintext SealPirClient::DecodeReply(
    const std::vector<seal::Ciphertext> &reply) {
  seal::EncryptionParameters parms;
  seal::parms_id_type parms_id;

  if (pir_params_.enable_mswitching) {
    parms = context_->last_context_data()->parms();
    parms_id = context_->last_parms_id();
  } else {
    parms = context_->first_context_data()->parms();
    parms_id = context_->first_parms_id();
  }

  uint32_t exp_ratio = ComputeExpansionRatio(parms);
  uint32_t recursion_level = pir_params_.d;

  std::vector<seal::Ciphertext> temp = reply;
  uint32_t ciphertext_size = temp[0].size();

  // uint64_t t = enc_params_->plain_modulus().value();

  for (uint32_t i = 0; i < recursion_level; i++) {
    std::vector<seal::Ciphertext> newtemp;
    std::vector<seal::Plaintext> tempplain;

    for (uint32_t j = 0; j < temp.size(); j++) {
      seal::Plaintext ptxt;
      decryptor_->decrypt(temp[j], ptxt);
#ifdef DEC_DEBUG_
      // SPDLOG_INFO("Client: reply noise budget = {}",
      //             decryptor_->invariant_noise_budget(temp[j]));
      //  SPDLOG_INFO("ptxt to_string: {}", ptxt.to_string());
#endif
      tempplain.push_back(ptxt);

#ifdef DEC_DEBUG_
      // SPDLOG_INFO("recursion level : {} noise budget : {}", i,
      //             decryptor_->invariant_noise_budget(temp[j]));
#endif

      if ((j + 1) % (exp_ratio * ciphertext_size) == 0 && j > 0) {
        // Combine into one ciphertext.
        seal::Ciphertext combined(*context_, parms_id);
        ComposeToCiphertext(parms, tempplain, combined);
        newtemp.push_back(combined);
        tempplain.clear();
      }
    }

    if (i == recursion_level - 1) {
      YACL_ENFORCE(temp.size() == 1);

      return tempplain[0];
    } else {
      tempplain.clear();
      temp = newtemp;
    }
  }

  // This should never be called
  assert(0);
  seal::Plaintext fail;
  return fail;
}

std::vector<uint8_t> SealPirClient::ExtractBytes(seal::Plaintext pt,
                                                 uint64_t offset) {
  // uint32_t N = enc_params_ -> poly_modulus_degree();
  uint32_t logt = floor(log2(enc_params_->plain_modulus().value()));
  uint32_t bytes_per_ptxt =
      pir_params_.elements_per_plaintext * pir_params_.ele_size;

  // Convert from FV plaintext (polynomial) to database element at the client
  vector<uint8_t> elems(bytes_per_ptxt);
  vector<uint64_t> coeffs;
  // encoder_->decode(pt, coeffs);

  CoeffsToBytes(logt, pt, elems.data(), bytes_per_ptxt, pir_params_.ele_size);
  return std::vector<uint8_t>(
      elems.begin() + offset * pir_params_.ele_size,
      elems.begin() + (offset + 1) * pir_params_.ele_size);
}

std::vector<uint32_t> SealPirClient::ExtractBytes(seal::Plaintext pt) {
  // uint32_t N = enc_params_ -> poly_modulus_degree();
  uint32_t logt = floor(log2(enc_params_->plain_modulus().value()));
  uint32_t bytes_per_ptxt =
      pir_params_.elements_per_plaintext * pir_params_.ele_size;

  // Convert from FV plaintext (polynomial) to database element at the client
  vector<uint32_t> elems(bytes_per_ptxt);
  vector<uint64_t> coeffs;
  // encoder_->decode(pt, coeffs);

  CoeffsToBytes(logt, pt, elems.data(), pir_params_.ele_size);
  return std::vector<uint32_t>(elems.begin(),
                               elems.begin() + pir_params_.ele_size);
}

void SealPirClient::ComposeToCiphertext(
    seal::EncryptionParameters params,
    std::vector<seal::Plaintext>::const_iterator pt_iter,
    const size_t ct_poly_count, seal::Ciphertext &ct) {
  const uint32_t pt_bits_per_coeff = log2(params.plain_modulus().value());
  const auto coeff_count = params.poly_modulus_degree();
  const auto coeff_mod_count = params.coeff_modulus().size();

  ct.resize(ct_poly_count);
  for (size_t poly_index = 0; poly_index < ct_poly_count; ++poly_index) {
    for (size_t coeff_mod_index = 0; coeff_mod_index < coeff_mod_count;
         ++coeff_mod_index) {
      const double coeff_bit_size =
          log2(params.coeff_modulus()[coeff_mod_index].value());
      const size_t local_expansion_ratio =
          ceil(coeff_bit_size / pt_bits_per_coeff);

      size_t shift = 0;
      for (size_t i = 0; i < local_expansion_ratio; ++i) {
        for (size_t c = 0; c < pt_iter->coeff_count(); ++c) {
          if (shift == 0) {
            ct.data(poly_index)[coeff_mod_index * coeff_count + c] =
                (*pt_iter)[c];
          } else {
            ct.data(poly_index)[coeff_mod_index * coeff_count + c] +=
                ((*pt_iter)[c] << shift);
          }
        }
        ++pt_iter;
        shift += pt_bits_per_coeff;
      }
    }
  }
}

void SealPirClient::ComposeToCiphertext(seal::EncryptionParameters params,
                                        const std::vector<seal::Plaintext> &pts,
                                        seal::Ciphertext &ct) {
  return ComposeToCiphertext(params, pts.begin(),
                             pts.size() / ComputeExpansionRatio(params), ct);
}

uint64_t SealPirClient::GetQueryIndex(uint64_t element_idx) {
  auto N = enc_params_->poly_modulus_degree();
  auto logt = std::floor(std::log2(enc_params_->plain_modulus().value()));

  auto ele_per_ptxt = ElementsPerPtxt(logt, N, options_.element_size);
  return static_cast<uint64_t>(element_idx / ele_per_ptxt);
}

uint64_t SealPirClient::GetQueryOffset(uint64_t element_idx) {
  uint32_t N = enc_params_->poly_modulus_degree();
  uint32_t logt = std::floor(std::log2(enc_params_->plain_modulus().value()));

  uint64_t ele_per_ptxt = ElementsPerPtxt(logt, N, options_.element_size);
  return element_idx % ele_per_ptxt;
}

std::vector<uint32_t> SealPirClient::PlaintextToBytes(
    const seal::Plaintext &plain, uint32_t ele_size) {
  uint32_t logt = std::floor(std::log2(enc_params_->plain_modulus().value()));

  // Convert from FV plaintext (polynomial) to database element at the client
  std::vector<uint32_t> elements(ele_size);

  // std::cout << ele_size << std::endl;
  CoeffsToBytes(logt, plain, elements.data(), ele_size);

  return elements;
}

void SealPirClient::SendPublicKey(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  seal::PublicKey pubkey = public_key_;

  std::string pubkey_str = SerializeSealObject<seal::PublicKey>(pubkey);
  yacl::Buffer pubkey_buffer(pubkey_str.data(), pubkey_str.length());

  link_ctx->SendAsync(
      link_ctx->NextRank(), pubkey_buffer,
      fmt::format("send public key to rank-{}", link_ctx->Rank()));
}

void SealPirClient::SendGaloisKeys(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  seal::GaloisKeys galkey = GenerateGaloisKeys();

  std::string galkey_str = SerializeSealObject<seal::GaloisKeys>(galkey);
  yacl::Buffer galkey_buffer(galkey_str.data(), galkey_str.length());

  link_ctx->SendAsync(
      link_ctx->NextRank(), galkey_buffer,
      fmt::format("send galios key to rank-{}", link_ctx->Rank()));
}

std::vector<uint32_t> SealPirClient::DoPirQuery(
    const std::shared_ptr<yacl::link::Context> &link_ctx, size_t db_index) {
  size_t query_index = db_index;
  size_t start_pos = 0;

  if (options_.query_size != 0) {
    query_index = db_index % options_.query_size;
    start_pos = db_index - query_index;
  }

  std::vector<std::vector<seal::Ciphertext>> query_ciphers =
      GenerateQuery(query_index);

  SealPirQueryProto query_proto;
  query_proto.set_query_size(options_.query_size);
  query_proto.set_start_pos(start_pos);

  yacl::Buffer query_buffer = SerializeQuery(&query_proto, query_ciphers);
  link_ctx->SendAsync(
      link_ctx->NextRank(), query_buffer,
      fmt::format("send query message({})", query_buffer.size()));
  SPDLOG_INFO("Client finished GenerateQuery and send");

  yacl::Buffer reply_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("send query message"));

  std::vector<seal::Ciphertext> reply_ciphers =
      DeSerializeCiphertexts(reply_buffer);

  SPDLOG_INFO("Client received reply");

  seal::Plaintext query_plain = DecodeReply(reply_ciphers);

  std::vector<uint32_t> query_reply_data = ExtractBytes(query_plain);

  YACL_ENFORCE(query_reply_data.size() == pir_params_.ele_size);

  return query_reply_data;
}
}  // namespace spu::seal_pir
