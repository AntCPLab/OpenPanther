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

#include "seal_mpir.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "experimental/ann/fix_pir_customed/serializable.pb.h"

namespace spu::seal_pir {

void MultiQueryServer::GenerateSimpleHash() {
  std::vector<uint128_t> query_index_hash(
      query_options_.seal_options.element_number);

  // generate item hash, server and client use same seed
  yacl::parallel_for(0, query_options_.seal_options.element_number,
                     [&](int64_t begin, int64_t end) {
                       for (int idx = begin; idx < end; ++idx) {
                         query_index_hash[idx] = HashItemIndex(idx);
                       }
                     });

  size_t num_bins = cuckoo_params_.NumBins();

  SPDLOG_INFO("Element_number:{}", query_options_.seal_options.element_number);

  for (size_t idx = 0; idx < query_options_.seal_options.element_number;
       ++idx) {
    spu::psi::CuckooIndex::HashRoom itemHash(query_index_hash[idx]);

    std::vector<uint64_t> bin_idx(query_options_.cuckoo_hash_number);
    for (size_t j = 0; j < query_options_.cuckoo_hash_number; ++j) {
      bin_idx[j] = itemHash.GetHash(j) % num_bins;
      size_t k = 0;
      for (; k < j; ++k) {
        if (bin_idx[j] == bin_idx[k]) {
          break;
        }
      }
      if (k < j) {
        continue;
      }

      simple_hash_[bin_idx[j]].push_back(idx);
    }
  }

  for (const auto &hash_ : simple_hash_) {
    max_bin_item_size_ = std::max(max_bin_item_size_, hash_.size());
  }
}

void MultiQueryServer::SetDatabase(yacl::ByteContainerView db_bytes) {
  std::vector<uint8_t> zero_bytes(query_options_.seal_options.element_size * 4,
                                  0);

  for (size_t idx = 0; idx < cuckoo_params_.NumBins(); ++idx) {
    std::vector<yacl::ByteContainerView> db_vec;

    for (size_t j : simple_hash_[idx]) {
      db_vec.emplace_back(
          &db_bytes[j * query_options_.seal_options.element_size * 4],
          query_options_.seal_options.element_size * 4);
    }
    for (size_t j = simple_hash_[idx].size(); j < max_bin_item_size_; ++j) {
      db_vec.emplace_back(zero_bytes);
    }

    pir_server_[idx]->SetDatabase(db_vec);
  }
}

void MultiQueryServer::RecvPublicKey(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  for (size_t i = 0; i < 2; i++) {
    yacl::Buffer pubkey_buffer = link_ctx->Recv(
        link_ctx->NextRank(),
        fmt::format("recv public key from rank-{}", link_ctx->Rank()));

    std::string pubkey_str(pubkey_buffer.size(), '\0');
    std::memcpy(pubkey_str.data(), pubkey_buffer.data(), pubkey_buffer.size());
    auto pubkey =
        pir_server_[0]->DeSerializeSealObject<seal::PublicKey>(pubkey_str, i);
    SetPublicKeys(pubkey, i);
  }
}

void MultiQueryServer::RecvGaloisKeys(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  for (size_t i = 0; i < 2; i++) {
    yacl::Buffer galkey_buffer = link_ctx->Recv(
        link_ctx->NextRank(),
        fmt::format("recv galios key from rank-{}", link_ctx->Rank()));

    std::string galkey_str(galkey_buffer.size(), '\0');
    std::memcpy(galkey_str.data(), galkey_buffer.data(), galkey_buffer.size());
    auto galkey =
        pir_server_[0]->DeSerializeSealObject<seal::GaloisKeys>(galkey_str, i);

    SetGaloisKeys(galkey, i);
  }
}

void MultiQueryServer::DoMultiPirAnswer(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  auto c0 = link_ctx->GetStats()->recv_bytes.load();
  const auto answer_time = std::chrono::system_clock::now();
  yacl::Buffer multi_query_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv multi pir query"));
  const auto recv_end = std::chrono::system_clock::now();

  auto c1 = link_ctx->GetStats()->recv_bytes.load();
  const DurationMillis recv_time = recv_end - answer_time;
  SPDLOG_INFO("recv time: {} ms", recv_time.count());
  SPDLOG_INFO("Query recv comm: {} KB", (c1 - c0) / 1024.0);

  SealMultiPirQueryProto multi_query_proto;
  multi_query_proto.ParseFromArray(multi_query_buffer.data(),
                                   multi_query_buffer.size());

  YACL_ENFORCE((uint64_t)multi_query_proto.querys().size() ==
               cuckoo_params_.NumBins());

  std::vector<yacl::Buffer> reply_cipher_buffers(
      multi_query_proto.querys().size());

  const auto compute_time = std::chrono::system_clock::now();
  const DurationMillis answer_query_time = compute_time - answer_time;
  std::cout << "before do reply time: " << answer_query_time.count() << " ms"
            << std::endl;
  yacl::parallel_for(
      0, multi_query_proto.querys().size(), [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          std::vector<std::vector<seal::Ciphertext>> query_ciphers =
              pir_server_[idx]->DeSerializeQuery(multi_query_proto.querys(idx));

          std::vector<seal::Ciphertext> query_reply =
              pir_server_[idx]->GenerateReply(query_ciphers);
          reply_cipher_buffers[idx] =
              pir_server_[idx]->SerializeCiphertexts(query_reply);
        }
      });
  const auto com_end = std::chrono::system_clock::now();
  const DurationMillis com_time = com_end - compute_time;
  SPDLOG_INFO("PIR: Compute time: {} ms", com_time.count());

  const auto proto_start = std::chrono::system_clock::now();
  SealMultiPirAnswerProto mpir_answer_reply_proto;
  for (int idx = 0; idx < multi_query_proto.querys().size(); ++idx) {
    SealPirAnswerProto *answer = mpir_answer_reply_proto.add_answers();
    answer->set_query_size(0);
    answer->set_start_pos(0);
    answer->mutable_answer()->ParseFromArray(reply_cipher_buffers[idx].data(),
                                             reply_cipher_buffers[idx].size());
  }

  yacl::Buffer mpir_answer_buffer(mpir_answer_reply_proto.ByteSizeLong());
  mpir_answer_reply_proto.SerializePartialToArray(mpir_answer_buffer.data(),
                                                  mpir_answer_buffer.size());

  const auto proto_end = std::chrono::system_clock::now();
  const DurationMillis proto_time = proto_end - proto_start;
  SPDLOG_INFO("PIR: Transfer to Protobuffer: {} ms", proto_time.count());

  const auto send_start = std::chrono::system_clock::now();
  auto s0 = link_ctx->GetStats()->sent_bytes.load();
  link_ctx->SendAsync(
      link_ctx->NextRank(), mpir_answer_buffer,
      fmt::format("send mpir reply buffer size:{}", mpir_answer_buffer.size()));

  auto s1 = link_ctx->GetStats()->sent_bytes.load();

  const auto send_end = std::chrono::system_clock::now();
  const DurationMillis send_time = send_end - send_start;
  SPDLOG_INFO("PIR: Send time: {} ms", send_time.count());
  SPDLOG_INFO("PIR: Query sent comm: {} KB", (s1 - s0) / 1024.0);

  const auto answer_time_end = std::chrono::system_clock::now();
  const DurationMillis time = answer_time_end - answer_time;
  SPDLOG_INFO("PIR: Answer generate: {} ms", time.count());
}

std::vector<std::vector<uint32_t>> MultiQueryServer::DoMultiPirAnswer(
    const std::shared_ptr<yacl::link::Context> &link_ctx, bool enable_H2A) {
  YACL_ENFORCE(enable_H2A == true);

  auto c0 = link_ctx->GetStats()->recv_bytes.load();
  const auto answer_time = std::chrono::system_clock::now();
  yacl::Buffer multi_query_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv multi pir query"));
  const auto recv_end = std::chrono::system_clock::now();

  auto c1 = link_ctx->GetStats()->recv_bytes.load();
  const DurationMillis recv_time = recv_end - answer_time;

  SPDLOG_INFO("PIR: Recv time: {} ms", recv_time.count());
  SPDLOG_INFO("PIR: Query recv comm: {} KB", (c1 - c0) / 1024.0);

  SealMultiPirQueryProto multi_query_proto;
  multi_query_proto.ParseFromArray(multi_query_buffer.data(),
                                   multi_query_buffer.size());

  YACL_ENFORCE((uint64_t)multi_query_proto.querys().size() ==
               cuckoo_params_.NumBins());

  std::vector<std::vector<uint64_t>> random_mask(
      cuckoo_params_.NumBins(),
      std::vector<uint64_t>(query_options_.seal_options.element_size, 0));

  std::vector<yacl::Buffer> reply_cipher_buffers(
      multi_query_proto.querys().size());

  const auto compute_time = std::chrono::system_clock::now();
  const DurationMillis answer_query_time = compute_time - answer_time;
  yacl::parallel_for(
      0, multi_query_proto.querys().size(), [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          std::vector<std::vector<seal::Ciphertext>> query_ciphers =
              pir_server_[idx]->DeSerializeQuery(multi_query_proto.querys(idx));
          std::vector<seal::Ciphertext> query_reply =
              pir_server_[idx]->GenerateReply(query_ciphers, random_mask[idx]);
          reply_cipher_buffers[idx] =
              pir_server_[idx]->SerializeCiphertexts(query_reply);
        }
      });

  const auto com_end = std::chrono::system_clock::now();
  const DurationMillis com_time = com_end - compute_time;
  SPDLOG_INFO("PIR: Compute time: {} ms", com_time.count());

  const auto proto_start = std::chrono::system_clock::now();
  SealMultiPirAnswerProto mpir_answer_reply_proto;
  for (int idx = 0; idx < multi_query_proto.querys().size(); ++idx) {
    SealPirAnswerProto *answer = mpir_answer_reply_proto.add_answers();
    answer->set_query_size(0);
    answer->set_start_pos(0);
    answer->mutable_answer()->ParseFromArray(reply_cipher_buffers[idx].data(),
                                             reply_cipher_buffers[idx].size());
  }

  yacl::Buffer mpir_answer_buffer(mpir_answer_reply_proto.ByteSizeLong());
  mpir_answer_reply_proto.SerializePartialToArray(mpir_answer_buffer.data(),
                                                  mpir_answer_buffer.size());

  const auto proto_end = std::chrono::system_clock::now();
  const DurationMillis proto_time = proto_end - proto_start;
  SPDLOG_INFO("PIR: Transfer to Protobuffer: {} ms", proto_time.count());

  const auto send_start = std::chrono::system_clock::now();
  auto s0 = link_ctx->GetStats()->sent_bytes.load();
  link_ctx->SendAsync(
      link_ctx->NextRank(), mpir_answer_buffer,
      fmt::format("send mpir reply buffer size:{}", mpir_answer_buffer.size()));

  auto s1 = link_ctx->GetStats()->sent_bytes.load();

  const auto send_end = std::chrono::system_clock::now();
  const DurationMillis send_time = send_end - send_start;

  SPDLOG_INFO("PIR: Send time: {} ms", send_time.count());
  SPDLOG_INFO("PIR: Query sent comm: {} KB", (s1 - s0) / 1024.0);

  const auto answer_time_end = std::chrono::system_clock::now();
  const DurationMillis time = answer_time_end - answer_time;

  SPDLOG_INFO("PIR: Answer generate: {} ms", time.count());
  std::vector<std::vector<uint32_t>> random(
      cuckoo_params_.NumBins(),
      std::vector<uint32_t>(query_options_.seal_options.element_size, 0));
  yacl::parallel_for(
      0, cuckoo_params_.NumBins(), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          for (size_t j = 0; j < query_options_.seal_options.element_size; j++)
            random[i][j] = random_mask[i][j];
        }
      });
  return random;
}

void MultiQueryClient::GenerateSimpleHashMap() {
  std::vector<uint128_t> query_index_hash(
      query_options_.seal_options.element_number);

  // generate item hash, server and client use same seed
  yacl::parallel_for(0, query_options_.seal_options.element_number,
                     [&](int64_t begin, int64_t end) {
                       for (int idx = begin; idx < end; ++idx) {
                         query_index_hash[idx] = HashItemIndex(idx);
                       }
                     });

  size_t num_bins = cuckoo_params_.NumBins();
  simple_hash_map_.resize(num_bins);

  std::vector<size_t> simple_hash_counter(num_bins);
  for (size_t idx = 0; idx < num_bins; ++idx) {
    simple_hash_counter[idx] = 0;
  }

  for (size_t idx = 0; idx < query_options_.seal_options.element_number;
       ++idx) {
    spu::psi::CuckooIndex::HashRoom itemHash(query_index_hash[idx]);

    std::vector<uint64_t> bin_idx(query_options_.cuckoo_hash_number);
    for (size_t j = 0; j < query_options_.cuckoo_hash_number; ++j) {
      bin_idx[j] = itemHash.GetHash(j) % num_bins;
      size_t k = 0;
      for (; k < j; ++k) {
        if (bin_idx[j] == bin_idx[k]) {
          break;
        }
      }
      if (k < j) {
        continue;
      }
      // SPDLOG_INFO("bin index[{}]:{}", j, bin_idx[j]);
      simple_hash_map_[bin_idx[j]].emplace(query_index_hash[idx],
                                           simple_hash_counter[bin_idx[j]]);
      simple_hash_counter[bin_idx[j]]++;
    }
  }
  for (const auto &simple_hash_ : simple_hash_map_) {
    max_bin_item_size_ = std::max(max_bin_item_size_, simple_hash_.size());
  }
}

std::vector<MultiQueryItem> MultiQueryClient::GenerateBatchQueryIndex(
    const std::vector<size_t> &multi_query_index) {
  std::vector<MultiQueryItem> multi_query(cuckoo_params_.NumBins());
  std::vector<uint128_t> query_index_hash(multi_query_index.size());

  yacl::parallel_for(
      0, multi_query_index.size(), [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          uint128_t item_hash = HashItemIndex(multi_query_index[idx]);

          query_index_hash[idx] = item_hash;
        }
      });

  spu::psi::CuckooIndex cuckoo_index(cuckoo_params_);
  cuckoo_index.Insert(query_index_hash);

  auto ck_bins = cuckoo_index.bins();

  std::random_device rd;

  std::mt19937 gen(rd());

  for (size_t idx = 0; idx < ck_bins.size(); ++idx) {
    if (ck_bins[idx].IsEmpty()) {
      // pad empty bin with random index
      multi_query[idx].db_index = 0;
      multi_query[idx].item_hash = 0;
      multi_query[idx].bin_item_index = gen() % simple_hash_map_[idx].size();

      continue;
    }
    size_t item_input_index = ck_bins[idx].InputIdx();

    uint128_t item_hash = query_index_hash[item_input_index];

    auto it = simple_hash_map_[idx].find(item_hash);
    if (it == simple_hash_map_[idx].end()) {
      continue;
    }

    multi_query[idx].db_index = multi_query_index[item_input_index];
    multi_query[idx].item_hash = item_hash;
    multi_query[idx].bin_item_index = it->second;
  }
  return multi_query;
}

void MultiQueryClient::SendPublicKey(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  pir_client_->SendPublicKey(link_ctx);
}

void MultiQueryClient::SendGaloisKeys(
    const std::shared_ptr<yacl::link::Context> &link_ctx) {
  pir_client_->SendGaloisKeys(link_ctx);
}

std::vector<std::vector<uint32_t>> MultiQueryClient::DoMultiPirQuery(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const std::vector<size_t> &multi_query_index, bool H2A) {
  const auto hash_s = std::chrono::system_clock::now();
  YACL_ENFORCE(H2A == true);
  std::vector<MultiQueryItem> multi_query =
      GenerateBatchQueryIndex(multi_query_index);

  test_query = multi_query;

  const auto hash_e = std::chrono::system_clock::now();
  const DurationMillis batch_time = hash_e - hash_s;

  SPDLOG_INFO("Hash time: {} ms", batch_time.count());

  SPDLOG_INFO("Query size: {}", multi_query.size());

  const auto query_s = std::chrono::system_clock::now();

  SealMultiPirQueryProto multi_query_proto;
  std::vector<SealPirQueryProto *> query_proto_vec(multi_query.size());
  for (size_t idx = 0; idx < multi_query.size(); ++idx) {
    query_proto_vec[idx] = multi_query_proto.add_querys();
  }
  // std::vector<std::vector<seal::Ciphertext>> query_ciphers =
  // GenerateMultiQuery(multi_query);

  yacl::parallel_for(0, multi_query.size(), [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      std::vector<std::vector<seal::Ciphertext>> query_ciphers =
          pir_client_->GenerateQuery(multi_query[idx].bin_item_index);
      query_proto_vec[idx]->set_query_size(0);
      query_proto_vec[idx]->set_start_pos(0);
      for (auto &query_cipher : query_ciphers) {
        ::spu::seal_pir::CiphertextsProto *ciphers_proto =
            query_proto_vec[idx]->add_query_cipher();

        for (size_t k = 0; k < query_cipher.size(); ++k) {
          std::string cipher_bytes =
              pir_client_->SerializeSealObject<seal::Ciphertext>(
                  query_cipher[k]);

          ciphers_proto->add_ciphers(cipher_bytes.data(),
                                     cipher_bytes.length());
        }
      }
    }
  });
  const auto query_e = std::chrono::system_clock::now();
  const DurationMillis query_time = query_e - query_s;
  SPDLOG_INFO("Query compute time: {} ms", query_time.count());

  auto s = multi_query_proto.SerializeAsString();
  yacl::Buffer multi_query_buffer(s.data(), s.size());

  link_ctx->SendAsync(
      link_ctx->NextRank(), multi_query_buffer,
      fmt::format("send multi pir query number:{}", multi_query.size()));

  yacl::Buffer reply_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv pir answer"));
  SealMultiPirAnswerProto multi_answer_proto;
  multi_answer_proto.ParseFromArray(reply_buffer.data(), reply_buffer.size());

  YACL_ENFORCE((uint64_t)multi_answer_proto.answers_size() ==
               multi_query.size());

  std::vector<std::vector<uint32_t>> answers(multi_answer_proto.answers_size());
  size_t answer_count = 0;
  SPDLOG_INFO("Answer size: {}", multi_answer_proto.answers_size());

  for (int idx = 0; idx < multi_answer_proto.answers_size(); ++idx) {
    CiphertextsProto answer_proto = multi_answer_proto.answers(idx).answer();
    std::vector<seal::Ciphertext> reply_ciphers =
        pir_client_->DeSerializeAnswers(answer_proto);

    seal::Plaintext query_plain = pir_client_->DecodeReply(reply_ciphers);

    std::vector<uint32_t> plaintext_bytes = pir_client_->PlaintextToBytes(
        query_plain, query_options_.seal_options.element_size);

    answers[idx].resize(query_options_.seal_options.element_size);

    answer_count++;

    std::memcpy(answers[idx].data(), plaintext_bytes.data(),
                query_options_.seal_options.element_size * 4);
  }

  return answers;
}

std::vector<std::vector<seal::Ciphertext>> MultiQueryClient::GenerateMultiQuery(
    std::vector<MultiQueryItem> &multi_query) {
  std::vector<std::vector<seal::Ciphertext>> query(
      multi_query.size(), std::vector<seal::Ciphertext>(2));
  for (size_t i = 0; i < multi_query.size(); i++) {
    for (size_t j = 0; j < query[i].size(); j++) {
      query[i][j].reserve(4096 * 3 * 2);
    }
  }
  for (size_t i = 0; i < multi_query.size(); i++) {
    // auto index = multi_query[i].bin_item_index;

    // indices_ = ComputeIndices(index, pir_params_.nvec);

    // // ComputeInverseScales();
    // YACL_ENFORCE(indices_.size() == 2);
    // // std::vector<std::vector<seal::Ciphertext>> result(pir_params_.d);
    // int N = 4096;

    // seal::Plaintext pt(N);
    // for (uint32_t i = 0; i < indices_.size(); i++) {
    //   uint32_t num_ptxts = ceil((pir_params_.nvec[i] + 0.0) / N);
    //   YACL_ENFORCE(num_ptxts == 1);
    //   uint64_t log_total;
    //   pt.set_zero();

    //   uint64_t real_index = indices_[i];
    //   int64_t n_i = pir_params_.nvec[i];
    //   uint64_t total = N;
    //   total = n_i % N;
    //   log_total = ceil(log2(total));
    //   pt[real_index] = 1;
    //   seal::Ciphertext dest;
    //   if (pir_params_.enable_symmetric) {
    //     encryptor_[i]->encrypt_symmetric(pt, dest);
    //   } else {
    //     encryptor_[i]->encrypt(pt, dest);
    //   }

    //   const auto &modulus = context_[i]
    //                             ->get_context_data(dest.parms_id())
    //                             ->parms()
    //                             .coeff_modulus();
    //   size_t num_modulus = dest.coeff_modulus_size();
    //   size_t num_coeff = dest.poly_modulus_degree();

    //   // std::cout << dest.size() << std::endl;
    //   for (size_t k = 0; k < dest.size(); k++) {
    //     uint64_t *dst_ptr = dest.data(k);
    //     for (size_t l = 0; l < num_modulus; ++l) {
    //       uint64_t inv_;
    //       seal::util::MultiplyUIntModOperand a;
    //       seal::util::try_invert_uint_mod(pow(2, log_total), modulus.at(l),
    //                                       inv_);
    //       a.set(inv_, modulus[l]);
    //       seal::util::multiply_poly_scalar_coeffmod(dst_ptr, num_coeff, a,
    //                                                 modulus.at(l), dst_ptr);
    //       dst_ptr += num_coeff;
    //     }
    //   }

    //   query[i].push_back(dest);
    // }
  }
  return query;
}

std::vector<std::vector<uint32_t>> MultiQueryClient::DoMultiPirQuery(
    const std::shared_ptr<yacl::link::Context> &link_ctx,
    const std::vector<size_t> &multi_query_index) {
  const auto query_s = std::chrono::system_clock::now();
  std::vector<MultiQueryItem> multi_query =
      GenerateBatchQueryIndex(multi_query_index);

  const auto query_stop = std::chrono::system_clock::now();
  const DurationMillis batch_time = query_stop - query_s;

  std::cout << "Hash: " << batch_time.count() << " ms" << std::endl;

  SealMultiPirQueryProto multi_query_proto;

  std::vector<SealPirQueryProto *> query_proto_vec(multi_query.size());
  for (size_t idx = 0; idx < multi_query.size(); ++idx) {
    query_proto_vec[idx] = multi_query_proto.add_querys();
  }
  yacl::parallel_for(0, multi_query.size(), [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      std::vector<std::vector<seal::Ciphertext>> query_ciphers =
          pir_client_->GenerateQuery(multi_query[idx].bin_item_index);

      query_proto_vec[idx]->set_query_size(0);
      query_proto_vec[idx]->set_start_pos(0);
      for (auto &query_cipher : query_ciphers) {
        ::spu::seal_pir::CiphertextsProto *ciphers_proto =
            query_proto_vec[idx]->add_query_cipher();

        for (size_t k = 0; k < query_cipher.size(); ++k) {
          std::string cipher_bytes =
              pir_client_->SerializeSealObject<seal::Ciphertext>(
                  query_cipher[k]);

          ciphers_proto->add_ciphers(cipher_bytes.data(),
                                     cipher_bytes.length());
        }
      }
    }
  });

  auto s = multi_query_proto.SerializeAsString();
  yacl::Buffer multi_query_buffer(s.data(), s.size());
  const auto query_com_e = std::chrono::system_clock::now();

  const DurationMillis qbuffer = query_com_e - query_stop;
  std::cout << "Query compute time: " << qbuffer.count() << " ms" << std::endl;

  const DurationMillis query_com_time = query_com_e - query_s;
  std::cout << "Query compute time: " << query_com_time.count() << " ms"
            << std::endl;
  link_ctx->SendAsync(
      link_ctx->NextRank(), multi_query_buffer,
      fmt::format("send multi pir query number:{}", multi_query.size()));
  const auto query_e = std::chrono::system_clock::now();
  const DurationMillis query_time = query_e - query_s;
  std::cout << "Query time: " << query_time.count() << " ms" << std::endl;
  yacl::Buffer reply_buffer =
      link_ctx->Recv(link_ctx->NextRank(), fmt::format("recv pir answer"));

  SealMultiPirAnswerProto multi_answer_proto;
  multi_answer_proto.ParseFromArray(reply_buffer.data(), reply_buffer.size());

  std::vector<std::vector<uint32_t>> answers(multi_query_index.size());
  size_t answer_count = 0;

  for (int idx = 0; idx < multi_answer_proto.answers_size(); ++idx) {
    if (multi_query[idx].item_hash == 0) {
      continue;
    }
    for (size_t j = 0; j < multi_query_index.size(); ++j) {
      if (multi_query_index[j] != multi_query[idx].db_index) {
        continue;
      }
      CiphertextsProto answer_proto = multi_answer_proto.answers(idx).answer();
      std::vector<seal::Ciphertext> reply_ciphers =
          pir_client_->DeSerializeAnswers(answer_proto);

      seal::Plaintext query_plain = pir_client_->DecodeReply(reply_ciphers);

      std::vector<uint32_t> plaintext_bytes = pir_client_->PlaintextToBytes(
          query_plain, query_options_.seal_options.element_size);

      answers[j].resize(query_options_.seal_options.element_size);

      answer_count++;

      std::memcpy(answers[j].data(), plaintext_bytes.data(),
                  query_options_.seal_options.element_size * 4);
      break;
    }
  }
  YACL_ENFORCE(answer_count == multi_query_index.size());

  return answers;
}

}  // namespace spu::seal_pir
