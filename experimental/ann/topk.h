#pragma once

#include "emp-sh2pc/emp-sh2pc.h"
namespace sanns::gc {
using namespace emp;
using namespace std;

using namespace std::chrono;

void Discard(Integer *a, int len, int discard_bits);

void Mux(Integer &a1, Integer &a2, const Bit &b);

void Tree_min(Integer *input, Integer *index, int len, Integer *res,
              Integer *res_id);

void Min(Integer *input, Integer *index, int len, Integer *res,
         Integer *res_id);

void Naive_topk(Integer *input, Integer *index, int len, int k, Integer *res,
                Integer *res_id);

void Approximate_topk(Integer *input, Integer *index, int len, int k, int l,
                      Integer *res, Integer *res_id);

std::vector<int32_t> NaiveTopK(size_t n, size_t k, size_t item_bits,
                               size_t discard_bits, size_t id_bits,
                               std::vector<uint32_t> &input,
                               std::vector<uint32_t> &index);

std::vector<int32_t> TopK(size_t n, size_t k, size_t item_bits, size_t id_bits,
                          std::vector<uint32_t> &input,
                          std::vector<uint32_t> &index);

std::vector<int32_t> EndTopK(size_t n, size_t k, size_t item_bits,
                             size_t id_bits, size_t n_stash, size_t stash_bits,
                             size_t discard_bits, std::vector<uint32_t> &input,
                             std::vector<uint32_t> &index,
                             std::vector<uint32_t> &stash,
                             std::vector<uint32_t> &stash_id);

template <typename T, typename D>
void BitonicTopk(T *key, D *data, int n, int k, Bit acc);
}  // namespace sanns::gc
