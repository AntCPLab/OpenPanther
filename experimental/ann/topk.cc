#include "topk.h"

#include "emp-sh2pc/emp-sh2pc.h"

namespace sanns::gc {
using namespace emp;
using namespace std;

using namespace std::chrono;

void Discard(Integer *a, int len, int discard_bits) {
  int32_t bit_size = a[0].bits.size();
  int32_t discarded_size = bit_size - discard_bits;
  for (int i = 0; i < len; i++) {
    a[i] = a[i] >> discard_bits;
    a[i].resize(discarded_size);
  }
}

// Mux swap:
// If b == 1, then swap a1 and a1;
// else return
void Mux(Integer &a1, Integer &a2, const Bit &b) {
  Integer zero = Integer(a1.bits.size(), 0, PUBLIC);
  Integer x = a1 ^ a2;
  Integer r = x.select(!b, zero);
  a1 = a1 ^ r;
  a2 = a2 ^ r;
}

void Tree_min(Integer *input, Integer *index, int len, Integer *res,
              Integer *res_id) {
  if (len <= 1) {
    res[0] = input[0];
    res_id[0] = index[0];
  } else {
    int half = len / 2;
    for (int i = 0; i < half; i++) {
      Bit b = input[i] > input[len - 1 - i];
      Mux(input[i], input[len - 1 - i], b);
      Mux(index[i], index[len - 1 - i], b);
    }
    Tree_min(input, index, half + len % 2, res, res_id);
  }
};

void Min(Integer *input, Integer *index, int len, Integer *res,
         Integer *res_id) {
  uint32_t item_size = input[0].bits.size();
  uint32_t max_item = ((1 << (item_size - 1)) - 1);

  res_id[0] = Integer(index[0].bits.size(), 0, PUBLIC);
  res[0] = Integer(item_size, max_item, PUBLIC);
  for (int i = 0; i < len; i++) {
    Bit b = input[i] < res[0];
    Mux(input[i], res[0], b);
    Mux(index[i], res_id[0], b);
  }
}

void mux(Integer &a1, Integer &a2, const Bit &b) {
  Integer r1 = a1.select(b, a2);
  Integer r2 = a2.select(b, a1);
  a1 = r1;
  a2 = r2;
}

void Naive_topk(Integer *input, Integer *index, int len, int k, Integer *res,
                Integer *res_id) {
  uint32_t item_size = input[0].bits.size();
  uint32_t max_item = ((1 << (item_size - 1)) - 1);

  for (int i = 0; i < k; i++) {
    res[i] = Integer(input[0].bits.size(), max_item, PUBLIC);
    res_id[i] = Integer(index[0].bits.size(), 0, PUBLIC);
  }
  for (int i = 0; i < len; i++) {
    Integer x = input[i];
    Integer id = index[i];
    for (int j = 0; j < k; j++) {
      Bit b = x < res[j];
      Mux(x, res[j], b);
      Mux(id, res_id[j], b);
    }
  }
}

void Approximate_topk(Integer *input, Integer *index, int len, int k, int l,
                      Integer *res, Integer *res_id) {
  Integer *bin_max = new Integer[l];
  Integer *bin_max_id = new Integer[l];
  int bin_size = len / l;
  for (int bin_index = 0; bin_index < l; bin_index++) {
    auto bin_len = bin_index == l - 1 ? len - bin_index * bin_size : bin_size;
    Min(input + (bin_index * bin_size), index + (bin_index * bin_size), bin_len,
        bin_max + (bin_index), bin_max_id + bin_index);
  }
  Naive_topk(bin_max, bin_max_id, l, k, res, res_id);
  delete[] bin_max;
  delete[] bin_max_id;
}

int greatestPowerOfTwoLessThan(int n) {
  int k = 1;
  while (k < n) k = k << 1;
  return k >> 1;
}

template <typename T, typename D = Bit>
void oddEvenMergeSort(T *key, D *data, int lo, int n, Bit acc) {
  auto t = greatestPowerOfTwoLessThan(n);
  uint32_t p = t;
  while (p > 0) {
    uint32_t q = t;
    uint32_t r = 0;
    uint32_t d = p;
    while (d > 0) {
      for (uint32_t i = 0; i < n - d; i++) {
        if ((i & p) == r) {
          cmp_swap(key, data, lo + i, lo + i + d, acc);
        }
      }
      d = q - p;
      q = q >> 1;
      r = p;
    }
    p = p >> 1;
  }
}

template <typename T, typename D = Bit>
int local_sort(T *key, D *data, int lo, int n, int k, Bit acc);

template <typename T, typename D = Bit>
int local_sort(T *key, D *data, int lo, int n, int k, Bit acc) {
  assert(n % k == 0);
  auto num = n / k;
  for (int i = 0; i < num; i++) {
    // bitonic_sort(key, data, i * k, k, acc);
    oddEvenMergeSort(key, data, i * k, k, acc);
  }
  return num;
}
template <typename T, typename D>

void topk_merge(T *key, D *data, int k, int number, Bit acc);

template <typename T, typename D>
void topk_merge(T *key, D *data, int k, int number, Bit acc) {
  if (number > 1) {
    auto merge_time = number / 2;
    for (int i = 0; i < merge_time; i++) {
      for (int j = 0; j < k; j++) {
        cmp_swap(key, data, i * k + j, number * k - i * k - j - 1, acc);
      }
    }
  }
}

template <typename T, typename D>
void BitonicTopk(T *key, D *data, int n, int k, Bit acc) {
  auto number = local_sort(key, data, 0, n, k, !acc);
  while (number != 1) {
    topk_merge(key, data, k, number, acc);
    number = std::ceil((double)number / 2);
    for (int i = 0; i < number; i++) {
      bitonic_merge(key, data, i * k, k, !acc);
    }
  }
}

template void BitonicTopk(Integer *key, Integer *data, int n, int k, Bit acc);

std::vector<int32_t> TopK(size_t n, size_t k, size_t item_bits, size_t id_bits,
                          std::vector<uint32_t> &input,
                          std::vector<uint32_t> &index) {
  std::vector<int32_t> gc_id(k);
  int32_t item_mask = (1 << item_bits) - 1;
  int32_t id_mask = (1 << id_bits) - 1;
  std::unique_ptr<emp::Integer[]> A = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> B = std::make_unique<emp::Integer[]>(n);

  std::unique_ptr<emp::Integer[]> A_idx = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> B_idx = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> INPUT = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> INDEX = std::make_unique<emp::Integer[]>(n);

  for (size_t i = 0; i < n; ++i) {
    input[i] &= item_mask;
    index[i] &= id_mask;

    A[i] = Integer(item_bits, input[i], ALICE);
    B[i] = Integer(item_bits, input[i], BOB);

    A_idx[i] = Integer(id_bits, index[i], ALICE);
    B_idx[i] = Integer(id_bits, index[i], BOB);
  }

  for (size_t i = 0; i < n; ++i) {
    INDEX[i] = A_idx[i] + B_idx[i];
    INPUT[i] = A[i] + B[i];
  }
  sanns::gc::BitonicTopk(INPUT.get(), INDEX.get(), n, k, true);
  for (size_t i = 0; i < k; i++) {
    gc_id[i] = INDEX[i].reveal<int32_t>(BOB);
    // std::cout << gc_id[i] << ":" << INPUT[i].reveal<int32_t>(BOB) <<
    // std::endl;
  };
  return gc_id;
}

std::vector<int32_t> EndTopK(size_t n, size_t k, size_t item_bits,
                             size_t id_bits, size_t n_stash, size_t stash_bits,
                             size_t discard_bits, std::vector<uint32_t> &input,
                             std::vector<uint32_t> &index,
                             std::vector<uint32_t> &stash,
                             std::vector<uint32_t> &stash_id) {
  std::vector<int32_t> gc_id(k);
  int32_t item_mask = (1 << item_bits) - 1;
  int32_t id_mask = (1 << id_bits) - 1;

  int32_t s_item_mask = (1 << stash_bits) - 1;
  int32_t s_id_mask = (1 << id_bits) - 1;
  size_t total_n = size_t(std::ceil(float(n + n_stash) / k) * k);
  std::unique_ptr<emp::Integer[]> A = std::make_unique<emp::Integer[]>(total_n);
  std::unique_ptr<emp::Integer[]> B = std::make_unique<emp::Integer[]>(total_n);

  std::unique_ptr<emp::Integer[]> A_idx =
      std::make_unique<emp::Integer[]>(total_n);
  std::unique_ptr<emp::Integer[]> B_idx =
      std::make_unique<emp::Integer[]>(total_n);
  std::unique_ptr<emp::Integer[]> INPUT =
      std::make_unique<emp::Integer[]>(total_n);
  std::unique_ptr<emp::Integer[]> INDEX =
      std::make_unique<emp::Integer[]>(total_n);
  std::unique_ptr<emp::Integer[]> SA =
      std::make_unique<emp::Integer[]>(n_stash);
  std::unique_ptr<emp::Integer[]> SB =
      std::make_unique<emp::Integer[]>(n_stash);

  std::unique_ptr<emp::Integer[]> SA_idx =
      std::make_unique<emp::Integer[]>(n_stash);
  std::unique_ptr<emp::Integer[]> SB_idx =
      std::make_unique<emp::Integer[]>(n_stash);
  std::unique_ptr<emp::Integer[]> SINPUT =
      std::make_unique<emp::Integer[]>(n_stash);
  std::unique_ptr<emp::Integer[]> SINDEX =
      std::make_unique<emp::Integer[]>(n_stash);
  // std::cout << n << " " << k << std::endl;
  for (size_t i = 0; i < n; ++i) {
    input[i] &= item_mask;
    index[i] &= id_mask;
    A[i] = Integer(item_bits, input[i], ALICE);
    B[i] = Integer(item_bits, input[i], BOB);
    A_idx[i] = Integer(id_bits, index[i], ALICE);
    B_idx[i] = Integer(id_bits, index[i], BOB);
  }

  for (size_t i = 0; i < n_stash; ++i) {
    stash[i] &= s_item_mask;
    stash_id[i] &= s_id_mask;
    SA[i] = Integer(stash_bits, stash[i], ALICE);
    SB[i] = Integer(stash_bits, stash[i], BOB);
    SA_idx[i] = Integer(id_bits, stash_id[i], ALICE);
    SB_idx[i] = Integer(id_bits, stash_id[i], BOB);
  }

  for (size_t i = 0; i < n_stash; ++i) {
    SINDEX[i] = SA_idx[i] + SB_idx[i];
    SINPUT[i] = SA[i] + SB[i];
  }
  // sanns::gc::BitonicTopk(SINPUT.get(), SINDEX.get(), n_stash, k, true);
  for (size_t i = 0; i < n; ++i) {
    INDEX[i] = A_idx[i] + B_idx[i];
    INPUT[i] = A[i] + B[i];
  }

  sanns::gc::Discard(SINPUT.get(), n_stash, discard_bits);
  for (size_t i = 0; i < n_stash; ++i) {
    INPUT[n + i] = SINPUT[i];
    INDEX[n + i] = SINDEX[i];
  }
  for (size_t i = n_stash + n; i < total_n; ++i) {
    INPUT[i] = Integer(item_bits, ((1 << (item_bits - 1)) - 1), PUBLIC);
    INDEX[i] = Integer(id_bits, 11111111, PUBLIC);
  }

  sanns::gc::BitonicTopk(INPUT.get(), INDEX.get(), total_n, k, true);
  for (size_t i = 0; i < k; i++) {
    gc_id[i] = INDEX[i].reveal<int32_t>(BOB);
    // std::cout << gc_id[i] << ":" << INPUT[i].reveal<int32_t>(PUBLIC)
    // << std::endl;
  }
  return gc_id;
}

std::vector<int32_t> NaiveTopK(size_t n, size_t k, size_t item_bits,
                               size_t discard_bits, size_t id_bits,
                               std::vector<uint32_t> &input,
                               std::vector<uint32_t> &index) {
  std::vector<int32_t> gc_id(k);
  int32_t item_mask = (1 << item_bits) - 1;
  int32_t id_mask = (1 << id_bits) - 1;
  std::unique_ptr<emp::Integer[]> A = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> B = std::make_unique<emp::Integer[]>(n);

  std::unique_ptr<emp::Integer[]> A_idx = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> B_idx = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> INPUT = std::make_unique<emp::Integer[]>(n);
  std::unique_ptr<emp::Integer[]> INDEX = std::make_unique<emp::Integer[]>(n);

  std::unique_ptr<emp::Integer[]> MIN_TOPK =
      std::make_unique<emp::Integer[]>(k);
  std::unique_ptr<emp::Integer[]> MIN_ID = std::make_unique<emp::Integer[]>(k);

  // Use for test

  for (size_t i = 0; i < n; ++i) {
    input[i] &= item_mask;
    index[i] &= id_mask;

    A[i] = Integer(item_bits, input[i], ALICE);
    B[i] = Integer(item_bits, input[i], BOB);

    A_idx[i] = Integer(id_bits, index[i], ALICE);
    B_idx[i] = Integer(id_bits, index[i], BOB);
  }

  for (size_t i = 0; i < n; ++i) {
    INPUT[i] = A[i] + B[i];
    INDEX[i] = A_idx[i] + B_idx[i];
  }
  sanns::gc::Discard(INPUT.get(), n, discard_bits);
  sanns::gc::Naive_topk(INPUT.get(), INDEX.get(), n, k, MIN_TOPK.get(),
                        MIN_ID.get());

  for (size_t i = 0; i < k; i++) {
    // gc_res[i] = MIN_TOPK[i].reveal<int32_t>(PUBLIC);
    gc_id[i] = MIN_ID[i].reveal<int32_t>(BOB);
  }
  return gc_id;
}

}  // namespace sanns::gc
