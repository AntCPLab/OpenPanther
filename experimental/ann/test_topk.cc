#include "assert.h"
#include "memory"
#include "topk.h"

// #define CORRECTNESS 0
using namespace std;
using namespace sanns::gc;
int32_t uint_to_int(const int32_t &x, int32_t bit_size) {
  int modulus = 1 << bit_size;
  if (x >= (modulus >> 1)) {
    return x - modulus;
  }
  return x;
}

void test_mux(int item_bits) {
  srand(0);

  int32_t item_mask = (1 << item_bits) - 1;

  uint32_t a = rand() & item_mask;
  uint32_t b = rand() & item_mask;
  bool c = 1;
  Integer A(item_bits, a, ALICE);
  Integer B(item_bits, b, ALICE);
  // Integer SUM = A + B;
  Mux(A, B, c);
  assert(a == B.reveal<uint32_t>(PUBLIC));
  assert(b == A.reveal<uint32_t>(PUBLIC));
}

void test_min(const int n, int l, int item_bits, int discard_bits,
              int id_bits) {
  int bin_size = n / l;

  int32_t item_mask = (1 << item_bits) - 1;

  unique_ptr<Integer[]> A = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> B = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> INPUT = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> INDEX = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> BIN_MIN = make_unique<Integer[]>(l);
  unique_ptr<Integer[]> MIN_ID = make_unique<Integer[]>(l);

  vector<int32_t> A_input(n);
  vector<int32_t> B_input(n);

  // Use for test
  // Don't need to communicate
  srand(0);
  for (int i = 0; i < n; ++i) {
    A_input[i] = rand() & item_mask;
    B_input[i] = rand() & item_mask;

    A[i] = Integer(item_bits, A_input[i], ALICE);
    B[i] = Integer(item_bits, B_input[i], BOB);
  }

  for (int i = 0; i < n; ++i) {
    INPUT[i] = A[i] + B[i];
    INDEX[i] = Integer(id_bits, i, ALICE);
  }

  Discard(INPUT.get(), n, discard_bits);

  for (int bin_index = 0; bin_index < l; ++bin_index) {
    auto bin_len = bin_index == l - 1 ? n - bin_index * bin_size : bin_size;
    Min(INPUT.get() + bin_index * bin_size, INDEX.get() + bin_index * bin_size,
        bin_len, BIN_MIN.get() + bin_index, MIN_ID.get() + bin_index);
  }
#ifdef CORRECTNESS
  vector<int32_t> plain_input(n);
  for (int i = 0; i < n; i++) {
    int32_t sum = (A_input[i] + B_input[i]) & item_mask;
    plain_input[i] = uint_to_int(sum, item_bits);
    plain_input[i] >>= discard_bits;
  }

  for (int i = 0; i < l; i++) {
    int32_t min_res = max_dist;
    int32_t p_min_id = 0;

    auto bin_len = i == l - 1 ? n - i * bin_size : bin_size;
    for (int j = 0; j < bin_len; j++) {
      if (min_res > plain_input[i * bin_size + j]) {
        min_res = plain_input[i * bin_size + j];
        p_min_id = i * bin_size + j;
      }
    }
    auto gc_res = BIN_MIN[i].reveal<int32_t>(PUBLIC);
    gc_res = uint_to_int(gc_res, item_bits - discard_bits);

    if (min_res != gc_res) {
      cout << " Min comp error! min res: " << i << " " << min_res << " "
           << gc_res << endl;
      // return;
    }
    if (p_min_id != MIN_ID[i].reveal<int32_t>(PUBLIC)) {
      cout << " Min comp error! "
           << " " << p_min_id << " " << i << " "
           << (MIN_ID[i].reveal<int32_t>(PUBLIC)) << std::endl;
    }
  }
  cout << "Min test correct! " << endl;
#endif
}

void test_bitonic_topk(int n, int k, int item_bits, int discard_bits,
                       int id_bits) {
  int32_t item_mask = (1 << item_bits) - 1;
  unique_ptr<Integer[]> A = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> B = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> AI = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> BI = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> INPUT = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> INDEX = make_unique<Integer[]>(n);

  vector<uint32_t> A_input(n);
  vector<uint32_t> B_input(n);

  vector<uint32_t> A_index(n);
  vector<uint32_t> B_index(n);
  // Use for test
  // srand(0);

  for (int i = 0; i < n; ++i) {
    A_input[i] = (rand() % 256) & item_mask;
    B_input[i] = 0 & item_mask;

    A_index[i] = i & item_mask;
    B_index[i] = 0 & item_mask;

    A[i] = Integer(item_bits, A_input[i], ALICE);
    B[i] = Integer(item_bits, B_input[i], BOB);

    AI[i] = Integer(id_bits, A_index[i], ALICE);
    BI[i] = Integer(id_bits, B_index[i], BOB);
  }

  for (int i = 0; i < n; ++i) {
    INPUT[i] = A[i] + B[i];
    INDEX[i] = AI[i] + BI[i];
  }

  Discard(INPUT.get(), n, discard_bits);
  sanns::gc::BitonicTopk(INPUT.get(), INDEX.get(), n, k, true);

  //   // std::cout << "test" << std::endl;
  //   vector<int32_t> plain_input(n);
  //   for (int i = 0; i < n; i++) {
  //     plain_input[i] = (A_input[i] + B_input[i]) & item_mask;
  //     plain_input[i] = uint_to_int(plain_input[i], item_bits);
  //     plain_input[i] >>= discard_bits;
  //   }
  //   std::sort(plain_input.begin(), plain_input.end());
  //   for (int i = 0; i < k; i++) {
  //     int32_t gc_res = INPUT[i].reveal<int32_t>(PUBLIC);
  //     gc_res = uint_to_int(gc_res, item_bits - discard_bits);
  //     if (gc_res != plain_input[k - 1 - i])
  //       std::cout << gc_res << " " << plain_input[k - 1 - i] << endl;

  //     // int32_t gc_id = INDEX[i].reveal<int32_t>(PUBLIC);
  //     // if (gc_id != plain_input[i])
  //     // std::cout << gc_id << " " << plain_topk_id[i] << endl;
  //   }
  //   std::cout << "Naive topk test correct!" << std::endl;
}

void test_naive_topk(int n, int k, int item_bits, int discard_bits,
                     int id_bits) {
  int32_t item_mask = (1 << item_bits) - 1;
  unique_ptr<Integer[]> A = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> B = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> AI = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> BI = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> INPUT = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> INDEX = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> MIN_TOPK = make_unique<Integer[]>(k);
  unique_ptr<Integer[]> MIN_ID = make_unique<Integer[]>(k);

  vector<uint32_t> A_input(n);
  vector<uint32_t> B_input(n);

  vector<uint32_t> A_index(n);
  vector<uint32_t> B_index(n);
  // Use for test
  srand(0);

  for (int i = 0; i < n; ++i) {
    A_input[i] = rand() & item_mask;
    B_input[i] = rand() & item_mask;

    A_index[i] = rand() & item_mask;
    B_index[i] = rand() & item_mask;

    A[i] = Integer(item_bits, A_input[i], ALICE);
    B[i] = Integer(item_bits, B_input[i], BOB);

    AI[i] = Integer(id_bits, A_input[i], ALICE);
    BI[i] = Integer(id_bits, B_input[i], BOB);
  }

  for (int i = 0; i < n; ++i) {
    INPUT[i] = A[i] + B[i];
    // INDEX[i] = Integer(id_bits, i, ALICE);
    INDEX[i] = AI[i] + BI[i];
  }

  Discard(INPUT.get(), n, discard_bits);
  Naive_topk(INPUT.get(), INDEX.get(), n, k, MIN_TOPK.get(), MIN_ID.get());

  // std::cout << "test" << std::endl;
#ifdef CORRECTNESS
  vector<int32_t> plain_input(n);
  vector<int32_t> plain_topk(k, max_dist);
  vector<int32_t> plain_topk_id(k, 0);
  for (int i = 0; i < n; i++) {
    plain_input[i] = (A_input[i] + B_input[i]) & item_mask;
    plain_input[i] = uint_to_int(plain_input[i], item_bits);
    plain_input[i] >>= discard_bits;
  }
  for (int i = 0; i < n; i++) {
    auto x = plain_input[i];
    auto id = i;
    for (int j = 0; j < k; j++) {
      if (x < plain_topk[j]) {
        swap(x, plain_topk[j]);
        swap(id, plain_topk_id[j]);
      }
    }
  }
  for (int i = 0; i < k; i++) {
    int32_t gc_res = MIN_TOPK[i].reveal<int32_t>(PUBLIC);
    gc_res = uint_to_int(gc_res, item_bits - discard_bits);
    if (gc_res != plain_topk[i]) cout << gc_res << " " << plain_topk[i] << endl;

    int32_t gc_id = MIN_ID[i].reveal<int32_t>(PUBLIC);
    if (gc_id != plain_topk_id[i]) cout << gc_id << " " << plain_topk_id[i];
  }
  std::cout << "Naive topk test correct!" << std::endl;
#endif
}

void test_approximate_topk(int n, int k, int l, uint32_t item_bits,
                           uint32_t discard_bits, uint32_t id_bits) {
  unique_ptr<Integer[]> A = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> B = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> INPUT = make_unique<Integer[]>(n);
  unique_ptr<Integer[]> INDEX = make_unique<Integer[]>(n);

  unique_ptr<Integer[]> MIN_TOPK = make_unique<Integer[]>(k);
  unique_ptr<Integer[]> MIN_ID = make_unique<Integer[]>(k);

  vector<uint32_t> A_input(n);
  vector<uint32_t> B_input(n);

  // Use for test
  srand(0);

  for (int i = 0; i < n; ++i) {
    A_input[i] = rand() & item_bits;
    B_input[i] = rand() & item_bits;

    A[i] = Integer(item_bits, A_input[i], ALICE);
    B[i] = Integer(item_bits, B_input[i], BOB);
  }

  for (int i = 0; i < n; ++i) {
    INPUT[i] = A[i] + B[i];
    INDEX[i] = Integer(id_bits, i, ALICE);
  }

  Discard(INPUT.get(), n, discard_bits);

  Approximate_topk(INPUT.get(), INDEX.get(), n, k, l, MIN_TOPK.get(),
                   MIN_ID.get());

#ifdef CORRECTNESS
  vector<int32_t> plain_input(n);
  vector<int32_t> min_bin(l);
  vector<int32_t> min_index(l);
  vector<int32_t> plain_topk(k, max_dist);
  vector<int32_t> plain_topk_id(k, 0);
  for (int i = 0; i < n; i++) {
    plain_input[i] = (A_input[i] + B_input[i]) & item_mask;
    plain_input[i] = uint_to_int(plain_input[i], item_bits);
    plain_input[i] >>= discard_bits;
  }
  for (int32_t i = 0; i < l; i++) {
    min_bin[i] = plain_input[i * bin_size];
    min_index[i] = i * bin_size;

    auto bin_len = i == l - 1 ? n - i * bin_size : bin_size;
    for (int32_t j = 0; j < bin_len; j++) {
      if (min_bin[i] > plain_input[i * bin_size + j]) {
        min_bin[i] = plain_input[i * bin_size + j];
        min_index[i] = i * bin_size + j;
      }
    }
  }

  for (int i = 0; i < l; i++) {
    auto x = min_bin[i];
    auto id = min_index[i];
    for (int j = 0; j < k; j++) {
      if (x < plain_topk[j]) {
        swap(x, plain_topk[j]);
        swap(id, plain_topk_id[j]);
      }
    }
  }
  for (int i = 0; i < k; i++) {
    int32_t gc_res = MIN_TOPK[i].reveal<int32_t>(PUBLIC);
    gc_res = uint_to_int(gc_res, item_bits - discard_bits);
    if (gc_res != plain_topk[i])
      std::cout << gc_res << " " << plain_topk[i] << std::endl;
    int32_t gc_id = MIN_ID[i].reveal<int32_t>(PUBLIC);
    if (gc_id != plain_topk_id[i])
      std::cout << "id: " << gc_id << " " << plain_topk_id[i] << std::endl;
  }
  std::cout << "Approximate_topk test correct! " << std::endl;
#endif
}

int test_all(int argc, char **argv) {
  int port, party;
  parse_party_and_port(argv, &party, &port);
  cout << "------------------------------" << endl;
  int n = 50810;
  int k = 50;
  int l = 458;
  int item_bits = 24;
  int id_bits = 24;
  int discard_bits = 5;

  if (argc > 3) {
    n = atoi(argv[3]);
    k = atoi(argv[4]);
    l = atoi(argv[5]);
    item_bits = atoi(argv[6]);
    discard_bits = atoi(argv[7]);
  } else if (argc < 3) {
    cout << " Please input: path/top_k party port" << endl;
  }
  cout << n << " " << k << " " << l << " " << item_bits << "" << discard_bits
       << endl;

  NetIO *io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port);

  setup_semi_honest(io, party);

  auto time_min_s = high_resolution_clock::now();
  size_t initial_counter = io->counter;  // 重置为当前计数，为下个测试准备
  test_min(n, l, item_bits, discard_bits, id_bits);
  auto time_min_e = high_resolution_clock::now();
  auto time_min_us =
      duration_cast<microseconds>(time_min_e - time_min_s).count();
  cout << "Top-k time: " << time_min_us / 1000 << " ms" << endl;
  size_t min_comm = io->counter - initial_counter;
  cout << "Communication for test_min: " << min_comm / 1024 << " KBs" << endl;

  initial_counter = io->counter;

  auto time_naive_topk_s = high_resolution_clock::now();
  test_naive_topk(l, k, item_bits, discard_bits, id_bits);
  auto time_naive_topk_e = high_resolution_clock::now();
  auto time_naive_topk_us =
      duration_cast<microseconds>(time_naive_topk_e - time_naive_topk_s)
          .count();

  cout << "Naive topk time: " << time_naive_topk_us / 1000 << " ms" << endl;
  size_t naive_topk_comm = io->counter - initial_counter;
  cout << "Communication for test_naive_topk: " << naive_topk_comm / 1024
       << " KBs" << endl;

  initial_counter = io->counter;
  auto time_approx_topk_s = high_resolution_clock::now();
  test_approximate_topk(n, k, l, item_bits, discard_bits, id_bits);

  auto time_approx_topk_e = high_resolution_clock::now();
  size_t appro_time = io->counter - initial_counter;
  auto time_approx_topk_us =
      duration_cast<microseconds>(time_approx_topk_e - time_approx_topk_s)
          .count();
  cout << "Approximate topk time: " << time_approx_topk_us / 1000 << " ms"
       << endl;

  cout << "Communication for test_approximate_topk: " << appro_time / 1024
       << " KBs" << endl;

  //   cout << "Number of And Gate: " <<
  //   CircuitExecution::circ_exec->num_and()
  //    << endl;

  finalize_semi_honest();
  cout << "Total communication: " << io->counter / 1024 << " KBs" << endl;
  delete io;
  return 0;
}

int main(int argc, char **argv) {
  int port, party;
  parse_party_and_port(argv, &party, &port);
  cout << "------------------------------" << endl;

  int k = 25;
  int l = 178;
  l = (l / k) * k + (l % k != 0) * k;
  int item_bits = 23;
  int id_bits = 20;
  int discard_bits = 5;

  if (argc > 3) {
    k = atoi(argv[4]);
    l = atoi(argv[5]);
    item_bits = atoi(argv[6]);
    discard_bits = atoi(argv[7]);
  } else if (argc < 3) {
    cout << " Please input: path/top_k party port" << endl;
  }
  cout << "From "
       << " " << l << " elements top-" << k << " " << item_bits << ""
       << discard_bits << endl;

  NetIO *io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port);

  setup_semi_honest(io, party);

  auto initial_counter = io->counter;

  auto time_naive_topk_s = high_resolution_clock::now();
  test_naive_topk(l, k, item_bits, discard_bits, id_bits);
  auto time_naive_topk_e = high_resolution_clock::now();
  auto time_naive_topk_us =
      duration_cast<microseconds>(time_naive_topk_e - time_naive_topk_s)
          .count();
  cout << "Naive topk time: " << time_naive_topk_us / 1000 << " ms" << endl;
  size_t naive_topk_comm = io->counter - initial_counter;
  cout << "Communication for test_naive_topk: " << naive_topk_comm / 1024
       << " KBs" << endl;

  initial_counter = io->counter;
  time_naive_topk_s = high_resolution_clock::now();
  test_bitonic_topk(l, k, item_bits, discard_bits, id_bits);
  time_naive_topk_e = high_resolution_clock::now();
  time_naive_topk_us =
      duration_cast<microseconds>(time_naive_topk_e - time_naive_topk_s)
          .count();

  cout << "Bitonic topk time: " << time_naive_topk_us / 1000 << " ms" << endl;
  naive_topk_comm = io->counter - initial_counter;
  cout << "Communication for test_bitonic_topk: " << naive_topk_comm / 1024
       << " KBs" << endl;

  finalize_semi_honest();
  cout << "Total communication: " << io->counter / 1024 << " KBs" << endl;
  delete io;
  return 0;
}