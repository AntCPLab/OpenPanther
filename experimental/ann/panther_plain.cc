#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "dist_cmp.h"
// #include "gtest/gtest.h"
#include "yacl/link/test_util.h"
const std::vector<int64_t> k_c = {50810, 25603, 9968, 3227, 29326};
const std::vector<int64_t> group_bin_number = {458, 270, 178, 84, 262};
const std::vector<int64_t> group_k_number = {50, 31, 19, 13, 10};

// const std::vector<uint32_t> k_c = {88821, 12122};
// const std::vector<int64_t> group_bin_number = {1000, 262};
// const std::vector<int64_t> group_k_number = {113, 10};
const std::vector<uint32_t> bw = {5, 5, 5, 5, 8};
// const std::vector<uint32_t> bw = {0, 0, 0, 0, 0};
const size_t knn = 10;

using namespace std;
vector<vector<uint32_t>> read_data(size_t n, size_t dim, string filename) {
  std::ifstream inputFile("./experimental/ann/" + filename);
  if (!inputFile.is_open()) {
    std::cerr << "Can't open it!" << std::endl;
  }
  std::vector<std::vector<uint32_t>> numbers(n, vector<uint32_t>(dim));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      if (!(inputFile >> numbers[i][j])) {
        cerr << "Read Error!" << endl;
        cerr << filename << endl;
      }
    }
  }

  inputFile.close();

  std::cout << "input data: (" << numbers.size() << ", " << numbers[0].size()
            << ")" << std::endl;
  return numbers;
}

vector<uint32_t> distance_compute(vector<uint32_t> &q,
                                  vector<vector<uint32_t>> &ps) {
  size_t N = 2048;
  size_t logt = 24;
  size_t n = ps.size();
  size_t points_dim = q.size();
  auto ctxs = yacl::link::test::SetupWorld(2);
  sanns::DisClient client(N, logt, ctxs[0]);
  sanns::DisServer server(N, logt, ctxs[1]);
  client.GenerateQuery(q);
  auto query = server.RecvQuery(points_dim);
  auto response = server.DoDistanceCmp(ps, query);
  auto vec_reply = client.RecvReply(n);
  const uint32_t MASK = (1 << logt) - 1;
  vector<uint32_t> value(n);
  for (size_t i = 0; i < n; i++) {
    uint32_t q_2 = 0;
    uint32_t p_2 = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      p_2 += ps[i][point_i] * ps[i][point_i];
      q_2 += q[point_i] * q[point_i];
    }
    auto get = (response[i] + vec_reply[i]) & MASK;
    value.at(i) = p_2 + q_2 - 2 * get;
  }
  return value;
}

vector<uint32_t> distance_compute_plain(vector<uint32_t> &q,
                                        vector<vector<uint32_t>> &ps) {
  size_t n = ps.size();
  size_t points_dim = q.size();
  auto ctxs = yacl::link::test::SetupWorld(2);
  vector<uint32_t> value(n);
  for (size_t i = 0; i < n; i++) {
    uint32_t distance = 0;
    for (size_t point_i = 0; point_i < points_dim; point_i++) {
      // p_2 += ps[i][point_i] * ps[i][point_i];
      // q_2 += q[point_i] * q[point_i];
      distance += (q[point_i] - ps[i][point_i]) * (q[point_i] - ps[i][point_i]);
    }
    value.at(i) = distance;
  }
  return value;
}

vector<pair<uint32_t, uint32_t>> approximate_topk(vector<uint32_t> &input,
                                                  size_t begin, size_t len,
                                                  size_t l, size_t k) {
  vector<pair<uint32_t, uint32_t>> vid(len);
  for (size_t i = 0; i < len; i++) {
    vid[i].first = input[begin + i];
    vid[i].second = begin + i;
  }
  random_shuffle(vid.begin(), vid.end());
  size_t bin_size = ceil(len / l);
  vector<pair<uint32_t, uint32_t>> indexs(l);
  for (size_t i = 0; i < l; i++) {
    uint32_t min_index = 0;
    uint32_t min_value = 1 << 24;
    for (size_t j = 0; j < min(bin_size, len - i * bin_size); j++) {
      size_t now = i * bin_size + j;
      if (min_value > vid.at(now).first) {
        min_index = vid.at(now).second;
        min_value = vid.at(now).first;
        // std::cout << now << std::endl;
      }
    }
    indexs.at(i).first = min_value;
    indexs.at(i).second = min_index;
    // std::cout << min_value << " " << min_index << std::endl;
  }
  std::partial_sort(indexs.begin(), indexs.begin() + k, indexs.end(),
                    less<pair<uint32_t, uint32_t>>());
  vector<pair<uint32_t, uint32_t>> topk(indexs.begin(), indexs.begin() + k);

  return topk;
}

void discard(vector<uint32_t> &v, size_t begin, size_t len, uint32_t bw) {
  for (size_t i = 0; i < len; i++) {
    v[begin + i] = v[begin + i] >> bw;
  }
}

uint32_t knns(vector<uint32_t> &q, vector<vector<uint32_t>> &ps,
              vector<vector<uint32_t>> &cluster_data,
              vector<vector<uint32_t>> &ptoc, vector<vector<uint32_t>> &stash,
              vector<uint32_t> &neighbors) {
  auto distance = distance_compute(q, cluster_data);
  size_t k_cluster = 0;
  for (size_t i = 0; i < k_c.size() - 1; i++) {
    k_cluster += group_bin_number[i];
  }
  vector<uint32_t> candidate(k_cluster);

  vector<pair<uint32_t, uint32_t>> v_id;

  size_t sum = 0;
  size_t sum_k = 0;
  for (size_t i = 0; i < k_c.size(); i++) {
    size_t begin = sum;
    size_t len = k_c[i];
    size_t l = group_bin_number[i];
    size_t k = group_k_number[i];
    discard(distance, begin, len, bw[i]);
    auto index = approximate_topk(distance, begin, len, l, k);
    if (i != k_c.size() - 1) {
      for (size_t j = 0; j < k; j++) {
        candidate[sum_k + j] = index[j].second;
      }
      sum += k_c[i];
      sum_k += k;
    } else {
      assert(index.size() == knn);
      v_id.insert(v_id.begin(), index.begin(), index.end());
    }
  }
  for (size_t i = 0; i < knn; i++) {
    v_id[i].second = stash[v_id[i].second - sum][0];
  }
  const size_t m = 20;
  vector<vector<uint32_t>> c_ps;

  for (size_t i = 0; i < k_cluster; i++) {
    for (size_t j = 0; j < m; j++) {
      auto index = candidate[i];
      if (ptoc[index][j] != 111111112) {
        c_ps.emplace_back(ps[ptoc[index][j]]);
        v_id.emplace_back(make_pair(0, ptoc[index][j]));
      }

      // std::cout << "TEST!" << std::endl;
    }
  }
  auto distance_p = distance_compute(q, c_ps);
  for (size_t i = knn; i < knn + c_ps.size(); i++) {
    v_id[i].first = distance_p[i - knn] >> bw[bw.size() - 1];
  }

  std::partial_sort(v_id.begin(), v_id.begin() + knn, v_id.end(),
                    less<pair<uint32_t, uint32_t>>());
  uint32_t correct = 0;
  for (size_t i = 0; i < knn; i++) {
    if (std::find(neighbors.begin(), neighbors.end(), v_id[i].second) !=
        neighbors.end())
      correct++;
    // std::cout << v_id[i].second << std::endl;
  }
  return correct;
}

int main() {
  std::srand(0);
  auto ps = read_data(1000000, 128, "dataset/dataset.txt");
  auto cluster_data = read_data(118934, 128, "dataset/centrios.txt");
  auto test_data = read_data(10000, 128, "dataset/test.txt");
  auto ptoc = read_data(89608, 20, "dataset/ptoc.txt");
  auto neighbors = read_data(10000, 10, "dataset/neighbors.txt");
  auto stash = read_data(29326, 1, "dataset/stash.txt");
  uint32_t sum = 0;
  uint32_t all = 0;
  for (size_t i = 0; i < 10000; i++) {
    auto r = knns(test_data[i], ps, cluster_data, ptoc, stash, neighbors[i]);
    all += knn;
    sum += r;
    std::cout << i << ": " << sum << "/" << all << " = " << float(sum) / all
              << std::endl;
  }
  return 0;
}