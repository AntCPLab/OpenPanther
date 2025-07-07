#include "common.h"

#include "yacl/link/test_util.h"

const std::vector<int64_t> k_c = {50810, 25603, 9968, 3227, 29326};
const std::vector<int64_t> group_bin_number = {458, 270, 178, 84, 262};
const std::vector<int64_t> group_k_number = {50, 31, 19, 13, 10};
const uint32_t dims = 128;
const uint32_t message = 3;
// Todo: read parameters from a json
using namespace spu;
using namespace panther;

// Test for the prepare for batch argmin
void test_prepare_batch_argmin() {
  size_t num_points = 0;
  int64_t num_bin = 0;
  int64_t max_bin_size = 0;
  for (size_t i = 0; i < k_c.size(); i++) {
    num_points += k_c[i];
    num_bin += group_bin_number[i];
    int64_t bin_size =
        std::ceil(static_cast<float>(k_c[i]) / group_bin_number[i]);
    max_bin_size = std::max(bin_size, max_bin_size);
  }
  std::vector<uint32_t> input(num_points);
  for (size_t i = 0; i < num_points; i++) {
    input[i] = i;
  }
  auto res = PrepareBatchArgmin(input, k_c, group_bin_number,
                                {num_bin, max_bin_size}, 1111111111);
  DISPATCH_ALL_FIELDS(spu::FM32, "test_prepare_argmin", [&]() {
    auto xres = NdArrayView<ring2k_t>(res);
    uint32_t count = 0;
    int64_t sum_bin = 0;
    for (size_t i = 0; i < k_c.size(); i++) {
      int64_t bin_size =
          std::ceil(static_cast<float>(k_c[i]) / group_bin_number[i]);
      for (int64_t j = 0; j < group_bin_number[i]; j++) {
        for (int64_t k = 0; k < min(bin_size, k_c[i] - j * bin_size); k++) {
          if (xres[(j + sum_bin) * max_bin_size + k] != count)
            std::cout << "Error: " << xres[(j + sum_bin) * max_bin_size + k]
                      << " " << count << std::endl;
          count++;
        }
      };
      sum_bin += group_bin_number[i];
    }
  });
  SPDLOG_INFO("Prepare for batch argmin passed! Test size:({}, {})", num_bin,
              max_bin_size);
}

int main() {
  auto ps = read_data(1000000, 128, "dataset/dataset.txt");
  auto ptoc = read_data(89608, 20, "dataset/ptoc.txt");
  test_prepare_batch_argmin();
}