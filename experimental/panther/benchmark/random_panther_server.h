#include "../protocol/common.h"
#include "yacl/link/test_util.h"

#include "libspu/mpc/utils/simulate.h"
// hyparameters:
// #define TEST_DEEP1M
// These parameter choices are taken from Appendix A of the SANNS paper for a
// fair comparison.
// Ref: https://www.usenix.org/system/files/sec20-chen-hao.pdf

#ifdef TEST_SIFT
const size_t pir_logt = 12;
const size_t pir_fixt = 2;
const size_t logt = 24;
const size_t N = 4096;
const size_t dis_N = 2048;
const size_t compare_radix = 5;
const size_t max_cluster_points = 20;
const std::vector<int64_t> k_c = {50810, 25603, 9968, 4227, 31412};
const std::vector<int64_t> group_bin_number = {458, 270, 178, 84, 262};
const std::vector<int64_t> group_k_number = {50, 31, 19, 13, 10};
const size_t total_points_num = 1000000;
const uint32_t dims = 128;
const size_t topk_k = 10;
const size_t pointer_dc_bits = 8;
const size_t cluster_dc_bits = 5;
const size_t message_size = 3;
const size_t ele_size = (dims + 2 * message_size) * max_cluster_points;
const uint32_t MASK = (1 << logt) - 1;
const uint32_t sum_k_c = 122020;
const uint32_t total_cluster_size = 90608;

#endif

// Amazon
#ifdef TEST_AMAZON
const size_t pir_logt = 12;
const size_t pir_fixt = 2;
const size_t logt = 24;
const size_t cluster_shift = 4;
const uint32_t dims = 50;
const size_t N = 4096;
const size_t dis_N = 2048;
const size_t compare_radix = 5;
const size_t max_cluster_points = 25;
const std::vector<int64_t> k_c = {41293, 24143, 9708, 3516, 1156, 8228};
const std::vector<int64_t> group_bin_number = {364, 364, 178, 84, 84, 84};
const std::vector<int64_t> group_k_number = {37, 37, 22, 10, 7, 10};
const size_t total_points_num = 1048576;
const size_t topk_k = 10;
const size_t pointer_dc_bits = 6;
const size_t cluster_dc_bits = 4;
const size_t message_size = 3;
const size_t ele_size = (dims + 2 * message_size) * max_cluster_points;
const uint32_t MASK = (1 << logt) - 1;
const uint32_t sum_k_c = 88044;
const uint32_t total_cluster_size = 79816;
#endif

// deep1m
#ifdef TEST_DEEP1M
const size_t pir_fixt = 2;
const size_t logt = 23;
const size_t pir_logt = 12;
const uint32_t dims = 96;
const size_t N = 4096;
const size_t dis_N = 2048;
const size_t compare_radix = 5;
const size_t max_cluster_points = 22;
const std::vector<int64_t> k_c = {44830, 25867, 11795, 5607, 2611, 25150};

const std::vector<int64_t> group_bin_number = {458, 270, 178, 84, 84, 210};
const std::vector<int64_t> group_k_number = {46, 31, 19, 13, 7, 10};
const size_t total_points_num = 1000000;
const size_t topk_k = 10;
const size_t pointer_dc_bits = 8;
const size_t cluster_dc_bits = 5;
const size_t message_size = 3;
const size_t ele_size = (dims + 2 * message_size) * max_cluster_points;
const uint32_t MASK = (1 << logt) - 1;
const uint32_t sum_k_c = 115860;
const uint32_t total_cluster_size = 90710;

#endif