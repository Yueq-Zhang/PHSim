#include "common_function.hpp"
#include "Scheduler/MyScheduler.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

static_assert(
    std::is_same_v<
        std::remove_reference_t<decltype(
            std::declval<MyScheduler&>()._estimated_all_cycle)>,
        cycle_type>,
    "scheduler prediction cycles must retain the 64-bit simulator timeline");
static_assert(
    sizeof(decltype(
        std::declval<MyScheduler&>()._stage_stats)::value_type::second_type) >=
        sizeof(cycle_type),
    "scheduler stage statistics must not truncate the simulator timeline");
namespace {

int failures = 0;

void expect_equal(const std::string& actual, const std::string& expected,
                  const std::string& name) {
    if (actual == expected) {
        std::cout << "  PASS: " << name << '\n';
        return;
    }

    ++failures;
    std::cerr << "  FAIL: " << name << " expected " << expected
              << ", got " << actual << '\n';
}

void test_pim_bandwidth_zero_cycles() {
    Config::system_config.core_freq = 1000;
    MemoryIOStat stat(0, 0, 0);
    stat.pim_reads = 64;
    stat.pim_writes = 64;

    expect_equal(stat.get_by_enum(MemoryIOStat::StatType::PIMBandwidth),
                 "0", "PIM bandwidth with zero cycles");
}

void test_ceil_div_boundaries() {
    expect_equal(std::to_string(ceil_div_u64(0, 64)), "0",
                 "ceil divide zero");
    expect_equal(std::to_string(ceil_div_u64(32, 64)), "1",
                 "ceil divide sub-burst");
    expect_equal(std::to_string(ceil_div_u64(64, 64)), "1",
                 "ceil divide exact burst");
    expect_equal(std::to_string(ceil_div_u64(65, 64)), "2",
                 "ceil divide burst tail");
}

void test_pim_bandwidth_wide_intermediate() {
    Config::system_config.core_freq = 4000;
    MemoryIOStat stat(0, 0, 5'000'000'000ULL);
    stat.pim_reads = 3'000'000'000ULL;
    stat.pim_writes = 2'000'000'000ULL;

    expect_equal(stat.get_by_enum(MemoryIOStat::StatType::PIMBandwidth),
                 "4000000000", "PIM bandwidth wide intermediate");
}

void test_pim_bandwidth_preserves_truncation() {
    Config::system_config.core_freq = 1;
    MemoryIOStat stat(0, 0, 3);
    stat.pim_reads = 2;

    expect_equal(stat.get_by_enum(MemoryIOStat::StatType::PIMBandwidth),
                 "666666", "PIM bandwidth preserves integer truncation");
}

OperationStat make_operation_stat() {
    OperationStat stat("statistics-edge-case");
    stat.num_calculation = 128;
    return stat;
}

void test_npu_utilization_zero_denominators() {
    Config::system_config.core_width = 4;
    Config::system_config.core_height = 8;

    OperationStat zero_cycles = make_operation_stat();
    zero_cycles.compute_cycles = 0;
    expect_equal(
        zero_cycles.get_by_enum(OperationStat::StatType::NpuUtilization),
        "0", "NPU utilization with zero compute cycles");

    OperationStat zero_width = make_operation_stat();
    zero_width.compute_cycles = 4;
    Config::system_config.core_width = 0;
    expect_equal(
        zero_width.get_by_enum(OperationStat::StatType::NpuUtilization),
        "0", "NPU utilization with zero core width");

    OperationStat zero_height = make_operation_stat();
    Config::system_config.core_width = 4;
    Config::system_config.core_height = 0;
    zero_height.compute_cycles = 4;
    expect_equal(
        zero_height.get_by_enum(OperationStat::StatType::NpuUtilization),
        "0", "NPU utilization with zero core height");
}

void test_npu_utilization_wide_intermediate() {
    Config::system_config.core_width = 2'000'000'000U;
    Config::system_config.core_height = 2;

    OperationStat stat("statistics-wide-intermediate");
    stat.compute_cycles = 5'000'000'000ULL;
    stat.num_calculation = 10'000'000'000'000'000'000ULL;

    expect_equal(stat.get_by_enum(OperationStat::StatType::NpuUtilization),
                 "0.500000", "NPU utilization wide intermediate");
}

}  // namespace

int main() {
    test_ceil_div_boundaries();
    test_pim_bandwidth_zero_cycles();
    test_pim_bandwidth_wide_intermediate();
    test_pim_bandwidth_preserves_truncation();
    test_npu_utilization_zero_denominators();
    test_npu_utilization_wide_intermediate();

    if (failures == 0) {
        std::cout << "RESULT PASS: 11 statistics edge checks\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
