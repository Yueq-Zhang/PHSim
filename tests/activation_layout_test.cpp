#include "common_function.hpp"
#include "tensor/MyTensor.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef PH_SIM_SOURCE_DIR
#define PH_SIM_SOURCE_DIR "."
#endif

namespace {

int failures = 0;

std::string source_path(const std::string& relative) {
    return std::string(PH_SIM_SOURCE_DIR) + "/" + relative;
}

void expect(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "  PASS: " << message << '\n';
        return;
    }
    ++failures;
    std::cerr << "  FAIL: " << message << '\n';
}

std::vector<std::vector<uint32_t>> make_indexes(
    uint32_t first_row, uint32_t last_row, uint32_t columns) {
    std::vector<std::vector<uint32_t>> indexes;
    indexes.reserve(
        static_cast<size_t>(last_row - first_row + 1) * columns);
    for (uint32_t row = first_row; row <= last_row; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            indexes.push_back({row, column});
        }
    }
    return indexes;
}

std::unordered_set<addr_type> expand_channels(
    const std::vector<addr_type>& logical_addresses) {
    std::unordered_set<addr_type> physical_addresses;
    for (const addr_type address : logical_addresses) {
        for (uint32_t channel = 0;
             channel < MyAddressAllocator::dram_channels; ++channel) {
            physical_addresses.insert(
                MyAddressAllocator::add_channel_index(address, channel));
        }
    }
    return physical_addresses;
}

bool disjoint(const std::unordered_set<addr_type>& lhs,
              const std::unordered_set<addr_type>& rhs) {
    for (const addr_type address : lhs) {
        if (rhs.count(address) != 0) return false;
    }
    return true;
}

uint64_t expected_logical_bursts(uint32_t rows, uint32_t columns) {
    const uint64_t elements_per_channel_burst =
        MyAddressAllocator::dram_burst_size /
        MyAddressAllocator::precision_activation;
    return ceil_div_u64(rows, MyAddressAllocator::dram_channels) *
           ceil_div_u64(columns, elements_per_channel_burst);
}

void verify_tensor_pair(uint32_t first_rows, uint32_t second_rows,
                        uint32_t columns, const std::string& name) {
    MyTensor first(name + "-first", {first_rows, columns},
                   TensorType::ACT, true);
    MyTensor second(name + "-second", {second_rows, columns},
                    TensorType::ACT, true);

    const auto first_logical = first.generate_addrs_based_on_indexes(
        make_indexes(0, first_rows - 1, columns));
    const auto second_logical = second.generate_addrs_based_on_indexes(
        make_indexes(0, second_rows - 1, columns));
    const auto first_physical = expand_channels(first_logical);
    const auto second_physical = expand_channels(second_logical);

    const uint64_t expected_first =
        expected_logical_bursts(first_rows, columns);
    const uint64_t expected_second =
        expected_logical_bursts(second_rows, columns);
    expect(first_logical.size() == expected_first,
           name + " first tensor has the expected logical burst count");
    expect(second_logical.size() == expected_second,
           name + " second tensor has the expected logical burst count");
    expect(first_physical.size() ==
               expected_first * MyAddressAllocator::dram_channels,
           name + " first tensor expands once to every channel");
    expect(second_physical.size() ==
               expected_second * MyAddressAllocator::dram_channels,
           name + " second tensor expands once to every channel");
    expect(disjoint(first_physical, second_physical),
           name + " adjacent request tensors do not overlap");

    if (first_rows > MyAddressAllocator::dram_channels) {
        const uint32_t tail_row = first_rows - 1;
        const auto tail_logical = first.generate_addrs_based_on_indexes(
            make_indexes(tail_row, tail_row, columns));
        const auto tail_physical = expand_channels(tail_logical);
        const uint64_t expected_columns = expected_logical_bursts(1, columns);
        expect(tail_logical.size() == expected_columns,
               name + " tail row keeps one logical channel group");
        expect(tail_physical.size() ==
                   expected_columns * MyAddressAllocator::dram_channels,
               name + " tail row preserves padded-channel traffic");
    }
}

}  // namespace

int main() {
    const std::filesystem::path output_path =
        std::filesystem::temp_directory_path() /
        "phsim-activation-layout-test";
    std::filesystem::create_directories(output_path);

    SysConfig config;
    config.initialize_from_config_path(
        source_path("tests/fixtures/smoke/compute_config.json"),
        source_path("tests/fixtures/smoke/memory_config.ini"),
        source_path("tests/fixtures/smoke/pim_config_iterative.json"),
        source_path(
            "tests/fixtures/smoke/inference_continuous_batching_cycle_accurate.json"),
        source_path("tests/fixtures/smoke/model_config_iterative_pim.json"),
        source_path("tests/fixtures/smoke/request_trace_continuous.csv"),
        output_path.string());
    config.effective_request_count = 3;
    MyAddressAllocator::init(config);
    MyAddressAllocator::activation_malloc();

    expect(MyAddressAllocator::dram_channels == 2,
           "fixture uses two DRAM channels");
    expect(MyAddressAllocator::dram_burst_size == 32,
           "fixture uses a 32-byte per-channel burst");
    expect(MyAddressAllocator::precision_activation == 2,
           "fixture uses two-byte activations");

    verify_tensor_pair(5, 4, 256, "continuous-batch fixture");
    verify_tensor_pair(5, 4, 4097, "multi-slice embedding");

    if (failures != 0) {
        std::cerr << failures << " activation layout checks failed\n";
        return 1;
    }
    std::cout << "All activation layout checks passed\n";
    return 0;
}
