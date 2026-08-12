#include "DRAM/DramDataContainer.h"
#include "common_function.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Burst = DramDataContainer::BurstData;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "  FAIL: " << message << '\n';
    }
}

void begin_case(const char* name) { std::cout << "CASE " << name << '\n'; }

std::unique_ptr<SysConfig> tiny_config(bool enabled = true) {
    auto config = std::make_unique<SysConfig>();
    config->dram_channels = 2;
    config->mem_config.ranks = 1;
    config->mem_config.bankgroups = 1;
    config->mem_config.banks_per_group = 2;
    config->mem_config.rows = 8;
    config->mem_config.columns = 16;
    config->mem_config.BL = 4;
    config->mem_config.bus_width = 16;
    config->dram_data_container_enable = enabled;
    return config;
}

bool all_zero(const Burst& burst) {
    for (const auto& column : burst) {
        for (uint8_t byte : column) {
            if (byte != 0) return false;
        }
    }
    return true;
}

void test_construct_is_sparse() {
    begin_case("construct_is_sparse");
    DramDataContainer container(*tiny_config());
    expect(container.enabled(), "enabled configuration should create an enabled store");
    expect(container.stored_column_count() == 0,
           "construction must not allocate any DRAM rows or columns");
}

void test_default_read_does_not_allocate() {
    begin_case("default_read_does_not_allocate");
    DramDataContainer container(*tiny_config());
    const Burst read = container.read_burst(1, 0, 0, 1, 3, 4);
    expect(read.size() == 4, "read should return one configured burst");
    expect(read.front().size() == 2,
           "each column should contain bus_width / 8 bytes");
    expect(all_zero(read), "an untouched location should read as zero");
    expect(container.stored_column_count() == 0,
           "a default-zero read must not materialize sparse storage");
}

void test_full_and_partial_round_trip() {
    begin_case("full_and_partial_round_trip");
    DramDataContainer container(*tiny_config());
    const Burst written{{0x10, 0x11}, {0x20, 0x21},
                        {0x30, 0x31}, {0x40, 0x41}};
    container.write_burst(written, 0, 0, 0, 0, 2, 4);
    const Burst read = container.read_burst(0, 0, 0, 0, 2, 4);
    expect(read == written, "a full burst should round-trip byte-for-byte");
    expect(container.flatten_burst(read) ==
               std::vector<uint8_t>({0x10, 0x11, 0x20, 0x21,
                                     0x30, 0x31, 0x40, 0x41}),
           "flattening should preserve DRAM column and byte order");

    container.write_burst({{0xA5}, {0x5A, 0xC3}}, 0, 0, 0, 0, 5, 8);
    expect(container.read_burst(0, 0, 0, 0, 5, 8) ==
               Burst({{0xA5, 0x00}, {0x5A, 0xC3},
                      {0x00, 0x00}, {0x00, 0x00}}),
           "a short write should pad bytes and preserve untouched columns");
}

void test_instance_and_coordinate_isolation() {
    begin_case("instance_and_coordinate_isolation");
    DramDataContainer first(*tiny_config());
    DramDataContainer second(*tiny_config());
    first.write_burst({{1, 2}, {3, 4}}, 0, 0, 0, 0, 1, 0);
    expect(all_zero(second.read_burst(0, 0, 0, 0, 1, 0)),
           "two simulator-owned stores must not share data");
    expect(all_zero(first.read_burst(1, 0, 0, 0, 1, 0)),
           "another channel must remain untouched");
    expect(all_zero(first.read_burst(0, 0, 0, 1, 1, 0)),
           "another bank must remain untouched");
}

void test_disabled_and_clear() {
    begin_case("disabled_and_clear");
    DramDataContainer disabled(*tiny_config(false));
    disabled.write_burst({{1, 2}}, 0, 0, 0, 0, 1, 0);
    expect(disabled.stored_column_count() == 0,
           "disabled DataContainer should not retain writes");

    DramDataContainer enabled(*tiny_config());
    enabled.write_burst({{1, 2}}, 0, 0, 0, 0, 1, 0);
    enabled.clear();
    expect(enabled.stored_column_count() == 0,
           "clear should release all sparse data");
}

void test_invalid_coordinates() {
    begin_case("invalid_coordinates");
    DramDataContainer container(*tiny_config());
    bool threw = false;
    try {
        (void)container.read_burst(2, 0, 0, 0, 0, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, "out-of-range hierarchy coordinates should be rejected");
}

void test_payload_limit() {
    begin_case("payload_limit");
    auto config = tiny_config();
    config->dram_data_container_max_payload_mb = 1;
    config->mem_config.bus_width = 8 * 1024 * 1024;
    DramDataContainer container(*config);

    container.write_burst({{1}}, 0, 0, 0, 0, 0, 0);
    expect(container.resident_payload_bytes() == 1024 * 1024,
           "one large column should consume the configured payload limit");

    bool threw = false;
    try {
        container.write_burst({{2}}, 0, 0, 0, 0, 0, 1);
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "a new column beyond the payload limit should be rejected");
}

}  // namespace

int main() {
    try {
        test_construct_is_sparse();
        test_default_read_does_not_allocate();
        test_full_and_partial_round_trip();
        test_instance_and_coordinate_isolation();
        test_disabled_and_clear();
        test_invalid_coordinates();
        test_payload_limit();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
    }

    if (failures == 0) {
        std::cout << "RESULT PASS: 7 instance-owned sparse cases\n";
        return 0;
    }
    std::cerr << "RESULT FAIL: " << failures << " assertion(s) failed\n";
    return 1;
}
