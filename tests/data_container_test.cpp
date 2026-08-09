#include "common_function.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Burst = std::vector<std::vector<uint8_t>>;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "  FAIL: " << message << '\n';
    }
}

void begin_case(const char* name) {
    std::cout << "CASE " << name << '\n';
}

void configure_tiny_system(SysConfig& config) {
    config.dram_channels = 2;
    config.mem_config.ranks = 1;
    config.mem_config.bankgroups = 1;
    config.mem_config.banks_per_group = 2;
    config.mem_config.rows = 8;
    config.mem_config.columns = 16;
    config.mem_config.BL = 4;
    config.mem_config.bus_width = 16;
}

bool all_zero(const Burst& burst) {
    for (const auto& column : burst) {
        for (uint8_t byte : column) {
            if (byte != 0) {
                return false;
            }
        }
    }
    return true;
}

void reset_container(const SysConfig& config) {
    DRAMDataContainer::cleanup();
    MyAddressAllocator::columns = config.mem_config.columns;
    DRAMDataContainer::init(config);
}

void test_init_is_sparse(const SysConfig& config) {
    begin_case("init_is_sparse");
    reset_container(config);

    expect(DRAMDataContainer::DRAMDataContainer.size() == 2,
           "channel dimension should match the tiny config");
    expect(DRAMDataContainer::DRAMDataContainer[0].size() == 1,
           "rank dimension should match the tiny config");
    expect(DRAMDataContainer::DRAMDataContainer[0][0].size() == 1,
           "bank-group dimension should match the tiny config");
    expect(DRAMDataContainer::DRAMDataContainer[0][0][0].size() == 2,
           "bank dimension should match the tiny config");
    expect(DRAMDataContainer::DRAMDataContainer[0][0][0][0].empty(),
           "rows should not be allocated eagerly");
    expect(DRAMDataContainer::DRAMDataContainer[1][0][0][1].empty(),
           "all banks should start without materialized rows");
}

void test_default_read_and_isolation(const SysConfig& config) {
    begin_case("default_read_and_isolation");
    reset_container(config);

    const Burst read = DRAMDataContainer::data_read(1, 0, 0, 1, 3, 4);
    expect(read.size() == 4, "read should return one configured burst");
    for (const auto& column : read) {
        expect(column.size() == 2,
               "each column should contain bus_width / 8 bytes");
    }
    expect(all_zero(read), "an untouched row should read as zero");
    expect(DRAMDataContainer::DRAMDataContainer[1][0][0][1].size() == 1,
           "only the accessed sparse row should be materialized");
    expect(DRAMDataContainer::DRAMDataContainer[0][0][0][0].empty(),
           "reading one bank must not allocate another bank");
}

void test_full_burst_round_trip(const SysConfig& config) {
    begin_case("full_burst_round_trip");
    reset_container(config);

    const Burst written{{0x10, 0x11}, {0x20, 0x21},
                        {0x30, 0x31}, {0x40, 0x41}};
    DRAMDataContainer::data_write(written, 0, 0, 0, 0, 2, 4);
    const Burst read = DRAMDataContainer::data_read(0, 0, 0, 0, 2, 4);
    expect(read == written, "a full burst should round-trip byte-for-byte");

    const std::vector<uint8_t> expected_flat{
        0x10, 0x11, 0x20, 0x21, 0x30, 0x31, 0x40, 0x41};
    expect(DRAMDataContainer::flatten_burst(read) == expected_flat,
           "flatten_burst should preserve column-major burst order");
}

void test_partial_write_and_coordinate_isolation(const SysConfig& config) {
    begin_case("partial_write_and_coordinate_isolation");
    reset_container(config);

    const Burst partial{{0xA5}, {0x5A, 0xC3}};
    DRAMDataContainer::data_write(partial, 0, 0, 0, 0, 5, 8);

    const Burst target = DRAMDataContainer::data_read(0, 0, 0, 0, 5, 8);
    const Burst expected{{0xA5, 0x00}, {0x5A, 0xC3},
                         {0x00, 0x00}, {0x00, 0x00}};
    expect(target == expected,
           "a short write should pad missing bytes and preserve untouched columns");

    const Burst other_channel =
        DRAMDataContainer::data_read(1, 0, 0, 0, 5, 8);
    const Burst other_bank =
        DRAMDataContainer::data_read(0, 0, 0, 1, 5, 8);
    expect(all_zero(other_channel),
           "the same coordinates in another channel should remain untouched");
    expect(all_zero(other_bank),
           "the same coordinates in another bank should remain untouched");
}

void test_cleanup_releases_storage(const SysConfig& config) {
    begin_case("cleanup_releases_storage");
    reset_container(config);
    DRAMDataContainer::data_write({{1, 2}, {3, 4}}, 0, 0, 0, 0, 1, 0);
    DRAMDataContainer::cleanup();
    expect(DRAMDataContainer::DRAMDataContainer.empty(),
           "cleanup should release the outer container storage");
}

}  // namespace

int main() {
    try {
        SysConfig config;
        configure_tiny_system(config);
        test_init_is_sparse(config);
        test_default_read_and_isolation(config);
        test_full_burst_round_trip(config);
        test_partial_write_and_coordinate_isolation(config);
        test_cleanup_releases_storage(config);
        DRAMDataContainer::cleanup();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "UNCAUGHT NON-STANDARD EXCEPTION\n";
    }

    if (failures == 0) {
        std::cout << "RESULT PASS: 5 cases, tiny sparse configuration\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " assertion(s) failed\n";
    return 1;
}
