#include "DRAM/DataContainer.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

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
    config.dram_data_container_max_payload_mb = 0;
    config.mem_config.co_pos = 0;
    config.mem_config.ba_pos = 2;
    config.mem_config.ro_pos = 3;
    config.mem_config.ch_pos = 6;
    config.mem_config.ra_pos = 0;
    config.mem_config.bg_pos = 0;
    config.mem_config.co_mask = 0x3;
    config.mem_config.ba_mask = 0x1;
    config.mem_config.ro_mask = 0x7;
    config.mem_config.ch_mask = 0x1;
    config.mem_config.ra_mask = 0;
    config.mem_config.bg_mask = 0;

    MyAddressAllocator::dram_channels = 2;
    MyAddressAllocator::ranks = 1;
    MyAddressAllocator::bankgroups = 1;
    MyAddressAllocator::banks = 2;
    MyAddressAllocator::rows = 8;
    MyAddressAllocator::columns = 16;
    MyAddressAllocator::burst_length = 4;
    MyAddressAllocator::dram_burst_size = 8;
    MyAddressAllocator::field_pos.clear();
    MyAddressAllocator::mask.clear();
    MyAddressAllocator::field_pos["co"] = 0;
    MyAddressAllocator::field_pos["ba"] = 2;
    MyAddressAllocator::field_pos["ro"] = 3;
    MyAddressAllocator::field_pos["ch"] = 6;
    MyAddressAllocator::field_pos["ra"] = 0;
    MyAddressAllocator::field_pos["bg"] = 0;
    MyAddressAllocator::mask["co"] = 0x3;
    MyAddressAllocator::mask["ba"] = 0x1;
    MyAddressAllocator::mask["ro"] = 0x7;
    MyAddressAllocator::mask["ch"] = 0x1;
    MyAddressAllocator::mask["ra"] = 0;
    MyAddressAllocator::mask["bg"] = 0;
}

addr_type address(uint32_t channel, uint32_t bank, uint32_t row,
                  uint32_t burst_column) {
    return MyAddressAllocator::make_address_by_index(
        0, 0, bank, row, burst_column, channel);
}

void test_capacity(const SysConfig& config) {
    begin_case("capacity");
    DramDataContainer container(config);
    expect(container.burst_bytes() == 8, "burst payload should be BL * bus width");
    expect(container.physical_capacity_bursts() == 128,
           "capacity should cover every configured burst coordinate");
    expect(container.physical_capacity_bytes() == 1024,
           "byte capacity should match burst count times burst bytes");
}

void test_zero_read_is_non_materializing(const SysConfig& config) {
    begin_case("zero_read_is_non_materializing");
    DramDataContainer container(config);
    const auto data = container.read_burst(address(1, 1, 3, 2));
    expect(data == DramDataContainer::Burst(8, 0),
           "untouched burst should read as zero");
    expect(container.resident_bursts() == 0,
           "zero-on-miss read should not allocate storage");
}

void test_round_trip_and_adjacent_isolation(const SysConfig& config) {
    begin_case("round_trip_and_adjacent_isolation");
    DramDataContainer container(config);
    const DramDataContainer::Burst first{1, 2, 3, 4, 5, 6, 7, 8};
    const DramDataContainer::Burst second{11, 12, 13, 14, 15, 16, 17, 18};
    const addr_type first_addr = address(0, 0, 2, 0);
    const addr_type second_addr = address(0, 0, 2, 1);
    container.write_burst(first_addr, first);
    container.write_burst(second_addr, second);
    expect(container.read_burst(first_addr) == first,
           "first adjacent burst should retain its payload");
    expect(container.read_burst(second_addr) == second,
           "second adjacent burst should not overlap the first");
    expect(container.resident_bursts() == 2,
           "two non-zero addresses should materialize two bursts");
}

void test_short_write_padding(const SysConfig& config) {
    begin_case("short_write_padding");
    DramDataContainer container(config);
    const addr_type target = address(0, 1, 5, 3);
    container.write_burst(target, {0xA5, 0x5A});
    expect(container.read_burst(target) ==
               DramDataContainer::Burst({0xA5, 0x5A, 0, 0, 0, 0, 0, 0}),
           "short write should be padded to one burst");
}

void test_zero_elision(const SysConfig& config) {
    begin_case("zero_elision");
    DramDataContainer container(config);
    const addr_type target = address(0, 0, 1, 0);
    container.write_burst(target, {9});
    expect(container.resident_payload_bytes() == 8,
           "one resident burst should account for one payload");
    container.write_burst(target, DramDataContainer::Burst(8, 0));
    expect(container.resident_bursts() == 0,
           "writing an all-zero burst should release sparse storage");
    expect(container.peak_resident_payload_bytes() == 8,
           "peak payload should retain the observed high-water mark");
}

void test_clear_and_instance_isolation(const SysConfig& config) {
    begin_case("clear_and_instance_isolation");
    const addr_type target = address(1, 1, 7, 3);
    DramDataContainer first(config);
    first.write_burst(target, {1});
    first.clear();
    expect(first.resident_bursts() == 0, "clear should release resident bursts");

    DramDataContainer second(config);
    expect(second.read_burst(target) == DramDataContainer::Burst(8, 0),
           "a new container instance must not inherit prior state");
}

void test_noncanonical_address_rejected(const SysConfig& config) {
    begin_case("noncanonical_address_rejected");
    DramDataContainer container(config);
    bool rejected = false;
    try {
        container.read_burst(address(0, 0, 0, 0) | (addr_type{1} << 20));
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    expect(rejected, "bits outside the configured layout should be rejected");
}

}  // namespace

int main() {
    try {
        SysConfig config;
        configure_tiny_system(config);
        test_capacity(config);
        test_zero_read_is_non_materializing(config);
        test_round_trip_and_adjacent_isolation(config);
        test_short_write_padding(config);
        test_zero_elision(config);
        test_clear_and_instance_isolation(config);
        test_noncanonical_address_rejected(config);
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
    }

    if (failures == 0) {
        std::cout << "RESULT PASS: 7 burst-sparse DataContainer cases\n";
        return 0;
    }
    std::cerr << "RESULT FAIL: " << failures << " assertion(s) failed\n";
    return 1;
}
