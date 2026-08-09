#include "common_function.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using PIMHashAddressing::AddressGroup;

bool expect_groups(const std::string& name,
                   const std::vector<addr_type>& addresses,
                   const std::vector<AddressGroup>& expected,
                   uint32_t dram_burst_size_bytes = 32) {
    const auto actual = PIMHashAddressing::group_comp_addresses(
        addresses, 1024, dram_burst_size_bytes);
    bool pass = actual.size() == expected.size();
    if (pass) {
        for (size_t i = 0; i < actual.size(); ++i) {
            pass = pass &&
                   actual[i].aligned_start_address ==
                       expected[i].aligned_start_address &&
                   actual[i].burst_count == expected[i].burst_count;
        }
    }
    std::cout << "CHECK " << name << ' ' << (pass ? "PASS" : "FAIL")
              << " groups=" << actual.size() << '\n';
    return pass;
}

bool expect_invalid_granularity() {
    bool zero_rejected = false;
    bool indivisible_rejected = false;
    try {
        (void)PIMHashAddressing::group_comp_addresses({0}, 1024, 0);
    } catch (const std::invalid_argument&) {
        zero_rejected = true;
    }
    try {
        (void)PIMHashAddressing::group_comp_addresses({0}, 1000, 32);
    } catch (const std::invalid_argument&) {
        indivisible_rejected = true;
    }
    const bool pass = zero_rejected && indivisible_rejected;
    std::cout << "CHECK invalid_granularity "
              << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

}  // namespace

int main() {
    bool pass = true;
    pass = expect_groups("empty", {}, {}) && pass;
    pass = expect_groups("single_address", {77}, {{64, 1}}) && pass;
    pass = expect_groups("same_chunk_with_duplicate", {64, 65, 65, 95},
                         {{64, 4}}) && pass;
    pass = expect_groups("last_address_crosses_chunk", {31, 32},
                         {{0, 1}, {32, 1}}) && pass;
    pass = expect_groups("multiple_chunks", {30, 31, 32, 33, 64},
                         {{0, 2}, {32, 2}, {64, 1}}) && pass;
    pass = expect_groups("noncontiguous_chunk_revisit", {64, 32, 33, 64},
                         {{64, 1}, {32, 2}, {64, 1}}) && pass;
    pass = expect_groups("server_64_byte_burst", {15, 16},
                         {{0, 1}, {16, 1}}, 64) && pass;
    pass = expect_invalid_granularity() && pass;
    return pass ? 0 : 1;
}
