#include "common_function.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

void test_known_memory_access_types() {
    const std::vector<std::pair<MemoryAccessType, std::string>> cases{
        {MemoryAccessType::READ, "READ"},
        {MemoryAccessType::WRITE, "WRITE"},
        {MemoryAccessType::GWRITE, "GWRITE"},
        {MemoryAccessType::COMP, "COMP"},
        {MemoryAccessType::COMP_HASH, "COMP_HASH"},
        {MemoryAccessType::READRES, "READRES"},
        {MemoryAccessType::P_HEADER, "P_HEADER"},
        {MemoryAccessType::COMPS_READRES, "COMPS_READRES"},
    };

    for (const auto& [type, expected] : cases) {
        expect_equal(memAccessTypeString(type), expected,
                     "MemoryAccessType " + expected);
    }
}

void test_invalid_memory_access_type() {
    try {
        (void)memAccessTypeString(static_cast<MemoryAccessType>(-1));
        ++failures;
        std::cerr << "  FAIL: invalid MemoryAccessType did not throw\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: invalid MemoryAccessType throws\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: invalid MemoryAccessType threw "
                  << error.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "  FAIL: invalid MemoryAccessType threw a non-standard exception\n";
    }
}

void test_dram_request_size_consistency() {
    const auto derived = ValidateDramRequestSizeConsistency(
        32, 16, 16, "pim.json", "memory.ini");
    if (derived == 32) {
        std::cout << "  PASS: matching DRAM request size is accepted\n";
    } else {
        ++failures;
        std::cerr << "  FAIL: derived DRAM request size should be 32 bytes\n";
    }

    try {
        (void)ValidateDramRequestSizeConsistency(
            64, 16, 16, "pim.json", "memory.ini");
        ++failures;
        std::cerr << "  FAIL: mismatching DRAM request size did not throw\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("dram_req_size=64 bytes") != std::string::npos &&
            message.find("= 32 bytes") != std::string::npos) {
            std::cout << "  PASS: mismatching DRAM request size reports both values\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: mismatch diagnostic omitted request sizes: "
                      << message << '\n';
        }
    }

    try {
        (void)ValidateDramRequestSizeConsistency(1, 16, 10);
        ++failures;
        std::cerr << "  FAIL: non-byte-aligned bus width did not throw\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: non-byte-aligned bus width throws\n";
    }
}

void test_dram_channel_consistency() {
    const auto channels = ValidateDramChannelConsistency(
        4, 4, "pim.json", "memory.ini");
    if (channels == 4) {
        std::cout << "  PASS: matching DRAM channel counts are accepted\n";
    } else {
        ++failures;
        std::cerr << "  FAIL: matching DRAM channel count should be 4\n";
    }

    try {
        (void)ValidateDramChannelConsistency(
            4, 2, "pim.json", "memory.ini");
        ++failures;
        std::cerr << "  FAIL: mismatching DRAM channel counts did not throw\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("dram_channels=4") != std::string::npos &&
            message.find("channels=2") != std::string::npos) {
            std::cout << "  PASS: channel mismatch reports both values\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: channel mismatch diagnostic omitted values: "
                      << message << '\n';
        }
    }

    try {
        (void)ValidateDramChannelConsistency(3, 3);
        ++failures;
        std::cerr << "  FAIL: non-power-of-two channel count did not throw\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: non-power-of-two channel count throws\n";
    }
}

void test_dram_frequency_consistency() {
    const double derived = ValidateDramFrequencyConsistency(
        2000, 0.5, "pim.json", "memory.ini");
    if (std::abs(derived - 2000.0) < 1.0e-9) {
        std::cout << "  PASS: matching DRAM frequency and tCK are accepted\n";
    } else {
        ++failures;
        std::cerr << "  FAIL: tCK=0.5 ns should imply 2000 MHz\n";
    }

    try {
        (void)ValidateDramFrequencyConsistency(
            1000, 0.5, "pim.json", "memory.ini");
        ++failures;
        std::cerr << "  FAIL: mismatching DRAM frequency did not throw\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("dram_freq=1000 MHz") != std::string::npos &&
            message.find("tCK=0.5 ns") != std::string::npos &&
            message.find("2000 MHz") != std::string::npos) {
            std::cout << "  PASS: frequency mismatch reports configured and derived values\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: frequency mismatch diagnostic omitted values: "
                      << message << '\n';
        }
    }

    try {
        (void)ValidateDramFrequencyConsistency(800, 0.0);
        ++failures;
        std::cerr << "  FAIL: zero tCK did not throw\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: zero tCK throws\n";
    }

    try {
        (void)ValidateDramFrequencyConsistency(1200, 0.833);
        std::cout << "  PASS: rounded tCK within 0.1 percent is accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: rounded tCK should be accepted: "
                  << error.what() << '\n';
    }
}

}  // namespace

int main() {
    test_known_memory_access_types();
    test_invalid_memory_access_type();
    test_dram_request_size_consistency();
    test_dram_channel_consistency();
    test_dram_frequency_consistency();

    if (failures == 0) {
        std::cout << "RESULT PASS: runtime guard checks\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
