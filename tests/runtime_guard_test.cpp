#include "common_function.hpp"
#include "Client/Client.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

int failures = 0;

static_assert(std::is_same_v<decltype(InferRequest::arrival_cycle), cycle_type>);
static_assert(std::is_same_v<decltype(InferRequest::completed_cycle), cycle_type>);
static_assert(std::is_same_v<decltype(std::declval<const Client&>().current_cycle()),
                             cycle_type>);

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

void test_pim_bank_organization_consistency() {
    try {
        phsim::ConfigValidator::ValidatePimBankOrganization(
            false, "SINGLE", "pim.json", "memory.ini");
        phsim::ConfigValidator::ValidatePimBankOrganization(
            true, " dual ", "pim.json", "memory.ini");
        phsim::ConfigValidator::ValidatePimBankOrganization(
            false, "", "pim.json", "memory.ini");
        std::cout << "  PASS: matching SINGLE/DUAL and legacy default are accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: valid PIM organization rejected: "
                  << error.what() << '\n';
    }

    try {
        phsim::ConfigValidator::ValidatePimBankOrganization(
            true, "SINGLE", "pim.json", "memory.ini");
        ++failures;
        std::cerr << "  FAIL: mismatching PIM organization did not throw\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("dual_bank=true") != std::string::npos &&
            message.find("pim_type=SINGLE") != std::string::npos) {
            std::cout << "  PASS: PIM organization mismatch reports both values\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: PIM organization diagnostic omitted values: "
                      << message << '\n';
        }
    }
}

void test_implemented_configuration_capabilities() {
    try {
        phsim::ConfigValidator::ValidateSupportedValue(
            "scheduler", "simple", "simple", "compute.json");
        phsim::ConfigValidator::ValidateLegacyNoOpValue(
            "sram_width", uint32_t{128}, uint32_t{128}, "compute.json");
        std::cout << "  PASS: implemented and legacy-compatible values are accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: supported configuration rejected: "
                  << error.what() << '\n';
    }

    try {
        phsim::ConfigValidator::ValidateSupportedValue(
            "scheduler", "experimental", "simple", "compute.json");
        ++failures;
        std::cerr << "  FAIL: unsupported scheduler did not throw\n";
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("implements only 'simple'") !=
            std::string::npos) {
            std::cout << "  PASS: unsupported capability is rejected explicitly\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: unsupported capability diagnostic unclear: "
                      << error.what() << '\n';
        }
    }

    try {
        phsim::ConfigValidator::ValidateLegacyNoOpValue(
            "scalar_add_latency", cycle_type{2}, cycle_type{1},
            "compute.json");
        ++failures;
        std::cerr << "  FAIL: changed no-op field did not throw\n";
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("not implemented") !=
            std::string::npos) {
            std::cout << "  PASS: changed no-op field cannot silently affect a run\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: no-op field diagnostic unclear: "
                      << error.what() << '\n';
        }
    }
}

void test_client_cycle_width() {
    InferRequest request{};
    request.arrival_cycle =
        static_cast<cycle_type>(std::numeric_limits<uint32_t>::max()) + 17;
    request.completed_cycle = request.arrival_cycle + 29;
    if (request.completed_cycle - request.arrival_cycle == 29 &&
        request.arrival_cycle > std::numeric_limits<uint32_t>::max()) {
        std::cout << "  PASS: Client request timestamps preserve 64-bit cycles\n";
    } else {
        ++failures;
        std::cerr << "  FAIL: Client request timestamps truncated above uint32\n";
    }
}

}  // namespace

int main() {
    test_known_memory_access_types();
    test_invalid_memory_access_type();
    test_dram_request_size_consistency();
    test_dram_channel_consistency();
    test_dram_frequency_consistency();
    test_pim_bank_organization_consistency();
    test_implemented_configuration_capabilities();
    test_client_cycle_width();

    if (failures == 0) {
        std::cout << "RESULT PASS: runtime guard checks\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
