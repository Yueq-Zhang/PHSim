#include "common_function.hpp"
#include "Client/Client.h"
#include "operations/NormalizationValidation.hpp"
#include "operations/SramTilingValidation.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
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

    try {
        const uint32_t single_pus =
            phsim::ConfigValidator::ValidatePimBankGeometry(
                false, 1, 4, 4, "memory.ini");
        const uint32_t dual_pus =
            phsim::ConfigValidator::ValidatePimBankGeometry(
                true, 1, 4, 4, "memory.ini");
        if (single_pus != 16 || dual_pus != 8) {
            throw std::runtime_error("unexpected PU count");
        }
        std::cout << "  PASS: single/dual PIM geometry derives 16/8 PUs\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: valid PIM geometry rejected: "
                  << error.what() << '\n';
    }

    for (const uint32_t banks_per_group : {1U, 3U}) {
        try {
            (void)phsim::ConfigValidator::ValidatePimBankGeometry(
                true, 1, 1, banks_per_group, "memory.ini");
            ++failures;
            std::cerr << "  FAIL: dual-bank geometry accepted "
                      << banks_per_group << " bank(s)\n";
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: dual-bank geometry rejects "
                      << banks_per_group << " unpairable bank(s)\n";
        }
    }
}

void test_implemented_configuration_capabilities() {
    try {
        phsim::ConfigValidator::ValidateSupportedValue(
            "scheduler", "simple", "simple", "compute.json");
        phsim::ConfigValidator::ValidateLegacyNoOpValue(
            "sram_width", uint32_t{128}, uint32_t{128}, "compute.json");
        phsim::ConfigValidator::ValidateLegacyNoOpValue(
            "layout", std::string{"NHWC"}, std::string{"NHWC"},
            "compute.json");
        phsim::ConfigValidator::ValidateRequiredField(
            true, "icnt_latency", "compute.json");
        phsim::ConfigValidator::ValidatePositiveValue(
            "icnt_latency", uint32_t{1}, "compute.json");
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

    try {
        phsim::ConfigValidator::ValidateRequiredField(
            false, "icnt_latency", "compute.json");
        ++failures;
        std::cerr << "  FAIL: missing icnt_latency was accepted\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: missing icnt_latency is rejected\n";
    }

    try {
        phsim::ConfigValidator::ValidatePositiveValue(
            "icnt_latency", uint32_t{0}, "compute.json");
        ++failures;
        std::cerr << "  FAIL: zero icnt_latency was accepted\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: zero icnt_latency is rejected\n";
    }

    try {
        phsim::ConfigValidator::ValidateLegacyNoOpValue(
            "layout", std::string{"NCHW"}, std::string{"NHWC"},
            "compute.json");
        ++failures;
        std::cerr << "  FAIL: inactive NCHW layout was accepted\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: inactive NCHW layout cannot silently alter a run\n";
    }
}

void test_acceleration_method_validation() {
    for (const std::string& method : {"naive", "Loop_wise", "Proportional"}) {
        try {
            phsim::ConfigValidator::ValidateAccelerationMethod(
                true, method, "inference.json");
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "  FAIL: valid acceleration method '" << method
                      << "' was rejected: " << error.what() << '\n';
        }
    }

    try {
        phsim::ConfigValidator::ValidateAccelerationMethod(
            false, "", "inference.json");
        std::cout << "  PASS: disabled acceleration accepts an omitted method\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: disabled acceleration rejected an omitted method: "
                  << error.what() << '\n';
    }

    for (const std::string& method : {"", "loop_wise", "unknown"}) {
        try {
            phsim::ConfigValidator::ValidateAccelerationMethod(
                true, method, "inference.json");
            ++failures;
            std::cerr << "  FAIL: invalid acceleration method '" << method
                      << "' was accepted\n";
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            if (message.find("naive") != std::string::npos &&
                message.find("Loop_wise") != std::string::npos &&
                message.find("Proportional") != std::string::npos) {
                std::cout << "  PASS: invalid acceleration method '" << method
                          << "' reports the supported values\n";
            } else {
                ++failures;
                std::cerr << "  FAIL: acceleration diagnostic omitted supported "
                             "values: "
                          << message << '\n';
            }
        }
    }
}

void test_json_configuration_schema() {
    using phsim::JsonFieldSpec;
    using phsim::JsonValueKind;
    const std::initializer_list<JsonFieldSpec> schema = {
        {"count", JsonValueKind::UnsignedInteger, true},
        {"enabled", JsonValueKind::Boolean, false},
    };

    try {
        phsim::ConfigValidator::ValidateJsonObject(
            nlohmann::json{{"count", 2}, {"enabled", true}}, "test",
            "test.json", schema);
        std::cout << "  PASS: valid JSON configuration schema is accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: valid JSON configuration was rejected: "
                  << error.what() << '\n';
    }

    for (const auto& invalid : {
             nlohmann::json{{"count", 2}, {"typo", true}},
             nlohmann::json{{"enabled", true}},
             nlohmann::json{{"count", "two"}},
             nlohmann::json{{"count",
                             static_cast<uint64_t>(
                                 std::numeric_limits<uint32_t>::max()) + 1}}}) {
        try {
            phsim::ConfigValidator::ValidateJsonObject(
                invalid, "test", "test.json", schema);
            ++failures;
            std::cerr << "  FAIL: invalid JSON configuration was accepted\n";
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: invalid JSON configuration is rejected\n";
        }
    }

    try {
        (void)load_config("__phsim_missing_configuration__.json");
        ++failures;
        std::cerr << "  FAIL: missing JSON configuration file was accepted\n";
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(
                "__phsim_missing_configuration__.json") !=
            std::string::npos) {
            std::cout << "  PASS: missing JSON configuration reports its path\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: missing JSON diagnostic omitted its path\n";
        }
    }
}

void test_feature_compatibility_validation() {
    try {
        phsim::ConfigValidator::ValidateFeatureCompatibility(
            false, true, true, "Proportional", false);
        std::cout << "  PASS: supported virtual-memory acceleration is accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: supported feature combination rejected: "
                  << error.what() << '\n';
    }

    for (const auto& combination : {
             std::tuple<bool, bool, bool, std::string, bool>{
                 true, false, true, "Proportional", false},
             {false, true, true, "Loop_wise", false},
             {true, false, false, "", true}}) {
        try {
            phsim::ConfigValidator::ValidateFeatureCompatibility(
                std::get<0>(combination), std::get<1>(combination),
                std::get<2>(combination), std::get<3>(combination),
                std::get<4>(combination));
            ++failures;
            std::cerr << "  FAIL: incompatible feature combination was accepted\n";
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: incompatible feature combination is rejected\n";
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

void test_layernorm_dimension_validation() {
    try {
        phsim::ValidateLayerNormInputDimensions(
            {2, 16}, {16}, "layernorm_test", 0);
        std::cout << "  PASS: valid two-dimensional LayerNorm input is accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: valid LayerNorm dimensions were rejected: "
                  << error.what() << '\n';
    }

    try {
        phsim::ValidateLayerNormInputDimensions(
            {16}, {16}, "layernorm_test", 0);
        ++failures;
        std::cerr << "  FAIL: one-dimensional LayerNorm input was accepted\n";
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("[tokens, hidden_size]") !=
            std::string::npos) {
            std::cout << "  PASS: LayerNorm input rank mismatch is rejected clearly\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: LayerNorm rank diagnostic omitted dimensions: "
                      << error.what() << '\n';
        }
    }

    try {
        phsim::ValidateLayerNormInputDimensions(
            {2, 3, 4}, {4}, "layernorm_test", 1);
        ++failures;
        std::cerr << "  FAIL: three-dimensional LayerNorm input was accepted\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: unsupported three-dimensional LayerNorm input is rejected\n";
    }

    try {
        phsim::ValidateLayerNormInputDimensions(
            {2, 8}, {16}, "layernorm_test", 3);
        ++failures;
        std::cerr << "  FAIL: mismatching LayerNorm hidden size was accepted\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("input 3") != std::string::npos &&
            message.find("hidden_size") != std::string::npos &&
            message.find("[2, 8]") != std::string::npos &&
            message.find("[16]") != std::string::npos) {
            std::cout << "  PASS: LayerNorm hidden-size mismatch is rejected clearly\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: LayerNorm mismatch diagnostic omitted values: "
                      << message << '\n';
        }
    }
}

void test_layernorm_parameter_dimension_validation() {
    try {
        const uint32_t one_dimensional =
            phsim::ValidateLayerNormParameterDimensions(
                {16}, {16}, "layernorm_test");
        if (one_dimensional != 16) {
            throw std::runtime_error("unexpected LayerNorm parameter size");
        }
        const uint32_t size_bytes =
            phsim::ValidateLayerNormParameterSizeBytes(
                one_dimensional, 2, "layernorm_test");
        if (size_bytes != 32) {
            throw std::runtime_error("unexpected LayerNorm parameter byte size");
        }
        std::cout << "  PASS: valid LayerNorm gamma/beta shapes are accepted\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: valid LayerNorm parameters were rejected: "
                  << error.what() << '\n';
    }

    for (const auto& invalid_shapes : {
             std::pair<std::vector<uint32_t>, std::vector<uint32_t>>{
                 {}, {}},
             {{3, 4}, {3, 4}},
             {{16}, {8}},
             {{0}, {0}}}) {
        try {
            (void)phsim::ValidateLayerNormParameterDimensions(
                invalid_shapes.first, invalid_shapes.second,
                "layernorm_test");
            ++failures;
            std::cerr << "  FAIL: invalid LayerNorm gamma/beta shapes were accepted\n";
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: invalid LayerNorm gamma/beta shapes are rejected\n";
        }
    }

    for (const auto& invalid_size : {
             std::pair<uint32_t, uint32_t>{16, 0},
             {std::numeric_limits<uint32_t>::max(), 2}}) {
        try {
            (void)phsim::ValidateLayerNormParameterSizeBytes(
                invalid_size.first, invalid_size.second, "layernorm_test");
            ++failures;
            std::cerr << "  FAIL: invalid LayerNorm parameter byte size was accepted\n";
        } catch (const std::invalid_argument&) {
            std::cout << "  PASS: zero LayerNorm parameter precision is rejected\n";
        } catch (const std::overflow_error&) {
            std::cout << "  PASS: overflowing LayerNorm parameter byte size is rejected\n";
        }
    }
}

void test_sram_tiling_guards() {
    if (phsim::AvailablePingPongSramBytes(8) != 4096) {
        ++failures;
        std::cerr << "  FAIL: ping-pong SRAM capacity conversion is incorrect\n";
    } else {
        std::cout << "  PASS: ping-pong SRAM capacity uses KiB and bytes consistently\n";
    }

    std::vector<uint32_t> inner{5, 16};
    std::vector<uint32_t> outer{1};
    phsim::HalveSramTileDimensionAndDoubleCount(
        inner, 0, outer, 0, "guard_test", 5000, 4096);
    if (inner[0] != 3 || outer[0] != 2) {
        ++failures;
        std::cerr << "  FAIL: SRAM tile split did not preserve ceil-halving\n";
    } else {
        std::cout << "  PASS: SRAM tile split preserves the existing ceil-halving policy\n";
    }

    try {
        std::vector<uint32_t> minimum_inner{1, 128};
        std::vector<uint32_t> minimum_outer{8};
        phsim::HalveSramTileDimensionAndDoubleCount(
            minimum_inner, 0, minimum_outer, 0, "guard_test",
            832, 512, "one token remains");
        ++failures;
        std::cerr << "  FAIL: oversized minimum SRAM tile did not throw\n";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("minimum tile cannot fit in SRAM") !=
                std::string::npos &&
            message.find("832 bytes") != std::string::npos &&
            message.find("512 bytes") != std::string::npos) {
            std::cout << "  PASS: oversized minimum SRAM tile fails clearly\n";
        } else {
            ++failures;
            std::cerr << "  FAIL: SRAM capacity diagnostic omitted values: "
                      << message << '\n';
        }
    }

    try {
        const std::vector<uint32_t> current{16, 8, 4};
        const std::vector<uint32_t> next{16, 8, 4};
        const std::vector<uint32_t> outer_counts{2, 2, 2};
        (void)phsim::SelectShrinkableSramDimension(
            current, next, {0, 1, 2}, outer_counts, "guard_test",
            8192, 4096, "all dimensions reached their minimum granularity");
        ++failures;
        std::cerr << "  FAIL: non-progressing aligned SRAM tile did not throw\n";
    } catch (const std::invalid_argument&) {
        std::cout << "  PASS: non-progressing aligned SRAM tile fails clearly\n";
    }

    try {
        const std::vector<uint32_t> current{64, 32, 16};
        const std::vector<uint32_t> next{64, 16, 8};
        const std::vector<uint32_t> outer_counts{2, 1, 1};
        const size_t selected = phsim::SelectShrinkableSramDimension(
            current, next, {0, 1, 2}, outer_counts, "guard_test",
            8192, 4096);
        if (selected != 1) {
            throw std::runtime_error("unexpected SRAM split dimension");
        }
        std::cout << "  PASS: aligned tiling skips a stalled dimension and continues shrinking\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: aligned tiling did not select another shrinkable dimension: "
                  << error.what() << '\n';
    }

    try {
        uint32_t count = std::numeric_limits<uint32_t>::max();
        phsim::DoubleSramTileCountOrThrow(count, "guard_test", 0);
        ++failures;
        std::cerr << "  FAIL: SRAM tile-count overflow did not throw\n";
    } catch (const std::overflow_error&) {
        std::cout << "  PASS: SRAM tile-count overflow is rejected\n";
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
    test_acceleration_method_validation();
    test_json_configuration_schema();
    test_feature_compatibility_validation();
    test_client_cycle_width();
    test_layernorm_dimension_validation();
    test_layernorm_parameter_dimension_validation();
    test_sram_tiling_guards();

    if (failures == 0) {
        std::cout << "RESULT PASS: runtime guard checks\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
