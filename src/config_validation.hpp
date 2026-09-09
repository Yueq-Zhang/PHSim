#pragma once

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "json.hpp"

namespace phsim {

enum class JsonValueKind {
    Boolean,
    String,
    UnsignedInteger,
    UnsignedInteger64,
    Number,
};

struct JsonFieldSpec {
    const char* name;
    JsonValueKind kind;
    bool required;
};

// Configuration contracts shared by the JSON front end, the DRAM INI parser,
// and tests. Keeping cross-file checks here prevents one backend from silently
// interpreting a duplicated setting differently from another backend.
class ConfigValidator {
public:
    static void ValidateJsonObject(
        const nlohmann::json& config, const std::string& config_role,
        const std::string& config_path,
        std::initializer_list<JsonFieldSpec> fields) {
        if (!config.is_object()) {
            throw std::invalid_argument(
                config_role + " config '" + config_path +
                "' must contain a JSON object");
        }

        for (const auto& entry : config.items()) {
            const JsonFieldSpec* field = nullptr;
            for (const auto& candidate : fields) {
                if (entry.key() == candidate.name) {
                    field = &candidate;
                    break;
                }
            }
            if (field == nullptr) {
                throw std::invalid_argument(
                    "Unknown field '" + entry.key() + "' in " +
                    config_role + " config '" + config_path + "'");
            }
            if (!JsonValueMatches(entry.value(), field->kind)) {
                throw std::invalid_argument(
                    "Field '" + entry.key() + "' in " + config_role +
                    " config '" + config_path + "' must be " +
                    JsonKindName(field->kind));
            }
        }

        for (const auto& field : fields) {
            if (field.required && !config.contains(field.name)) {
                throw std::invalid_argument(
                    "Missing required field '" + std::string(field.name) +
                    "' in " + config_role + " config '" + config_path +
                    "'");
            }
        }
    }

    static void ValidateBackendCapabilities(bool event_driven,
                                            bool enable_self_refresh) {
        if (event_driven && enable_self_refresh) {
            throw std::invalid_argument(
                "EventDriven DRAM does not model self-refresh timing; disable "
                "enable_self_refresh or use the CycleAccurate backend");
        }
    }

    static void ValidateFeatureCompatibility(
        bool data_container_enabled, bool virtual_memory_enabled,
        bool acceleration_enabled, const std::string& acceleration_method,
        bool decode_pruning_enabled) {
        if (data_container_enabled && acceleration_enabled) {
            throw std::invalid_argument(
                "dram_data_container_enable=true is incompatible with "
                "accelerate_ctrl=true because DataContainer requires exact "
                "per-request data side effects");
        }
        if (virtual_memory_enabled && acceleration_enabled &&
            acceleration_method != "Proportional") {
            throw std::invalid_argument(
                "virtual_mem_hash_enable=true is incompatible with "
                "accelerate_method='" + acceleration_method +
                "'; virtual-memory side-effect replay supports only "
                "Proportional acceleration");
        }
        if (data_container_enabled && decode_pruning_enabled) {
            throw std::invalid_argument(
                "dram_data_container_enable=true is incompatible with "
                "decode_pruning_enabled=true because DataContainer requires "
                "exact per-request data side effects");
        }
    }

    static void ValidateAccelerationMethod(
        bool acceleration_enabled, const std::string& method,
        const std::string& config_path = {}) {
        if (!acceleration_enabled && method.empty()) {
            return;
        }
        if (method == "naive" || method == "Loop_wise" ||
            method == "Proportional") {
            return;
        }

        std::ostringstream message;
        if (method.empty()) {
            message << "Missing accelerate_method";
        } else {
            message << "Unsupported accelerate_method='" << method << "'";
        }
        if (!config_path.empty()) {
            message << " in '" << config_path << "'";
        }
        message << "; supported methods are 'naive', 'Loop_wise', and "
                   "'Proportional'";
        throw std::invalid_argument(message.str());
    }

    static uint32_t ValidateRequestSize(
        uint32_t configured_request_size_bytes, uint32_t burst_length,
        uint32_t bus_width_bits, const std::string& pim_config_path = {},
        const std::string& memory_config_path = {}) {
        if (burst_length == 0) {
            throw std::invalid_argument(
                "DRAM burst length must be greater than zero");
        }
        if (bus_width_bits == 0 || bus_width_bits % 8 != 0) {
            throw std::invalid_argument(
                "DRAM bus_width must be a positive multiple of 8 bits");
        }

        const uint64_t derived_request_size_bytes =
            static_cast<uint64_t>(burst_length) * bus_width_bits / 8;
        if (derived_request_size_bytes >
            std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "Derived DRAM burst size does not fit in uint32_t");
        }
        if (configured_request_size_bytes == 0) {
            throw std::invalid_argument(
                "PIM config dram_req_size must be greater than zero");
        }
        if (configured_request_size_bytes != derived_request_size_bytes) {
            std::ostringstream message;
            message << "DRAM request-size mismatch";
            AppendConfigPath(message, "PIM config", pim_config_path);
            message << " sets dram_req_size=" << configured_request_size_bytes
                    << " bytes";
            AppendComparedConfig(message, memory_config_path);
            message << " implies BL(" << burst_length << ") * bus_width("
                    << bus_width_bits << " bits) / 8 = "
                    << derived_request_size_bytes << " bytes";
            throw std::invalid_argument(message.str());
        }
        return static_cast<uint32_t>(derived_request_size_bytes);
    }

    static uint32_t ValidateChannels(
        uint32_t configured_channels, int memory_channels,
        const std::string& pim_config_path = {},
        const std::string& memory_config_path = {}) {
        if (configured_channels == 0) {
            throw std::invalid_argument(
                "PIM config dram_channels must be greater than zero");
        }
        if (memory_channels <= 0) {
            throw std::invalid_argument(
                "Memory config channels must be greater than zero");
        }

        const auto memory_channels_unsigned =
            static_cast<uint32_t>(memory_channels);
        if (!IsPowerOfTwo(configured_channels)) {
            throw std::invalid_argument(
                "PIM config dram_channels must be a power of two");
        }
        if (!IsPowerOfTwo(memory_channels_unsigned)) {
            throw std::invalid_argument(
                "Memory config channels must be a power of two");
        }
        if (configured_channels != memory_channels_unsigned) {
            std::ostringstream message;
            message << "DRAM channel-count mismatch";
            AppendConfigPath(message, "PIM config", pim_config_path);
            message << " sets dram_channels=" << configured_channels;
            AppendComparedConfig(message, memory_config_path);
            message << " sets channels=" << memory_channels_unsigned;
            throw std::invalid_argument(message.str());
        }
        return configured_channels;
    }

    static double ValidateFrequency(
        uint32_t configured_frequency_mhz, double tck_ns,
        const std::string& pim_config_path = {},
        const std::string& memory_config_path = {}) {
        constexpr double relative_tolerance = 1.0e-3;
        if (configured_frequency_mhz == 0) {
            throw std::invalid_argument(
                "PIM config dram_freq must be greater than zero");
        }
        if (!std::isfinite(tck_ns) || tck_ns <= 0.0) {
            throw std::invalid_argument(
                "Memory config tCK must be a finite value greater than zero");
        }

        const double derived_frequency_mhz = 1000.0 / tck_ns;
        const double relative_error =
            std::abs(static_cast<double>(configured_frequency_mhz) -
                     derived_frequency_mhz) /
            derived_frequency_mhz;
        if (relative_error > relative_tolerance) {
            std::ostringstream message;
            message << "DRAM frequency mismatch";
            AppendConfigPath(message, "PIM config", pim_config_path);
            message << " sets dram_freq=" << configured_frequency_mhz
                    << " MHz";
            AppendComparedConfig(message, memory_config_path);
            message << " sets tCK=" << tck_ns << " ns, which implies "
                    << derived_frequency_mhz
                    << " MHz (allowed relative error 0.1%)";
            throw std::invalid_argument(message.str());
        }
        return derived_frequency_mhz;
    }

    static std::string NormalizePimType(std::string pim_type) {
        pim_type.erase(pim_type.begin(),
                       std::find_if(pim_type.begin(), pim_type.end(),
                                    [](unsigned char value) {
                                        return !std::isspace(value);
                                    }));
        pim_type.erase(
            std::find_if(pim_type.rbegin(), pim_type.rend(),
                         [](unsigned char value) {
                             return !std::isspace(value);
                         })
                .base(),
            pim_type.end());
        std::transform(pim_type.begin(), pim_type.end(), pim_type.begin(),
                       [](unsigned char value) {
                           return static_cast<char>(std::toupper(value));
                       });
        return pim_type.empty() ? "SINGLE" : pim_type;
    }

    static void ValidatePimBankOrganization(
        bool dual_bank, const std::string& raw_pim_type,
        const std::string& pim_config_path = {},
        const std::string& memory_config_path = {}) {
        const std::string pim_type = NormalizePimType(raw_pim_type);
        if (pim_type != "SINGLE" && pim_type != "DUAL") {
            throw std::invalid_argument(
                "Unsupported memory-config pim_type='" + pim_type +
                "'; PIM simulation requires SINGLE or DUAL");
        }
        const bool ini_dual_bank = pim_type == "DUAL";
        if (dual_bank != ini_dual_bank) {
            std::ostringstream message;
            message << "PIM bank-organization mismatch";
            AppendConfigPath(message, "PIM config", pim_config_path);
            message << " sets dual_bank=" << (dual_bank ? "true" : "false");
            AppendComparedConfig(message, memory_config_path);
            message << " sets pim_type=" << pim_type
                    << " (expected " << (dual_bank ? "DUAL" : "SINGLE")
                    << ")";
            throw std::invalid_argument(message.str());
        }
    }

    static uint32_t ValidatePimBankGeometry(
        bool dual_bank, uint32_t ranks, uint32_t bankgroups,
        uint32_t banks_per_group,
        const std::string& memory_config_path = {}) {
        if (ranks == 0 || bankgroups == 0 || banks_per_group == 0) {
            throw std::invalid_argument(
                "PIM bank geometry requires positive ranks, bankgroups, and "
                "banks_per_group");
        }
        const uint64_t banks_per_channel =
            static_cast<uint64_t>(ranks) * bankgroups * banks_per_group;
        if (banks_per_channel > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "PIM banks per channel do not fit in uint32_t");
        }
        if (dual_bank &&
            (banks_per_channel < 2 || banks_per_channel % 2 != 0)) {
            std::ostringstream message;
            message << "Invalid dual-bank PIM geometry";
            AppendConfigPath(message, "memory config", memory_config_path);
            message << ": " << banks_per_channel
                    << " banks per channel cannot form complete bank pairs";
            throw std::invalid_argument(message.str());
        }
        return static_cast<uint32_t>(
            dual_bank ? banks_per_channel / 2 : banks_per_channel);
    }

    static void ValidateRequiredField(bool present, const std::string& field,
                                      const std::string& config_path = {}) {
        if (present) {
            return;
        }
        std::ostringstream message;
        message << "Missing required configuration field '" << field << "'";
        if (!config_path.empty()) {
            message << " in '" << config_path << "'";
        }
        throw std::invalid_argument(message.str());
    }

    template <typename T>
    static void ValidatePositiveValue(const std::string& field,
                                      const T& value,
                                      const std::string& config_path = {}) {
        if (value > T{0}) {
            return;
        }
        std::ostringstream message;
        message << "Configuration field " << field
                << " must be greater than zero";
        if (!config_path.empty()) {
            message << " in '" << config_path << "'";
        }
        throw std::invalid_argument(message.str());
    }

    static void ValidateSupportedValue(const std::string& field,
                                       const std::string& actual,
                                       const std::string& supported,
                                       const std::string& config_path = {}) {
        if (actual == supported) {
            return;
        }
        std::ostringstream message;
        message << "Unsupported " << field << "='" << actual << "'";
        if (!config_path.empty()) {
            message << " in '" << config_path << "'";
        }
        message << "; this build implements only '" << supported << "'";
        throw std::invalid_argument(message.str());
    }

    template <typename T>
    static void ValidateLegacyNoOpValue(const std::string& field,
                                        const T& actual,
                                        const T& legacy_value,
                                        const std::string& config_path = {}) {
        if (actual == legacy_value) {
            return;
        }
        std::ostringstream message;
        message << "Configuration field " << field << "=" << actual;
        if (!config_path.empty()) {
            message << " in '" << config_path << "'";
        }
        message << " is not implemented; only the legacy compatibility value "
                << legacy_value << " is accepted";
        throw std::invalid_argument(message.str());
    }

private:
    static bool JsonValueMatches(const nlohmann::json& value,
                                 JsonValueKind kind) {
        switch (kind) {
            case JsonValueKind::Boolean:
                return value.is_boolean();
            case JsonValueKind::String:
                return value.is_string();
            case JsonValueKind::UnsignedInteger:
                return IsNonNegativeInteger(value) &&
                       value.get<uint64_t>() <=
                           std::numeric_limits<uint32_t>::max();
            case JsonValueKind::UnsignedInteger64:
                return IsNonNegativeInteger(value);
            case JsonValueKind::Number:
                return value.is_number();
        }
        return false;
    }

    static const char* JsonKindName(JsonValueKind kind) {
        switch (kind) {
            case JsonValueKind::Boolean:
                return "a boolean";
            case JsonValueKind::String:
                return "a string";
            case JsonValueKind::UnsignedInteger:
                return "a non-negative integer in the uint32 range";
            case JsonValueKind::UnsignedInteger64:
                return "a non-negative integer in the uint64 range";
            case JsonValueKind::Number:
                return "a number";
        }
        return "a valid value";
    }

    static bool IsNonNegativeInteger(const nlohmann::json& value) {
        if (value.is_number_unsigned()) {
            return true;
        }
        return value.is_number_integer() && value.get<int64_t>() >= 0;
    }

    static bool IsPowerOfTwo(uint32_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    static void AppendConfigPath(std::ostringstream& message,
                                 const char* label,
                                 const std::string& path) {
        if (!path.empty()) {
            message << ": " << label << " '" << path << "'";
        }
    }

    static void AppendComparedConfig(std::ostringstream& message,
                                     const std::string& path) {
        if (!path.empty()) {
            message << ", but memory config '" << path << "'";
        } else {
            message << ", but the memory config";
        }
    }
};

}  // namespace phsim
