#ifndef PHSIM_DRAM_GEOMETRY_HPP
#define PHSIM_DRAM_GEOMETRY_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace phsim {

struct DramGeometryInput {
    uint64_t channel_size_mib;
    uint64_t channels;
    uint64_t bankgroups;
    uint64_t banks_per_group;
    uint64_t rows;
    uint64_t columns;
    uint64_t device_width_bits;
    uint64_t bus_width_bits;
    uint64_t burst_length;
    bool bankgroup_enable;
};

struct DramGeometryResult {
    uint32_t channel_size_mib;
    uint32_t ranks;
    uint32_t banks;
    uint32_t bankgroups;
    uint32_t banks_per_group;
    uint32_t devices_per_rank;
    uint32_t request_size_bytes;
    uint32_t shift_bits;
    uint32_t channel_bits;
    uint32_t rank_bits;
    uint32_t bankgroup_bits;
    uint32_t bank_bits;
    uint32_t row_bits;
    uint32_t column_bits;
    uint32_t address_bits;
    uint64_t rank_size_mib;
    bool channel_size_increased;
};

struct DramAddressLayout {
    uint32_t channel_width;
    uint32_t rank_width;
    uint32_t bankgroup_width;
    uint32_t bank_width;
    uint32_t row_width;
    uint32_t column_width;

    int channel_pos;
    int rank_pos;
    int bankgroup_pos;
    int bank_pos;
    int row_pos;
    int column_pos;

    uint64_t channel_mask;
    uint64_t rank_mask;
    uint64_t bankgroup_mask;
    uint64_t bank_mask;
    uint64_t row_mask;
    uint64_t column_mask;

    // Inner, middle, and outer iteration order for rank/bankgroup/bank.
    std::array<std::string, 3> row_loop_order;
};

inline bool IsPositivePowerOfTwo(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

inline uint32_t IntegerLog2(uint64_t power_of_two) {
    uint32_t result = 0;
    while (power_of_two > 1) {
        power_of_two >>= 1;
        ++result;
    }
    return result;
}

inline uint64_t MaskForWidth(uint32_t width) {
    if (width == 0) {
        return 0;
    }
    if (width >= 64) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t{1} << width) - 1;
}

inline uint64_t CheckedGeometryMultiply(uint64_t left, uint64_t right,
                                        const std::string& prefix,
                                        const std::string& expression) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
        throw std::overflow_error(prefix + expression + " overflows uint64_t");
    }
    return left * right;
}

inline DramGeometryResult CalculateDramGeometry(
    const DramGeometryInput& input, const std::string& config_path) {
    const std::string geometry_prefix =
        "Invalid DRAM geometry in '" + config_path + "': ";
    const std::string capacity_prefix =
        "Invalid DRAM capacity in '" + config_path + "': ";
    const std::string address_prefix =
        "Invalid DRAM address layout in '" + config_path + "': ";
    const uint64_t max_config_integer =
        static_cast<uint64_t>(std::numeric_limits<int>::max());

    const auto require_power_of_two = [&](uint64_t value,
                                          const std::string& field) {
        if (!IsPositivePowerOfTwo(value) || value > max_config_integer) {
            throw std::invalid_argument(
                geometry_prefix + field + " must be a positive power of two");
        }
    };

    if (input.channel_size_mib == 0 ||
        input.channel_size_mib > max_config_integer) {
        throw std::invalid_argument(
            geometry_prefix +
            "channel_size must be a positive representable integer");
    }
    require_power_of_two(input.channels, "channels");
    require_power_of_two(input.bankgroups, "bankgroups");
    require_power_of_two(input.banks_per_group, "banks_per_group");
    require_power_of_two(input.rows, "rows");
    require_power_of_two(input.columns, "columns");
    require_power_of_two(input.burst_length, "effective BL");

    if (input.device_width_bits == 0 ||
        input.device_width_bits > max_config_integer) {
        throw std::invalid_argument(
            geometry_prefix + "device_width must be greater than zero");
    }
    if (input.bus_width_bits == 0 || input.bus_width_bits % 8 != 0 ||
        input.bus_width_bits > max_config_integer) {
        throw std::invalid_argument(
            geometry_prefix +
            "bus_width must be a positive multiple of 8 bits");
    }
    if (input.bus_width_bits % input.device_width_bits != 0) {
        throw std::invalid_argument(
            geometry_prefix +
            "bus_width must be divisible by device_width");
    }
    if (input.columns < input.burst_length ||
        input.columns % input.burst_length != 0) {
        throw std::invalid_argument(
            geometry_prefix +
            "columns must be divisible by and no smaller than BL");
    }

    const uint64_t bank_count = CheckedGeometryMultiply(
        input.bankgroups, input.banks_per_group, geometry_prefix,
        "bankgroups * banks_per_group");
    if (bank_count > max_config_integer) {
        throw std::overflow_error(
            geometry_prefix +
            "bankgroups * banks_per_group does not fit in int");
    }
    const uint64_t effective_bankgroups =
        input.bankgroup_enable ? input.bankgroups : 1;
    const uint64_t effective_banks_per_group =
        input.bankgroup_enable ? input.banks_per_group : bank_count;

    const uint64_t page_bits = CheckedGeometryMultiply(
        input.columns, input.device_width_bits, capacity_prefix,
        "columns * device_width");
    if (page_bits % 8 != 0) {
        throw std::invalid_argument(
            geometry_prefix +
            "columns * device_width must represent whole bytes");
    }
    const uint64_t page_bytes = page_bits / 8;
    const uint64_t bank_bytes = CheckedGeometryMultiply(
        page_bytes, input.rows, capacity_prefix, "page_bytes * rows");
    const uint64_t rank_device_bytes = CheckedGeometryMultiply(
        bank_bytes, bank_count, capacity_prefix, "bank_bytes * banks");
    const uint64_t devices_per_rank =
        input.bus_width_bits / input.device_width_bits;
    const uint64_t rank_bytes = CheckedGeometryMultiply(
        rank_device_bytes, devices_per_rank, capacity_prefix,
        "bank_bytes * banks * devices_per_rank");

    constexpr uint64_t bytes_per_mib = 1024ULL * 1024ULL;
    if (rank_bytes == 0 || rank_bytes % bytes_per_mib != 0) {
        throw std::invalid_argument(
            capacity_prefix +
            "one rank must have a positive whole-MiB capacity");
    }
    const uint64_t rank_size_mib = rank_bytes / bytes_per_mib;
    if (rank_size_mib > max_config_integer) {
        throw std::overflow_error(
            capacity_prefix +
            "one-rank capacity does not fit in the configuration integer type");
    }

    const bool channel_size_increased =
        rank_size_mib > input.channel_size_mib;
    const uint64_t ranks = channel_size_increased
                               ? 1
                               : input.channel_size_mib / rank_size_mib;
    const uint64_t channel_size_mib = ranks * rank_size_mib;
    if (!IsPositivePowerOfTwo(ranks)) {
        throw std::invalid_argument(
            address_prefix +
            "derived ranks must be a positive power of two");
    }

    const uint64_t request_size_bytes = CheckedGeometryMultiply(
        input.bus_width_bits / 8, input.burst_length, address_prefix,
        "bus_width / 8 * BL");
    if (!IsPositivePowerOfTwo(request_size_bytes) ||
        request_size_bytes > max_config_integer) {
        throw std::invalid_argument(
            address_prefix +
            "bus_width / 8 * BL must be a positive power of two");
    }
    const uint64_t bursts_per_row = input.columns / input.burst_length;
    if (!IsPositivePowerOfTwo(bursts_per_row)) {
        throw std::invalid_argument(
            address_prefix +
            "columns / BL must be a positive power of two");
    }

    DramGeometryResult result = {};
    result.channel_size_mib = static_cast<uint32_t>(channel_size_mib);
    result.ranks = static_cast<uint32_t>(ranks);
    result.banks = static_cast<uint32_t>(bank_count);
    result.bankgroups = static_cast<uint32_t>(effective_bankgroups);
    result.banks_per_group =
        static_cast<uint32_t>(effective_banks_per_group);
    result.devices_per_rank = static_cast<uint32_t>(devices_per_rank);
    result.request_size_bytes = static_cast<uint32_t>(request_size_bytes);
    result.shift_bits = IntegerLog2(request_size_bytes);
    result.channel_bits = IntegerLog2(input.channels);
    result.rank_bits = IntegerLog2(ranks);
    result.bankgroup_bits = IntegerLog2(effective_bankgroups);
    result.bank_bits = IntegerLog2(effective_banks_per_group);
    result.row_bits = IntegerLog2(input.rows);
    result.column_bits = IntegerLog2(bursts_per_row);
    const uint64_t address_bits =
        static_cast<uint64_t>(result.shift_bits) + result.channel_bits +
        result.rank_bits + result.bankgroup_bits + result.bank_bits +
        result.row_bits + result.column_bits;
    if (address_bits > 64) {
        throw std::invalid_argument(
            address_prefix + "combined byte-address layout exceeds 64 bits");
    }
    result.address_bits = static_cast<uint32_t>(address_bits);
    result.rank_size_mib = rank_size_mib;
    result.channel_size_increased = channel_size_increased;
    return result;
}

inline DramAddressLayout CalculateDramAddressLayout(
    const DramGeometryResult& geometry, const std::string& address_mapping,
    const std::string& config_path) {
    const std::string prefix =
        "Invalid DRAM address mapping in '" + config_path + "': ";
    if (address_mapping.size() != 12) {
        throw std::invalid_argument(
            prefix + "exactly 6 two-character fields are required");
    }

    DramAddressLayout layout = {};
    layout.channel_width = geometry.channel_bits;
    layout.rank_width = geometry.rank_bits;
    layout.bankgroup_width = geometry.bankgroup_bits;
    layout.bank_width = geometry.bank_bits;
    layout.row_width = geometry.row_bits;
    layout.column_width = geometry.column_bits;

    std::array<std::string, 6> tokens;
    std::array<bool, 6> seen = {};
    const auto field_index = [&](const std::string& token) -> std::size_t {
        if (token == "ch") return 0;
        if (token == "ra") return 1;
        if (token == "bg") return 2;
        if (token == "ba") return 3;
        if (token == "ro") return 4;
        if (token == "co") return 5;
        throw std::invalid_argument(prefix + "unknown address field '" +
                                    token + "'");
    };
    const auto field_width = [&](std::size_t index) -> uint32_t {
        switch (index) {
            case 0: return layout.channel_width;
            case 1: return layout.rank_width;
            case 2: return layout.bankgroup_width;
            case 3: return layout.bank_width;
            case 4: return layout.row_width;
            case 5: return layout.column_width;
            default: throw std::logic_error("Invalid DRAM address field index");
        }
    };
    const auto set_position = [&](std::size_t index, int position) {
        switch (index) {
            case 0: layout.channel_pos = position; break;
            case 1: layout.rank_pos = position; break;
            case 2: layout.bankgroup_pos = position; break;
            case 3: layout.bank_pos = position; break;
            case 4: layout.row_pos = position; break;
            case 5: layout.column_pos = position; break;
            default: throw std::logic_error("Invalid DRAM address field index");
        }
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = address_mapping.substr(index * 2, 2);
        const std::size_t index_value = field_index(tokens[index]);
        if (seen[index_value]) {
            throw std::invalid_argument(prefix + "duplicate address field '" +
                                        tokens[index] + "'");
        }
        seen[index_value] = true;
    }

    int position = 0;
    std::size_t row_loop_index = 0;
    for (auto token = tokens.rbegin(); token != tokens.rend(); ++token) {
        const std::size_t index = field_index(*token);
        set_position(index, position);
        position += static_cast<int>(field_width(index));
        if (*token == "ra" || *token == "bg" || *token == "ba") {
            layout.row_loop_order.at(row_loop_index++) = *token;
        }
    }
    if (static_cast<uint32_t>(position) + geometry.shift_bits !=
        geometry.address_bits) {
        throw std::logic_error(
            prefix + "field widths do not match the calculated geometry");
    }

    layout.channel_mask = MaskForWidth(layout.channel_width);
    layout.rank_mask = MaskForWidth(layout.rank_width);
    layout.bankgroup_mask = MaskForWidth(layout.bankgroup_width);
    layout.bank_mask = MaskForWidth(layout.bank_width);
    layout.row_mask = MaskForWidth(layout.row_width);
    layout.column_mask = MaskForWidth(layout.column_width);
    return layout;
}

}  // namespace phsim

#endif  // PHSIM_DRAM_GEOMETRY_HPP
