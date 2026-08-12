#pragma once

#include "../common_function.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class DramDataContainer {
public:
    using Burst = std::vector<uint8_t>;

    explicit DramDataContainer(const SysConfig& config);

    Burst read_burst(addr_type address) const;
    void write_burst(addr_type address, const Burst& data);
    void write_burst(addr_type address, Burst&& data);
    void clear();

    uint32_t burst_bytes() const { return burst_bytes_; }
    uint64_t physical_capacity_bursts() const { return physical_capacity_bursts_; }
    uint64_t physical_capacity_bytes() const { return physical_capacity_bytes_; }
    size_t resident_bursts() const { return bursts_.size(); }
    uint64_t resident_payload_bytes() const;
    uint64_t peak_resident_payload_bytes() const { return peak_resident_payload_bytes_; }

private:
    struct AddressField {
        AddressField() = default;
        AddressField(uint32_t field_position, addr_type field_mask)
            : position(field_position), mask(field_mask) {}

        uint32_t position = 0;
        addr_type mask = 0;
    };

    uint32_t decode(addr_type address, const AddressField& field) const;
    void validate_address(addr_type address) const;
    void write_normalized(addr_type address, Burst data);

    uint32_t channels_ = 0;
    uint32_t ranks_ = 0;
    uint32_t bankgroups_ = 0;
    uint32_t banks_per_group_ = 0;
    uint32_t rows_ = 0;
    uint32_t bursts_per_row_ = 0;
    uint32_t burst_bytes_ = 0;
    uint64_t physical_capacity_bursts_ = 0;
    uint64_t physical_capacity_bytes_ = 0;
    uint64_t max_resident_payload_bytes_ = 0;
    uint64_t peak_resident_payload_bytes_ = 0;
    AddressField channel_field_;
    AddressField rank_field_;
    AddressField bankgroup_field_;
    AddressField bank_field_;
    AddressField row_field_;
    AddressField column_field_;
    std::unordered_map<addr_type, Burst> bursts_;
};
