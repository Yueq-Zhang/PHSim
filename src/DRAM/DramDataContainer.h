#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class SysConfig;
struct MemoryAccess;

// Sparse, instance-owned storage for the data values maintained by the DRAM
// simulator. Address values passed to this class have already been decoded
// into DRAM hierarchy fields; the class itself does not perform VM mapping.
class DramDataContainer {
public:
    using ColumnData = std::vector<uint8_t>;
    using BurstData = std::vector<ColumnData>;

    explicit DramDataContainer(const SysConfig& config);

    bool enabled() const noexcept { return enabled_; }
    uint32_t burst_length() const noexcept { return burst_length_; }
    uint32_t dq_bytes() const noexcept { return dq_bytes_; }
    uint32_t columns() const noexcept { return columns_; }

    void write_burst(BurstData data, uint32_t channel, uint32_t rank,
                     uint32_t bankgroup, uint32_t bank, uint32_t row,
                     uint32_t column);
    BurstData read_burst(uint32_t channel, uint32_t rank,
                         uint32_t bankgroup, uint32_t bank, uint32_t row,
                         uint32_t column) const;

    std::vector<uint8_t> flatten_burst(const BurstData& data) const;

    // Apply the common READ/WRITE completion semantics used by both DRAM
    // backends. The response is left untouched when the container is disabled.
    void apply_response(MemoryAccess* response);

    void clear() noexcept {
        columns_data_.clear();
        peak_resident_payload_bytes_ = 0;
    }
    std::size_t stored_column_count() const noexcept {
        return columns_data_.size();
    }
    uint64_t resident_payload_bytes() const noexcept {
        return static_cast<uint64_t>(columns_data_.size()) * dq_bytes_;
    }
    uint64_t peak_resident_payload_bytes() const noexcept {
        return peak_resident_payload_bytes_;
    }

private:
    struct ColumnAddress {
        uint32_t channel;
        uint32_t rank;
        uint32_t bankgroup;
        uint32_t bank;
        uint32_t row;
        uint32_t column;

        bool operator==(const ColumnAddress& other) const noexcept;
    };

    struct ColumnAddressHash {
        std::size_t operator()(const ColumnAddress& address) const noexcept;
    };

    void validate_location(uint32_t channel, uint32_t rank,
                           uint32_t bankgroup, uint32_t bank, uint32_t row,
                           uint32_t column, uint32_t column_count) const;

    bool enabled_ = false;
    uint32_t channels_ = 0;
    uint32_t ranks_ = 0;
    uint32_t bankgroups_ = 0;
    uint32_t banks_per_group_ = 0;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t burst_length_ = 0;
    uint32_t dq_bytes_ = 0;
    uint64_t max_resident_payload_bytes_ = 0;
    uint64_t peak_resident_payload_bytes_ = 0;

    std::unordered_map<ColumnAddress, ColumnData, ColumnAddressHash>
        columns_data_;
};
