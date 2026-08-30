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

    // Apply the common completion semantics used by both DRAM backends. In
    // addition to ordinary READ/WRITE traffic, this maintains opaque PIM
    // input/output bytes across P_HEADER/GWRITE/COMP/READRES. COMP timing is
    // still modeled by the DRAM backend; this class does not invent numerical
    // MAC behavior.
    void apply_response(MemoryAccess* response);

    void clear() noexcept;
    std::size_t stored_column_count() const noexcept {
        return columns_data_.size();
    }
    uint64_t resident_payload_bytes() const noexcept;
    uint64_t peak_resident_payload_bytes() const noexcept {
        return peak_resident_payload_bytes_;
    }
    uint64_t max_resident_payload_bytes() const noexcept {
        return max_resident_payload_bytes_;
    }
    uint64_t dram_payload_bytes() const noexcept {
        return static_cast<uint64_t>(columns_data_.size()) * dq_bytes_;
    }
    uint64_t pim_payload_bytes() const noexcept;
    uint64_t burst_bytes() const noexcept { return burst_bytes_; }
    uint32_t channel_count() const noexcept { return channels_; }
    std::size_t peak_stored_column_count() const noexcept {
        return peak_stored_column_count_;
    }
    uint64_t pim_input_payload_bytes(uint32_t channel) const;
    uint64_t pim_output_payload_bytes(uint32_t channel) const;
    uint64_t peak_pim_input_payload_bytes(uint32_t channel) const;
    uint64_t peak_pim_output_payload_bytes(uint32_t channel) const;

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

    struct PimChannelState {
        std::vector<uint8_t> input;
        std::vector<uint8_t> output;
        uint64_t read_offset = 0;
        uint64_t peak_input_bytes = 0;
        uint64_t peak_output_bytes = 0;
    };

    void validate_location(uint32_t channel, uint32_t rank,
                           uint32_t bankgroup, uint32_t bank, uint32_t row,
                           uint32_t column, uint32_t column_count) const;
    void validate_channel(uint32_t channel) const;
    void ensure_growth_fits(uint64_t additional_bytes) const;
    void update_peak() noexcept;
    std::vector<uint8_t> normalized_burst(
        const std::vector<uint8_t>& data) const;
    void reset_pim_state(uint32_t channel) noexcept;
    void append_pim_input(uint32_t channel,
                          const std::vector<uint8_t>& data);
    void materialize_pim_output(uint32_t channel,
                                const std::vector<uint8_t>& supplied_result);
    std::vector<uint8_t> read_pim_output_burst(uint32_t channel);

    bool enabled_ = false;
    uint32_t channels_ = 0;
    uint32_t ranks_ = 0;
    uint32_t bankgroups_ = 0;
    uint32_t banks_per_group_ = 0;
    uint32_t rows_ = 0;
    uint32_t columns_ = 0;
    uint32_t burst_length_ = 0;
    uint32_t dq_bytes_ = 0;
    uint64_t burst_bytes_ = 0;
    uint64_t pim_input_capacity_bytes_ = 0;
    uint64_t pim_output_capacity_bytes_ = 0;
    uint64_t max_resident_payload_bytes_ = 0;
    uint64_t peak_resident_payload_bytes_ = 0;
    std::size_t peak_stored_column_count_ = 0;

    std::unordered_map<ColumnAddress, ColumnData, ColumnAddressHash>
        columns_data_;
    std::vector<PimChannelState> pim_channels_;
};
