#include "DramDataContainer.h"

#include "../common_function.hpp"

#include <stdexcept>
#include <utility>
#include <algorithm>
#include <limits>

namespace {

void hash_combine(std::size_t& seed, uint32_t value) noexcept {
    seed ^= std::hash<uint32_t>{}(value) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
}

}  // namespace

DramDataContainer::DramDataContainer(const SysConfig& config)
    : enabled_(config.dram_data_container_enable),
      channels_(config.dram_channels),
      ranks_(config.mem_config.ranks),
      bankgroups_(config.mem_config.bankgroups),
      banks_per_group_(config.mem_config.banks_per_group),
      rows_(config.mem_config.rows),
      columns_(config.mem_config.columns),
      burst_length_(config.mem_config.BL) {
    if (config.mem_config.bus_width == 0 ||
        config.mem_config.bus_width % 8 != 0) {
        throw std::invalid_argument(
            "DRAM DataContainer bus width must be a non-zero byte multiple");
    }
    dq_bytes_ = config.mem_config.bus_width / 8;
    burst_bytes_ = static_cast<uint64_t>(burst_length_) * dq_bytes_;

    if (config.mem_config.input_buffer_size < 0 ||
        config.mem_config.output_buffer_size < 0 ||
        config.mem_config.PU_num < 0) {
        throw std::invalid_argument(
            "DRAM DataContainer PIM buffer geometry must be non-negative");
    }
    pim_input_capacity_bytes_ =
        static_cast<uint64_t>(config.mem_config.input_buffer_size);
    if (static_cast<uint64_t>(config.mem_config.output_buffer_size) >
            std::numeric_limits<uint64_t>::max() /
                static_cast<uint64_t>(std::max(config.mem_config.PU_num, 1))) {
        throw std::overflow_error(
            "DRAM DataContainer PIM output capacity overflows bytes");
    }
    pim_output_capacity_bytes_ =
        static_cast<uint64_t>(config.mem_config.output_buffer_size) *
        static_cast<uint64_t>(config.mem_config.PU_num);
    pim_channels_.resize(channels_);

    if (config.dram_data_container_max_payload_mb >
        std::numeric_limits<uint64_t>::max() / (1024ULL * 1024ULL)) {
        throw std::overflow_error(
            "DRAM DataContainer resident payload limit overflows bytes");
    }
    max_resident_payload_bytes_ =
        config.dram_data_container_max_payload_mb * 1024ULL * 1024ULL;

    if (channels_ == 0 || ranks_ == 0 || bankgroups_ == 0 ||
        banks_per_group_ == 0 || rows_ == 0 || columns_ == 0 ||
        burst_length_ == 0 || burst_length_ > columns_) {
        throw std::invalid_argument("Invalid DRAM DataContainer geometry");
    }
}

void DramDataContainer::clear() noexcept {
    columns_data_.clear();
    for (auto& state : pim_channels_) {
        state.input.clear();
        state.output.clear();
        state.read_offset = 0;
        state.peak_input_bytes = 0;
        state.peak_output_bytes = 0;
    }
    peak_resident_payload_bytes_ = 0;
    peak_stored_column_count_ = 0;
}

uint64_t DramDataContainer::resident_payload_bytes() const noexcept {
    uint64_t bytes =
        static_cast<uint64_t>(columns_data_.size()) * dq_bytes_;
    for (const auto& state : pim_channels_) {
        bytes += state.input.size();
        bytes += state.output.size();
    }
    return bytes;
}

uint64_t DramDataContainer::pim_input_payload_bytes(uint32_t channel) const {
    validate_channel(channel);
    return pim_channels_[channel].input.size();
}

uint64_t DramDataContainer::pim_payload_bytes() const noexcept {
    uint64_t bytes = 0;
    for (const auto& state : pim_channels_) {
        bytes += state.input.size();
        bytes += state.output.size();
    }
    return bytes;
}

uint64_t DramDataContainer::pim_output_payload_bytes(uint32_t channel) const {
    validate_channel(channel);
    return pim_channels_[channel].output.size();
}

uint64_t DramDataContainer::peak_pim_input_payload_bytes(
    uint32_t channel) const {
    validate_channel(channel);
    return pim_channels_[channel].peak_input_bytes;
}

uint64_t DramDataContainer::peak_pim_output_payload_bytes(
    uint32_t channel) const {
    validate_channel(channel);
    return pim_channels_[channel].peak_output_bytes;
}

bool DramDataContainer::ColumnAddress::operator==(
    const ColumnAddress& other) const noexcept {
    return channel == other.channel && rank == other.rank &&
           bankgroup == other.bankgroup && bank == other.bank &&
           row == other.row && column == other.column;
}

std::size_t DramDataContainer::ColumnAddressHash::operator()(
    const ColumnAddress& address) const noexcept {
    std::size_t seed = 0;
    hash_combine(seed, address.channel);
    hash_combine(seed, address.rank);
    hash_combine(seed, address.bankgroup);
    hash_combine(seed, address.bank);
    hash_combine(seed, address.row);
    hash_combine(seed, address.column);
    return seed;
}

void DramDataContainer::validate_location(
    uint32_t channel, uint32_t rank, uint32_t bankgroup, uint32_t bank,
    uint32_t row, uint32_t column, uint32_t column_count) const {
    if (channel >= channels_ || rank >= ranks_ ||
        bankgroup >= bankgroups_ || bank >= banks_per_group_ || row >= rows_) {
        throw std::out_of_range("DRAM DataContainer hierarchy index out of range");
    }
    if (column > columns_ || column_count > columns_ - column) {
        throw std::out_of_range("DRAM DataContainer column range out of bounds");
    }
}

void DramDataContainer::validate_channel(uint32_t channel) const {
    if (channel >= channels_) {
        throw std::out_of_range(
            "DRAM DataContainer PIM channel index out of range");
    }
}

void DramDataContainer::ensure_growth_fits(uint64_t additional_bytes) const {
    const uint64_t resident_bytes = resident_payload_bytes();
    if (max_resident_payload_bytes_ != 0 &&
        (resident_bytes > max_resident_payload_bytes_ ||
         additional_bytes > max_resident_payload_bytes_ - resident_bytes)) {
        throw std::length_error(
            "DRAM DataContainer resident payload limit exceeded");
    }
}

void DramDataContainer::update_peak() noexcept {
    peak_resident_payload_bytes_ =
        std::max(peak_resident_payload_bytes_, resident_payload_bytes());
}

std::vector<uint8_t> DramDataContainer::normalized_burst(
    const std::vector<uint8_t>& data) const {
    std::vector<uint8_t> result(static_cast<std::size_t>(burst_bytes_), 0);
    std::copy_n(data.begin(), std::min<uint64_t>(data.size(), burst_bytes_),
                result.begin());
    return result;
}

void DramDataContainer::reset_pim_state(uint32_t channel) noexcept {
    auto& state = pim_channels_[channel];
    state.input.clear();
    state.output.clear();
    state.read_offset = 0;
}

void DramDataContainer::append_pim_input(
    uint32_t channel, const std::vector<uint8_t>& data) {
    validate_channel(channel);
    if (data.size() > burst_bytes_) {
        throw std::invalid_argument(
            "DRAM DataContainer GWRITE payload exceeds one burst");
    }
    auto& state = pim_channels_[channel];
    if (burst_bytes_ > pim_input_capacity_bytes_ -
                           std::min<uint64_t>(state.input.size(),
                                              pim_input_capacity_bytes_)) {
        throw std::length_error(
            "DRAM DataContainer PIM input buffer capacity exceeded");
    }
    ensure_growth_fits(burst_bytes_);
    const auto burst = normalized_burst(data);
    state.input.insert(state.input.end(), burst.begin(), burst.end());
    state.peak_input_bytes =
        std::max<uint64_t>(state.peak_input_bytes, state.input.size());
    update_peak();
}

void DramDataContainer::materialize_pim_output(
    uint32_t channel, const std::vector<uint8_t>& supplied_result) {
    validate_channel(channel);
    auto& state = pim_channels_[channel];

    std::vector<uint8_t> next_output;
    if (!supplied_result.empty()) {
        if (supplied_result.size() > pim_output_capacity_bytes_) {
            throw std::length_error(
                "DRAM DataContainer PIM output buffer capacity exceeded");
        }
        next_output = supplied_result;
    } else if (state.output.empty()) {
        const uint64_t bytes =
            std::min<uint64_t>(state.input.size(),
                               pim_output_capacity_bytes_);
        next_output.assign(state.input.begin(), state.input.begin() + bytes);
    } else {
        return;
    }

    const uint64_t growth = next_output.size() > state.output.size()
                                ? next_output.size() - state.output.size()
                                : 0;
    ensure_growth_fits(growth);
    state.output = std::move(next_output);
    state.read_offset = 0;
    state.peak_output_bytes =
        std::max<uint64_t>(state.peak_output_bytes, state.output.size());
    update_peak();
}

std::vector<uint8_t> DramDataContainer::read_pim_output_burst(
    uint32_t channel) {
    validate_channel(channel);
    auto& state = pim_channels_[channel];
    std::vector<uint8_t> result(static_cast<std::size_t>(burst_bytes_), 0);
    if (state.read_offset < state.output.size()) {
        const uint64_t available = state.output.size() - state.read_offset;
        const uint64_t count = std::min<uint64_t>(available, burst_bytes_);
        std::copy_n(state.output.begin() + state.read_offset, count,
                    result.begin());
    }
    if (state.read_offset >
        std::numeric_limits<uint64_t>::max() - burst_bytes_) {
        state.read_offset = std::numeric_limits<uint64_t>::max();
    } else {
        state.read_offset += burst_bytes_;
    }
    return result;
}

void DramDataContainer::write_burst(
    BurstData data, uint32_t channel, uint32_t rank, uint32_t bankgroup,
    uint32_t bank, uint32_t row, uint32_t column) {
    if (!enabled_) {
        return;
    }
    if (data.size() > burst_length_) {
        throw std::invalid_argument(
            "DRAM DataContainer write exceeds one burst");
    }
    validate_location(channel, rank, bankgroup, bank, row, column,
                      static_cast<uint32_t>(data.size()));

    uint64_t new_columns = 0;
    for (uint32_t offset = 0; offset < data.size(); ++offset) {
        const ColumnAddress address{channel, rank, bankgroup, bank, row,
                                    column + offset};
        if (columns_data_.find(address) == columns_data_.end()) {
            ++new_columns;
        }
    }
    if (new_columns > std::numeric_limits<uint64_t>::max() / dq_bytes_) {
        throw std::overflow_error(
            "DRAM DataContainer write payload size overflows bytes");
    }
    ensure_growth_fits(new_columns * dq_bytes_);

    for (uint32_t offset = 0; offset < data.size(); ++offset) {
        auto& bytes = data[offset];
        if (bytes.size() > dq_bytes_) {
            bytes.resize(dq_bytes_);
        } else if (bytes.size() < dq_bytes_) {
            bytes.resize(dq_bytes_, 0);
        }
        columns_data_[ColumnAddress{channel, rank, bankgroup, bank, row,
                                    column + offset}] = std::move(bytes);
    }
    peak_stored_column_count_ =
        std::max(peak_stored_column_count_, columns_data_.size());
    update_peak();
}

DramDataContainer::BurstData DramDataContainer::read_burst(
    uint32_t channel, uint32_t rank, uint32_t bankgroup, uint32_t bank,
    uint32_t row, uint32_t column) const {
    validate_location(channel, rank, bankgroup, bank, row, column,
                      burst_length_);

    BurstData result(burst_length_, ColumnData(dq_bytes_, 0));
    if (!enabled_) {
        return result;
    }

    for (uint32_t offset = 0; offset < burst_length_; ++offset) {
        const auto it = columns_data_.find(ColumnAddress{
            channel, rank, bankgroup, bank, row, column + offset});
        if (it != columns_data_.end()) {
            result[offset] = it->second;
        }
    }
    return result;
}

std::vector<uint8_t> DramDataContainer::flatten_burst(
    const BurstData& data) const {
    std::vector<uint8_t> flattened;
    flattened.reserve(data.size() * dq_bytes_);
    for (const auto& column : data) {
        for (uint32_t byte = 0; byte < dq_bytes_; ++byte) {
            flattened.push_back(byte < column.size() ? column[byte] : 0);
        }
    }
    return flattened;
}

void DramDataContainer::apply_response(MemoryAccess* response) {
    if (!enabled_ || response == nullptr || response->request ||
        response->data_ready) {
        return;
    }

    const addr_type address = response->dram_address;
    const uint32_t channel = MyAddressAllocator::get_channel_index(address);
    const uint32_t rank = MyAddressAllocator::get_rank_index(address);
    const uint32_t bankgroup =
        MyAddressAllocator::get_bankgroup_index(address);
    const uint32_t bank = MyAddressAllocator::get_bank_index(address);
    const uint32_t row = MyAddressAllocator::get_row_index(address);
    const uint32_t column = MyAddressAllocator::get_col_index(address);

    switch (response->req_type) {
        case MemoryAccessType::READ:
            response->data = flatten_burst(
                read_burst(channel, rank, bankgroup, bank, row, column));
            break;
        case MemoryAccessType::WRITE:
            if (!response->data.empty()) {
                BurstData burst(burst_length_, ColumnData(dq_bytes_, 0));
                for (uint32_t column_offset = 0;
                     column_offset < burst_length_; ++column_offset) {
                    for (uint32_t byte = 0; byte < dq_bytes_; ++byte) {
                        const uint64_t source =
                            static_cast<uint64_t>(column_offset) * dq_bytes_ +
                            byte;
                        if (source < response->data.size()) {
                            burst[column_offset][byte] =
                                response->data[source];
                        }
                    }
                }
                write_burst(std::move(burst), channel, rank, bankgroup, bank,
                            row, column);
            }
            break;
        case MemoryAccessType::P_HEADER:
            reset_pim_state(channel);
            response->data.clear();
            break;
        case MemoryAccessType::GWRITE:
            append_pim_input(channel, response->data);
            break;
        case MemoryAccessType::COMP:
        case MemoryAccessType::COMP_HASH:
            materialize_pim_output(channel, response->data);
            break;
        case MemoryAccessType::READRES:
            response->data = read_pim_output_burst(channel);
            break;
        case MemoryAccessType::COMPS_READRES:
            materialize_pim_output(channel, response->data);
            response->data = read_pim_output_burst(channel);
            break;
        case MemoryAccessType::SIZE:
            throw std::invalid_argument(
                "DRAM DataContainer cannot apply SIZE request type");
    }

    response->data_ready = true;
}
