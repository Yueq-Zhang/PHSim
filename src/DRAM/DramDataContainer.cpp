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
    const uint64_t resident_bytes = resident_payload_bytes();
    if (max_resident_payload_bytes_ != 0 &&
        (resident_bytes > max_resident_payload_bytes_ ||
         new_columns >
             (max_resident_payload_bytes_ - resident_bytes) / dq_bytes_)) {
        throw std::length_error(
            "DRAM DataContainer resident payload limit exceeded");
    }

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
    peak_resident_payload_bytes_ =
        std::max(peak_resident_payload_bytes_, resident_payload_bytes());
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

    if (response->req_type == MemoryAccessType::READ) {
        response->data = flatten_burst(
            read_burst(channel, rank, bankgroup, bank, row, column));
    } else if (response->req_type == MemoryAccessType::WRITE &&
               !response->data.empty()) {
        BurstData burst(burst_length_, ColumnData(dq_bytes_, 0));
        for (uint32_t column_offset = 0; column_offset < burst_length_;
             ++column_offset) {
            for (uint32_t byte = 0; byte < dq_bytes_; ++byte) {
                const uint64_t source =
                    static_cast<uint64_t>(column_offset) * dq_bytes_ + byte;
                if (source < response->data.size()) {
                    burst[column_offset][byte] = response->data[source];
                }
            }
        }
        write_burst(std::move(burst), channel, rank, bankgroup, bank, row,
                    column);
    }

    response->data_ready = true;
}
