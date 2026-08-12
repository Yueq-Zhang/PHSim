#include "DataContainer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

uint64_t checked_multiply(uint64_t lhs, uint64_t rhs, const char* description) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(std::string("DataContainer capacity overflow: ") + description);
    }
    return lhs * rhs;
}

bool is_all_zero(const DramDataContainer::Burst& data) {
    return std::all_of(data.begin(), data.end(), [](uint8_t value) { return value == 0; });
}

}  // namespace

DramDataContainer::DramDataContainer(const SysConfig& config)
    : channels_(config.dram_channels),
      ranks_(config.mem_config.ranks),
      bankgroups_(config.mem_config.bankgroups),
      banks_per_group_(config.mem_config.banks_per_group),
      rows_(config.mem_config.rows),
      channel_field_{static_cast<uint32_t>(config.mem_config.ch_pos),
                     static_cast<addr_type>(config.mem_config.ch_mask)},
      rank_field_{static_cast<uint32_t>(config.mem_config.ra_pos),
                  static_cast<addr_type>(config.mem_config.ra_mask)},
      bankgroup_field_{static_cast<uint32_t>(config.mem_config.bg_pos),
                       static_cast<addr_type>(config.mem_config.bg_mask)},
      bank_field_{static_cast<uint32_t>(config.mem_config.ba_pos),
                  static_cast<addr_type>(config.mem_config.ba_mask)},
      row_field_{static_cast<uint32_t>(config.mem_config.ro_pos),
                 static_cast<addr_type>(config.mem_config.ro_mask)},
      column_field_{static_cast<uint32_t>(config.mem_config.co_pos),
                    static_cast<addr_type>(config.mem_config.co_mask)} {
    if (channels_ == 0 || ranks_ == 0 || bankgroups_ == 0 ||
        banks_per_group_ == 0 || rows_ == 0) {
        throw std::invalid_argument("DataContainer requires non-zero DRAM geometry");
    }
    if (config.mem_config.BL == 0 || config.mem_config.columns == 0 ||
        config.mem_config.columns % config.mem_config.BL != 0) {
        throw std::invalid_argument(
            "DataContainer requires columns to be a non-zero multiple of BL");
    }
    if (config.mem_config.bus_width == 0 || config.mem_config.bus_width % 8 != 0) {
        throw std::invalid_argument(
            "DataContainer requires bus_width to be a non-zero multiple of 8");
    }

    bursts_per_row_ = config.mem_config.columns / config.mem_config.BL;
    const uint64_t burst_bytes = checked_multiply(
        config.mem_config.BL, config.mem_config.bus_width / 8, "burst bytes");
    if (burst_bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("DataContainer burst size exceeds uint32_t");
    }
    burst_bytes_ = static_cast<uint32_t>(burst_bytes);

    uint64_t capacity = channels_;
    capacity = checked_multiply(capacity, ranks_, "ranks");
    capacity = checked_multiply(capacity, bankgroups_, "bank groups");
    capacity = checked_multiply(capacity, banks_per_group_, "banks");
    capacity = checked_multiply(capacity, rows_, "rows");
    capacity = checked_multiply(capacity, bursts_per_row_, "bursts per row");
    physical_capacity_bursts_ = capacity;
    physical_capacity_bytes_ = checked_multiply(capacity, burst_bytes_, "capacity bytes");

    max_resident_payload_bytes_ = checked_multiply(
        config.dram_data_container_max_payload_mb, 1024ULL * 1024ULL,
        "resident payload limit");
}

DramDataContainer::Burst DramDataContainer::read_burst(addr_type address) const {
    validate_address(address);
    const auto it = bursts_.find(address);
    if (it == bursts_.end()) {
        return Burst(burst_bytes_, 0);
    }
    return it->second;
}

void DramDataContainer::write_burst(addr_type address, const Burst& data) {
    write_normalized(address, data);
}

void DramDataContainer::write_burst(addr_type address, Burst&& data) {
    write_normalized(address, std::move(data));
}

void DramDataContainer::clear() {
    bursts_.clear();
    peak_resident_payload_bytes_ = 0;
}

uint64_t DramDataContainer::resident_payload_bytes() const {
    return static_cast<uint64_t>(bursts_.size()) * burst_bytes_;
}

uint32_t DramDataContainer::decode(addr_type address,
                                   const AddressField& field) const {
    return static_cast<uint32_t>((address >> field.position) & field.mask);
}

void DramDataContainer::validate_address(addr_type address) const {
    const uint32_t channel = decode(address, channel_field_);
    const uint32_t rank = decode(address, rank_field_);
    const uint32_t bankgroup = decode(address, bankgroup_field_);
    const uint32_t bank = decode(address, bank_field_);
    const uint32_t row = decode(address, row_field_);
    const uint32_t burst_column = decode(address, column_field_);

    assert(channel < channels_);
    assert(rank < ranks_);
    assert(bankgroup < bankgroups_);
    assert(bank < banks_per_group_);
    assert(row < rows_);
    assert(burst_column < bursts_per_row_);

    addr_type canonical = 0;
    canonical |= (static_cast<addr_type>(channel) & channel_field_.mask)
                 << channel_field_.position;
    canonical |= (static_cast<addr_type>(rank) & rank_field_.mask)
                 << rank_field_.position;
    canonical |= (static_cast<addr_type>(bankgroup) & bankgroup_field_.mask)
                 << bankgroup_field_.position;
    canonical |= (static_cast<addr_type>(bank) & bank_field_.mask)
                 << bank_field_.position;
    canonical |= (static_cast<addr_type>(row) & row_field_.mask)
                 << row_field_.position;
    canonical |= (static_cast<addr_type>(burst_column) & column_field_.mask)
                 << column_field_.position;
    if (canonical != address) {
        throw std::out_of_range("DataContainer address is outside the configured DRAM layout");
    }
}

void DramDataContainer::write_normalized(addr_type address, Burst data) {
    validate_address(address);
    if (data.size() > burst_bytes_) {
        throw std::invalid_argument("DataContainer write exceeds one DRAM burst");
    }
    data.resize(burst_bytes_, 0);

    if (is_all_zero(data)) {
        bursts_.erase(address);
        return;
    }

    const bool is_new_burst = bursts_.find(address) == bursts_.end();
    if (is_new_burst && max_resident_payload_bytes_ != 0 &&
        resident_payload_bytes() + burst_bytes_ > max_resident_payload_bytes_) {
        throw std::length_error("DataContainer resident payload limit exceeded");
    }
    auto it = bursts_.find(address);
    if (it == bursts_.end()) {
        bursts_.emplace(address, std::move(data));
    } else {
        it->second = std::move(data);
    }
    peak_resident_payload_bytes_ =
        std::max(peak_resident_payload_bytes_, resident_payload_bytes());
}
