#include "common_function.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef PH_SIM_SOURCE_DIR
#define PH_SIM_SOURCE_DIR "."
#endif

namespace {

using RuntimeTuple = std::array<uint32_t, 6>;
using BankTuple = std::array<uint32_t, 4>;

std::string source_path(const std::string& relative) {
    return std::string(PH_SIM_SOURCE_DIR) + "/" + relative;
}

void configure_address_decoder(const MemConfig& mem) {
    MyAddressAllocator::dram_channels = mem.channels;
    MyAddressAllocator::ranks = mem.ranks;
    MyAddressAllocator::bankgroups = mem.bankgroups;
    MyAddressAllocator::banks = mem.banks_per_group;
    MyAddressAllocator::rows = mem.rows;
    MyAddressAllocator::columns = mem.columns;
    MyAddressAllocator::burst_length = mem.BL;
    MyAddressAllocator::channel_width = mem.bus_width;
    MyAddressAllocator::dram_burst_size = mem.BL * mem.bus_width / 8;

    MyAddressAllocator::field_pos.clear();
    MyAddressAllocator::mask.clear();
    MyAddressAllocator::field_pos["ch"] = mem.ch_pos;
    MyAddressAllocator::field_pos["ra"] = mem.ra_pos;
    MyAddressAllocator::field_pos["bg"] = mem.bg_pos;
    MyAddressAllocator::field_pos["ba"] = mem.ba_pos;
    MyAddressAllocator::field_pos["ro"] = mem.ro_pos;
    MyAddressAllocator::field_pos["co"] = mem.co_pos;
    MyAddressAllocator::mask["ch"] = static_cast<int>(mem.ch_mask);
    MyAddressAllocator::mask["ra"] = static_cast<int>(mem.ra_mask);
    MyAddressAllocator::mask["bg"] = static_cast<int>(mem.bg_mask);
    MyAddressAllocator::mask["ba"] = static_cast<int>(mem.ba_mask);
    MyAddressAllocator::mask["ro"] = static_cast<int>(mem.ro_mask);
    MyAddressAllocator::mask["co"] = static_cast<int>(mem.co_mask);
}

RuntimeTuple runtime_tuple(addr_type address) {
    return {
        MyAddressAllocator::get_channel_index(address),
        MyAddressAllocator::get_rank_index(address),
        MyAddressAllocator::get_bankgroup_index(address),
        MyAddressAllocator::get_bank_index(address),
        MyAddressAllocator::get_row_index(address),
        MyAddressAllocator::get_col_index(address),
    };
}

BankTuple runtime_bank_tuple(addr_type address) {
    const RuntimeTuple decoded = runtime_tuple(address);
    return {decoded[0], decoded[1], decoded[2], decoded[3]};
}

BankTuple mapper_bank_tuple(const TwoLevelDeterministicMapper& mapper,
                            addr_type address) {
    const auto decoded = mapper.extract_bank_info(address);
    return {decoded.channel, decoded.rank, decoded.bankgroup, decoded.bank};
}

void print_check(const std::string& name, bool pass) {
    std::cout << "CHECK " << name << ' ' << (pass ? "PASS" : "FAIL")
              << '\n';
}

struct ConfigCase {
    const char* name;
    const char* memory_config;
    const char* pim_config;
};

bool verify_config_layout(const ConfigCase& config_case) {
    const MemConfig mem(source_path(config_case.memory_config),
                        source_path(config_case.pim_config), "/tmp");
    configure_address_decoder(mem);

    const uint64_t total_bytes =
        static_cast<uint64_t>(mem.channels) * mem.channel_size * 1024 * 1024;
    const uint64_t burst_bytes =
        static_cast<uint64_t>(mem.BL) * mem.bus_width / 8;
    const uint64_t physical_pages =
        total_bytes / TwoLevelDeterministicMapper::PAGE_SIZE_BYTES;
    TwoLevelDeterministicMapper mapper(physical_pages);
    mapper.configure(mem);

    bool healthy =
        mapper.request_size_bytes() == burst_bytes &&
        mapper.page_size_address_units() ==
            TwoLevelDeterministicMapper::PAGE_SIZE_BYTES / burst_bytes &&
        mapper.hash_unit_address_units() ==
            TwoLevelDeterministicMapper::HASH_UNIT_BYTES / burst_bytes &&
        mapper.physical_capacity_address_units() == total_bytes / burst_bytes &&
        mapper.num_physical_pages() == physical_pages;

    uint64_t decoder_disagreements = 0;
    uint64_t bank_changes = 0;
    uint64_t channel_changes = 0;
    uint64_t row_changes = 0;
    constexpr uint64_t kChunksToProbe = 256;
    for (uint64_t chunk = 0; chunk < kChunksToProbe; ++chunk) {
        const addr_type logical = chunk * mapper.hash_unit_address_units();
        const addr_type physical = mapper.map(logical);
        decoder_disagreements +=
            mapper_bank_tuple(mapper, physical) != runtime_bank_tuple(physical);
        bank_changes +=
            runtime_bank_tuple(physical) != runtime_bank_tuple(logical);
        channel_changes +=
            MyAddressAllocator::get_channel_index(physical) !=
            MyAddressAllocator::get_channel_index(logical);
        row_changes +=
            MyAddressAllocator::get_row_index(physical) !=
            MyAddressAllocator::get_row_index(logical);
    }

    std::map<RuntimeTuple, uint64_t> first_page_for_tuple;
    uint64_t page_aliases = 0;
    constexpr uint64_t kPagesToProbe = 17;
    for (uint64_t logical_page = 0; logical_page < kPagesToProbe;
         ++logical_page) {
        const addr_type logical =
            logical_page * mapper.page_size_address_units();
        const RuntimeTuple decoded = runtime_tuple(mapper.map(logical));
        page_aliases +=
            !first_page_for_tuple.emplace(decoded, logical_page).second;
    }

    healthy = healthy && decoder_disagreements == 0 && bank_changes > 0 &&
              channel_changes == 0 && row_changes == 0 && page_aliases == 0;
    std::cout << "MATRIX " << config_case.name << ' '
              << (healthy ? "PASS" : "FAIL")
              << " burst_bytes=" << burst_bytes
              << " page_units=" << mapper.page_size_address_units()
              << " hash_units=" << mapper.hash_unit_address_units()
              << " pages=" << physical_pages
              << " decoder_disagreements=" << decoder_disagreements
              << " bank_changes=" << bank_changes
              << " channel_changes=" << channel_changes
              << " row_changes=" << row_changes
              << " page_aliases=" << page_aliases << '\n';
    return healthy;
}

}  // namespace

int main() {
    const std::vector<ConfigCase> config_cases = {
        {"Nano", "configs/Cases/Nano/Nano_2xLPDDR5.ini",
         "configs/Cases/Nano/Nano_pim_config.json"},
        {"Mobile", "configs/Cases/Mobile/Mobile_2xLPDDR5.ini",
         "configs/Cases/Mobile/Mobile_pim_config.json"},
        {"Server", "configs/Cases/Server/Server_8xHBM2.ini",
         "configs/Cases/Server/Server_pim_config.json"},
        {"Base", "configs/Cases/Base/Base_8xGDDR6.ini",
         "configs/Cases/Base/Base_pim_config.json"},
    };
    bool config_matrix_healthy = true;
    for (const auto& config_case : config_cases) {
        config_matrix_healthy =
            verify_config_layout(config_case) && config_matrix_healthy;
    }

    const std::string memory_config =
        source_path("configs/Cases/Nano/Nano_2xLPDDR5.ini");
    const std::string pim_config =
        source_path("configs/Cases/Nano/Nano_pim_config.json");
    const MemConfig mem(memory_config, pim_config, "/tmp");
    configure_address_decoder(mem);

    const uint64_t total_bytes =
        static_cast<uint64_t>(mem.channels) * mem.channel_size * 1024 * 1024;
    const uint64_t physical_pages =
        total_bytes / TwoLevelDeterministicMapper::PAGE_SIZE_BYTES;
    const uint64_t dram_burst_bytes =
        static_cast<uint64_t>(mem.BL) * mem.bus_width / 8;
    const uint64_t runtime_address_capacity_units =
        total_bytes / dram_burst_bytes;
    const uint64_t expected_page_size_address_units =
        TwoLevelDeterministicMapper::PAGE_SIZE_BYTES / dram_burst_bytes;
    const uint64_t expected_hash_unit_address_units =
        TwoLevelDeterministicMapper::HASH_UNIT_BYTES / dram_burst_bytes;

    const addr_type probe_address = expected_hash_unit_address_units;
    const bool disabled_pass_through =
        TwoLevelPageMapper::map_logical_address(probe_address) == probe_address;

    Config::system_config.mem_config = mem;
    Config::system_config.dram_channels = mem.channels;
    TwoLevelPageMapper::init_two_level_mapper();
    const addr_type wrapper_first =
        TwoLevelPageMapper::map_logical_address(probe_address);
    const addr_type wrapper_second =
        TwoLevelPageMapper::map_logical_address(probe_address);
    const bool wrapper_deterministic = wrapper_first == wrapper_second;

    std::ostringstream page_table;
    TwoLevelPageMapper::dump_page_table(page_table);
    const bool wrapper_page_allocated =
        page_table.str().find("LPage 0  ->  PPage 0") != std::string::npos;

    TwoLevelDeterministicMapper mapper(physical_pages);
    mapper.configure(mem);
    const uint64_t runtime_pages_with_configured_page_units =
        mapper.physical_capacity_address_units() /
        mapper.page_size_address_units();
    const bool address_units_match_config =
        mapper.request_size_bytes() == dram_burst_bytes &&
        mapper.page_size_address_units() == expected_page_size_address_units &&
        mapper.hash_unit_address_units() == expected_hash_unit_address_units &&
        mapper.num_physical_pages() == physical_pages &&
        runtime_pages_with_configured_page_units == physical_pages;

    uint64_t numeric_changes = 0;
    uint64_t runtime_bank_changes = 0;
    uint64_t runtime_channel_changes = 0;
    uint64_t runtime_row_changes = 0;
    uint64_t decoder_disagreements = 0;
    uint64_t non_idempotent_remaps = 0;
    const uint64_t kHashChunksPerPage =
        TwoLevelDeterministicMapper::PAGE_SIZE_BYTES /
        TwoLevelDeterministicMapper::HASH_UNIT_BYTES;
    for (uint64_t chunk = 0; chunk < kHashChunksPerPage; ++chunk) {
        const addr_type logical = chunk * mapper.hash_unit_address_units();
        const addr_type physical = mapper.map(logical);
        const addr_type remapped_physical = mapper.map(physical);
        numeric_changes += physical != logical;
        runtime_bank_changes +=
            runtime_bank_tuple(physical) != runtime_bank_tuple(logical);
        runtime_channel_changes +=
            MyAddressAllocator::get_channel_index(physical) !=
            MyAddressAllocator::get_channel_index(logical);
        runtime_row_changes +=
            MyAddressAllocator::get_row_index(physical) !=
            MyAddressAllocator::get_row_index(logical);
        decoder_disagreements +=
            mapper_bank_tuple(mapper, physical) != runtime_bank_tuple(physical);
        non_idempotent_remaps += remapped_physical != physical;
    }

    std::map<RuntimeTuple, uint64_t> first_page_for_tuple;
    uint64_t decoded_page_aliases = 0;
    uint64_t first_alias_page = 0;
    uint64_t first_alias_with = 0;
    constexpr uint64_t kPagesToProbe = 17;
    for (uint64_t logical_page = 0; logical_page < kPagesToProbe;
         ++logical_page) {
        const addr_type logical =
            logical_page * mapper.page_size_address_units();
        const addr_type physical = mapper.map(logical);
        const RuntimeTuple decoded = runtime_tuple(physical);
        const auto [it, inserted] =
            first_page_for_tuple.emplace(decoded, logical_page);
        if (!inserted) {
            if (decoded_page_aliases == 0) {
                first_alias_page = logical_page;
                first_alias_with = it->second;
            }
            ++decoded_page_aliases;
        }
    }

    const bool decoders_agree = decoder_disagreements == 0;
    const bool hash_changes_runtime_bank = runtime_bank_changes > 0;
    const bool channel_is_preserved = runtime_channel_changes == 0;
    const bool row_is_preserved_within_page = runtime_row_changes == 0;
    const bool mapping_is_idempotent = non_idempotent_remaps == 0;
    const bool pages_are_isolated = decoded_page_aliases == 0;

    print_check("disabled_mapping_is_identity", disabled_pass_through);
    print_check("enabled_mapping_is_deterministic", wrapper_deterministic);
    print_check("page_table_records_first_touch", wrapper_page_allocated);
    print_check("address_units_match_mem_config", address_units_match_config);
    print_check("mapper_and_dram_decoders_agree", decoders_agree);
    print_check("hash_changes_runtime_bank_tuple", hash_changes_runtime_bank);
    print_check("mapping_preserves_runtime_channel", channel_is_preserved);
    print_check("hash_preserves_runtime_row_within_page",
                row_is_preserved_within_page);
    print_check("first_17_pages_are_dram_isolated", pages_are_isolated);
    print_check("active_dram_config_matrix", config_matrix_healthy);
    std::cout << "OBSERVATION mapping_is_idempotent_under_retry "
              << (mapping_is_idempotent ? "TRUE" : "FALSE") << '\n';

    std::cout << "METRIC physical_capacity_bytes " << total_bytes << '\n';
    std::cout << "METRIC mapper_physical_pages " << physical_pages << '\n';
    std::cout << "METRIC dram_burst_bytes " << dram_burst_bytes << '\n';
    std::cout << "METRIC runtime_address_capacity_units "
              << runtime_address_capacity_units << '\n';
    std::cout << "METRIC configured_page_size_address_units "
              << mapper.page_size_address_units() << '\n';
    std::cout << "METRIC configured_hash_unit_address_units "
              << mapper.hash_unit_address_units() << '\n';
    std::cout << "METRIC runtime_pages_with_configured_page_units "
              << runtime_pages_with_configured_page_units << '\n';
    std::cout << "METRIC sampled_1kb_chunks " << kHashChunksPerPage << '\n';
    std::cout << "METRIC numeric_address_changes " << numeric_changes << '\n';
    std::cout << "METRIC runtime_bank_tuple_changes "
              << runtime_bank_changes << '\n';
    std::cout << "METRIC runtime_channel_changes "
              << runtime_channel_changes << '\n';
    std::cout << "METRIC runtime_row_changes " << runtime_row_changes << '\n';
    std::cout << "METRIC mapper_dram_decoder_disagreements "
              << decoder_disagreements << '\n';
    std::cout << "METRIC non_idempotent_remaps "
              << non_idempotent_remaps << '\n';
    std::cout << "METRIC decoded_page_aliases " << decoded_page_aliases << '\n';
    if (decoded_page_aliases != 0) {
        std::cout << "METRIC first_page_alias " << first_alias_page
                  << "->" << first_alias_with << '\n';
    }

    const bool healthy = disabled_pass_through && wrapper_deterministic &&
                         wrapper_page_allocated && address_units_match_config &&
                         decoders_agree &&
                         hash_changes_runtime_bank && channel_is_preserved &&
                         row_is_preserved_within_page && pages_are_isolated &&
                         config_matrix_healthy;
    std::cout << "RESULT " << (healthy ? "PASS" : "FAIL") << '\n';
    return healthy ? 0 : 2;
}
