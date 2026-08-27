#include "common_function.hpp"

#include <boost/mpl/base.hpp>

// #include "DRAM/Dram.h"
#include "fmt/format.h"
#include "NewtonSim/ext/headers/catch.hpp"

SysConfig Config::system_config;

bool is_llama_model_name(const std::string& model_name) {
    return model_name.find("LLAMA") != std::string::npos ||
           model_name.find("Llama") != std::string::npos ||
           model_name.find("llama") != std::string::npos;
}

SysConfig::SysConfig() {
    // std::cout << "Create System config Object" << std::endl;
}


void SysConfig::initialize_from_config_path(std::string const sys_config_path, std::string const memory_config_path, std::string const pim_config_path,
    std::string const inference_config_path, std::string const model_config_path, std::string const request_dataset_path, std::string const output_path) {

    system_config_path_ = sys_config_path;
    memory_config_path_ = memory_config_path;
    inference_config_path_ = inference_config_path;
    model_config_path_ = model_config_path;
    output_path_ = output_path;
    request_dataset_path_ = request_dataset_path;
    operation_log_output_path = output_path;
    log_dir = output_path;

    mem_config = MemConfig(memory_config_path, pim_config_path, output_path);
    // initialize_pim_config(pim_config_path);
    initialize_PIM_config(pim_config_path);
    initialize_compute_die_system_config(sys_config_path);
    initialize_inference_config(inference_config_path_);
    initialize_model_config(model_config_path_);
}


void SysConfig::initialize_compute_die_system_config(std::string sys_config_path) {
    nlohmann::json config = load_config(sys_config_path);

    /* Core configs */
    num_cores = config["num_cores"];
    if ((std::string)config["core_type"] == "systolic_os")
        core_type = CoreType::SYSTOLIC_OS;
    else if ((std::string)config["core_type"] == "systolic_ws")
        core_type = CoreType::SYSTOLIC_WS;
    else
        throw std::runtime_error(
            fmt::format("Not implemented core type {} ",
                (std::string)config["core_type"]));
    core_freq = config["core_freq"];
    core_width = config["core_width"];
    core_height = config["core_height"];

    /* Vector configs */
    vector_core_count = config["vector_core_count"];
    vector_core_width = config["vector_core_width"];
    add_latency = config["add_latency"];
    mul_latency = config["mul_latency"];
    exp_latency = config["exp_latency"];
    gelu_latency = config["gelu_latency"];
    add_tree_latency = config["add_tree_latency"];
    scalar_sqrt_latency = config["scalar_sqrt_latency"];
    scalar_add_latency = config["scalar_add_latency"];
    scalar_mul_latency = config["scalar_mul_latency"];

    /* SRAM configs */
    // sram_size = config["sram_size"];
    sram_width = config["sram_width"];
    spad_size = config["spad_size"];
    accum_spad_size = config["accum_spad_size"];

    /* log config*/
    operation_log_output_path = config["operation_log_output_path"];

    /* Icnt config */
    if ((std::string)config["icnt_type"] == "simple")
        icnt_type = IcntType::SIMPLE;
    else if ((std::string)config["icnt_type"] == "booksim2")
        icnt_type = IcntType::BOOKSIM2;
    else
        throw std::runtime_error(
            fmt::format("Not implemented icnt type {} ", (std::string)config["icnt_type"]));
    icnt_freq = config["icnt_freq"];
    if (config.contains("icnt_latency")) icnt_latency = config["icnt_latency"];
    if (config.contains("icnt_config_path"))
        icnt_config_path = config["icnt_config_path"];

    precision = config["precision"];
    layout = config["layout"];
    scheduler_type = config["scheduler"];

    precision_weight= config["precision_weight"];
    precision_activation= config["precision_activation"];
    precision_cache = config["precision_cache"];
    precision_psum = config["precision_psum"];
}

void SysConfig::initialize_inference_config(std::string inference_config_path) {
    nlohmann::json inference_config = load_config(inference_config_path);

    max_seq_len = inference_config["max_seq_len"];
    kv_cache_entry_size = inference_config["kv_cache_entry_size"];
    max_active_reqs = inference_config["max_active_reqs"];
    max_batch_size = inference_config["max_batch_size"];

    allocation_scheme = inference_config["allocation_scheme"];
    virtual_mem_hash_enable = inference_config.value("virtual_mem_hash_enable", false);
    dram_data_container_enable = inference_config.value("dram_data_container_enable", false);
    dram_data_container_max_payload_mb =
        inference_config.value("dram_data_container_max_payload_mb", 0ULL);

    // Single Test Model
    test_single_op = inference_config["test_single_op"];
    test_single_op_name = inference_config["test_single_op_name"];

    // Multi Layer Test
    test_multi_layer = inference_config["test_multi_layer"];
    test_multi_layer_name  = inference_config["test_multi_layer_name"];

    //Simulation Acceleration
    accelerate_ctrl = inference_config.value("accelerate_ctrl", false);
    accelerate_method = inference_config.value("accelerate_method", std::string{});
    accelerate_sample_ratio = inference_config.value("accelerate_sample_ratio", 0.25);
    attention_command_warmup_weight =
        inference_config.value("attention_command_warmup_weight", 0.0);
    if (attention_command_warmup_weight < 0.0 ||
        attention_command_warmup_weight > 1.0) {
        throw std::invalid_argument(
            "attention_command_warmup_weight must be between 0 and 1");
    }
    softmax_warmup_rounds = inference_config.value("softmax_warmup_rounds", 2U);
    softmax_sample_rounds = inference_config.value("softmax_sample_rounds", 4U);
    compile_time_tile_pruning =
        inference_config.value("compile_time_tile_pruning", false);
    decode_pruning_enabled =
        inference_config.value("decode_pruning_enabled", false);
    decode_pruning_iterations =
        inference_config.value("decode_pruning_iterations", 1U);
    if (decode_pruning_iterations == 0) {
        throw std::invalid_argument(
            "decode_pruning_iterations must be greater than zero");
    }
    decode_pruning_sample_iterations =
        inference_config.value("decode_pruning_sample_iterations", 1U);
    if (decode_pruning_sample_iterations == 0) {
        throw std::invalid_argument(
            "decode_pruning_sample_iterations must be greater than zero");
    }
    if (decode_pruning_sample_iterations > decode_pruning_iterations) {
        throw std::invalid_argument(
            "decode_pruning_sample_iterations cannot exceed "
            "decode_pruning_iterations");
    }

    dram_trace_simulation_mode = inference_config.value("dram_trace_simulation_mode", false);
    record_dram_completion_trace = inference_config.value("record_dram_completion_trace", false);

    // Preset Request Scale
    gen_request = inference_config["gen_request"];
    gen_request_count = inference_config["gen_request_count"];
    gen_request_input_size = inference_config["gen_request_input_size"];
    gen_request_output_size =
        inference_config.value("gen_request_output_size", 0U);
    output_token_iteration_enable =
        inference_config.value("output_token_iteration_enable", false);

    gen_random_request = inference_config["gen_random_request"];
    request_interval = inference_config["request_interval"];
}


void SysConfig::initialize_model_config(std::string model_config_path) {
    nlohmann::json model_config = load_config(model_config_path);
    /* GPT configs */
    model_name = model_config["model_name"];
    model_params_b = model_config["model_params_b"];
    model_vocab_size = model_config["model_vocab_size"];
    model_n_layer = model_config["model_n_layer"];
    model_n_head = model_config["model_n_head"];
    model_n_kv_head = model_config.contains("model_n_kv_head") ? model_config["model_n_kv_head"].get<uint32_t>() : model_n_head;
    model_n_embd = model_config["model_n_embd"];
    if (model_n_kv_head == 0 || model_n_head % model_n_kv_head != 0) {
        throw std::runtime_error(fmt::format("Invalid model_n_kv_head {} for model_n_head {}",
            model_n_kv_head, model_n_head));
    }
    if (model_n_embd % model_n_head != 0) {
        throw std::runtime_error(fmt::format("Invalid model_n_embd {} for model_n_head {}",
            model_n_embd, model_n_head));
    }
    /* parallelism config */
    n_tp = model_config["n_tp"];
}


void SysConfig::initialize_PIM_config(std::string pim_config) {
    nlohmann::json mem_config = load_config(pim_config);
    // DRAM config
    if ((std::string)mem_config["dram_type"] == "dram")
        dram_type = DramType::DRAM;
    else if ((std::string)mem_config["dram_type"] == "newton")
        dram_type = DramType::NEWTON;
    else
        throw std::runtime_error(
            fmt::format("Not implemented dram type {} ", (std::string)mem_config["dram_type"]));
    dram_freq = mem_config["dram_freq"];
    ValidateDramFrequencyConsistency(
        dram_freq, this->mem_config.tCK, pim_config, memory_config_path_);
    DRAM_act_buf_size = (uint64_t)(mem_config["DRAM_act_buf_size_MB"])MB;
    dram_channels = mem_config["dram_channels"];
    ValidateDramChannelConsistency(
        dram_channels, this->mem_config.channels, pim_config,
        memory_config_path_);
    dram_req_size = mem_config.value("dram_req_size", 0U);
    ValidateDramRequestSizeConsistency(
        dram_req_size, this->mem_config.BL, this->mem_config.bus_width,
        pim_config, memory_config_path_);
    // PIM config
    pim_PE_num = mem_config["pim_PE_num"]; // PE_num in each PIM PU
    pim_input_buffer_size = mem_config["pim_input_buffer_size"];  // PIM Input Buffer size, Global Input Buffer(Byte)
    pim_output_buffer_size = mem_config["pim_output_buffer_size"]; // PIM Output Buffer size, output buffer(Byte)
    // area
    hybrid_bonding_bw_area_ratio = mem_config["hybrid_bonding_bw_area_ratio"];
    pim_pu_area = mem_config["pim_pu_area"];
    pim_controller_area_overhead = mem_config["pim_controller_area_overhead"];
    pim_buffer_area_per_kb = mem_config["pim_buffer_area_per_kb"];    
    // power
    pim_static_power_per_pu= mem_config["pim_static_power_per_pu"];
    pim_dynamic_power_per_pu_comp = mem_config["pim_dynamic_power_per_pu_comp"];
    pim_buffer_static_power_per_kb = mem_config["pim_buffer_static_power_per_kb"];
    pim_buffer_dynamic_power_per_bit = mem_config["pim_buffer_dynamic_power_per_bit"];

    pim_parallel_bank_accesses = mem_config.value("pim_parallel_bank_accesses", 0);
}


SysConfig& initialized_config(std::string const sys_config_path, std::string const memory_config_path, std::string const pim_config_path, std::string const inference_config_path, const std::string model_config_path, const std::string output_path) {

    auto system_config = new SysConfig();
    // system_config->initialize_from_config_path(sys_config_path, memory_config_path, pim_config_path, inference_config_path, model_config_path, output_path);

    return *system_config;
}


// ===================== TwoLevelDeterministicMapper Implementation  =====================
constexpr TwoLevelDeterministicMapper::addr_type TwoLevelDeterministicMapper::INVALID_PAGE;

TwoLevelDeterministicMapper::TwoLevelDeterministicMapper(addr_type num_physical_pages)
    : num_physical_pages_(num_physical_pages),
      physical_used_(num_physical_pages_, false),
      next_free_physical_page_(0) {
}

// initialize from config files
bool TwoLevelDeterministicMapper::initFromConfig(const std::string& /*config_path*/) {
    configure(Config::system_config.mem_config);
    return true;
}

void TwoLevelDeterministicMapper::configure(const MemConfig& mem_config) {
    channels_ = mem_config.channels;
    ranks_ = mem_config.ranks;
    bankgroups_ = mem_config.bankgroups;
    banks_per_group_ = mem_config.banks_per_group;
    rows_ = mem_config.rows;
    columns_ = mem_config.columns;
    bus_width_ = mem_config.bus_width;
    burst_length_ = mem_config.BL;

    ch_bits_ = log2_power_of_two(channels_);
    ra_bits_ = log2_power_of_two(ranks_);
    bg_bits_ = log2_power_of_two(bankgroups_);
    ba_bits_ = log2_power_of_two(banks_per_group_);

    ch_pos_ = mem_config.ch_pos;
    ra_pos_ = mem_config.ra_pos;
    bg_pos_ = mem_config.bg_pos;
    ba_pos_ = mem_config.ba_pos;
    ro_pos_ = mem_config.ro_pos;
    co_pos_ = mem_config.co_pos;

    ch_mask_ = mem_config.ch_mask;
    ra_mask_ = mem_config.ra_mask;
    bg_mask_ = mem_config.bg_mask;
    ba_mask_ = mem_config.ba_mask;
    ro_mask_ = mem_config.ro_mask;
    co_mask_ = mem_config.co_mask;

    const addr_type request_size_bits =
        static_cast<addr_type>(burst_length_) * bus_width_;
    if (request_size_bits == 0 || request_size_bits % 8 != 0) {
        throw std::runtime_error(
            "DRAM burst length and bus width do not form a whole-byte request");
    }
    request_size_bytes_ = request_size_bits / 8;
    if (
        PAGE_SIZE_BYTES % request_size_bytes_ != 0 ||
        HASH_UNIT_BYTES % request_size_bytes_ != 0) {
        throw std::runtime_error(
            "Virtual-memory page/hash sizes must be divisible by the DRAM burst size");
    }
    page_size_address_units_ = PAGE_SIZE_BYTES / request_size_bytes_;
    hash_unit_address_units_ = HASH_UNIT_BYTES / request_size_bytes_;

    const addr_type total_bytes =
        static_cast<addr_type>(mem_config.channels) *
        mem_config.channel_size * 1024ull * 1024ull;
    if (total_bytes % request_size_bytes_ != 0 ||
        total_bytes % PAGE_SIZE_BYTES != 0) {
        throw std::runtime_error(
            "DRAM capacity must contain whole burst requests and virtual pages");
    }
    physical_capacity_address_units_ = total_bytes / request_size_bytes_;
    const addr_type configured_pages = total_bytes / PAGE_SIZE_BYTES;
    if (configured_pages != num_physical_pages_ ||
        configured_pages * page_size_address_units_ >
            physical_capacity_address_units_) {
        throw std::runtime_error(
            "Virtual-memory page count is inconsistent with MemConfig capacity");
    }

    const uint32_t page_offset_bits =
        log2_power_of_two(static_cast<uint32_t>(page_size_address_units_));
    if (page_offset_bits == 0 ||
        ch_pos_ + ch_bits_ > page_offset_bits ||
        ra_pos_ + ra_bits_ > page_offset_bits ||
        bg_pos_ + bg_bits_ > page_offset_bits ||
        ba_pos_ + ba_bits_ > page_offset_bits) {
        throw std::runtime_error(
            "DRAM routing/bank fields must lie within the virtual page offset");
    }
}

void TwoLevelDeterministicMapper::reset() {
    logical_to_physical_.assign(logical_to_physical_.size(), INVALID_PAGE);
    physical_used_.assign(physical_used_.size(), false);
    next_free_physical_page_ = 0;
}

void TwoLevelDeterministicMapper::free_page(addr_type logical_page) {
    if (logical_page >= logical_to_physical_.size()) {
        return;
    }
    addr_type phys = logical_to_physical_[logical_page];
    if (phys == INVALID_PAGE) {
        return;
    }
    // release the physical page and remove mapping
    if (phys < physical_used_.size()) {
        physical_used_[phys] = false;
    }
    logical_to_physical_[logical_page] = INVALID_PAGE;
    // Optional optimization: prioritize reallocating this physical page
    if (phys < next_free_physical_page_) {
        next_free_physical_page_ = phys;
    }
}

uint32_t TwoLevelDeterministicMapper::log2_power_of_two(uint32_t x) {
    if (x == 0 || (x & (x - 1)) != 0) {
        return 0;
    }
    uint32_t r = 0;
    while ((x >> r) != 1) {
        ++r;
    }
    return r;
}

// Layer One: Page-level Mapping
TwoLevelDeterministicMapper::addr_type TwoLevelDeterministicMapper::get_or_alloc_physical_page(addr_type logical_page) {
    // If the logical page number exceeds the existing table size, expand the capacity
    if (logical_page >= logical_to_physical_.size()) {
        logical_to_physical_.resize(logical_page + 1, INVALID_PAGE);
    }

    addr_type &entry = logical_to_physical_[logical_page];
    if (entry != INVALID_PAGE) {
        return entry;  // return directly, if the mapping exist
    }

    // Find the next available physical page
    while (next_free_physical_page_ < num_physical_pages_ &&
           physical_used_[next_free_physical_page_]) {
        ++next_free_physical_page_;
    }

    if (next_free_physical_page_ >= num_physical_pages_) {
        throw std::runtime_error("No more physical pages available");
    }

    entry = next_free_physical_page_;
    physical_used_[next_free_physical_page_] = true;
    ++next_free_physical_page_;

    return entry;
}

// The main mapping function
TwoLevelDeterministicMapper::addr_type TwoLevelDeterministicMapper::map(addr_type logical_addr) {
    if (page_size_address_units_ == 0 || hash_unit_address_units_ == 0) {
        throw std::runtime_error("TwoLevelDeterministicMapper is not configured");
    }

    // Stage1: Page level mapping
    const addr_type logical_page = logical_addr / page_size_address_units_;
    const addr_type offset_in_page = logical_addr % page_size_address_units_;

    // get or allocate physical pages
    const addr_type physical_page = get_or_alloc_physical_page(logical_page);

    // Combine initial physical addresses
    const addr_type physical_addr =
        physical_page * page_size_address_units_ + offset_in_page;

    // Extract the previous bank information from the complete address
    const uint32_t logical_ch = extractChannel(logical_addr);
    const uint32_t old_ra = extractRank(physical_addr);
    const uint32_t old_bg = extractBankGroup(physical_addr);
    const uint32_t old_ba = extractBank(physical_addr);

    // Generate mix values using combinational logic (based on logical_page + chunk_index)
    const addr_type chunk_index = offset_in_page / hash_unit_address_units_;
    addr_type mix = chunk_index +
        0x9e3779b97f4a7c15ull * (logical_page + 1);
    mix = (mix ^ (mix >> 30)) * 0xbf58476d1ce4e5b9ull;
    mix = (mix ^ (mix >> 27)) * 0x94d049bb133111ebull;
    mix ^= mix >> 31;

    // Extract perturbation values for each field by bit width from mix (pure combinational logic)
    uint32_t used_bits = 0;
    auto take_bits = [&](uint32_t n_bits) -> uint32_t {
        if (n_bits == 0) return 0;
        const addr_type mask = (static_cast<addr_type>(1) << n_bits) - 1;
        const uint32_t v = static_cast<uint32_t>((mix >> used_bits) & mask);
        used_bits += n_bits;
        return v;
    };

    const uint32_t mix_ba = take_bits(ba_bits_);
    const uint32_t mix_bg = take_bits(bg_bits_);
    const uint32_t mix_ra = take_bits(ra_bits_);

    // Breaking up banks through XOR (combinational logic)
    const uint32_t new_ra = (old_ra ^ mix_ra) & ra_mask_;
    const uint32_t new_bg = (old_bg ^ mix_bg) & bg_mask_;
    const uint32_t new_ba = (old_ba ^ mix_ba) & ba_mask_;

    // Channel is preserved because the request has already been routed to a
    // per-channel Core/ICNT queue before virtual-memory translation.
    const addr_type final_addr = write_bank_tuple(
        physical_addr, logical_ch, new_ra, new_bg, new_ba);

    if (final_addr / page_size_address_units_ != physical_page ||
        final_addr >= physical_capacity_address_units_) {
        throw std::runtime_error(
            "Virtual-memory bank hashing escaped its configured physical page");
    }

    return final_addr;
}

TwoLevelDeterministicMapper::addr_type TwoLevelDeterministicMapper::write_bank_tuple(
    addr_type full_addr, uint32_t new_ch, uint32_t new_ra, uint32_t new_bg, uint32_t new_ba) const {

    // Clear old bank positions
    addr_type clear_mask = ~((ch_mask_ << ch_pos_) | (ra_mask_ << ra_pos_) |
                             (bg_mask_ << bg_pos_) | (ba_mask_ << ba_pos_));
    addr_type addr = full_addr & clear_mask;

    // Write back the new value
    if (ch_bits_ > 0) {
        addr |= (static_cast<addr_type>(new_ch) & ch_mask_) << ch_pos_;
    }
    if (ra_bits_ > 0) {
        addr |= (static_cast<addr_type>(new_ra) & ra_mask_) << ra_pos_;
    }
    if (bg_bits_ > 0) {
        addr |= (static_cast<addr_type>(new_bg) & bg_mask_) << bg_pos_;
    }
    if (ba_bits_ > 0) {
        addr |= (static_cast<addr_type>(new_ba) & ba_mask_) << ba_pos_;
    }

    return addr;
}

TwoLevelDeterministicMapper::BankInfo TwoLevelDeterministicMapper::extract_bank_info(addr_type physical_addr) const {
    BankInfo info{0, 0, 0, 0};
    info.channel = extractChannel(physical_addr);
    info.rank = extractRank(physical_addr);
    info.bankgroup = extractBankGroup(physical_addr);
    info.bank = extractBank(physical_addr);
    return info;
}

uint32_t TwoLevelDeterministicMapper::extractChannel(addr_type addr) const {
    return static_cast<uint32_t>((addr >> ch_pos_) & ch_mask_);
}

uint32_t TwoLevelDeterministicMapper::extractRank(addr_type addr) const {
    return static_cast<uint32_t>((addr >> ra_pos_) & ra_mask_);
}

uint32_t TwoLevelDeterministicMapper::extractBankGroup(addr_type addr) const {
    return static_cast<uint32_t>((addr >> bg_pos_) & bg_mask_);
}

uint32_t TwoLevelDeterministicMapper::extractBank(addr_type addr) const {
    return static_cast<uint32_t>((addr >> ba_pos_) & ba_mask_);
}

// 页表调试输出
void TwoLevelDeterministicMapper::dump_page_table(std::ostream &os) const {
    os << "==== Page Mapping Table (LogicalPage -> PhysicalPage) ====\n";
    for (addr_type lp = 0; lp < logical_to_physical_.size(); ++lp) {
        addr_type pp = logical_to_physical_[lp];
        if (pp == INVALID_PAGE) continue; // 只打印有效映??
        os << "  LPage " << lp << "  ->  PPage " << pp << "\n";
    }
    os << "=========================================================\n";
}

namespace AddressConfig {
    addr_type alignment = MyAddressAllocator::BL_num_per_row;   // align to row
    addr_type channel_mask;                                     // not used
    addr_type channel_offset;                                   // not used
}


// FIXME: Magic Numbers
uint32_t AddressConfig::mask_channel(addr_type address) {
    const int col_bits = 4;
    const int offset = 6;

    int ch = (address >> (col_bits + offset)) & channel_mask;
    return ch;
}

// Address switch for channel and column
addr_type AddressConfig::switch_co_ch(addr_type addr) {
    const int num_col_bits = 4;
    const int num_ch_bits = 5;
    const int num_offset = 6;

    const addr_type ch_mask = ((1 << num_ch_bits) - 1) << (num_col_bits + num_offset);
    const addr_type col_mask = ((1 << num_col_bits) - 1) << num_offset;

    const addr_type mask = ch_mask | col_mask;

    addr_type new_col_bits = (addr & (col_mask << num_ch_bits)) >> num_ch_bits;
    addr_type new_ch_bits = (addr & (ch_mask >> num_col_bits)) << num_col_bits;

    addr = addr & (~mask);
    addr = addr | new_col_bits;
    addr = addr | new_ch_bits;

    return addr;
}

// used in NPU-only
// this is creating dram address.
// align cachline size to 4B
// ex) allocate 31 bytes => align to 32 bytes
addr_type AddressConfig::allocate_address(uint32_t size) {
    static addr_type base_addr{0};

    addr_type result = base_addr;
    base_addr += size;
    if (base_addr & (alignment - 1)) {
        base_addr += alignment - (base_addr & (alignment - 1));
    }

    return result;
}

addr_type AddressConfig::align(addr_type addr) {
    addr_type aligned_addr;
    if (addr & (alignment - 1)==0) {
        aligned_addr = addr - alignment;
    }
    else {
        aligned_addr = addr - (addr & (alignment - 1));
    }

    return aligned_addr;
}


// ===================== TwoLevelDeterministicMapper Integration =====================
// 说明：
// - Two layers of address mapping modules are integrated here, but the existing behavior is not changed by default.c
// - When two-layer mapping needs to be enabled, you can call init_two_level_mapper() after system initialization,
// - And at the appropriate location, call mapw_logicaladdress() on the logical address.

namespace TwoLevelPageMapper {

// Global two-layer mapper
static std::unique_ptr<TwoLevelDeterministicMapper> g_mapper;

// Do you want to enable two-layer mapping (disabled by default to avoid affecting existing behavior)
static bool g_enabled = false;

// Initialize Two Level Deterministic Mapper by MemConfig
static void init_from_mem_config(const MemConfig& mem_cfg) {
    using Mapper      = TwoLevelDeterministicMapper;
    using addr_type   = Mapper::addr_type;
    const addr_type total_bytes =
        static_cast<addr_type>(mem_cfg.channels) *
        mem_cfg.channel_size * 1024ull * 1024ull;
    const addr_type num_pages = total_bytes / Mapper::PAGE_SIZE_BYTES;
    if (num_pages == 0) {
        throw std::runtime_error(
            "Configured DRAM capacity is smaller than one virtual-memory page");
    }

    g_mapper.reset(new Mapper(num_pages));
    g_mapper->configure(mem_cfg);
    g_enabled = true;
    spdlog::info(
        "Virtual memory initialized from MemConfig: {} physical pages, "
        "{} bytes/burst, {} address units/page, {} address units/hash chunk",
        g_mapper->num_physical_pages(), g_mapper->request_size_bytes(),
        g_mapper->page_size_address_units(),
        g_mapper->hash_unit_address_units());
}

// The externally exposed initialization interface is called after being initialized by SysConfig
void init_two_level_mapper() {
    try {
        init_from_mem_config(Config::system_config.mem_config);
    } catch (const std::exception& e) {
        spdlog::error("init_two_level_mapper failed: {}", e.what());
        g_enabled = false;
    }
}

// if enabled, Logical addresses are mapped to physical addresses; else the original address is returned directly
addr_type map_logical_address(addr_type logical_addr) {
    if (!g_enabled || !g_mapper) return logical_addr;
    addr_type physical_address = g_mapper->map(logical_addr);
    // spdlog::info("Data Convert from Virtual Address: {} to Physical Address {}", logical_addr, physical_address);
    return physical_address;
}

// print page mapping table
void dump_page_table(std::ostream& os) {
    if (!g_enabled || !g_mapper) {
        os << "TwoLevelDeterministicMapper is not enabled.\n";
        return;
    }
    g_mapper->dump_page_table(os);
}

} // namespace TwoLevelPageMapper



uint64_t AddressConfig::make_address(int channel, int rank, int bankgroup, int bank, int row, int col) {
    // rorabgbachco
    // HBM2_8Gb_s128_pim.ini
    uint64_t addr = 0;

    // int row_bits = 15;
    uint32_t rank_bits = 1;
    uint32_t bankgroup_bits = 2;
    uint32_t bank_bits = 2;
    uint32_t channel_bits = LogBase2(Config::system_config.dram_channels);
    uint32_t col_bits = 4;
    uint32_t offset = 6;

    addr |= row;

    addr <<= rank_bits;
    addr |= rank;

    addr <<= bankgroup_bits;
    addr |= bankgroup;

    addr <<= bank_bits;
    addr |= bank;

    addr <<= channel_bits;
    addr |= channel;

    addr <<= col_bits;
    addr |= (col & 15);

    addr <<= offset;

    return addr;
}

uint64_t AddressConfig::encode_pim_header(int channel, int row, bool for_gwrite, int num_comps, int num_readres) {
    int gwrite_bit = for_gwrite ? 1 : 0;

    // we can use only 4 bits for column bit
    // use it to shift_amount
    int log_comps = (gwrite_bit << 3) + LogBase2(num_comps);
    int log_readres = LogBase2(num_readres);

    return make_address(channel, log_readres / 16, (log_readres / 4) & 3, log_readres % 4, row,log_comps);
}

uint64_t AddressConfig::encode_pim_comps_readres(int ch, int row, int num_comps, bool last_cmd) {
    int ra_bits = 1;
    int bg_bits = 2;
    int ba_bits = 2;

    int bg_mask = (1 << bg_bits) - 1;
    int ba_mask = (1 << ba_bits) - 1;

    num_comps -= 1;

    assert(num_comps < (1 << (ra_bits + bg_bits + ba_bits)));

    int rank = num_comps >> (bg_bits + ba_bits);
    int bankgroup = (num_comps >> ba_bits) & bg_mask;
    int bank = num_comps & ba_mask;
    int col = last_cmd ? 1 : 0;

    return make_address(ch, rank, bankgroup, bank, row, col);
}

namespace PIM_Parameters {
    bool dual_bank;
    uint32_t PU_num_per_channel;
    uint32_t global_input_buffer_size;
    uint32_t output_buffer_size;

    double pim_static_power_per_pu;
    double pim_dynamic_power_per_pu_comp;
    double pim_buffer_static_power_per_kb;
    double pim_buffer_dynamic_power_per_bit;
}

bool PIM_Parameters::init(const SysConfig& config) {
    dual_bank = config.mem_config.dual_bank;
    if (dual_bank) {
        PU_num_per_channel = config.mem_config.ranks * config.mem_config.bankgroups * config.mem_config.banks_per_group / 2;
    }
    else {
        PU_num_per_channel = config.mem_config.ranks * config.mem_config.bankgroups * config.mem_config.banks_per_group;
    }

    global_input_buffer_size = config.pim_input_buffer_size;
    output_buffer_size = config.pim_output_buffer_size;

    pim_static_power_per_pu = config.pim_static_power_per_pu;
    pim_dynamic_power_per_pu_comp = config.pim_dynamic_power_per_pu_comp;
    pim_buffer_static_power_per_kb = config.pim_buffer_static_power_per_kb;
    pim_buffer_dynamic_power_per_bit = config.pim_buffer_dynamic_power_per_bit;
    return true;
}


namespace MyAddressAllocator {
    uint32_t dram_channels;
    uint32_t ranks;
    uint32_t devices_per_rank; // For DDR4 or DDR5,concat device to channel width
    uint32_t bankgroups;
    uint32_t banks;
    uint32_t rows;
    uint32_t columns;

    uint32_t total_banks;
    uint32_t banks_per_channel;
    uint32_t DQ_width;
    uint32_t burst_length;
    uint32_t BL_num_per_row; 
    uint32_t page_size_bytes;
    uint32_t channel_width;
    uint32_t dram_burst_size;
    uint32_t memory_burst_size;

    // PIM relevant parameters
    bool dual_bank;
    uint32_t PU_num_per_channel;
    uint32_t global_input_buffer_size;
    uint32_t output_buffer_size;

    uint32_t channel_bits;
    uint32_t rank_bits;
    uint32_t bankgroup_bits;
    uint32_t bank_bits;
    uint32_t row_bits;
    uint32_t col_bits;
    uint32_t offset;
    uint32_t shift_bits;

    bool virtual_mem_hash_enable;
    uint32_t AddrGranularity_Hash_Bytes;

    std::vector<std::string> fields;
    std::map<std::string, int> field_widths;
    std::map<std::string, int> field_pos;

    std::map<std::string, uint64_t> mask;
    int ch_pos, ra_pos, bg_pos, ba_pos, ro_pos, co_pos;

    uint32_t precision_weight;
    uint32_t precision_activation;
    uint32_t precision_cache;
    uint32_t precision_psum;

    addr_type _base_addr;
    uint64_t _top_addr;

    uint32_t activation_buf_size;
    AllocationScheme allocation_scheme;
    std::map<std::string, uint32_t> row_loop_size;
    std::string outer_row_loop, middle_row_loop, inner_row_loop; // The mapping of Ba, BG, Ra

    uint32_t base_row, base_column, base_outer_row_loop, base_middle_row_loop, base_inner_row_loop;
    uint32_t base_row_1D_weight, base_column_1D_weight, base_outer_row_loop_1D_weight, base_middle_row_loop_1D_weight, base_inner_row_loop_1D_weight;
    uint32_t act_column, act_row, act_outer_row_loop, act_middle_row_loop, act_inner_row_loop;
    uint32_t activation_start_column, activation_start_row, act_start_inner_row_loop, act_start_middle_row_loop, act_start_outer_row_loop;
    uint32_t cache_column, cache_row, cache_outer_row_loop, cache_middle_row_loop, cache_inner_row_loop;

    std::vector<uint32_t> weight_1D_space; // row index of 1D Weight
    uint32_t row_1D_weight_index; 

    std::vector<uint32_t> activation_space; // row index of Activation

    uint32_t tile_height;
    uint32_t tile_width;
    uint32_t pim_unit;

    uint32_t kv_cache_entry_size;
    uint32_t max_active_reqs;
    uint32_t max_seq_len;
    uint32_t h;  // Q head count, kept for existing attention compute loops
    uint32_t h_q;
    uint32_t h_kv;
    uint32_t d_k;
    uint32_t kv_dim;
    uint32_t kv_group_size;
    uint32_t kv_cache_entry_burst_num;
    uint32_t layers;

    std::vector<Ptr<std::deque<uint64_t>>> _rows;
    bool activation_space_malloced;  // The Activation is allocated after all weight have been allocated
    uint32_t act_column_slice_size = 1024;
    uint32_t weight_column_slice_size = 1024;

    bool IANUS_channel_parallel = false;

    // 1D weight
    uint32_t weight_1D_allocate_row = 1;
    uint32_t available_weight_1D_burst = 0;

    std::vector<std::vector<uint32_t>> rababg_bank_index;

    // 2D weight
    uint32_t interleaved_banks_per_tile; // Mapping Banks of each Weight Tile
    uint32_t allocated_tiles_per_iteration;
    uint32_t interleaved_row_offset;
    std::vector<std::vector<std::vector<uint32_t>>> interleaved_bank_index; // Index of Mapping bank index of each tile (Ra, BG, Ba)
    uint32_t burst_weight_row_unit;
    uint32_t burst_times_per_tile;
    uint32_t bank_allocated_columns_per_tile;
    uint32_t weight_interleave_columns = 1;
    uint32_t weight_rows_per_bank_row = 1; // capacity of one bank row for weight rows of multi weight interleave columns

    // KCache mapping
    uint32_t KVCache_allocate_row = 1;
    uint32_t parallel_KCache_head_per_channel = 1;
    uint32_t parallel_VCache_head_per_channel = 1;

    // K Cache
    uint32_t KCache_interleaved_banks_per_head;
    uint32_t allocated_KCache_head_per_iteration;
    uint32_t KCache_rows_per_bank_row;
    uint32_t burst_times_per_KCache_row;
    uint32_t KCache_rows_per_allocation;

    std::vector<std::vector<std::vector<uint32_t>>> KCache_interleaved_bank_index; // Index of Mapping bank index of each KCache head (Ra, BG, Ba)
    // V Cache
    uint32_t VCache_interleaved_banks_per_head;
    uint32_t allocated_VCache_head_per_iteration;
    uint32_t VCache_columns_per_bank;
    uint32_t VCache_burst_row_unit;
    uint32_t VCache_row_units_per_allocation;
    uint32_t VCache_rows_per_allocation;
    uint32_t VCache_rows_per_bank_row;

    std::vector<std::vector<std::vector<uint32_t>>> VCache_interleaved_bank_index; // Index of Mapping bank index of each VCache head (Ra, BG, Ba)
}

bool MyAddressAllocator::init(const SysConfig& config) {
    /*
    virtual_alloc_ptr = 0;
    page_table.clear();
    bank_alloc_ptr.clear();
    */
    dram_channels = config.mem_config.channels;
    ranks = config.mem_config.ranks;
    devices_per_rank = config.mem_config.devices_per_rank;
    bankgroups = config.mem_config.bankgroups;
    banks = config.mem_config.banks_per_group;
    rows = config.mem_config.rows;
    columns = config.mem_config.columns;
    DQ_width = config.mem_config.device_width;  // DQ of each device = DQ of each banks
    burst_length = config.mem_config.BL;
    BL_num_per_row = columns / burst_length;   // the contains operation for each row

    page_size_bytes =  DQ_width * columns / 8;

    total_banks = dram_channels * ranks * bankgroups * banks;
    banks_per_channel = ranks * bankgroups * banks;

    channel_width = config.mem_config.bus_width;
    dram_burst_size = burst_length * channel_width / 8;  // Byte
    memory_burst_size = dram_burst_size * dram_channels;

    spdlog::critical("Initializing MyAddressAllocator of DRAM system with "
                     "{} channels, {} ranks, {} devices, {} bankgroups, {} banks, {} rows, {} columns",
                     dram_channels, ranks, devices_per_rank, bankgroups, banks, rows, columns);

    spdlog::critical("DQ = {}, burst_length = {}, Channel_Burst_size = {}, DRAM_Burst_size = {}, Each Bank Row Contains {} Bytes, the supported BL time of one row is {}",
        DQ_width, burst_length, dram_burst_size, memory_burst_size, columns * DQ_width / 8, BL_num_per_row);

    precision_weight = config.precision_weight;
    precision_activation = config.precision_activation;
    precision_cache = config.precision_cache;
    precision_psum = config.precision_psum;

    // Address widths, positions, and masks are calculated once by MemConfig.
    const phsim::DramAddressLayout& layout =
        config.mem_config.address_layout();
    channel_bits = layout.channel_width;
    rank_bits = layout.rank_width;
    bankgroup_bits = layout.bankgroup_width;
    bank_bits = layout.bank_width;
    row_bits = layout.row_width;
    col_bits = LogBase2(columns);
    offset = LogBase2(burst_length);
    shift_bits = config.mem_config.shift_bits;

    virtual_mem_hash_enable = config.virtual_mem_hash_enable;

    AddrGranularity_Hash_Bytes = 1024;

    if (virtual_mem_hash_enable) {
        spdlog::info("");
    }

    fields.clear();
    field_widths.clear();
    field_pos.clear();
    mask.clear();
    row_loop_size.clear();
    field_widths = {{"ch", static_cast<int>(layout.channel_width)},
                    {"ra", static_cast<int>(layout.rank_width)},
                    {"bg", static_cast<int>(layout.bankgroup_width)},
                    {"ba", static_cast<int>(layout.bank_width)},
                    {"ro", static_cast<int>(layout.row_width)},
                    {"co", static_cast<int>(layout.column_width)}};
    field_pos = {{"ch", layout.channel_pos},
                 {"ra", layout.rank_pos},
                 {"bg", layout.bankgroup_pos},
                 {"ba", layout.bank_pos},
                 {"ro", layout.row_pos},
                 {"co", layout.column_pos}};
    mask = {{"ch", layout.channel_mask},
            {"ra", layout.rank_mask},
            {"bg", layout.bankgroup_mask},
            {"ba", layout.bank_mask},
            {"ro", layout.row_mask},
            {"co", layout.column_mask}};

    _base_addr = 0;
    _top_addr = 0;

    inner_row_loop = layout.row_loop_order[0];
    middle_row_loop = layout.row_loop_order[1];
    outer_row_loop = layout.row_loop_order[2];

    // spdlog::info("************************************************************************************************************************************");
    // spdlog::info("The Bank Iteration: Outer Row Loop: {}, Middle Row Loop: {}, Inner_Row_Loop: {}", outer_row_loop, middle_row_loop, inner_row_loop );
    // spdlog::info("************************************************************************************************************************************");

    ch_pos = field_pos.at("ch");
    ra_pos = field_pos.at("ra");
    bg_pos = field_pos.at("bg");
    ba_pos = field_pos.at("ba");
    ro_pos = field_pos.at("ro");
    co_pos = field_pos.at("co");

#ifndef NDEBUG
    // Keep independent calculations only in Debug builds so configuration
    // drift is caught without adding comparison overhead to Release runs.
    assert(config.dram_channels == dram_channels);
    assert(LogBase2(dram_channels) == static_cast<int>(layout.channel_width));
    assert(LogBase2(ranks) == static_cast<int>(layout.rank_width));
    assert(LogBase2(bankgroups) == static_cast<int>(layout.bankgroup_width));
    assert(LogBase2(banks) == static_cast<int>(layout.bank_width));
    assert(LogBase2(rows) == static_cast<int>(layout.row_width));
    assert(col_bits - offset == static_cast<int>(layout.column_width));
    assert(ch_pos == config.mem_config.ch_pos);
    assert(ra_pos == config.mem_config.ra_pos);
    assert(bg_pos == config.mem_config.bg_pos);
    assert(ba_pos == config.mem_config.ba_pos);
    assert(ro_pos == config.mem_config.ro_pos);
    assert(co_pos == config.mem_config.co_pos);
    assert(mask.at("ch") == config.mem_config.ch_mask);
    assert(mask.at("ra") == config.mem_config.ra_mask);
    assert(mask.at("bg") == config.mem_config.bg_mask);
    assert(mask.at("ba") == config.mem_config.ba_mask);
    assert(mask.at("ro") == config.mem_config.ro_mask);
    assert(mask.at("co") == config.mem_config.co_mask);
#endif

    activation_buf_size = config.DRAM_act_buf_size;

    row_loop_size["ra"] = ranks;
    row_loop_size["bg"] = bankgroups;
    row_loop_size["ba"] = banks;


    if (config.allocation_scheme == "NPU") {
        allocation_scheme = AllocationScheme::NPU;
    }
    else if(config.allocation_scheme == "NeuPIM") {
        allocation_scheme = AllocationScheme::NeuPIM;
    } else if (config.allocation_scheme == "IANUS") {
        allocation_scheme = AllocationScheme::IANUS;
        // TODO
    } else if (config.allocation_scheme == "DASH") {
        allocation_scheme = AllocationScheme::DASH;
    } else {
        throw std::runtime_error("Unsupported allocation scheme");
    }

    activation_space_malloced = false;

    base_row = 0;
    base_column = 0;
    base_outer_row_loop = 0;
    base_middle_row_loop = 0;
    base_inner_row_loop = 0;

    base_row = 0;
    base_column = 0;
    base_outer_row_loop = 0;
    base_middle_row_loop = 0;
    base_inner_row_loop = 0;

    row_1D_weight_index = 0;
    base_row_1D_weight = 0;
    base_column_1D_weight = 0;
    base_outer_row_loop_1D_weight = 0;
    base_middle_row_loop_1D_weight = 0;
    base_inner_row_loop_1D_weight  = 0;

    act_column = 0;
    act_row = 0;
    act_outer_row_loop = 0;
    act_middle_row_loop = 0;
    act_inner_row_loop = 0;

    cache_column = 0;
    cache_row = 0;
    cache_outer_row_loop = 0;
    cache_middle_row_loop = 0;
    cache_inner_row_loop = 0;

    max_active_reqs = config.max_active_reqs;
    max_seq_len = config.max_seq_len;
    h_q = config.model_n_head;
    h_kv = config.model_n_kv_head;
    h = h_q;
    d_k = config.model_n_embd / h_q;
    kv_dim = h_kv * d_k;
    if (h_kv == 0 || h_q % h_kv != 0) {
        throw std::runtime_error(fmt::format("Invalid GQA head configuration: h_q={}, h_kv={}", h_q, h_kv));
    }
    kv_group_size = h_q / h_kv;
    layers = config.model_n_layer;

    // SIZE of each tile
    tile_height = config.core_height;
    tile_width = config.core_width;
    pim_unit = config.pim_PE_num;

    assert(tile_width % 2 == 0);

    interleaved_banks_per_tile = 8; // The set value is 4, What is the number of banks corresponding to a single Tile mapping in a channel
    // *******************************************************************************************
    // The following settings are used to set the interlove_manks_per_tile to determine the acceleration effect of FFN in the decode stage
    std::string buffer_key = std::to_string(config.pim_input_buffer_size) + "_" + std::to_string(config.pim_output_buffer_size);
    int batch_size = config.gen_request_count;

    // Construct Lookup Table
    std::map<std::string, std::map<int, int>> weight_interleave_map = {
        {"512_32", {{1, 8}, {2, 8}, {4, 8}}},
        {"512_64", {{1, 8}, {2, 8}, {4, 8}}},
        {"1024_32", {{1, 8}, {2, 8}, {4, 8}}},
        {"1024_64", {{1, 8}, {2, 8}, {4, 8}}}
    };

    // Find the corresponding weight_interleave_comlumns
    if (weight_interleave_map.find(buffer_key) != weight_interleave_map.end()) {
        auto& batch_map = weight_interleave_map[buffer_key];
        if (batch_map.find(batch_size) != batch_map.end()) {
            weight_interleave_columns = batch_map[batch_size];
            spdlog::info("weight_interleave_columns set to {} for buffer {} and batch size {}",
                         weight_interleave_columns, buffer_key, batch_size);
        } else {
            spdlog::error("Invalid batch size: {}", batch_size);
            weight_interleave_columns = 8;
        }
    } else {
        spdlog::error("Invalid buffer configuration: {}", buffer_key);
        weight_interleave_columns = 8;
    }

    // Based on the solved weight_interleave_columns, reverse calculate the interleaved_banks_per_tile
    interleaved_banks_per_tile = tile_width / weight_interleave_columns / dram_channels;
    spdlog::info("Current weight interleaved_banks_per_tile = {} based on weight_interleave_columns {} for tile_width {} and dram_channels {}",
                 interleaved_banks_per_tile, weight_interleave_columns, tile_width, dram_channels);

    // *******************************************************************************************

    allocated_tiles_per_iteration = ranks * bankgroups * banks / interleaved_banks_per_tile;  // the allocated tile number of current interleave number
    assert(ranks * bankgroups * banks % interleaved_banks_per_tile == 0);
    weight_interleave_columns = 0;  //

    rababg_bank_index.reserve(ranks * bankgroups * banks);
    for (uint32_t rank_index = 0; rank_index < ranks; rank_index++) {
        for (uint32_t bank_index = 0; bank_index < banks; bank_index++) {
            for (uint32_t bankgroup_index = 0; bankgroup_index < bankgroups; bankgroup_index++) {
                auto index = std::vector<uint32_t>{rank_index, bankgroup_index, bank_index};
                rababg_bank_index.push_back(index);
            }
        }
    }

    // arrangement for KV cache
    if (allocation_scheme == AllocationScheme::NPU) {
        kv_cache_entry_size = columns * DQ_width / 8 / (d_k * precision_cache);  // Kcache token num in one DRAM Page
    }
    else {
        kv_cache_entry_size = config.kv_cache_entry_size;  // the preset token number of each KV Cache entry
        kv_cache_entry_burst_num = kv_cache_entry_size;
    }

    // weight allocation scheme
    if (allocation_scheme == AllocationScheme::DASH) {
        // load the data time with size of tile_width * tile_height
        bank_allocated_columns_per_tile = std::ceil((double)tile_width / (dram_channels * interleaved_banks_per_tile));
        burst_weight_row_unit = dram_burst_size / precision_weight;  // one column * burst_weight_row_unit rows weight elements are loaded
        burst_times_per_tile = std::ceil(tile_width / dram_channels * (tile_height / burst_weight_row_unit)); // burst times for tile_width col × burst_weight_row_unit row

        spdlog::info("Current Allocation Scheme is DASH, {} Tile width, {} DRAM Channels, {} interleaved_banks_per_tile", tile_width, dram_channels, interleaved_banks_per_tile);
        spdlog::info("interleave columns per tile is {}, burst_weight_row_unit is {}", bank_allocated_columns_per_tile, burst_weight_row_unit);

        for (uint32_t tile_offset = 0; tile_offset < allocated_tiles_per_iteration; tile_offset++) {
            // location of current Rank, BankGroup, Bank
            uint32_t tile_banks = tile_offset * interleaved_banks_per_tile / bankgroups;
            uint32_t bankgroup_offset = (tile_offset * interleaved_banks_per_tile) % bankgroups;

            uint32_t tile_ranks = tile_banks / banks;
            uint32_t bank_offset = tile_banks % banks;
            uint32_t rank_offset = tile_ranks % ranks;

            std::vector<std::vector<uint32_t>> accessed_bank_index;
            for (uint32_t i=0; i<interleaved_banks_per_tile; i++) {
                uint32_t bankgroup_index = (bankgroup_offset + i) % bankgroups;
                uint32_t bank_index = (bank_offset + (bankgroup_offset + i) / bankgroups);
                assert(bank_index < banks);
                uint32_t rank_index = rank_offset;
                accessed_bank_index.push_back(std::vector<uint32_t>{rank_index, bank_index, bankgroup_index});
            }
            interleaved_bank_index.push_back(accessed_bank_index);
        }
        weight_interleave_columns = bank_allocated_columns_per_tile;  // The column Number for interleave stored in each bank
        weight_rows_per_bank_row = BL_num_per_row / weight_interleave_columns * burst_weight_row_unit;
        // data size per bank row = weight_interleave_columns × weight_rows_per_bank_row
    }
    else if (allocation_scheme == AllocationScheme::IANUS) {
        burst_times_per_tile = std::floor(tile_width / dram_channels * (tile_height * precision_weight / dram_burst_size));
        bank_allocated_columns_per_tile = 1;
        weight_rows_per_bank_row = BL_num_per_row / bank_allocated_columns_per_tile * burst_weight_row_unit;
    }

    /*
    if (ranks % 2 == 0) {
        parallel_KCache_head_per_channel = 2;
        parallel_VCache_head_per_channel = 2;
    }
    else {
        parallel_KCache_head_per_channel = ranks;
        parallel_VCache_head_per_channel = ranks;
    }
    */

    parallel_KCache_head_per_channel = 2;
    parallel_VCache_head_per_channel = 2;

    // KCache allocation scheme
    allocated_KCache_head_per_iteration = parallel_KCache_head_per_channel * dram_channels; // at least one KCache head per channel
    KCache_interleaved_banks_per_head = std::floor( static_cast<double>(ranks * bankgroups * banks) / parallel_KCache_head_per_channel);
    burst_times_per_KCache_row = std::floor( static_cast<double>(d_k * precision_cache) / dram_burst_size);
    KCache_rows_per_bank_row = (dram_burst_size * BL_num_per_row) / (d_k * precision_cache);  // One KCache column of the head  continuously maps to the Bank
    KCache_rows_per_allocation = KCache_rows_per_bank_row * KCache_interleaved_banks_per_head; // The capacity of KCache columns for banks of each head

    // bank index of each mapping head of KCache
    for (uint32_t head_offset = 0; head_offset < parallel_KCache_head_per_channel; head_offset++) {
        uint32_t bankgroup_offset = (head_offset * KCache_interleaved_banks_per_head) % bankgroups;
        uint32_t head_banks = head_offset * KCache_interleaved_banks_per_head / bankgroups;  // BG-Ba-Ra，the initial bank loop of current KCache Head
        uint32_t bank_offset = head_banks % banks;

        uint32_t head_ranks = head_banks / banks;
        uint32_t rank_offset = head_ranks % ranks;

        std::vector<std::vector<uint32_t>> accessed_bank_index;
        for (uint32_t i = 0; i < KCache_interleaved_banks_per_head; i++) {
            uint32_t bankgroup_index = (bankgroup_offset + i) % bankgroups;
            uint32_t bank_index = (bank_offset + (bankgroup_offset + i) / bankgroups) % banks;
            uint32_t rank_index = rank_offset + (bank_offset + (bankgroup_offset + i) / bankgroups) / banks;
            assert(bank_index < banks);
            assert(bankgroup_index < bankgroups);
            assert(rank_index < ranks);
            accessed_bank_index.push_back(std::vector<uint32_t>{rank_index, bank_index, bankgroup_index});
        }
        KCache_interleaved_bank_index.push_back(accessed_bank_index);
    }

    // VCache allocation scheme
    allocated_VCache_head_per_iteration = parallel_VCache_head_per_channel * dram_channels;
    VCache_interleaved_banks_per_head = std::floor( static_cast<double>(ranks * bankgroups * banks) / parallel_VCache_head_per_channel);
    VCache_columns_per_bank = ceil(static_cast<double>(d_k) / (VCache_interleaved_banks_per_head)); // d_k VCache columns of each head through interleaving
    VCache_burst_row_unit = dram_burst_size / precision_cache;
    VCache_row_units_per_allocation = BL_num_per_row / VCache_columns_per_bank; // the capacity of VCache columns of each head after allocation
    VCache_rows_per_allocation = VCache_row_units_per_allocation * VCache_burst_row_unit;
    VCache_rows_per_bank_row = BL_num_per_row / VCache_columns_per_bank * VCache_burst_row_unit;

    // bank index of each mapping head of V Cache
    for (uint32_t head_offset = 0; head_offset < parallel_VCache_head_per_channel; head_offset++) {
        uint32_t bankgroup_offset = (head_offset * VCache_interleaved_banks_per_head) % bankgroups;
        uint32_t head_banks = head_offset * VCache_interleaved_banks_per_head / bankgroups;
        uint32_t bank_offset = head_banks % banks;

        uint32_t head_ranks = head_banks / banks;
        uint32_t rank_offset = head_ranks % ranks;

        std::vector<std::vector<uint32_t>> accessed_bank_index;
        for (uint32_t i = 0; i < VCache_interleaved_banks_per_head; i++) {
            uint32_t bankgroup_index = (bankgroup_offset + i) % bankgroups;
            uint32_t bank_index = (bank_offset + (bankgroup_offset + i) / bankgroups) % banks;
            uint32_t rank_index = rank_offset + (bank_offset + (bankgroup_offset + i) / bankgroups) / banks;
            assert(bank_index < banks);
            assert(bankgroup_index < bankgroups);
            assert(rank_index < ranks);
            accessed_bank_index.push_back(std::vector<uint32_t>{rank_index, bank_index, bankgroup_index});
        }
        VCache_interleaved_bank_index.push_back(accessed_bank_index);
    }

    weight_malloc_1D(); // allocate one row of all banks for 1D weight
    return true;
}

uint32_t MyAddressAllocator::get_kv_head_index(uint32_t q_head_index) {
    if (q_head_index >= h_q) {
        throw std::runtime_error(fmt::format("Invalid q head index {} for h_q {}", q_head_index, h_q));
    }
    if (kv_group_size == 0) {
        throw std::runtime_error("Invalid kv_group_size 0");
    }
    uint32_t kv_head_index = q_head_index / kv_group_size;
    if (kv_head_index >= h_kv) {
        throw std::runtime_error(fmt::format("Invalid kv head index {} mapped from q head {}, h_kv {}", kv_head_index, q_head_index, h_kv));
    }
    return kv_head_index;
}

void MyAddressAllocator::cleanup() {
    std::vector<std::string>().swap(fields);
    std::map<std::string, int>().swap(field_widths);
    std::map<std::string, int>().swap(field_pos);
    std::map<std::string, uint64_t>().swap(mask);
    std::map<std::string, uint32_t>().swap( row_loop_size);
    std::vector<std::vector<uint32_t>>().swap(rababg_bank_index);
    std::vector<std::vector<std::vector<uint32_t>>>().swap(interleaved_bank_index);
    std::vector<std::vector<std::vector<uint32_t>>>().swap(KCache_interleaved_bank_index);
    std::vector<std::vector<std::vector<uint32_t>>>().swap(VCache_interleaved_bank_index);
    std::vector<uint32_t>().swap(weight_1D_space);
    std::vector<uint32_t>().swap(activation_space);
    /*
    std::map<uint64_t, uint64_t>().swap(page_table);
    std::map<uint32_t, uint32_t>().swap(bank_alloc_ptr);
    */
}

bool MyAddressAllocator::weight_malloc_1D(){
    if (base_column + base_inner_row_loop + base_middle_row_loop + base_outer_row_loop != 0) {
        base_column = 0;
        base_inner_row_loop = 0;
        base_middle_row_loop = 0;
        base_outer_row_loop = 0;
        base_row++;
    }
    // spdlog::info("current row is align to {} ", base_row);
    for (uint32_t i=0; i< weight_1D_allocate_row; i++) {
        weight_1D_space.push_back(base_row);
        base_row++;
        // spdlog::info("Allocate row {} of all bank for 1D Weight", weight_1D_space.back());
    }
    available_weight_1D_burst = available_weight_1D_burst + weight_1D_allocate_row * row_loop_size[inner_row_loop] * row_loop_size[middle_row_loop] * row_loop_size[outer_row_loop] * columns;
    // spdlog::info("After allocate of {} row, current available_weight_1D burst time is {}", weight_1D_allocate_row,available_weight_1D_burst);

    return true;
}

std::vector<uint32_t> MyAddressAllocator::weight_allocate_1D(std::vector<uint32_t> dims, uint32_t precision) {
    assert(dims.size() == 1);
    assert(activation_space_malloced == false);

    uint64_t size = dims[0] * precision;
    uint32_t allocate_column_num = ceil(size/memory_burst_size);  // required burst time of current tensor

    while (available_weight_1D_burst < allocate_column_num) {
        weight_malloc_1D();  // malloc new space until required burst time is satisfied
    }

    // col offset
    uint32_t col_offset = (base_column_1D_weight + allocate_column_num) % BL_num_per_row;
    uint32_t allocate_row_num = (base_column_1D_weight + allocate_column_num) / BL_num_per_row;
    // inner_row_loop
    uint32_t inner_row_offset = (base_inner_row_loop_1D_weight + allocate_row_num) % row_loop_size[inner_row_loop];
    uint32_t allocate_row_middle = (base_inner_row_loop_1D_weight + allocate_row_num) / row_loop_size[inner_row_loop];
    // middle_loop
    uint32_t middle_row_offset = (base_middle_row_loop_1D_weight + allocate_row_middle) % row_loop_size[middle_row_loop];
    uint32_t allocate_row_outer = (base_middle_row_loop_1D_weight + allocate_row_middle) / row_loop_size[middle_row_loop];
    // outer_loop
    uint32_t outer_row_offset = (base_outer_row_loop_1D_weight + allocate_row_outer) % row_loop_size[outer_row_loop];
    uint32_t allocate_row = (base_outer_row_loop_1D_weight + allocate_row_outer) / row_loop_size[outer_row_loop];

    std::vector<uint32_t> allocate_row_indexes;
    allocate_row_indexes.push_back(weight_1D_space[row_1D_weight_index]);
    for (uint32_t i=0; i < allocate_row; i++) {
        // spdlog::info("row {} is utilized for 1-D weight");
        row_1D_weight_index++;
        allocate_row_indexes.push_back(weight_1D_space[row_1D_weight_index]);
    }

    base_column_1D_weight = col_offset;
    base_inner_row_loop_1D_weight = inner_row_offset;
    base_middle_row_loop_1D_weight = middle_row_offset;
    base_outer_row_loop_1D_weight = outer_row_offset;
    base_row_1D_weight = weight_1D_space.back();

    // spdlog::info("The allocated 1-D Weight size is {}", size);
    // spdlog::info("The allocate size corresponding to {} times burst, {} row", allocate_column_num, allocate_row_num);

    return allocate_row_indexes;
}

std::vector<uint32_t> MyAddressAllocator::weight_allocate_2D(std::vector<uint32_t> dims, uint32_t precision) {
    // allocate space for each 2D weight Tensor
    std::vector<uint32_t> allocate_row_indexes;
    if (allocation_scheme == AllocationScheme::NPU or allocation_scheme == AllocationScheme::NeuPIM) {
        allocate_row_indexes = weight_2D_allocate_in_sequence(dims, precision);  // allocate based on address mapping
    }
    else if (allocation_scheme == AllocationScheme::IANUS) {
        // IANUS scheme, allocate one column consecutively in one bank
        allocate_row_indexes = weight_2D_allocate_in_IANUS(dims, precision);
    }
    else if (allocation_scheme == AllocationScheme::DASH) {
        allocate_row_indexes = weight_2D_allocate_in_DASH(dims, precision);
    }
    return allocate_row_indexes;
}


std::vector<uint32_t> MyAddressAllocator::weight_2D_allocate_in_sequence(std::vector<uint32_t> dims, uint32_t precision) {
    assert(dims.size() == 2);

    uint64_t size = precision;
    for (auto dim : dims) {
        size *= dim;
    }

    spdlog::info("The allocated size of Weight 2D is {}", size);
    spdlog::info("The base index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", base_row, outer_row_loop, base_outer_row_loop,
        middle_row_loop, base_middle_row_loop, inner_row_loop, base_inner_row_loop, base_column);
    uint32_t allocate_column_num = ceil(size/memory_burst_size);  // The required burst times for allocated data
    uint32_t col_offset = (base_column + allocate_column_num) % BL_num_per_row; // the new col offset
    uint32_t allocate_row_num = (base_column + allocate_column_num) / BL_num_per_row;
    spdlog::info("The allocate size corresponding to {} times burst, {} row", allocate_column_num, allocate_row_num);
    spdlog::info("The new column offset is {} from the basic column offset {}", col_offset, base_column);

    // calculate in the loop of Bank, BankGroup, Rank
    spdlog::info("The Bank Sequence for row allocation is {}, {}, {}",outer_row_loop, middle_row_loop,inner_row_loop);
    // inner_row_loop offset
    uint32_t inner_row_offset = (base_inner_row_loop + allocate_row_num) % row_loop_size[inner_row_loop];  // new inner_loop index
    uint32_t allocate_row_middle = (base_inner_row_loop + allocate_row_num) / row_loop_size[inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (base_middle_row_loop + allocate_row_middle) % row_loop_size[middle_row_loop]; // new middle_loop index
    uint32_t allocate_row_outer = (base_middle_row_loop + allocate_row_middle) / row_loop_size[middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (base_outer_row_loop + allocate_row_outer) % row_loop_size[outer_row_loop];  // new outer_loop index
    uint32_t allocate_row = (base_outer_row_loop + allocate_row_outer) / row_loop_size[outer_row_loop];

    // row offset
    uint32_t row_offset = base_row + allocate_row;

    std::vector<uint32_t> allocate_row_indexes;
    for (uint32_t i=0; i < allocate_row; i++) {
        allocate_row_indexes.push_back(base_row);
        base_row++;
    }

    // based on the address maping scheme, malloc new row or record the allocated row
    spdlog::info("The new index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", row_offset, outer_row_loop, outer_row_offset,
        middle_row_loop, middle_row_offset, inner_row_loop, inner_row_offset, col_offset);

    base_column = col_offset;
    base_inner_row_loop = inner_row_offset;
    base_middle_row_loop = middle_row_offset;
    base_outer_row_loop = outer_row_offset;

    return allocate_row_indexes;
}


std::vector<uint32_t> MyAddressAllocator::weight_2D_allocate_in_DASH(std::vector<uint32_t> dims, uint32_t precision) {
    assert(dims.size() == 2);
    uint32_t K = dims[0];
    uint32_t N = dims[1];

    /* uint32_t data_per_burst = dram_burst_size / precision;    // 一次Burst操作加载的数据量
     * uint32_t max_tile_num = 4096; // embedding维度是最大的Tile大小，映射的最大??
     * uint32_t column_concat_num = data_per_burst / pim_unit; // 每次Burst取出的数据可以供近存单元完成的次??
     * uint32_t max_interleave_bank_num = dram_channels * bankgroups * banks; // 当前内存系统NPU加载过程中，至少交织的bank数，当前只考虑每个channel中一个Device的情况，没有数据位宽方向上的拼接
     * uint32_t min_interleave_bank_num = dram_channels * 2; // 最少的交织数据，每个dram channel中至用到两个
     *
     * 当前数据映射的约束条件：
     * 1. Tile width 列权值数据，至少映射在两个Bank，至多映射在Ba×BG个Bank中，不跨Rank映射，避免读取一个Tile的过程中产生额外的读取开销
     * 2. 如果tile width < BG×Ba，按照优先channel，之后BankGroup，最后Bank的顺序进行填充，如果Tile width>BG×Ba，则连续进行映射过程
     * 3. 映射到一个Bank中同一个Tile width的数据，在行中排布过程按照每次Burst加载的数量，划分相应的行数，多列以该数量为单位在Bank中的一行交织产生排??
     * 4. 计算出几列数据在Bank中排布所占用的行数量，完成更新，之后继续对相应的行继??
     *
     * 为了有利于PIM计算，计算过程中在一行排布的最小单位是 pim_PE_num, 另外一个因素是Global Input Buffer的大小，加载的数据可以支持多少个Weight进行计算过程
     * 数据类型是FP16，不需要考虑额外的量化参数的问题，PSUM的位宽为FP32，最终输出的时候截断为FP16??
     * 映射过程中具有两个约束：
     * 1. 一个矩阵的Tile，经过列映射可以排布映射到所有的Bank中，在近存计算过程中可以并行GEMV，因此每一Bank中映射的列数具有一个最大值，每个Tile映射的Bank数量有限制；
     * 2. 映射的Tile，填充到Bank中的时候，每一个Bank中具有对应的列数，而且需要交织进行排布，为了保证近存计算的效率，交织的最小粒度是PE一次计算的数据量，
     *
     * NPU要求映射的最大值：加载core_width的数据时，需要从多个并行的channel中交织取出： max × (dram_channel × 2) <= tile
     * NPU要求映射的最小?? 一个Bank一列数??min >=1, 对于单个Tile的加载，避免出现rank切换，min × channel × BankGroup × Bank > core_width
     * PIM要求映射的最大值：保证内存系统的每一个Bank中都有填充数??从而实现并行计?? max × dram_channel × BankGroup × Bank × Rank <= Demb
     * PIM要求映射的最小值：越小越好，几列数据在这里进行交织排布，PIM计算过程中不会有行切换开销(可以取四个Bank，对应因素为FAW)
     *
     * 数据加载过程中，同一个Bank中的数据越多, 在Bank中每行的空间被交织填充，每行的含有的权值数据量至少要与Input Global Buffer大小相等，保证能做连续的乘累加计??
     */

    uint32_t tile_column_interleaved_per_bank = tile_width / (dram_channels * interleaved_banks_per_tile);
    uint32_t allocate_rows = K * tile_column_interleaved_per_bank * precision_weight / (dram_burst_size * BL_num_per_row);
    uint32_t allocate_iteration = std::ceil(((double)N / tile_width) / allocated_tiles_per_iteration);

    spdlog::info("{} columns 2D weight data interleaved to {} bank * {} channel, {} bank rows is occupied, total {} iteration for 2D Weight with size {}",
                 tile_column_interleaved_per_bank, interleaved_banks_per_tile, dram_channels, allocate_rows, allocate_iteration, dims);

    std::vector<uint32_t> allocate_row_index;
    for (uint32_t i=0; i < allocate_iteration; i++) {
        allocate_row_index.push_back(base_row);
        base_row = base_row + allocate_rows;
    }
    // 以Tile为单位，进行循环填充过程
    // 每个Tile进行填充过程，完成了循环填充
    // 计算出来一个单个Tile??core_width 列数据，最终到底映射到了内存系统的几个Bank中，此后所有的2D Weight都以按照这种交织方式完成映射过程
    // 向四个BankGroup，每个BankGroup中的Bank完成填充；如果有多个rank，则以每一个Tile 交替向两个Rank中填充，按列向BankGroup，再向Bank中进行填充

    return allocate_row_index;
}


std::vector<uint32_t> MyAddressAllocator::weight_2D_allocate_in_IANUS(std::vector<uint32_t> dims, uint32_t precision) {
    assert(dims.size() == 2);
    uint32_t K = dims[0];
    uint32_t N = dims[1];

    // 每一个Bank中连续存放一列数据，首先计算出一次迭代的column??
    uint32_t column_allocate_iteration = std::ceil(static_cast<double>(N)/ MyAddressAllocator::total_banks);
    uint32_t required_rows_per_column = std::ceil(static_cast<double>(K * precision) / (dram_burst_size * BL_num_per_row));

    std::vector<uint32_t> allocate_row_index;
    for (uint32_t iteration_index=0; iteration_index < column_allocate_iteration; iteration_index++) {
        allocate_row_index.push_back(base_row);
        base_row += required_rows_per_column;
    }
    return allocate_row_index;
    /*
    uint32_t IANUS_tile_columns = dram_channels * ranks * bankgroups * banks;
    uint32_t req_rows_per_IANUS_tile = std::ceil( (double)K * precision / (dram_burst_size * BL_num_per_row));
    spdlog::info("Per IANUS weight tile occupy {} rows", req_rows_per_IANUS_tile);
    std::vector<uint32_t> allocate_row_index;
    uint32_t allocate_iteration = std::ceil(((double)N / IANUS_tile_columns));
    for (uint32_t i=0; i < allocate_iteration; i++) {
        allocate_row_index.push_back(base_row);
        base_row = base_row + req_rows_per_IANUS_tile;
    }
    return allocate_row_index;
    */
}


addr_type MyAddressAllocator::get_sequence_address(uint64_t burst_offset, uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, uint32_t row, uint32_t column) {
    // calculate current index for data fetch
    uint32_t col_offset = (column + burst_offset) % BL_num_per_row;
    uint32_t allocate_row_num = (column + burst_offset) / BL_num_per_row;

    // inner_row_loop offset
    uint32_t inner_row_offset = (row_loop_inner + allocate_row_num) % row_loop_size[inner_row_loop];
    uint32_t allocate_row_middle = (row_loop_inner + allocate_row_num) / row_loop_size[inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (row_loop_middle + allocate_row_middle) % row_loop_size[middle_row_loop];
    uint32_t allocate_row_outer = (row_loop_middle + allocate_row_middle) / row_loop_size[middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (row_loop_outer + allocate_row_outer) % row_loop_size[outer_row_loop];
    uint32_t allocate_row = (row_loop_outer + allocate_row_outer) / row_loop_size[outer_row_loop];

    // row offset
    uint32_t row_offset = row + allocate_row;

    auto addr = make_address(outer_row_offset, middle_row_offset, inner_row_offset, row_offset, col_offset, 0);

    return addr;
}


addr_type MyAddressAllocator::get_sequence_address(uint64_t burst_offset, uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, std::vector<uint32_t> allocated_rows, uint32_t column) {
    uint32_t col_offset = (column + burst_offset) % BL_num_per_row;
    uint32_t allocate_row_num = (column + burst_offset) / BL_num_per_row;

    // inner_row_loop offset
    uint32_t inner_row_offset = (row_loop_inner + allocate_row_num) % row_loop_size[inner_row_loop];
    uint32_t allocate_row_middle = (row_loop_inner + allocate_row_num) / row_loop_size[inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (row_loop_middle + allocate_row_middle) % row_loop_size[middle_row_loop];
    uint32_t allocate_row_outer = (row_loop_middle + allocate_row_middle) / row_loop_size[middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (row_loop_outer + allocate_row_outer) % row_loop_size[outer_row_loop];
    uint32_t row_offset = (row_loop_outer + allocate_row_outer) / row_loop_size[outer_row_loop];

    // row offset
    uint32_t row_index = allocated_rows[row_offset];

    return make_address(outer_row_offset, middle_row_offset, inner_row_offset, row_index, col_offset, 0);
}


addr_type MyAddressAllocator::get_sequence_address_pim(uint64_t burst_offset, uint32_t row, uint32_t column) {
    uint32_t col_index = (column + burst_offset) % BL_num_per_row;
    uint32_t row_offset = (column + burst_offset) / BL_num_per_row;
    uint32_t row_index = row + row_offset;
    auto addr = make_address(0, 0, 0, row_index, col_index, 0);
    return addr;
}



addr_type MyAddressAllocator::get_sequence_address_ianus(uint64_t burst_offset, uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, uint32_t row, uint32_t column) {

    uint32_t col_offset = (column + burst_offset) % BL_num_per_row;
    uint32_t allocate_row_num = (column + burst_offset) / BL_num_per_row;

    uint32_t row_index = allocate_row_num / total_banks;
    uint32_t channel_index = (allocate_row_num % total_banks) / banks_per_channel;
    uint32_t allocate_row_num_channel_offset = allocate_row_num % banks_per_channel;

    // inner_row_loop offset
    uint32_t inner_row_offset = (row_loop_inner + allocate_row_num_channel_offset) % row_loop_size[inner_row_loop];
    uint32_t allocate_row_middle = (row_loop_inner + allocate_row_num_channel_offset) / row_loop_size[inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (row_loop_middle + allocate_row_middle) % row_loop_size[middle_row_loop];
    uint32_t allocate_row_outer = (row_loop_middle + allocate_row_middle) / row_loop_size[middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (row_loop_outer + allocate_row_outer) % row_loop_size[outer_row_loop];
    uint32_t row_offset = (row_loop_outer + allocate_row_outer) / row_loop_size[outer_row_loop];

    return make_address(outer_row_offset, middle_row_offset, inner_row_offset, row_index, col_offset, channel_index);
}



addr_type MyAddressAllocator::get_sequence_address_pim(uint64_t burst_offset, std::vector<uint32_t> allocated_rows, uint32_t column) {
    uint32_t col_index = (column + burst_offset) % BL_num_per_row;
    uint32_t row_offset = (column + burst_offset) / BL_num_per_row;
    uint32_t row_index = allocated_rows[row_offset];
    auto addr = make_address(0, 0, 0, row_index, col_index, 0);
    return addr;
}

// reserve the activation space after weight allocation
bool MyAddressAllocator::activation_malloc() {
    assert(activation_space_malloced == false);
    activation_space_malloced = true;

    row_align();
    // Begin of Activation
    activation_start_column = base_column;
    activation_start_row = base_row;
    act_start_inner_row_loop = base_inner_row_loop;
    act_start_middle_row_loop = base_middle_row_loop;
    act_start_outer_row_loop = base_outer_row_loop;

    act_column = base_column;
    act_row = base_row;
    act_inner_row_loop = base_inner_row_loop;
    act_middle_row_loop = base_middle_row_loop;
    act_outer_row_loop = base_outer_row_loop;

    allocate_in_sequence(activation_buf_size);

    for (uint32_t i=0; i<cache_row; i++) {
        activation_space.push_back(i);
    }

    // Begin of KV Cache after Activation
    row_align();
    cache_column = base_column;
    cache_row = base_row;
    cache_inner_row_loop = base_inner_row_loop;
    cache_middle_row_loop = base_middle_row_loop;
    cache_outer_row_loop = base_outer_row_loop;

    base_column = act_column;
    base_row = act_row;
    base_inner_row_loop = act_inner_row_loop;
    base_middle_row_loop = act_middle_row_loop;
    base_outer_row_loop = act_outer_row_loop;

    return true;
}

bool MyAddressAllocator::activation_refresh() {
    assert(activation_space_malloced == true);
    // refresh activation location after each transformer block
    act_column = activation_start_column;
    act_row = activation_start_row;
    act_inner_row_loop = act_start_inner_row_loop;
    act_middle_row_loop = act_start_middle_row_loop;
    act_outer_row_loop = act_start_outer_row_loop;
    return true;
}

// allocate the activation
std::vector<uint32_t> MyAddressAllocator::activation_allocate(std::vector<uint32_t> dims, uint32_t precision) {
    assert(activation_space_malloced == true);
    auto allocate_row_indexes = activation_allocate_in_sequence(dims, precision);
    return allocate_row_indexes;
}

std::vector<uint32_t> MyAddressAllocator::activation_allocate_in_sequence(std::vector<uint32_t> dims, uint32_t precision) {
    uint64_t size = precision;
    if (dims.size() == 1) {
        for (auto dim : dims) {
            size *= dim;
        }
    }
    else if (dims.size() == 2 && dims[0] == 1) {
        for (auto dim : dims) {
            size *= dim;
        }
    }
    else { // dims.size() == 2 or higher dims size = 3, flatten to 2D dim
        dims.back() = std::ceil(static_cast<double>(dims[0]) / (dram_burst_size/precision)) * (dram_burst_size/precision);  // align the last dim to channel burst size
        uint32_t dim = 1;
        for (size_t i = 0; i + 1 < dims.size(); i++) {
            dim *= dims[i];
        }
        dim = std::ceil(static_cast<double>(dim)/dram_channels) * dram_channels;
        size = size * dim * dims.back();
    }

    uint32_t allocate_column_num = ceil(size/memory_burst_size);  // The required burst times for allocated data
    uint32_t col_offset = (act_column + allocate_column_num) % BL_num_per_row; // the new col offset
    uint32_t allocate_row_num = (act_column + allocate_column_num) / BL_num_per_row;

    // Calculate the loop of Bank, BankGroup and Rank, allocate the memory space
    // inner_row_loop offset
    uint32_t inner_row_offset = (act_inner_row_loop + allocate_row_num) % row_loop_size[inner_row_loop];  // inner_loop
    uint32_t allocate_row_middle = (act_inner_row_loop + allocate_row_num) / row_loop_size[inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (act_middle_row_loop + allocate_row_middle) % row_loop_size[middle_row_loop];  // middle_loop
    uint32_t allocate_row_outer = (act_middle_row_loop + allocate_row_middle) / row_loop_size[middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (act_outer_row_loop + allocate_row_outer) % row_loop_size[outer_row_loop];  // outer_loop
    uint32_t allocate_row = (act_outer_row_loop + allocate_row_outer) / row_loop_size[outer_row_loop];

    // row offset
    uint32_t row_offset = act_row + allocate_row;

    std::vector<uint32_t> allocate_row_indexes;
    if (allocate_row == 0) {
        allocate_row_indexes.push_back(row_offset);
    }
    else {
        for (uint32_t i=0; i<allocate_row; i++) {
            allocate_row_indexes.push_back(act_row);
            act_row++;
        }
    }

    act_row = row_offset;
    act_column = col_offset;
    act_inner_row_loop = inner_row_offset;
    act_middle_row_loop = middle_row_offset;
    act_outer_row_loop = outer_row_offset;

    return allocate_row_indexes;
}

std::vector<std::vector<uint32_t>> MyAddressAllocator::kcache_allocate(std::vector<uint32_t> dims, uint32_t precision) {
    std::vector<std::vector<uint32_t>> kcache_allocate_rows;
    uint32_t Lin = dims[0];
    // spdlog::info("The allocated K cache with Lin = {}", Lin);
    if (allocation_scheme == AllocationScheme::NeuPIM) {
        // uint32_t ch = 0;
        // addr_type row = _rows[ch]->front();
        allocate_in_sequence(kv_cache_entry_size * h_kv * d_k * precision);
    }
    else if (allocation_scheme == AllocationScheme::NPU) {
        uint32_t entry_num_per_head = std::ceil(static_cast<double>(Lin)/kv_cache_entry_size);
        uint32_t allocated_page_num = entry_num_per_head * h_kv;

        // 循环完成对于这些的分配内容，每次分配的row的数量基于total Bank获得, cache_row, cache_outer_row_loop, cache_middle_row_loop, cache_inner_row_loop;
        std::vector<uint32_t> allocate_row_indexes;
        for (uint32_t row_offset=0; row_offset< std::ceil(static_cast<double>(allocated_page_num) / total_banks); row_offset++) {
            allocate_row_indexes.push_back(cache_row);
            cache_row++;
        }
        kcache_allocate_rows.push_back(allocate_row_indexes);
    }
    else if (allocation_scheme == AllocationScheme::DASH or allocation_scheme == AllocationScheme::IANUS) {
        // Calculate the number of rows in each bank required to accommodate KCache at the current interleaving granularity,
        // and allocate rows that are rounded up. If they are exactly integers, add 1 to ensure there is new allocation space
        uint32_t allocated_rows = Lin / KCache_rows_per_allocation + 1;
        // spdlog::info("{} rows are allocated for each head of K Cache", allocated_rows);
        for (uint32_t i=0; i<h_kv; i++) {  // h_kv is the number of KV cache heads
            uint32_t head_iteration_index = i / allocated_KCache_head_per_iteration;
            uint32_t row_offset = cache_row + head_iteration_index * allocated_rows;
            std::vector<uint32_t> allocate_row_indexes;
            for (uint32_t j=0; j<allocated_rows; j++) {
                allocate_row_indexes.push_back(row_offset + j);
            }
            kcache_allocate_rows.push_back(allocate_row_indexes);
        }
        assert(kcache_allocate_rows.size() == h_kv and kcache_allocate_rows[0].size() == allocated_rows);
        cache_row = cache_row + std::ceil((double)h_kv / allocated_KCache_head_per_iteration) * allocated_rows;
    }
    return kcache_allocate_rows;
}


std::vector<std::vector<uint32_t>> MyAddressAllocator::vcache_allocate(std::vector<uint32_t> dims, uint32_t precision){
    std::vector<std::vector<uint32_t>> vcache_allocate_rows;
    uint32_t Lin = dims[0];

    if (allocation_scheme == AllocationScheme::NeuPIM) {
        uint32_t ch = 0;
        // addr_type row = _rows[ch]->front();
        allocate_in_sequence(kv_cache_entry_size * h_kv * d_k * precision_cache);
        _rows[ch]->pop_front();
    }
    else if (allocation_scheme == AllocationScheme::NPU) {
        const uint32_t vcache_entry_size =
            columns * DQ_width / 8 / (d_k * precision);
        if (vcache_entry_size != kv_cache_entry_size) {
            throw std::runtime_error(fmt::format(
                "NPU K/V cache entry-size mismatch: initialized {}, "
                "VCache requires {}",
                kv_cache_entry_size, vcache_entry_size));
        }
        uint32_t entry_num_per_head =
            std::ceil(static_cast<double>(Lin) / vcache_entry_size);
        uint32_t allocated_page_num = entry_num_per_head * h_kv;
        std::vector<uint32_t> allocate_row_indexes;
        for (uint32_t row_offset=0; row_offset< std::ceil(static_cast<double>(allocated_page_num) / total_banks); row_offset++) {
            allocate_row_indexes.push_back(cache_row);
            cache_row++;
        }
        vcache_allocate_rows.push_back(allocate_row_indexes);
    }
    else if (allocation_scheme == AllocationScheme::DASH or allocation_scheme == AllocationScheme::IANUS) {
        uint32_t allocation_burst_times = std::ceil((double)Lin/VCache_burst_row_unit) * VCache_columns_per_bank;
        uint32_t allocated_rows = allocation_burst_times / BL_num_per_row + 1;
        // available_VCache_rows = allocated_rows * BL_num_per_row / VCache_columns_per_bank  * VCache_burst_row_unit - Lin;
        for (uint32_t i=0; i<h_kv; i++) {
            uint32_t head_iteration_index = i / allocated_VCache_head_per_iteration;
            uint32_t row_offset = cache_row + head_iteration_index * allocated_rows;
            std::vector<uint32_t> allocate_row_indexes;
            for (uint32_t j=0; j<allocated_rows; j++) {
                allocate_row_indexes.push_back(row_offset + j);
            }
            vcache_allocate_rows.push_back(allocate_row_indexes);
        }
        assert(vcache_allocate_rows.size() == h_kv and vcache_allocate_rows[0].size() == allocated_rows);
        cache_row = cache_row + std::ceil((double)h_kv / allocated_VCache_head_per_iteration) * allocated_rows;
    }
    return vcache_allocate_rows;
}


uint32_t MyAddressAllocator::kvcache_append(std::vector<std::vector<uint32_t>>* allocated_kvcache_rows, TensorType tensor_type) {
    if (allocation_scheme == AllocationScheme::NPU) {
        spdlog::info("In NPU Allocation scheme, the KVCache append operation is A");
    }
    else if (allocation_scheme == AllocationScheme::IANUS or allocation_scheme == AllocationScheme::DASH) {
        // input the allocated KCache rows, return the new KV cache capacity
        if (tensor_type == TensorType::KCache) {
            for (uint32_t head_index=0; head_index<h_kv; head_index++) {
                uint32_t head_iteration_index = head_index / allocated_KCache_head_per_iteration;
                uint32_t row_offset = cache_row + head_iteration_index * KVCache_allocate_row;
                (*allocated_kvcache_rows)[head_index].push_back(row_offset);
            }
            cache_row = cache_row + std::ceil(static_cast<double>(h_kv) / allocated_KCache_head_per_iteration) * KVCache_allocate_row;
            return KCache_rows_per_allocation * KVCache_allocate_row;
        }
        else if (tensor_type == TensorType::VCache) {
            for (uint32_t head_index=0; head_index<h_kv; head_index++) {
                uint32_t head_iteration_index = head_index / allocated_VCache_head_per_iteration;
                uint32_t row_offset = cache_row + head_iteration_index * KVCache_allocate_row;
                (*allocated_kvcache_rows)[head_index].push_back(row_offset);
            }
            cache_row = cache_row + std::ceil(static_cast<double>(h_kv) / allocated_VCache_head_per_iteration) * KVCache_allocate_row;
            return VCache_rows_per_allocation * KVCache_allocate_row;
        }
        else {
            throw std::runtime_error("Invalid Tensor Type");
        }
    }
    return 0;
}


bool MyAddressAllocator::allocate_in_sequence(uint64_t size) {
    spdlog::info("The allocated size is {}", size);
    spdlog::info("The base index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", base_row, outer_row_loop, base_outer_row_loop,
        middle_row_loop, base_middle_row_loop, inner_row_loop, base_inner_row_loop, base_column);
    uint32_t allocate_column_num = ceil(size/memory_burst_size);  // The required burst times for allocated data
    uint32_t col_offset = (base_column + allocate_column_num) % BL_num_per_row; // the new col offset
    uint32_t allocate_row_num = (base_column + allocate_column_num) / BL_num_per_row;
    spdlog::info("The allocate size corresponding to {} times burst, {} row", allocate_column_num, allocate_row_num);
    spdlog::info("The new column offset is {} from the basic column offset {}", col_offset, base_column);

    // 计算在Bank, BankGroup, Rank
    spdlog::info("The loop allocate for row allocation is {}, {}, {}",outer_row_loop, middle_row_loop,inner_row_loop);
    // inner_row_loop offset
    uint32_t inner_row_offset = (base_inner_row_loop + allocate_row_num) % row_loop_size[inner_row_loop];  // new inner_loop index
    uint32_t allocate_row_middle = (base_inner_row_loop + allocate_row_num) / row_loop_size[inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (base_middle_row_loop + allocate_row_middle) % row_loop_size[middle_row_loop];  // new middle_loop index
    uint32_t allocate_row_outer = (base_middle_row_loop + allocate_row_middle) / row_loop_size[middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (base_outer_row_loop + allocate_row_outer) % row_loop_size[outer_row_loop];  // new outer_loop index
    uint32_t allocate_row = (base_outer_row_loop + allocate_row_outer) / row_loop_size[outer_row_loop];

    // row offset
    uint32_t row_offset = base_row + allocate_row;

    spdlog::info("The new index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", row_offset, outer_row_loop, outer_row_offset,
        middle_row_loop, middle_row_offset, inner_row_loop, inner_row_offset, col_offset);

    // update the activation information
    base_row = row_offset;
    base_column = col_offset;
    base_inner_row_loop = inner_row_offset;
    base_middle_row_loop = middle_row_offset;
    base_outer_row_loop = outer_row_offset;

    return true;
}


bool MyAddressAllocator::malloc_kvcache_space() {
    assert(activation_space_malloced == true);

    // KV Cache space reservation
    base_column = cache_column;
    base_row = cache_row;
    base_inner_row_loop = cache_inner_row_loop;
    base_middle_row_loop = cache_middle_row_loop;
    base_outer_row_loop = cache_outer_row_loop;

    if (allocation_scheme == AllocationScheme::NPU) {
        // NPU scheme，calculation KV Cache scales and reserve the memory space
        uint64_t kv_cache_max_size = max_active_reqs * max_seq_len * h_kv * d_k * precision_cache;
        for (uint32_t i=0; i<layers; i++) {
            allocate_in_sequence(kv_cache_max_size);  // Kcache
            allocate_in_sequence(kv_cache_max_size);  // Vcache
        }
    }
    else if (allocation_scheme == AllocationScheme::NeuPIM) {
        uint32_t free_rows_size = rows - base_row;  // The free rows
        // _rows stores the free rows of each channel
        for (uint32_t i = 0; i < dram_channels; ++i) {
            _rows.push_back(std::make_shared<std::deque<uint64_t>>());
            for (uint32_t j = 0; j < free_rows_size; ++j) {
                if (base_row + j < rows) {
                    _rows[i]->push_back(base_row + j);
                }
            }
        }
    }
    else if (allocation_scheme == AllocationScheme::DASH){
        spdlog::info("DASH allocation scheme: DASH");
    }
    return true;
}

bool MyAddressAllocator::check_addrs(const std::vector<addr_type>& addrs) {
    spdlog::info("********************** Check addrs **********************");
    spdlog::info("Total {} address are checked", addrs.size());
    for (auto addr : addrs) {
        spdlog::info("current addr: {}, for channel {}, Rank {}, BankGroup {}, Bank {}, Row {}, Column {}",
            addr, get_channel_index(addr), get_rank_index(addr), get_bankgroup_index(addr), get_bank_index(addr), get_row_index(addr), get_col_index(addr));
    }
    return true;
}


addr_type MyAddressAllocator::activation_address_allocate(uint32_t size) {
    uint32_t alignment = AddressConfig::alignment;
    addr_type result = _top_addr;
    _top_addr += size;
    if (_top_addr & (alignment - 1)) {
        _top_addr += alignment - (_top_addr & (alignment - 1));
    }
    return result;
}

addr_type MyAddressAllocator::kvcache_address_allocate(uint32_t size) {

    return size;
}

addr_type MyAddressAllocator::make_address(uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, uint32_t row, uint32_t column, uint32_t channel) {
    // make a address based on the row loon rank indexes
    assert(channel < dram_channels);
    assert(column < BL_num_per_row);
    assert(row < rows);
    assert(row_loop_inner < row_loop_size.at(inner_row_loop));
    assert(row_loop_middle < row_loop_size.at(middle_row_loop));
    assert(row_loop_outer < row_loop_size.at(outer_row_loop));
    addr_type address = 0;
    address |= (channel & mask["ch"]) << field_pos["ch"];
    address |= (column & mask["co"]) << field_pos["co"];
    address |= (row_loop_inner & mask[inner_row_loop])  << field_pos[inner_row_loop];
    address |= (row_loop_middle & mask[middle_row_loop])  << field_pos[middle_row_loop];
    address |= (row_loop_outer & mask[outer_row_loop]) << field_pos[outer_row_loop];
    address |= (row & mask["ro"]) << field_pos["ro"];
    // address <<= shift_bits;

    return address;
}

addr_type MyAddressAllocator::make_address() {
    // make a address based on the current based indexes
    assert(base_column < BL_num_per_row);
    assert(base_row < rows);
    assert(base_inner_row_loop < row_loop_size.at(inner_row_loop));
    assert(base_middle_row_loop < row_loop_size.at(middle_row_loop));
    assert(base_outer_row_loop < row_loop_size.at(outer_row_loop));
    addr_type address = 0;
    uint32_t channel = 0;
    address |= (channel & mask["ch"]) << field_pos["ch"];
    address |= (base_column & mask["co"]) << field_pos["co"];
    address |= (base_inner_row_loop & mask[inner_row_loop])  << field_pos[inner_row_loop];
    address |= (base_middle_row_loop & mask[middle_row_loop])  << field_pos[middle_row_loop];
    address |= (base_outer_row_loop & mask[outer_row_loop]) << field_pos[outer_row_loop];
    address |= (base_row & mask["ro"]) << field_pos["ro"];
    // address <<= shift_bits;
    return address;
}


addr_type MyAddressAllocator::make_address_by_index(uint32_t rank_index, uint32_t bankgroup_index, uint32_t bank_index, uint32_t row_index, uint32_t column_index, uint32_t channel_index) {
    // make a address based on the specific indexes
    assert(channel_index < dram_channels);
    assert(rank_index < ranks);
    assert(bankgroup_index < bankgroups);
    assert(bank_index < banks);
    assert(row_index < rows);
    assert(column_index < BL_num_per_row);
    addr_type addr = 0;
    addr |= (channel_index & mask["ch"]) << field_pos["ch"];
    addr |= (rank_index & mask["ra"]) << field_pos["ra"];
    addr |= (bankgroup_index & mask["bg"]) << field_pos["bg"];
    addr |= (bank_index & mask["ba"]) << field_pos["ba"];
    addr |= (row_index & mask["ro"]) << field_pos["ro"];
    addr |= (column_index & mask["co"]) << field_pos["co"];
    // addr <<= shift_bits;
    return addr;
}

addr_type MyAddressAllocator::make_address_by_index_with_shift(uint32_t rank_index, uint32_t bankgroup_index, uint32_t bank_index, uint32_t row_index, uint32_t column_index, uint32_t channel_index) {
    // make address based on the specific indexes with shift
    assert(channel_index < dram_channels);
    assert(rank_index < ranks);
    assert(bankgroup_index < bankgroups);
    assert(bank_index < banks);
    assert(row_index < rows);
    assert(column_index < BL_num_per_row);
    addr_type addr = 0;
    addr |= (channel_index & mask["ch"]) << field_pos["ch"];
    addr |= (rank_index & mask["ra"]) << field_pos["ra"];
    addr |= (bankgroup_index & mask["bg"]) << field_pos["bg"];
    addr |= (bank_index & mask["ba"]) << field_pos["ba"];
    addr |= (row_index & mask["ro"]) << field_pos["ro"];
    addr |= (column_index & mask["co"]) << field_pos["co"];
    addr <<= shift_bits;
    return addr;
}


addr_type MyAddressAllocator::add_channel_index(addr_type addr, uint32_t channel_index) {
    assert(channel_index < dram_channels);
    addr |= (channel_index & mask["ch"]) << field_pos["ch"];
    return addr;
}

uint32_t MyAddressAllocator::get_channel_index(addr_type addr) {
    uint32_t channel_index = (addr >> field_pos["ch"]) & mask["ch"];
    return channel_index;
}

uint32_t MyAddressAllocator::get_rank_index(addr_type addr) {
    uint32_t rank_index = (addr >> field_pos["ra"]) & mask["ra"];
    return rank_index;
}

uint32_t MyAddressAllocator::get_bankgroup_index(addr_type addr) {
    uint32_t bankgroup_index = (addr >> field_pos["bg"]) & mask["bg"];
    return bankgroup_index;
}

uint32_t MyAddressAllocator::get_bank_index(addr_type addr) {
    uint32_t bank_index = (addr >> field_pos["ba"]) & mask["ba"];
    return bank_index;
}

uint32_t MyAddressAllocator::get_row_index(addr_type addr) {
    uint32_t row_index = (addr >> field_pos["ro"]) & mask["ro"];
    return row_index;
}

uint32_t MyAddressAllocator::get_col_index(addr_type addr) {
    uint32_t col_index = (addr >> field_pos["co"]) & mask["co"];
    return col_index;
}


uint32_t MyAddressAllocator::get_data_offset(addr_type addr, uint32_t offset_size) {
    return addr / offset_size;
}

std::vector<PIMHashAddressing::AddressGroup>
PIMHashAddressing::group_comp_addresses(
    const std::vector<addr_type>& addresses,
    uint32_t hash_granularity_bytes,
    uint32_t dram_burst_size_bytes) {
    if (hash_granularity_bytes == 0 || dram_burst_size_bytes == 0) {
        throw std::invalid_argument(
            "PIM hash granularity and DRAM burst size must be non-zero");
    }
    if (hash_granularity_bytes % dram_burst_size_bytes != 0) {
        throw std::invalid_argument(
            "PIM hash granularity must be divisible by the DRAM burst size");
    }

    const addr_type hash_granularity_address_units =
        hash_granularity_bytes / dram_burst_size_bytes;
    std::vector<AddressGroup> groups;
    groups.reserve(addresses.size());

    for (const addr_type address : addresses) {
        const addr_type aligned_start_address =
            (address / hash_granularity_address_units) *
            hash_granularity_address_units;
        if (groups.empty() ||
            groups.back().aligned_start_address != aligned_start_address) {
            groups.push_back(AddressGroup{aligned_start_address, 1});
        } else {
            if (groups.back().burst_count ==
                std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error(
                    "PIM hash address group exceeds the supported burst count");
            }
            ++groups.back().burst_count;
        }
    }
    return groups;
}


addr_type MyAddressAllocator::make_address_with_channel(uint32_t inner_row_loop_index, uint32_t middle_row_loop_index, uint32_t outer_row_loop_index, uint32_t row, uint32_t col, uint32_t channel) {
    assert(channel < dram_channels);
    assert(col < BL_num_per_row);
    assert(row < rows);
    assert(inner_row_loop_index < row_loop_size.at(inner_row_loop));
    assert(middle_row_loop_index < row_loop_size.at(middle_row_loop));
    assert(outer_row_loop_index < row_loop_size.at(outer_row_loop));
    addr_type addr = 0;

    addr |= (channel & mask["ch"]) << field_pos["ch"];
    addr |= (inner_row_loop_index & mask[inner_row_loop]) << field_pos[inner_row_loop];
    addr |= (middle_row_loop_index & mask[middle_row_loop]) << field_pos[middle_row_loop];
    addr |= (outer_row_loop_index & mask[outer_row_loop]) << field_pos[outer_row_loop];
    addr |= (row & mask["ro"]) << field_pos["ro"];
    addr |= (col & mask["co"]) << field_pos["co"];
    addr <<= shift_bits;
    return addr;
}


addr_type MyAddressAllocator::weight_address_allocate(uint32_t size) {
    addr_type unit = memory_burst_size;
    addr_type result = _top_addr;
    _top_addr += (size + unit - 1) / unit;
    return result;
}


addr_type MyAddressAllocator::get_next_aligned_addr() {
    ast(_top_addr > 0);
    return AddressConfig::align(_top_addr) + AddressConfig::alignment;
}


addr_type MyAddressAllocator::addr_align(addr_type addr) {
    // Discard the unused columns of current bank row and allocate a new row to achieve address alignment
    return addr - (addr & (BL_num_per_row - 1)) + BL_num_per_row;
}


addr_type MyAddressAllocator::addr_row_align(addr_type addr) {
    return addr - (addr & (BL_num_per_row * row_loop_size[inner_row_loop] * row_loop_size[middle_row_loop] * row_loop_size[outer_row_loop] - 1))
    + (BL_num_per_row * row_loop_size[inner_row_loop] * row_loop_size[middle_row_loop] * row_loop_size[outer_row_loop]);
}


bool MyAddressAllocator::column_align() {
    if (base_column == 0) {
        // spdlog::info("Current column is aligned");
    }
    else {
        spdlog::info("Align the memory space of column");
        spdlog::info("before column align, the base index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", base_row, outer_row_loop, base_outer_row_loop,
            middle_row_loop, base_middle_row_loop, inner_row_loop, base_inner_row_loop, base_column);
        base_column = 0;
        base_inner_row_loop = (base_inner_row_loop + 1) % row_loop_size[inner_row_loop];
        if (base_inner_row_loop+1 >= row_loop_size[inner_row_loop]) {
            base_middle_row_loop = (base_middle_row_loop + 1) % row_loop_size[middle_row_loop];
            if (base_middle_row_loop + 1 >= row_loop_size[middle_row_loop]) {
                base_outer_row_loop = (base_outer_row_loop + 1) % row_loop_size[outer_row_loop];
                if (base_outer_row_loop + 1 >= row_loop_size[outer_row_loop]) {
                    base_row = base_row + 1;
                }
            }
        }
        spdlog::info("after column align, the base index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", base_row, outer_row_loop, base_outer_row_loop,
        middle_row_loop, base_middle_row_loop, inner_row_loop, base_inner_row_loop, base_column);
    }
    return true;
}

bool MyAddressAllocator::row_align() {
    if (base_column + base_inner_row_loop + base_middle_row_loop + base_outer_row_loop == 0) {
        // spdlog::info("Current column is aligned");
    }
    else {
        spdlog::info("Align the memory space of row");
        spdlog::info("before row align, the base index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", base_row, outer_row_loop, base_outer_row_loop,
            middle_row_loop, base_middle_row_loop, inner_row_loop, base_inner_row_loop, base_column);
        base_column = 0;
        base_inner_row_loop = 0;
        base_middle_row_loop = 0;
        base_outer_row_loop = 0;
        base_row +=1;
        spdlog::info("after row align, the base index is row:{}, {}:{}, {}:{}, {}:{}, column:{}", base_row, outer_row_loop, base_outer_row_loop,
            middle_row_loop, base_middle_row_loop, inner_row_loop, base_inner_row_loop, base_column);
    }
    return true;
}


uint32_t MyAddressAllocator::get_precision(TensorType tensor_type){
    if (tensor_type == TensorType::WGT) {
        return precision_weight;
    }
    else if (tensor_type == TensorType::ACT) {
        return precision_activation;
    }
    else if (tensor_type == TensorType::KCache or tensor_type == TensorType::VCache) {
        return precision_cache;
    }
    else {
        return precision_psum;
    }
}


std::string Instruction::repr() {
    std::string ret;
    switch (opcode) {
        case (Opcode::MOVIN):
            ret += "MOVIN";
            break;
        case (Opcode::MOVOUT):
            ret += "MOVOUT";
            break;
        case (Opcode::GEMM_PRELOAD):
            ret += "GEMM_PRELOAD";
            break;
        case (Opcode::GEMM):
            ret += "GEMM";
            break;
        case (Opcode::BAR):
            ret += "BAR";
            break;
        case (Opcode::LAYERNORM):
            ret += "LAYERNORM";
            break;
        case (Opcode::RMSNORM):
            ret += "RMSNORM";
            break;
        case (Opcode::ROPE):
            ret += "ROPE";
            break;
        case (Opcode::DATA_CONVERT):
            ret += "DATA_CONVERT";
            break;
        case (Opcode::GELU):
            ret += "GELU";
            break;
        case (Opcode::SILU):
            ret += "SILU";
            break;
        case (Opcode::SOFTMAX):
            ret += "SOFTMAX";
            break;
        case (Opcode::ADD):
            ret += "ADD";
            break;
        case (Opcode::MUL):
            ret += "MUL";
            break;
        case (Opcode::DUMMY):
            ret += "DUMMY";
            break;
        default:
            ret += "UNSUPPORTED";
            break;
    }
    ret += " / src_addrs.size() : ";
    ret += std::to_string(src_addrs.size());
    ret += " / dest_addrs : ";
    //ret += to_hex(dest_addr);
    ret += dest_addr;
    return ret;
}

std::string Instruction::print_optype() {
    std::string ret;
    switch (opcode) {
        case (Opcode::MOVIN):
            ret += "MOVIN";
            break;
        case (Opcode::MOVOUT):
            ret += "MOVOUT";
            break;
        case (Opcode::GEMM_PRELOAD):
            ret += "GEMM_PRELOAD";
            break;
        case (Opcode::GEMM):
            ret += "GEMM";
            break;
        case (Opcode::BAR):
            ret += "BAR";
            break;
        case (Opcode::LAYERNORM):
            ret += "LAYERNORM";
            break;
        case (Opcode::RMSNORM):
            ret += "RMSNORM";
            break;
        case (Opcode::ROPE):
            ret += "ROPE";
            break;
        case (Opcode::DATA_CONVERT):
            ret += "DATA_CONVERT";
            break;
        case (Opcode::GELU):
            ret += "GELU";
            break;
        case (Opcode::SILU):
            ret += "SILU";
            break;
        case (Opcode::SOFTMAX):
            ret += "SOFTMAX";
            break;
        case (Opcode::ADD):
            ret += "ADD";
            break;
        case (Opcode::MUL):
            ret += "MUL";
            break;
        case (Opcode::DUMMY):
            ret += "DUMMY";
            break;
        default :
            ret += "UNSUPPORTED";
            break;
    }
    return ret;
}


std::string Tile::repr() {
    std::string ret;
    switch (status) {
        case Status::INITIALIZED:
            ret += "init ";
            break;
        case Status::RUNNING:
            ret += "run ";
            break;
        case Status::FINISH:
            ret += "fin ";
            break;
        case Status::BAR:
            ret += "bar ";
            break;
        case Status::EMPTY:
            ret += "emp ";
            break;
    }
    ret += optype + " ";
    ret += std::to_string(operation_id) + " ";

    return ret;
}


std::unique_ptr<MemoryAccess> TransToMemoryAccess(
    Instruction &inst, uint32_t size, uint32_t core_id,
    cycle_type start_cycle, int buffer_id, StagePlatform stage_platform) {
    MemoryAccessType req_type;
    switch (inst.opcode){
        case Opcode::PIM_HEADER:
            req_type = MemoryAccessType::P_HEADER;
            break;
        case Opcode::PIM_GWRITE:
            req_type = MemoryAccessType::GWRITE;
            break;
        case Opcode::PIM_COMP:
            req_type = MemoryAccessType::COMP;
            break;
        case Opcode::PIM_READRES:
            req_type = MemoryAccessType::READRES;
            break;
        case Opcode::PIM_COMPS_READRES:
            req_type = MemoryAccessType::COMPS_READRES;
            break;
        case Opcode::MOVIN:
            req_type = MemoryAccessType::READ;
            break;
        case Opcode::MOVOUT:
            req_type = MemoryAccessType::WRITE;
            break;
        default:
            req_type = MemoryAccessType::SIZE;
            spdlog::error("Fail to translate unknown Instruction to MemoryAccessType");
            break;
    }

    auto it = inst.src_addrs.begin();
    assert(it != inst.src_addrs.end());
    addr_type dram_addr = *it;

    auto mem_request = std::unique_ptr<MemoryAccess>(new MemoryAccess{
        .id = generate_mem_access_id(),
        .logical_dram_address = dram_addr,
        .dram_address = dram_addr,
        .spad_address = inst.dest_addr,
        .size = size,          //
        .logical_start_addr = (inst.logical_start_addr != 0 ? inst.logical_start_addr : dram_addr),
        .exec_len_bytes = (inst.exec_len_bytes != 0 ? inst.exec_len_bytes : inst.size),
        .req_type = req_type,  //
        .request = true,
        .core_id = core_id,
        .start_cycle = start_cycle,
        .buffer_id = buffer_id,
        .parent_tile = inst.parent_tile,
        .stage_platform = stage_platform,
    });
    return mem_request;
}

std::string memAccessTypeString(MemoryAccessType type) {
    switch (type) {
        case (MemoryAccessType::READ):
            return "READ";
        case (MemoryAccessType::WRITE):
            return "WRITE";
        case (MemoryAccessType::GWRITE):
            return "GWRITE";
        case (MemoryAccessType::COMP):
            return "COMP";
        case (MemoryAccessType::COMP_HASH):
            return "COMP_HASH";
        case (MemoryAccessType::READRES):
            return "READRES";
        case (MemoryAccessType::P_HEADER):
            return "P_HEADER";
        case (MemoryAccessType::COMPS_READRES):
            return "COMPS_READRES";
        default:
            throw std::invalid_argument(
                "Unknown MemoryAccessType: " +
                std::to_string(static_cast<int>(type)));
    }
}

std::string opcodeTypeString(Opcode opcode) {
    switch (opcode) {
        case (Opcode::MOVIN):
            return "MOVIN";
        case (Opcode::MOVOUT):
            return "MOVOUT";
        case (Opcode::PIM_HEADER):
            return "PIM_HEADER";
        case (Opcode::PIM_COMP):
            return "PIM_COMP";
        case (Opcode::PIM_GWRITE):
            return "PIM_GWRITE";
        case (Opcode::PIM_READRES):
            return "PIM_READRES";
        case (Opcode::PIM_COMPS_READRES):
            return "PIM_COMPS_READRES";
        default:
            return "Unknown";
    }
}

std::vector<std::unique_ptr<MemoryAccess>> MemoryAccess::from_instruction(Instruction &inst, uint32_t id, uint32_t size, MemoryAccessType req_type, bool request, uint32_t core_id,
                                                                  cycle_type start_cycle, int buffer_id, StagePlatform stage_platform) {
    // generate trace from the generated instructions, add the channel index when loading the Core,
    // where the number of src_add is the number of bursts required during the loading process
    std::unordered_set<addr_type> aligned_src_addrs;
    for (auto addr : inst.src_addrs) {
        for (uint32_t ch = 0; ch < MyAddressAllocator::dram_channels; ch++) {
            auto convert_address = MyAddressAllocator::add_channel_index(addr, ch);
            auto channel = MyAddressAllocator::get_channel_index(convert_address);
            assert(channel == ch);
            aligned_src_addrs.insert(convert_address);
        }
    }


    std::vector<std::unique_ptr<MemoryAccess>> ret;
    for (auto &addr : aligned_src_addrs) {
        req_count++;
        auto mem_access = std::unique_ptr<MemoryAccess>(new MemoryAccess{
            .id = id,
            .dram_address = addr,
            .spad_address = inst.dest_addr,
            .size = size,
            .req_type = req_type,
            .request = request,
            .core_id = core_id,
            .start_cycle = start_cycle,
            .buffer_id = buffer_id,
            .parent_tile = inst.parent_tile,
            .stage_platform = stage_platform,
        });
        ret.push_back(std::move(mem_access));
    }

    return ret;
}


std::vector<std::unique_ptr<MemoryAccess>> MemoryAccess::gen_trace_from_instruction(Instruction &inst, uint32_t id, uint32_t size, MemoryAccessType req_type, bool request, uint32_t core_id,
                                                                  cycle_type start_cycle, int buffer_id, StagePlatform stage_platform) {

    std::vector<std::unique_ptr<MemoryAccess>> ret;
    std::vector<addr_type> address_with_channel_index;

    if (inst.per_ch_inst) {
        for (auto addr : inst.src_addrs) {
            auto ch = MyAddressAllocator::get_channel_index(addr);
            req_count++;
            auto mem_access = std::unique_ptr<MemoryAccess>(new MemoryAccess{
                .id = ch,
                .dram_address = addr,
                .spad_address = inst.dest_addr,
                .size = size,
                .req_type = req_type,
                .request = request,
                .core_id = core_id,
                .start_cycle = start_cycle,
                .buffer_id = buffer_id,
                .parent_tile = inst.parent_tile,
                .stage_platform = stage_platform,
            });
            ret.push_back(std::move(mem_access));
        }
    }
    else {
        for (auto addr : inst.src_addrs) {
            for (uint32_t ch = 0; ch < MyAddressAllocator::dram_channels; ch++) {
                auto convert_address = MyAddressAllocator::add_channel_index(addr, ch);
                auto channel = MyAddressAllocator::get_channel_index(convert_address);
                assert(channel == ch);
                address_with_channel_index.push_back(convert_address);
                req_count++;
                auto mem_access = std::unique_ptr<MemoryAccess>(new MemoryAccess{
                    .id = ch,
                    .dram_address = convert_address,
                    .spad_address = inst.dest_addr,
                    .size = size,
                    .req_type = req_type,
                    .request = request,
                    .core_id = core_id,
                    .start_cycle = start_cycle,
                    .buffer_id = buffer_id,
                    .parent_tile = inst.parent_tile,
                    .stage_platform = stage_platform,
                });
                ret.push_back(std::move(mem_access));
            }
        }
    }
    return ret;
}


std::vector<std::unique_ptr<MemoryAccess>> MemoryAccess::gen_pim_trace_from_instruction(Instruction &inst, uint32_t id, uint32_t size, MemoryAccessType req_type, bool request, uint32_t core_id,
                                                                  cycle_type start_cycle, int buffer_id, StagePlatform stage_platform) {
    std::vector<std::unique_ptr<MemoryAccess>> ret;
    const uint32_t parallel_n = Config::system_config.pim_parallel_bank_accesses;
    // const bool pim_comp_bank_stagger = (parallel_n > 0 && req_type == MemoryAccessType::COMP && (parallel_n % MyAddressAllocator::dram_channels) == 0);
    for (auto addr : inst.src_addrs) {
        if (req_type == MemoryAccessType::COMP_HASH) {
            // spdlog::info("Add the channel and Bank index to all the COMP Hash Operation");
            const uint32_t row = MyAddressAllocator::get_row_index(addr);
            const uint32_t col = MyAddressAllocator::get_col_index(addr);

            for (uint32_t rank_index = 0; rank_index < MyAddressAllocator::ranks; rank_index++) {
                for (uint32_t bank_index = 0; bank_index < MyAddressAllocator::banks; bank_index++) {
                    for (uint32_t bankgroup_index = 0; bankgroup_index < MyAddressAllocator::bankgroups; bankgroup_index++) {
                        for (uint32_t channel_index = 0; channel_index < MyAddressAllocator::dram_channels; channel_index++) {
                            const addr_type dram_addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row, col, channel_index);
                            req_count++;
                            auto mem_access = std::unique_ptr<MemoryAccess>(new MemoryAccess{
                                .id = channel_index,
                                .logical_dram_address = dram_addr,
                                .dram_address = dram_addr,
                                .spad_address = inst.dest_addr,
                                .size = size,
                                .logical_start_addr = (inst.logical_start_addr != 0 ? inst.logical_start_addr : dram_addr),
                                .exec_len_bytes = (inst.exec_len_bytes != 0 ? inst.exec_len_bytes : inst.size),
                                .req_type = req_type,
                                .request = request,
                                .core_id = core_id,
                                .start_cycle = start_cycle,
                                .buffer_id = buffer_id,
                                .parent_tile = inst.parent_tile,
                                .stage_platform = stage_platform,
                            });
                            ret.push_back(std::move(mem_access));
                        }
                    }
                }
            }
            /*
            for (uint32_t ch = 0; ch < MyAddressAllocator::dram_channels; ch++) {
                uint32_t bank_linear = (ch * bank_step) % MyAddressAllocator::banks_per_channel;
                uint32_t bk = bank_linear;
                const uint32_t ba_s = bk % MyAddressAllocator::banks;
                bk /= MyAddressAllocator::banks;
                const uint32_t bg_s = bk % MyAddressAllocator::bankgroups;
                bk /= MyAddressAllocator::bankgroups;
                const uint32_t ra_s = bk % MyAddressAllocator::ranks;
                const addr_type dram_addr =
                    MyAddressAllocator::make_address_by_index(ra_s, bg_s, ba_s, row, col, ch);
                req_count++;
                auto mem_access = std::unique_ptr<MemoryAccess>(new MemoryAccess{
                    .id = ch,
                    .logical_dram_address = dram_addr,
                    .dram_address = dram_addr,
                    .spad_address = inst.dest_addr,
                    .size = size,
                    .logical_start_addr = (inst.logical_start_addr != 0 ? inst.logical_start_addr : dram_addr),
                    .exec_len_bytes = (inst.exec_len_bytes != 0 ? inst.exec_len_bytes : inst.size),
                    .req_type = req_type,
                    .request = request,
                    .core_id = core_id,
                    .start_cycle = start_cycle,
                    .buffer_id = buffer_id,
                    .parent_tile = inst.parent_tile,
                    .stage_platform = stage_platform,
                });
                ret.push_back(std::move(mem_access));
            }
            */
        }
        else {
            if (parallel_n > 0 && req_type == MemoryAccessType::COMP) {
                static bool warned = false;
                if (!warned) {
                    spdlog::warn(
                        "pim_parallel_bank_accesses={}: N%%dram_channels!=0 for PIM_COMP; using legacy add_channel_index path",
                        parallel_n);
                    warned = true;
                }
            }
            for (uint32_t ch = 0; ch < MyAddressAllocator::dram_channels; ch++) {
                auto convert_address = MyAddressAllocator::add_channel_index(addr, ch);
                auto channel = MyAddressAllocator::get_channel_index(convert_address);
                assert(channel == ch);
                req_count++;
                auto mem_access = std::unique_ptr<MemoryAccess>(new MemoryAccess{
                    .id = ch,
                    .logical_dram_address = convert_address,
                    .dram_address = convert_address,
                    .spad_address = inst.dest_addr,
                    .size = size,
                    .logical_start_addr = (inst.logical_start_addr != 0 ? inst.logical_start_addr : convert_address),
                    .exec_len_bytes = (inst.exec_len_bytes != 0 ? inst.exec_len_bytes : inst.size),
                    .req_type = req_type,
                    .request = request,
                    .core_id = core_id,
                    .start_cycle = start_cycle,
                    .buffer_id = buffer_id,
                    .parent_tile = inst.parent_tile,
                    .stage_platform = stage_platform,
                });
                ret.push_back(std::move(mem_access));
            }
        }
    }
    /* 这里是之前的写法
    std::vector<addr_type> address_with_channel_index;
    for (auto addr : inst.src_addrs) {
        for (uint32_t ch = 0; ch < MyAddressAllocator::dram_channels; ch++) {
            auto convert_address = MyAddressAllocator::add_channel_index(addr, ch);
            auto channel = MyAddressAllocator::get_channel_index(convert_address);
            assert(channel == ch);
            address_with_channel_index.push_back(convert_address);
            req_count++;
            MemoryAccess *mem_access = new MemoryAccess{
                .id = ch,
                .dram_address = convert_address,
                .spad_address = inst.dest_addr,
                .size = size,
                .req_type = req_type,
                .request = request,
                .core_id = core_id,
                .start_cycle = start_cycle,
                .buffer_id = buffer_id,
                .parent_tile = inst.parent_tile,
                .stage_platform = stage_platform,
            };
            ret.push_back(mem_access);
        }
    }
    */
    return ret;
}




void print_backtrace() {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    std::raise(SIGINT);
    exit(1);
}


void ast(bool cond) {
    if (cond) return;

    print_backtrace();
    spdlog::error("assertion failed");
    std::raise(SIGSEGV);
    // exit(-1);
}


std::string to_hex(uint32_t input) {
    std::stringstream addr_as_hex;
    addr_as_hex << std::hex << input;
    return addr_as_hex.str();
}



namespace EventDrivenParams{

    int activate_to_activate;
    int activate_to_activate_l;
    int activate_to_activate_s;
    int activate_to_precharge;
    int activate_to_read;
    int activate_to_write;

    int precharge_to_activate;
    int precharge_to_precharge;

    int refresh_to_activate;

    int read_to_read_l;
    int read_to_read_s;
    int read_to_read_o;
    int read_to_write;
    int read_to_write_o;
    int read_to_precharge;

    int write_to_read_l;
    int write_to_read_s;
    int write_to_read_o;
    int write_to_write_l;
    int write_to_write_s;
    int write_to_write_o;
    int write_to_precharge;

    int burst_cycle;
    int read_delay;
    int write_delay;
}


bool EventDrivenParams::init(const SysConfig& config) {

    // Command Activate
    activate_to_activate = config.mem_config.tRC ;    // same bank
    activate_to_activate_l = config.mem_config.tRRD_L;  // same bankgroup
    activate_to_activate_s = config.mem_config.tRRD_S;  // other bankgroup
    activate_to_precharge = config.mem_config.tRAS;
    if (config.mem_config.IsGDDR() || config.mem_config.IsHBM()) {
        activate_to_read = config.mem_config.tRCDRD;
        activate_to_write = config.mem_config.tRCDWR;
    } else {
        activate_to_read = config.mem_config.tRCD - config.mem_config.AL;
        activate_to_write = config.mem_config.tRCD - config.mem_config.AL;
    }

    // Command Precharge
    precharge_to_activate = config.mem_config.tRP;
    precharge_to_precharge = config.mem_config.tPPD;

    // Command Refresh
    refresh_to_activate = config.mem_config.tRFC;;

    // Read
    read_to_read_l = std::max(config.mem_config.burst_cycle, config.mem_config.tCCD_L);
    read_to_read_s = std::max(config.mem_config.burst_cycle, config.mem_config.tCCD_S);
    read_to_read_o = config.mem_config.burst_cycle + config.mem_config.tRTRS;
    read_to_write = config.mem_config.RL + config.mem_config.burst_cycle - config.mem_config.WL + config.mem_config.tRTRS;
    read_to_write_o = config.mem_config.read_delay + config.mem_config.burst_cycle + config.mem_config.tRTRS - config.mem_config.write_delay;
    read_to_precharge = config.mem_config.AL + config.mem_config.tRTP;

    // Write
    write_to_read_l = config.mem_config.write_delay + config.mem_config.tWTR_L;
    write_to_read_s = config.mem_config.write_delay + config.mem_config.tWTR_S;
    write_to_read_o = config.mem_config.write_delay + config.mem_config.burst_cycle + config.mem_config.tRTRS - config.mem_config.read_delay;
    write_to_write_l = std::max(config.mem_config.burst_cycle, config.mem_config.tCCD_L);
    write_to_write_s = std::max(config.mem_config.burst_cycle, config.mem_config.tCCD_S);
    write_to_write_o = config.mem_config.burst_cycle;
    write_to_precharge = config.mem_config.WL + config.mem_config.burst_cycle + config.mem_config.tWR;

    // 后面都是PIM计算内容, PIM部分完全是顺序执行
    burst_cycle = config.mem_config.burst_cycle;
    read_delay = config.mem_config.read_delay;
    write_delay = config.mem_config.write_delay;

    return true;
}
