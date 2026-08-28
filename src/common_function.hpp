#ifndef common_function_hpp
#define common_function_hpp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
// For backtrace
#include <execinfo.h>
#include <unistd.h>

#include <csignal>
#include <memory>

#include <sys/stat.h>

#include "json.hpp"
#include "INIReader.h"
#include "dram_geometry.hpp"

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>


#define SPAD_BASE 0x10000000
#define ACCUM_SPAD_BASE 0x20000000
#define GARBAGE_ADDR 0xFFFFFFFFFFFFFFF
#define KB *1024

#define PAGE_SIZE 4096 KB
#define HASH_SIZE 1024


#define ADDR_ALIGN 256

#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define MIN3(x, y, z) MIN(MIN(x, y), z)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define GB *1024 * 1024 * 1024
#define MB *1024 * 1024
#define MHz *1000 * 1000

///////////////////////////////////////////////////
// Simulation configuration types and parameters.
///////////////////////////////////////////////////
enum class CoreType { SYSTOLIC_OS, SYSTOLIC_WS };

enum class DramType { DRAM, NEWTON };

enum class IcntType { SIMPLE, BOOKSIM2 };

enum class RunMode { NPU_ONLY, NPU_PIM };

enum class DramMode {
    CYCLE_ACCURATE,  // simulation based on newton
    EVENT_DRIVEN     // simulation based on event driven simulator
};

#include "config_validation.hpp"

typedef uint64_t cycle_type;

inline int ReadTCKESRWithLegacyFallback(const INIReader& reader) {
    constexpr int default_tckesr = 12;
    const std::string canonical_text = reader.Get("timing", "tCKESR", "");
    const std::string legacy_text = reader.Get("timing", "tCKSRE", "");
    const bool has_canonical = !canonical_text.empty();
    const bool has_legacy = !legacy_text.empty();

    const auto read_nonnegative_timing = [&](const std::string& key) {
        constexpr long invalid_value = std::numeric_limits<long>::min();
        const long value = reader.GetInteger("timing", key, invalid_value);
        if (value == invalid_value || value < 0 ||
            value > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Invalid non-negative DRAM timing value for " + key);
        }
        return static_cast<int>(value);
    };

    if (has_canonical && has_legacy) {
        const int canonical_value = read_nonnegative_timing("tCKESR");
        const int legacy_value = read_nonnegative_timing("tCKSRE");
        if (canonical_value != legacy_value) {
            throw std::runtime_error(
                "Conflicting DRAM timing values: tCKESR=" +
                std::to_string(canonical_value) + " and deprecated tCKSRE=" +
                std::to_string(legacy_value));
        }
        spdlog::warn(
            "Deprecated DRAM timing key tCKSRE duplicates tCKESR; remove tCKSRE");
        return canonical_value;
    }

    if (has_canonical) {
        return read_nonnegative_timing("tCKESR");
    }
    if (has_legacy) {
        const int legacy_value = read_nonnegative_timing("tCKSRE");
        spdlog::warn(
            "Deprecated DRAM timing key tCKSRE is used as tCKESR={}; migrate the INI to tCKESR",
            legacy_value);
        return legacy_value;
    }
    return default_tckesr;
}

inline void ValidateDramBackendCapabilities(DramMode mode,
                                            bool enable_self_refresh) {
    phsim::ConfigValidator::ValidateBackendCapabilities(
        mode == DramMode::EVENT_DRIVEN, enable_self_refresh);
}

inline uint32_t ValidateDramRequestSizeConsistency(
    uint32_t configured_request_size_bytes, uint32_t burst_length,
    uint32_t bus_width_bits, const std::string& pim_config_path = {},
    const std::string& memory_config_path = {}) {
    return phsim::ConfigValidator::ValidateRequestSize(
        configured_request_size_bytes, burst_length, bus_width_bits,
        pim_config_path, memory_config_path);
}

inline uint32_t ValidateDramChannelConsistency(
    uint32_t configured_channels, int memory_channels,
    const std::string& pim_config_path = {},
    const std::string& memory_config_path = {}) {
    return phsim::ConfigValidator::ValidateChannels(
        configured_channels, memory_channels, pim_config_path,
        memory_config_path);
}

inline double ValidateDramFrequencyConsistency(
    uint32_t configured_frequency_mhz, double tck_ns,
    const std::string& pim_config_path = {},
    const std::string& memory_config_path = {}) {
    return phsim::ConfigValidator::ValidateFrequency(
        configured_frequency_mhz, tck_ns, pim_config_path,
        memory_config_path);
}


///////////////////////////////////////////////////
//  Memory_Config: 生成对于PIM 计算单元的配置类
///////////////////////////////////////////////////
enum class DRAMProtocol {
    DDR3,
    DDR4,
    GDDR5,
    GDDR5X,
    GDDR6,
    LPDDR,
    LPDDR3,
    LPDDR4,
    LPDDR5,
    HBM,
    HBM2,
    HMC,
    SIZE
};

enum class RefreshPolicy {
    RANK_LEVEL_SIMULTANEOUS,  // impractical due to high power requirement
    RANK_LEVEL_STAGGERED,
    BANK_LEVEL_STAGGERED,
    SIZE
};


inline int LogBase2(int power_of_two) {
    int i = 0;
    while (power_of_two > 1) {
        power_of_two /= 2;
        i++;
    }
    return i;
}

inline bool DirExist(std::string dir) {
    // courtesy to stackoverflow
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        return false;
    } else if (info.st_mode & S_IFDIR) {
        return true;
    } else {  // exists but is a file
        return false;
    }
}

inline nlohmann::json load_config(std::string config_path) {
    nlohmann::json config_json;
    std::ifstream config_file(config_path);
    config_file >> config_json;
    config_file.close();
    return config_json;
}


class MemConfig {
public:
    MemConfig();
    MemConfig(std::string memory_config_path, std::string pim_config_path, std::string out_dir);
    // Address AddressMapping(uint64_t hex_addr) const;  // Legacy address-mapping declaration.
    // DRAM physical structure
    DRAMProtocol protocol;
    uint32_t channel_size;
    uint32_t channels;
    uint32_t ranks;
    uint32_t banks;
    uint32_t bankgroups;
    uint32_t banks_per_group;
    uint32_t rows;
    uint32_t columns;
    uint32_t device_width;
    uint32_t bus_width;
    uint32_t devices_per_rank;
    uint32_t BL;

    // Address mapping numbers
    int shift_bits;
    int ch_pos, ra_pos, bg_pos, ba_pos, ro_pos, co_pos;
    uint64_t ch_mask, ra_mask, bg_mask, ba_mask, ro_mask, co_mask;

    // Generic DRAM timing parameters
    double tCK;
    int burst_cycle;  // seperate BL with timing since for GDDRx it's not BL/2
    int AL;
    int CL;
    int CWL;
    int RL;
    int WL;
    int tCCD_L;
    int tCCD_S;
    int tRTRS;
    int tRTP;
    int tWTR_L;
    int tWTR_S;
    int tWR;
    int tRP;
    int tRRD_L;
    int tRRD_S;
    int tRAS;
    int tRCD;
    int tRFC;
    int tRC;
    // tCKSRE and tCKSRX are only useful for changing clock freq after entering
    // SRE mode we are not doing that, so tCKESR is sufficient
    int tCKE;
    int tCKESR;
    int tXS;
    int tXP;
    int tRFCb;
    int tREFI;
    int tREFIb;
    int tFAW;
    int tRPRE;  // read preamble and write preamble are important
    int tWPRE;
    int read_delay;
    int write_delay;
    int activate_read; // precharge to activate + active to read
    int activate_write; // precharge to activate + active to write

    int read_to_precharge;
    int write_to_precharge;

    int same_bankgroup_read_to_read_latency;
    int same_bankgroup_read_to_write_latency;
    int same_bankgroup_write_to_read_latency;
    int same_bankgroup_write_to_write_latency;

    int other_bankgroup_read_to_read_latency;
    int other_bankgroup_read_to_write_latency;
    int other_bankgroup_write_to_read_latency;
    int other_bankgroup_write_to_write_latency;

    int other_rank_read_to_read_latency;
    int other_rank_read_to_write_latency;
    int other_rank_write_to_read_latency;
    int other_rank_write_to_write_latency;

    // LPDDR4 and GDDR5
    int tPPD;
    // GDDR5
    int t32AW;
    int tRCDRD;
    int tRCDWR;

    // pre calculated power parameters
    double act_energy_inc;
    double pre_energy_inc;
    double read_energy_inc;
    double write_energy_inc;
    double ref_energy_inc;
    double refb_energy_inc;
    double act_stb_energy_inc;
    double pre_stb_energy_inc;
    double pre_pd_energy_inc;
    double sref_energy_inc;

    // HMC
    int num_links;
    int num_dies;
    int link_width;
    int link_speed;
    int num_vaults;
    int block_size;  // block size in bytes
    int xbar_queue_depth;

    // System
    std::string address_mapping;
    std::string queue_structure;
    std::string row_buf_policy;
    RefreshPolicy refresh_policy;
    int cmd_queue_size;
    bool unified_queue;
    int trans_queue_size;
    int write_buf_size;
    bool enable_self_refresh;
    int sref_threshold;
    bool aggressive_precharging_enabled;
    bool enable_hbm_dual_cmd;

    int epoch_period;
    int output_level;
    std::string output_dir;
    std::string output_prefix;
    std::string json_stats_name;
    std::string json_epoch_name;
    std::string txt_stats_name;


    // PIM PU Parameter
    std::string PU_location;
    std::string pim_type;  // NewtonSim organization from the memory INI.
    bool dual_bank;  // Whether one PIM unit spans two banks.


    int PU_buffer_write_delay;

    int PU_DRAM_read_delay;
    int PU_DRAM_write_delay;

    int Host_PU_read_delay;
    int Host_PU_write_delay;

    int PU_read_to_read_latency;
    int PU_write_to_write_latency;
    int PU_Buffer_Register_move_delay;

    int PE_num;
    int input_buffer_size;
    int output_buffer_size;

    int PU_num;

    double hybrid_bonding_bw_area_ratio;
    double pim_pu_area;
    double pim_controller_area_overhead;
    double pim_buffer_area_per_kb;

    double pim_static_power_per_pu;
    double pim_compute_power_per_mac;
    double pim_buffer_static_power_per_kb;
    double pim_buffer_dynamic_power_per_bit;

    // Computed parameters
    int request_size_bytes;

    const phsim::DramAddressLayout& address_layout() const {
        return address_layout_;
    }

    bool IsGDDR() const {
        return (protocol == DRAMProtocol::GDDR5 || protocol == DRAMProtocol::GDDR5X || protocol == DRAMProtocol::GDDR6);
    }
    bool IsHBM() const {
        return (protocol == DRAMProtocol::HBM || protocol == DRAMProtocol::HBM2);
    }
    bool IsHMC() const { return (protocol == DRAMProtocol::HMC); }
    // yzy: add another function
    bool IsDDR4() const { return (protocol == DRAMProtocol::DDR4); }

    int ideal_memory_latency;

private:
    std::shared_ptr<INIReader> reader_;
    std::string memory_config_path_;
    bool bankgroup_enable_;
    phsim::DramGeometryResult geometry_;
    phsim::DramAddressLayout address_layout_;
    int GetInteger(const std::string& sec, const std::string& opt, int default_val) const;  //
    static DRAMProtocol GetDRAMProtocol(std::string protocol_str);

    void InitSystemParams();
    void InitDRAMParams();
    void CalculateSize();
    void SetAddressMapping();
    void InitTimingParams();
    void InitPowerParams();
    void InitOtherParams();

    void InitPIMParams(std::string pim_config_path);
};


inline MemConfig::MemConfig() {
    // std::cout << "Initializing memory config" << std::endl;
}


inline MemConfig::MemConfig(std::string memory_config_path, std::string pim_config_path, std::string out_dir){

    output_dir = out_dir;
    memory_config_path_ = memory_config_path;
    // Load the Memory Configs
    reader_ = std::make_shared<INIReader>(memory_config_path);
    if (reader_->ParseError() < 0) {
        throw std::runtime_error("Can't load memory_config file - " + memory_config_path);
    }
    // The initialization of the parameters has to be strictly in this order
    // because of internal dependencies
    InitSystemParams();
    InitDRAMParams();
    CalculateSize();
    SetAddressMapping();
    InitTimingParams();
    InitPowerParams();
    InitOtherParams();

    InitPIMParams(pim_config_path);
#ifdef THERMAL
    InitThermalParams();
#endif  // THERMAL
    reader_.reset();
}

inline void MemConfig::CalculateSize() {
    phsim::DramGeometryInput input = {};
    input.channel_size_mib = channel_size;
    input.channels = channels;
    input.bankgroups = bankgroups;
    input.banks_per_group = banks_per_group;
    input.rows = rows;
    input.columns = columns;
    input.device_width_bits = device_width;
    input.bus_width_bits = bus_width;
    input.burst_length = BL;
    input.bankgroup_enable = bankgroup_enable_;
    geometry_ = phsim::CalculateDramGeometry(input, memory_config_path_);

    if (geometry_.channel_size_increased) {
        std::cout << "WARNING: Cannot create memory system of size "
                  << channel_size
                  << "MB with given device choice! Using default size "
                  << geometry_.rank_size_mib << " instead!" << std::endl;
    }
    channel_size = geometry_.channel_size_mib;
    ranks = geometry_.ranks;
    banks = geometry_.banks;
    bankgroups = geometry_.bankgroups;
    banks_per_group = geometry_.banks_per_group;
    devices_per_rank = geometry_.devices_per_rank;
    request_size_bytes = static_cast<int>(geometry_.request_size_bytes);
    shift_bits = static_cast<int>(geometry_.shift_bits);
}

inline DRAMProtocol MemConfig::GetDRAMProtocol(std::string protocol_str) {
    std::map<std::string, DRAMProtocol> protocol_pairs = {
        {"DDR3", DRAMProtocol::DDR3},     {"DDR4", DRAMProtocol::DDR4},
        {"GDDR5", DRAMProtocol::GDDR5},   {"GDDR5X", DRAMProtocol::GDDR5X},  {"GDDR6", DRAMProtocol::GDDR6},
        {"LPDDR", DRAMProtocol::LPDDR},   {"LPDDR3", DRAMProtocol::LPDDR3},
        {"LPDDR4", DRAMProtocol::LPDDR4}, {"LPDDR5", DRAMProtocol::LPDDR5}, {"HBM", DRAMProtocol::HBM},
        {"HBM2", DRAMProtocol::HBM2},     {"HMC", DRAMProtocol::HMC}};

    if (protocol_pairs.find(protocol_str) == protocol_pairs.end()) {
        throw std::runtime_error("Invalid DRAM protocol: " + protocol_str);
    }

    return protocol_pairs[protocol_str];
}

inline int MemConfig::GetInteger(const std::string& sec, const std::string& opt, int default_val) const {
    return static_cast<int>(reader_->GetInteger(sec, opt, default_val));
}

inline void MemConfig::InitSystemParams() {
    const auto& reader = *reader_;
    channel_size = GetInteger("system", "channel_size", 1024);
    channels = GetInteger("system", "channels", 1);
    bus_width = GetInteger("system", "bus_width", 64);
    address_mapping = reader.Get("system", "address_mapping", "chrobabgraco");
    queue_structure = reader.Get("system", "queue_structure", "PER_BANK");
    row_buf_policy = reader.Get("system", "row_buf_policy", "OPEN_PAGE");
    cmd_queue_size = GetInteger("system", "cmd_queue_size", 16);
    trans_queue_size = GetInteger("system", "trans_queue_size", 32);
    unified_queue = reader.GetBoolean("system", "unified_queue", false);
    write_buf_size = GetInteger("system", "write_buf_size", 16);
    std::string ref_policy = reader.Get("system", "refresh_policy", "RANK_LEVEL_STAGGERED");

    if (ref_policy == "RANK_LEVEL_SIMULTANEOUS") {
        refresh_policy = RefreshPolicy::RANK_LEVEL_SIMULTANEOUS;
    } else if (ref_policy == "RANK_LEVEL_STAGGERED") {
        refresh_policy = RefreshPolicy::RANK_LEVEL_STAGGERED;
    } else if (ref_policy == "BANK_LEVEL_STAGGERED") {
        refresh_policy = RefreshPolicy::BANK_LEVEL_STAGGERED;
    } else {
        throw std::runtime_error("Invalid refresh_policy: " + ref_policy);
    }
    enable_self_refresh = reader.GetBoolean("system", "enable_self_refresh", false);
    sref_threshold = GetInteger("system", "sref_threshold", 1000);
    aggressive_precharging_enabled = reader.GetBoolean("system", "aggressive_precharging_enabled", false);
}

inline void MemConfig::InitDRAMParams() {
    const auto& reader = *reader_;
    protocol = GetDRAMProtocol(reader.Get("dram_structure", "protocol", "DDR3"));
    bankgroups = GetInteger("dram_structure", "bankgroups", 2);
    banks_per_group = GetInteger("dram_structure", "banks_per_group", 2);
    bankgroup_enable_ =
        reader.GetBoolean("dram_structure", "bankgroup_enable", true);
    rows = GetInteger("dram_structure", "rows", 1 << 16);
    columns = GetInteger("dram_structure", "columns", 1 << 10);
    device_width = GetInteger("dram_structure", "device_width", 8);
    BL = GetInteger("dram_structure", "BL", 8);
    num_dies = GetInteger("dram_structure", "num_dies", 1);
    pim_type = reader.Get("dram_structure", "pim_type", "SINGLE");

    // HBM specific parameters
    enable_hbm_dual_cmd = reader.GetBoolean("dram_structure", "hbm_dual_cmd", true);
    enable_hbm_dual_cmd &= IsHBM();  // Make sure only HBM enables this

    // HMC specific parameters
    num_links = GetInteger("hmc", "num_links", 4);
    link_width = GetInteger("hmc", "link_width", 16);
    link_speed = GetInteger("hmc", "link_speed", 15000);  //MHz
    block_size = GetInteger("hmc", "block_size", 64);
    xbar_queue_depth = GetInteger("hmc", "xbar_queue_depth", 16);

    if (IsHMC()) {
        // the BL for HMC is determined by max block_size, which is a multiple
        // of 32B, each "device" transfer 32b per half cycle therefore BL is 8
        // for 32B block size
        if (block_size <= 0 || device_width == 0) {
            throw std::invalid_argument(
                "Invalid HMC geometry in '" + memory_config_path_ +
                "': block_size and device_width must be greater than zero");
        }
        const uint64_t block_bits = static_cast<uint64_t>(block_size) * 8;
        if (block_bits % device_width != 0) {
            throw std::invalid_argument(
                "Invalid HMC geometry in '" + memory_config_path_ +
                "': block_size * 8 must be divisible by device_width");
        }
        const uint64_t derived_bl = block_bits / device_width;
        if (derived_bl > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "Invalid HMC geometry in '" + memory_config_path_ +
                "': derived BL overflows uint32_t");
        }
        BL = static_cast<uint32_t>(derived_bl);
    }
    // set burst cycle according to protocol
    // We use burst_cycle for timing and use BL for capacity calculation
    // BL = 0 simulate perfect BW
    if (protocol == DRAMProtocol::GDDR5) {
        burst_cycle = (BL == 0) ? 0 : BL / 4;
        BL = (BL == 0) ? 8 : BL;
    } else if (protocol == DRAMProtocol::GDDR5X) {
        burst_cycle = (BL == 0) ? 0 : BL / 8;
        BL = (BL == 0) ? 8 : BL;
    } else if (protocol == DRAMProtocol::GDDR6) {
        burst_cycle = (BL == 0) ? 0 : BL / 8;
        BL = (BL == 0) ? 8 : BL;
    } else if (protocol == DRAMProtocol::LPDDR5) {
        burst_cycle = (BL == 0) ? 0 : BL / 8;
        BL = (BL == 0) ? 8 : BL;
    }
    else {
        burst_cycle = (BL == 0) ? 0 : BL / 2;
        BL = (BL == 0) ? (IsHBM() ? 4 : 8) : BL;
    }

    // every protocol has a different definition of "column",
    // in DDR3/4, each column is exactly device_width bits,
    // but in GDDR5, a column is device_width * BL bits
    // and for HBM each column is device_width * 2 (prefetch)
    // as a result, different protocol has different method of calculating
    // page size, and address mapping...
    // To make life easier, we regulate the use of the term "column"
    // to only represent physical column (device width)
    /*
    if (IsGDDR()) {
        columns *= BL;
    }
    */
    //else if (IsHBM()) {
    //    columns *= 2;
    //}
}

inline void MemConfig::InitOtherParams() {
    const auto& reader = *reader_;
    epoch_period = GetInteger("other", "epoch_period", 100000);
    // determine how much output we want:
    // -1: no file output at all (NOT implemented yet)
    // 0: no epoch file output, only outputs the summary in the end
    // 1: default value, adds epoch CSV output on level 0
    // 2: adds histogram outputs in a different CSV format
    output_level = reader.GetInteger("other", "output_level", 1);
    // Other Parameters
    // give a prefix instead of specify the output name one by one...
    // this would allow outputing to a directory and you can always override
    // these values
    if (!DirExist(output_dir)) {
        std::cout << "WARNING: Output directory " << output_dir
                  << " not exists! Using current directory for output!"
                  << std::endl;
        output_dir = "./";
    } else {
        output_dir = output_dir + "/";
    }
    output_prefix =
        output_dir + reader.Get("other", "output_prefix", "dramsim3");
    json_stats_name = output_prefix + ".json";
    json_epoch_name = output_prefix + "epoch.json";
    txt_stats_name = output_prefix + ".txt";
}


inline void MemConfig::SetAddressMapping() {
    address_layout_ = phsim::CalculateDramAddressLayout(
        geometry_, address_mapping, memory_config_path_);
    ch_pos = address_layout_.channel_pos;
    ra_pos = address_layout_.rank_pos;
    bg_pos = address_layout_.bankgroup_pos;
    ba_pos = address_layout_.bank_pos;
    ro_pos = address_layout_.row_pos;
    co_pos = address_layout_.column_pos;
    ch_mask = address_layout_.channel_mask;
    ra_mask = address_layout_.rank_mask;
    bg_mask = address_layout_.bankgroup_mask;
    ba_mask = address_layout_.bank_mask;
    ro_mask = address_layout_.row_mask;
    co_mask = address_layout_.column_mask;
}

inline void MemConfig::InitTimingParams() {
    // Timing Parameters
    // TODO there is no need to keep all of these variables, they should
    // just be temporary, ultimately we only need cmd to cmd Timing
    const auto& reader = *reader_;
    tCK = reader.GetReal("timing", "tCK", 1.0);
    AL = GetInteger("timing", "AL", 0);
    CL = GetInteger("timing", "CL", 12);
    CWL = GetInteger("timing", "CWL", 12);
    tCCD_L = GetInteger("timing", "tCCD_L", 6);
    tCCD_S = GetInteger("timing", "tCCD_S", 4);
    tRTRS = GetInteger("timing", "tRTRS", 2);
    tRTP = GetInteger("timing", "tRTP", 5);
    tWTR_L = GetInteger("timing", "tWTR_L", 5);
    tWTR_S = GetInteger("timing", "tWTR_S", 5);
    tWR = GetInteger("timing", "tWR", 10);
    tRP = GetInteger("timing", "tRP", 10);
    tRRD_L = GetInteger("timing", "tRRD_L", 4);
    tRRD_S = GetInteger("timing", "tRRD_S", 4);
    tRAS = GetInteger("timing", "tRAS", 24);
    tRCD = GetInteger("timing", "tRCD", 10);
    tRFC = GetInteger("timing", "tRFC", 74);
    tRC = tRAS + tRP;
    tCKE = GetInteger("timing", "tCKE", 6);
    tCKESR = ReadTCKESRWithLegacyFallback(reader);
    tXS = GetInteger("timing", "tXS", 432);
    tXP = GetInteger("timing", "tXP", 8);
    tRFCb = GetInteger("timing", "tRFCb", 20);
    tREFI = GetInteger("timing", "tREFI", 7800);
    tREFIb = GetInteger("timing", "tREFIb", 1950);
    tFAW = GetInteger("timing", "tFAW", 50);
    tRPRE = GetInteger("timing", "tRPRE", 1);
    tWPRE = GetInteger("timing", "tWPRE", 1);

    // LPDDR4 and GDDR5/6
    tPPD = GetInteger("timing", "tPPD", 0);

    // GDDR5/6
    t32AW = GetInteger("timing", "t32AW", 330);
    tRCDRD = GetInteger("timing", "tRCDRD", 24);
    tRCDWR = GetInteger("timing", "tRCDWR", 20);

    ideal_memory_latency = GetInteger("timing", "ideal_memory_latency", 10);

    // latency and duration of read and write
    RL = AL + CL;
    WL = AL + CWL;
    read_delay = RL + burst_cycle;
    write_delay = WL + burst_cycle;
    // precharge to activate + active to read or write
    activate_read = tRCD - AL;
    activate_write = tRCD - AL; //

    // operation to precharge
    read_to_precharge = AL + tRTP;
    write_to_precharge = WL + burst_cycle + tWR;

    // Timing constrain between two read or write transaction
    // banks in same bankgroup
    same_bankgroup_read_to_read_latency = std::max(burst_cycle, tCCD_L);
    same_bankgroup_read_to_write_latency = RL + burst_cycle - WL + tRTRS;
    same_bankgroup_write_to_read_latency = write_delay + tWTR_L;
    same_bankgroup_write_to_write_latency = std::max(burst_cycle, tCCD_L);

    // banks in other bankgroup
    other_bankgroup_read_to_read_latency = std::max(burst_cycle, tCCD_S);
    other_bankgroup_read_to_write_latency = RL + burst_cycle - WL + tRTRS;
    other_bankgroup_write_to_read_latency = write_delay + tWTR_S;
    other_bankgroup_write_to_write_latency = std::max(burst_cycle, tCCD_S);

    // other Rank
    other_rank_read_to_read_latency = burst_cycle + tRTRS;
    other_rank_read_to_write_latency = read_delay + burst_cycle + tRTRS - write_delay;
    other_rank_write_to_read_latency = write_delay + burst_cycle + tRTRS - read_delay;
    other_rank_write_to_write_latency = burst_cycle;
}


inline void MemConfig::InitPowerParams() {
    const auto& reader = *reader_;
    // Power-related parameters
    double VDD = reader.GetReal("power", "VDD", 1.2);
    double IDD0 = reader.GetReal("power", "IDD0", 48);
    double IDD2P = reader.GetReal("power", "IDD2P", 25);
    double IDD2N = reader.GetReal("power", "IDD2N", 34);
    // double IDD3P = reader.GetReal("power", "IDD3P", 37);
    double IDD3N = reader.GetReal("power", "IDD3N", 43);
    double IDD4W = reader.GetReal("power", "IDD4W", 123);
    double IDD4R = reader.GetReal("power", "IDD4R", 135);
    double IDD5AB = reader.GetReal("power", "IDD5AB", 250);  // all-bank ref
    double IDD5PB = reader.GetReal("power", "IDD5PB", 5);    // per-bank ref
    double IDD6x = reader.GetReal("power", "IDD6x", 31);

    // energy increments per command/cycle, calculated as voltage * current *
    // time(in cycles) units are V * mA * Cycles and if we convert cycles to ns
    // then it's exactly pJ in energy and because a command take effects on all
    // devices per rank, also multiply that number
    double devices = static_cast<double>(devices_per_rank);
    act_energy_inc =
        VDD * (IDD0 * tRC - (IDD3N * tRAS + IDD2N * tRP)) * devices;
    read_energy_inc = VDD * (IDD4R - IDD3N) * burst_cycle * devices;
    write_energy_inc = VDD * (IDD4W - IDD3N) * burst_cycle * devices;
    ref_energy_inc = VDD * (IDD5AB - IDD3N) * tRFC * devices;
    refb_energy_inc = VDD * (IDD5PB - IDD3N) * tRFCb * devices;
    // the following are added per cycle
    act_stb_energy_inc = VDD * IDD3N * devices;
    pre_stb_energy_inc = VDD * IDD2N * devices;
    pre_pd_energy_inc = VDD * IDD2P * devices;
    sref_energy_inc = VDD * IDD6x * devices;
}


inline void MemConfig::InitPIMParams(std::string pim_config_path) {
    nlohmann::json pim_config = load_config(pim_config_path);

    PE_num = pim_config["pim_PE_num"];
    input_buffer_size = pim_config["pim_input_buffer_size"];
    output_buffer_size = pim_config["pim_output_buffer_size"];

    // The Operation Latency for PUs
    PU_DRAM_read_delay = RL + burst_cycle;
    PU_DRAM_write_delay = WL + burst_cycle;

    Host_PU_read_delay = burst_cycle;
    Host_PU_write_delay = burst_cycle;

    // 两次操作间的Offset
    PU_read_to_read_latency = std::max(burst_cycle, tCCD_L);
    PU_write_to_write_latency = std::max(burst_cycle, tCCD_L);

    PU_Buffer_Register_move_delay = 0;
    PU_location = pim_config["PU_location"];
    dual_bank = pim_config["dual_bank"];

    if (dual_bank) {
        PU_num = ranks * banks / 2;
    }
    else {
        PU_num = ranks * banks;
    }

    // area config
    hybrid_bonding_bw_area_ratio = pim_config["hybrid_bonding_bw_area_ratio"];
    pim_pu_area = pim_config["pim_pu_area"];
    pim_controller_area_overhead = pim_config["pim_controller_area_overhead"];
    pim_buffer_area_per_kb = pim_config["pim_buffer_area_per_kb"];

    // power config
    pim_static_power_per_pu = pim_config["pim_static_power_per_pu"];
    pim_compute_power_per_mac = pim_config["pim_dynamic_power_per_pu_comp"];
    pim_buffer_static_power_per_kb = pim_config["pim_buffer_static_power_per_kb"];
    pim_buffer_dynamic_power_per_bit = pim_config["pim_buffer_dynamic_power_per_bit"];
}


class SysConfig {
public:
    SysConfig();
    ~SysConfig() = default;
    SysConfig(const SysConfig&) = delete;
    SysConfig& operator=(const SysConfig&) = delete;
    SysConfig(SysConfig&&) = delete;
    SysConfig& operator=(SysConfig&&) = delete;
    void initialize_from_config_path(std::string sys_config_path, std::string memory_config_path, std::string pim_config_path, std::string inference_config_path, std::string model_config_path, std::string request_dataset_path, std::string output_path);
    void initialize_compute_die_system_config(std::string sys_config_path);
    void initialize_inference_config(std::string inference_config_path);
    void initialize_model_config(std::string model_config_path);
    void validate_configuration_contracts();
    // void initialize_pim_config(std::string pim_config);

    void initialize_PIM_config(std::string pim_config);

    std::string system_config_path_;
    std::string memory_config_path_;
    std::string inference_config_path_;
    std::string model_config_path_;
    std::string request_dataset_path_;
    std::string output_path_;

    MemConfig mem_config;

    // gpt model SysConfig
    std::string model_name;
    uint32_t model_params_b;
    uint32_t model_block_size;
    uint32_t model_vocab_size;
    uint32_t model_n_layer;
    uint32_t model_n_head;
    uint32_t model_n_kv_head;
    uint32_t model_n_embd;

    /* Custom SysConfig */
    uint32_t max_batch_size;
    uint32_t max_active_reqs;  // max size of (ready_queue + running_queue) in scheduler
    uint32_t max_seq_len;
    uint32_t kv_cache_entry_size;
    uint64_t DRAM_act_buf_size;  // Size of Preset DRAM space for activation buffer in bytes

    std::string allocation_scheme;

    bool dram_data_container_enable;
    uint64_t dram_data_container_max_payload_mb = 0;
    bool virtual_mem_hash_enable;

    // For Single Operation Test
    bool test_single_op;
    std::string test_single_op_name;

    //For Multi Layer Test
    bool test_multi_layer;
    std::string test_multi_layer_name;

    //For Simulation Acceleration
    bool accelerate_ctrl;
    std::string accelerate_method;
    double accelerate_sample_ratio;
    double attention_command_warmup_weight;
    uint32_t softmax_warmup_rounds;
    uint32_t softmax_sample_rounds;
    bool compile_time_tile_pruning;
    bool decode_pruning_enabled;
    uint32_t decode_pruning_iterations;
    uint32_t decode_pruning_sample_iterations;
    // Transient build context controlled by the Scheduler. This prevents
    // Decode compile-time pruning from deferring same-named Prefill FFN ops.
    bool decode_pruning_compile_context = false;

    bool dram_trace_simulation_mode;
    bool record_dram_completion_trace;

    /* Client SysConfig for Request */
    bool gen_request;
    u_int32_t gen_request_count;
    // Actual number of requests that this run will issue. For generated
    // workloads this equals gen_request_count; trace-driven workloads fill it
    // from the parsed CSV before address allocation starts.
    uint32_t effective_request_count = 0;
    u_int32_t gen_request_input_size;
    u_int32_t gen_request_output_size = 0;
    // Legacy mode (false) executes the existing fixed stage sequence once.
    // When enabled, Prefill runs once and the selected Decode backend repeats
    // until each request reaches InferRequest::output_size.
    bool output_token_iteration_enable = false;
    bool gen_random_request;

    uint32_t request_interval;
    std::string request_dataset_path;

    /* Core SysConfig */
    uint32_t num_cores;
    CoreType core_type;
    uint32_t core_freq;
    uint32_t core_width;
    uint32_t core_height;

    uint32_t n_tp;

    uint32_t vector_core_count;
    uint32_t vector_core_width;

    /* Vector SysConfig*/
    uint32_t process_bit;

    cycle_type gemv_latency;
    cycle_type layernorm_latency;
    cycle_type softmax_latency;
    cycle_type add_latency;
    cycle_type mul_latency;
    cycle_type exp_latency;
    cycle_type gelu_latency;
    cycle_type add_tree_latency;
    cycle_type scalar_sqrt_latency;
    cycle_type scalar_add_latency;
    cycle_type scalar_mul_latency;

    /* SRAM SysConfig */
    uint32_t sram_width;
    // uint32_t sram_size;
    uint32_t spad_size;
    uint32_t accum_spad_size;

    /* DRAM SysConfig */
    DramType dram_type;
    uint32_t dram_freq;
    uint32_t dram_channels;
    uint32_t dram_req_size;

    /* PIM SysConfig */
    std::string pim_config_path;
    uint32_t dram_banks_per_ch;
    uint32_t pim_comp_coverage;  // # params per PIM_COMP command

    uint32_t pim_PE_num;
    uint32_t pim_input_buffer_size;
    uint32_t pim_output_buffer_size;

    // 0: legacy mode; replicate the PIM template address across dram_channels via add_channel_index.
    // N>0 (PIM_COMP only): still exactly dram_channels DRAM requests (no N-fold traffic); bank_linear per channel is
    // (ch * (N / dram_channels)) % banks_per_channel to spread across banks without multiplying completion time.
    // Requires N % dram_channels == 0; otherwise legacy path. Data must be laid out for staggered banks to be correct.
    uint32_t pim_parallel_bank_accesses;


    double hybrid_bonding_bw_area_ratio;
    double pim_pu_area;
    double pim_controller_area_overhead;
    double pim_buffer_area_per_kb;

    double pim_buffer_static_power_per_kb;
    double pim_buffer_dynamic_power_per_bit;
    double pim_static_power_per_pu;
    double pim_dynamic_power_per_pu_comp;

    /* Log SysConfig */
    std::string operation_log_output_path;
    std::string log_dir;

    /* ICNT SysConfig */
    IcntType icnt_type;
    std::string icnt_config_path;
    uint32_t icnt_freq;
    uint32_t icnt_latency;

    /* Sheduler SysConfig */
    std::string scheduler_type;

    /* Other configs */
    uint32_t precision;

    uint32_t precision_weight;
    uint32_t precision_activation;
    uint32_t precision_cache;
    uint32_t precision_psum;

    std::string layout;

    uint64_t align_address(uint64_t addr) { return addr - (addr % dram_req_size); }
};



namespace Config {
    extern SysConfig system_config;
}



SysConfig& initialized_config(std::string const sys_config_path, std::string const memory_config_path, std::string const pim_config_path, std::string const inference_config_path,
                             const std::string model_config_path, const std::string output_path);


///////////////////////////////////////////////////
// Statistics collected from simulator components during execution.
///////////////////////////////////////////////////

// TODO: num_cycles is a magic number. it counts memory load store for 50 core-cycles
// TODO: convert global cycle to core cycle
//  global time / core freq -> core cycle
// memory_reads, memory_writes: bytes
typedef struct NPUStat {
    NPUStat() = default;
    NPUStat(uint64_t core_cycle_)
        : start_cycle(core_cycle_), num_calculations(0) {}

    uint64_t start_cycle;
    // uint64_t num_cycles;
    uint64_t num_calculations;

    enum class StatType {
        StartCycle,
        NumCalculations,
    };

    static std::vector<StatType> get_stat_types() {
        return {
            StatType::StartCycle,
            StatType::NumCalculations,
        };
    }

    static std::string enum_to_string(StatType stat_type) {
        switch (stat_type) {
            case StatType::StartCycle:
                return "StartCycle";
            case StatType::NumCalculations:
                return "NumCalculations";
        }
        throw std::invalid_argument("Unknown NPUStat::StatType");
    }

    std::string get_by_enum(StatType stat_type) {
        // uint64_t core_cycle = Config::system_config.core_freq * 1000000;  // Mhz

        switch (stat_type) {
            case StatType::StartCycle:
                return std::to_string(start_cycle);
            case StatType::NumCalculations:
                return std::to_string(num_calculations);
        }
        throw std::invalid_argument("Unknown NPUStat::StatType");
    }

    static std::string get_columns() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += enum_to_string(type) + "\t";
        }
        return ret + "\n";
    }

    std::string repr() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += get_by_enum(type) + "\t";
        }
        return ret + "\n";
    }
} NPUStat;

// TODO: num_cycles is a magic number. it counts memory load store for 50 core-cycles
// TODO: convert global cycle to core cycle
//  global time / core freq -> core cycle
// memory_reads, memory_writes: bytes
typedef struct MemoryIOStat {
    MemoryIOStat() = default;
    MemoryIOStat(uint64_t core_cycle_, uint64_t channel_id_, uint64_t num_cycles_)
        : start_cycle(core_cycle_),
          channel_id(channel_id_),
          num_cycles(num_cycles_),
          memory_reads(0),
          memory_writes(0),
          pim_reads(0),
          pim_writes(0),
          pim_comps(0),

          pim_energy(0) {}

    uint64_t start_cycle;
    uint64_t channel_id;
    uint64_t num_cycles;
    uint64_t memory_reads;
    uint64_t memory_writes;

    uint64_t pim_reads;
    uint64_t pim_writes;
    uint64_t pim_comps;
    double pim_energy;

    enum class StatType {
        StartCycle,
        ChannelID,
        MemoryReads,
        MemoryWrites,
        MemoryReadBandwidth,
        MemoryWriteBandwidth,
        PIMReads,
        PIMWrites,
        PIMComps,
        PIMBandwidth,
        PIMEnergy,
    };

    static std::vector<StatType> get_stat_types() {
        return {
            StatType::StartCycle,   StatType::ChannelID,           StatType::MemoryReads,
            StatType::MemoryWrites, StatType::MemoryReadBandwidth, StatType::MemoryWriteBandwidth,
            StatType::PIMReads,     StatType::PIMWrites,           StatType::PIMComps,
            StatType::PIMBandwidth, StatType::PIMEnergy,
        };
    }

    static std::string enum_to_string(StatType stat_type) {
        switch (stat_type) {
            case StatType::StartCycle:
                return "StartCycle";
            case StatType::ChannelID:
                return "ChannelID";
            case StatType::MemoryReads:
                return "MemoryReads";
            case StatType::MemoryWrites:
                return "MemoryWrites";
            case StatType::MemoryReadBandwidth:
                return "MemoryReadBandwidth";
            case StatType::MemoryWriteBandwidth:
                return "MemoryWriteBandwidth";
            case StatType::PIMReads:
                return "PIMReads";
            case StatType::PIMWrites:
                return "PIMWrites";
            case StatType::PIMComps:
                return "PIMComps";
            case StatType::PIMBandwidth:
                return "PIMBandwidth";
            case StatType::PIMEnergy:
                return "PIMEnergy";
        }
        throw std::invalid_argument("Unknown MemoryIOStat::StatType");
    }

    std::string get_by_enum(StatType stat_type) {
        uint64_t core_cycle = Config::system_config.core_freq * 1000000;  // Mhz

        switch (stat_type) {
            case StatType::StartCycle:
                return std::to_string(start_cycle);
            case StatType::ChannelID:
                return std::to_string(channel_id);
            case StatType::MemoryReads:
                return std::to_string(memory_reads);
            case StatType::MemoryWrites:
                return std::to_string(memory_writes);
            case StatType::MemoryReadBandwidth:
                return num_cycles == 0 ? "0" : std::to_string(
                    static_cast<uint64_t>(std::llround(
                        static_cast<long double>(memory_reads) * core_cycle /
                        num_cycles)));
            case StatType::MemoryWriteBandwidth:
                return num_cycles == 0 ? "0" : std::to_string(
                    static_cast<uint64_t>(std::llround(
                        static_cast<long double>(memory_writes) * core_cycle /
                        num_cycles)));
            case StatType::PIMReads:
                return std::to_string(pim_reads);
            case StatType::PIMWrites:
                return std::to_string(pim_writes);
            case StatType::PIMComps:
                return std::to_string(pim_comps);
            case StatType::PIMBandwidth:
                if (num_cycles == 0) {
                    return "0";
                }
                return std::to_string(static_cast<uint64_t>(
                    (static_cast<long double>(pim_reads) + pim_writes) *
                    core_cycle / num_cycles));
            case StatType::PIMEnergy:
                return std::to_string(pim_energy);
        }
        throw std::invalid_argument("Unknown MemoryIOStat::StatType");
    }

    static std::string get_columns() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += enum_to_string(type) + "\t";
        }
        return ret + "\n";
    }

    std::string repr() {
        std::string ret = "";
        for (auto type : get_stat_types()) {
            ret += get_by_enum(type) + "\t";
        }
        return ret + "\n";
    }
} MemoryIOStat;

typedef struct TileStat {
    TileStat() = default;
    TileStat(uint64_t core_cycle)
        : start_cycle(core_cycle),
          end_cycle(0),
          compute_cycles(0),
          weight_load_cycles(0),
          memory_stalls(0),
          systolic_memory_stalls(0),
          memory_reads(0),
          memory_writes(0),
          sram_reads(0),
          sram_writes(0),
          num_calculation(0),
          logical_tiles(1),
          estimated(false) {}

    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t compute_cycles;
    uint64_t weight_load_cycles;
    uint64_t memory_stalls;
    // stall cycles between weight-resued GEMMs
    uint64_t systolic_memory_stalls;

    // unit: bytes
    uint64_t memory_reads;
    uint64_t memory_writes;

    // todo
    uint64_t sram_reads;
    uint64_t sram_writes;

    // todo: vector
    uint64_t num_calculation;

    // A normal completed tile contributes one measured logical tile.  A
    // pruning compensation record can aggregate multiple estimated tiles.
    uint64_t logical_tiles = 1;
    bool estimated = false;

    uint64_t dependency_stall;
} TileStat;

typedef struct OperationStat {
    OperationStat() = default;
    OperationStat(std::string name)
        : op_name(name),
          start_cycle(-1),
          end_cycle(0),
          compute_cycles(0),
          memory_reads(0),
          memory_writes(0),
          num_calculation(0),
          measured_tiles(0),
          estimated_tiles(0),
          estimated_memory_reads(0),
          estimated_memory_writes(0),
          estimated_num_calculation(0) {}

    void update_stat(TileStat tile_stat) {
        // std::cout << "tile stat: " << repr() << std::endl;
        // spdlog::info("Tile finished update stat of the Operation");
        if (tile_stat.start_cycle < start_cycle) start_cycle = tile_stat.start_cycle;
        end_cycle = end_cycle > tile_stat.end_cycle ? end_cycle : tile_stat.end_cycle;
        compute_cycles += tile_stat.compute_cycles;

        memory_reads += tile_stat.memory_reads;
        memory_writes += tile_stat.memory_writes;

        num_calculation += tile_stat.num_calculation;
        if (tile_stat.estimated) {
            estimated_tiles += tile_stat.logical_tiles;
            estimated_memory_reads += tile_stat.memory_reads;
            estimated_memory_writes += tile_stat.memory_writes;
            estimated_num_calculation += tile_stat.num_calculation;
        } else {
            measured_tiles += tile_stat.logical_tiles;
        }
    }

    enum class StatType {
        OpName,
        StartCycle,
        EndCycle,
        TotalCycle,
        ComputeCycles,
        MemoryReads,
        MemoryWrites,
        ReadBandwidth,
        WriteBandwidth,
        TotalMemoryBandwidth,
        NumCalculation,
        LogicalTiles,
        MeasuredTiles,
        EstimatedTiles,
        EstimatedMemoryReads,
        EstimatedMemoryWrites,
        EstimatedNumCalculation,
        NpuUtilization,
    };

    static std::vector<StatType> get_stat_types() {
        return {
            StatType::OpName,
            StatType::StartCycle,
            StatType::EndCycle,
            StatType::TotalCycle,
            StatType::ComputeCycles,
            StatType::MemoryReads,
            StatType::MemoryWrites,
            StatType::ReadBandwidth,
            StatType::WriteBandwidth,
            StatType::TotalMemoryBandwidth,
            StatType::NumCalculation,
            StatType::LogicalTiles,
            StatType::MeasuredTiles,
            StatType::EstimatedTiles,
            StatType::EstimatedMemoryReads,
            StatType::EstimatedMemoryWrites,
            StatType::EstimatedNumCalculation,
            StatType::NpuUtilization,
        };
    }

    static std::string enum_to_string(StatType stat_type) {
        switch (stat_type) {
            case StatType::OpName:
                return "OpName";
            case StatType::StartCycle:
                return "StartCycle";
            case StatType::EndCycle:
                return "EndCycle";
            case StatType::TotalCycle:
                return "TotalCycle";
            case StatType::ComputeCycles:
                return "ComputeCycles";
            case StatType::MemoryReads:
                return "MemoryReads";
            case StatType::MemoryWrites:
                return "MemoryWrites";
            case StatType::ReadBandwidth:
                return "ReadBandwidth";
            case StatType::WriteBandwidth:
                return "WriteBandwidth";
            case StatType::TotalMemoryBandwidth:
                return "TotalMemoryBandwidth";
            case StatType::NumCalculation:
                return "NumCalculation";
            case StatType::LogicalTiles:
                return "LogicalTiles";
            case StatType::MeasuredTiles:
                return "MeasuredTiles";
            case StatType::EstimatedTiles:
                return "EstimatedTiles";
            case StatType::EstimatedMemoryReads:
                return "EstimatedMemoryReads";
            case StatType::EstimatedMemoryWrites:
                return "EstimatedMemoryWrites";
            case StatType::EstimatedNumCalculation:
                return "EstimatedNumCalculation";
            case StatType::NpuUtilization:
                return "NpuUtilization";
        }
        throw std::invalid_argument("Unknown OperationStat::StatType");
    }

    std::string get_by_enum(StatType stat_type) {
        uint64_t total_cycle = end_cycle - start_cycle;
        uint64_t core_cycle = Config::system_config.core_freq * 1000000;

        switch (stat_type) {
            case StatType::OpName:
                return op_name;
            case StatType::StartCycle:
                return std::to_string(start_cycle);
            case StatType::EndCycle:
                return std::to_string(end_cycle);
            case StatType::TotalCycle:
                return std::to_string(total_cycle);
            case StatType::ComputeCycles:
                return std::to_string(compute_cycles);
            case StatType::MemoryReads:
                return std::to_string(memory_reads);
            case StatType::MemoryWrites:
                return std::to_string(memory_writes);
            case StatType::ReadBandwidth:
                return total_cycle == 0 ? "0" : std::to_string(
                    static_cast<uint64_t>(std::llround(
                        static_cast<long double>(memory_reads) * core_cycle /
                        total_cycle)));
            case StatType::WriteBandwidth:
                return total_cycle == 0 ? "0" : std::to_string(
                    static_cast<uint64_t>(std::llround(
                        static_cast<long double>(memory_writes) * core_cycle /
                        total_cycle)));
            case StatType::TotalMemoryBandwidth:
                return total_cycle == 0 ? "0" : std::to_string(
                    static_cast<uint64_t>(std::llround(
                        (static_cast<long double>(memory_reads) +
                         memory_writes) * core_cycle / total_cycle)));
            case StatType::NumCalculation:
                return std::to_string(num_calculation);
            case StatType::LogicalTiles:
                return std::to_string(measured_tiles + estimated_tiles);
            case StatType::MeasuredTiles:
                return std::to_string(measured_tiles);
            case StatType::EstimatedTiles:
                return std::to_string(estimated_tiles);
            case StatType::EstimatedMemoryReads:
                return std::to_string(estimated_memory_reads);
            case StatType::EstimatedMemoryWrites:
                return std::to_string(estimated_memory_writes);
            case StatType::EstimatedNumCalculation:
                return std::to_string(estimated_num_calculation);
            case StatType::NpuUtilization:
                if (compute_cycles == 0 || Config::system_config.core_width == 0 ||
                    Config::system_config.core_height == 0) {
                    return "0";
                }
                return std::to_string(static_cast<double>(
                    static_cast<long double>(num_calculation) /
                    (static_cast<long double>(compute_cycles) *
                     Config::system_config.core_width *
                     Config::system_config.core_height)));
        }
        throw std::invalid_argument("Unknown OperationStat::StatType");
    }

    static std::string get_columns() {
        std::string ret;
        for (auto type : get_stat_types()) {
            ret += enum_to_string(type) + "\t";
        }
        return ret + "\n";
    }

    std::string repr() {
        std::string ret;
        if (end_cycle == 0) {
            return ret;
        }

        for (auto type : get_stat_types()) {
            ret += get_by_enum(type) + "\t";
        }
        return ret + "\n";
    }

    std::string op_name;

    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t compute_cycles;

    uint64_t memory_reads;
    uint64_t memory_writes;

    // TODO: count num_calculation for vector operations
    uint64_t num_calculation;

    uint64_t measured_tiles;
    uint64_t estimated_tiles;
    uint64_t estimated_memory_reads;
    uint64_t estimated_memory_writes;
    uint64_t estimated_num_calculation;
} OperationStat;

// xxx not used
typedef struct {
    uint64_t op_cycles;
    std::vector<TileStat> tile_stats;
} OpStat;

// xxx not used
typedef struct {
    uint64_t total_cycles;
    std::vector<OpStat> op_stats;
} ModelStat;

///////////////////////////////////////////////////
// Common: Data structure utilized in simulation
///////////////////////////////////////////////////

// Short alias for shared pointers used by simulator objects.
template <typename T>
using Ptr = std::shared_ptr<T>;

typedef uint64_t addr_type;


namespace AddressConfig {

extern addr_type alignment;
extern addr_type channel_mask;
extern addr_type channel_offset;

uint32_t mask_channel(addr_type address);
addr_type allocate_address(uint32_t size);
addr_type align(addr_type addr);

uint64_t make_address(int channel, int rank, int bankgroup, int bank, int row, int col);
uint64_t encode_pim_header(int channel, int row, bool for_gwrite, int num_comps, int num_readres);
uint64_t encode_pim_comps_readres(int ch, int row, int num_comps, bool last_cmd);
addr_type switch_co_ch(addr_type addr);

}  // namespace AddressConfig

// ===================== TwoLevelDeterministicMapper 本体定义 =====================
//
// 原本位于 TwoLevelDeterministicMapper.hpp / TwoLevelDeterministicMapper.cpp 中，
// Kept here for compatibility with simulator modules that include common_function.hpp.
//
class TwoLevelDeterministicMapper {
public:
    using addr_type = std::uint64_t;

    struct BankInfo {
        uint32_t channel;
        uint32_t rank;
        uint32_t bankgroup;
        uint32_t bank;
    };

    // Construct the mapper with the available physical-page count.
    TwoLevelDeterministicMapper(addr_type num_physical_pages);

    // Initialize from a configuration file; retained for compatibility.
    bool initFromConfig(const std::string& config_path);
    void configure(const MemConfig& mem_config);

    // Configure directly from the active DRAM configuration.

    // 核心映射函数：逻辑地址 -> 物理地址（两层变换）
    addr_type map(addr_type logical_addr);

    // Clear all page mappings; the next map call allocates them again.
    void reset();

    // Release a logical page and make its physical page reusable.
    void free_page(addr_type logical_page);

    // Print the current logical-to-physical page table.
    void dump_page_table(std::ostream &os) const;

    // 从物理地址提取 bank 信息
    BankInfo extract_bank_info(addr_type physical_addr) const;

    // 获取配置信息
    int channels() const { return channels_; }
    int ranks() const { return ranks_; }
    int bankgroups() const { return bankgroups_; }
    int banks_per_group() const { return banks_per_group_; }
    int total_banks() const { return channels_ * ranks_ * bankgroups_ * banks_per_group_; }
    addr_type request_size_bytes() const { return request_size_bytes_; }
    addr_type page_size_address_units() const { return page_size_address_units_; }
    addr_type hash_unit_address_units() const { return hash_unit_address_units_; }
    addr_type physical_capacity_address_units() const {
        return physical_capacity_address_units_;
    }
    addr_type num_physical_pages() const { return num_physical_pages_; }

    // Fixed virtual-memory page and hash-unit sizes in bytes.
    static constexpr addr_type PAGE_SIZE_BYTES = 4096ull * 1024;  // 4MB
    static constexpr addr_type HASH_UNIT_BYTES = 1024ull;        // 1KB

private:
    // 第一层：页表相关
    addr_type num_physical_pages_;
    std::vector<addr_type> logical_to_physical_;  // 逻辑页号 -> 物理页号
    std::vector<bool> physical_used_;            // 物理页是否已使用
    addr_type next_free_physical_page_ = 0;
    static constexpr addr_type INVALID_PAGE = ~0ull;

    // DRAM 结构参数
    uint32_t channels_ = 1;
    uint32_t ranks_ = 1;
    uint32_t bankgroups_ = 1;
    uint32_t banks_per_group_ = 8;
    uint32_t rows_ = 16384;
    uint32_t columns_ = 1024;
    uint32_t bus_width_ = 8;
    uint32_t burst_length_ = 1;

    // PHSim DRAM addresses use one unit per complete burst transaction.
    addr_type request_size_bytes_ = 0;
    addr_type page_size_address_units_ = 0;
    addr_type hash_unit_address_units_ = 0;
    addr_type physical_capacity_address_units_ = 0;

    // 计算出的位宽
    uint32_t ch_bits_ = 0;
    uint32_t ra_bits_ = 0;
    uint32_t bg_bits_ = 0;
    uint32_t ba_bits_ = 0;

    // Bit positions used by the DRAM address layout.
    uint32_t ch_pos_ = 0;
    uint32_t ra_pos_ = 0;
    uint32_t bg_pos_ = 0;
    uint32_t ba_pos_ = 0;
    uint32_t ro_pos_ = 0;
    uint32_t co_pos_ = 0;

    // 掩码
    addr_type ch_mask_ = 0;
    addr_type ra_mask_ = 0;
    addr_type bg_mask_ = 0;
    addr_type ba_mask_ = 0;
    addr_type ro_mask_ = 0;
    addr_type co_mask_ = 0;

    // 辅助函数
    static uint32_t log2_power_of_two(uint32_t x);

    // 第一层：页级映射
    addr_type get_or_alloc_physical_page(addr_type logical_page);

    // Second level: deterministic bank transformation within a physical page.

    // Extract individual DRAM fields from an address.
    uint32_t extractChannel(addr_type addr) const;
    uint32_t extractRank(addr_type addr) const;
    uint32_t extractBankGroup(addr_type addr) const;
    uint32_t extractBank(addr_type addr) const;

    // 构建地址

    // 写回 bank 信息到地址
    addr_type write_bank_tuple(addr_type full_addr, uint32_t new_ch, uint32_t new_ra,
                               uint32_t new_bg, uint32_t new_ba) const;
};


// ===================== TwoLevelDeterministicMapper 接口声明 =====================
//
// Compatibility interface for the two-level deterministic mapper.
// Wrap TwoLevelDeterministicMapper behind a namespace-level compatibility API.
// It provides initialization and mapping entry points for simulator modules.
//
namespace TwoLevelPageMapper {

// 使用当前 Config::system_config.mem_config 初始化两层映射器
void init_two_level_mapper();

// Map a logical address to a physical address; return it unchanged when disabled.
addr_type map_logical_address(addr_type logical_addr);

// Print the logical-to-physical page table for diagnostics.
void dump_page_table(std::ostream& os);

} // namespace TwoLevelPageMapper




enum class TensorType{WGT, ACT, KCache, VCache, PSUM};
enum class AllocationScheme{NPU, AttAcc, NeuPIM, AttenPIM, IANUS, DASH};

namespace MyAddressAllocator {
    extern uint32_t dram_channels;
    extern uint32_t ranks;
    extern uint32_t devices_per_rank; // For DDR4 or DDR5,concat device to channel width
    extern uint32_t bankgroups;
    extern uint32_t banks;
    extern uint32_t rows;
    extern uint32_t columns;
    extern uint32_t total_banks;
    extern uint32_t banks_per_channel;
    extern uint32_t DQ_width;
    extern uint32_t burst_length;
    extern uint32_t BL_num_per_row; // 这个地方是按照Bank的位宽DQ从Row Buffer中取出数据，对于3D DRAM可能会存在更高位宽的情况
    extern uint32_t page_size_bytes;

    extern uint32_t channel_width;
    extern uint32_t dram_burst_size;
    extern uint32_t memory_burst_size;

    extern bool virtual_mem_hash_enable;
    extern uint32_t AddrGranularity_Hash_Bytes;

    extern std::vector<std::string> fields;
    extern std::map<std::string, int> field_widths;
    extern std::map<std::string, int> field_pos;

    extern std::map<std::string, uint64_t> mask;
    extern int ch_pos, ra_pos, bg_pos, ba_pos, ro_pos, co_pos;

    extern uint32_t precision_weight;
    extern uint32_t precision_activation;
    extern uint32_t precision_cache;
    extern uint32_t precision_psum;

    extern addr_type _base_addr;
    extern uint64_t _top_addr;

    extern uint32_t activation_buf_size;

    extern AllocationScheme allocation_scheme;
    extern uint32_t base_column, base_row, base_outer_row_loop, base_middle_row_loop, base_inner_row_loop;
    extern uint32_t act_column, act_row, act_outer_row_loop, act_middle_row_loop, act_inner_row_loop;
    extern uint32_t activation_start_column, activation_start_row, act_start_inner_row_loop, act_start_middle_row_loop, act_start_outer_row_loop;
    extern uint32_t cache_column, cache_row, cache_outer_row_loop, cache_middle_row_loop, cache_inner_row_loop;

    extern uint32_t base_row_1D_weight, base_column_1D_weight, base_outer_row_loop_1D_weight, base_middle_row_loop_1D_weight, base_inner_row_loop_1D_weight;

    extern std::map<std::string, uint32_t> row_loop_size;
    extern std::string outer_row_loop, middle_row_loop, inner_row_loop;

    // The 1D data and activation is aligned
    extern std::vector<uint32_t> weight_1D_space; // 分配空间的过程中更新
    extern std::vector<uint32_t> activation_space;

    extern uint32_t act_column_slice_size;  // Activation slice size along the X dimension.
    extern uint32_t weight_column_slice_size;  // Weight slice size along the X dimension.
    extern bool IANUS_channel_parallel;

    // Initialize only the DRAM geometry and address-decoder state.  This is
    // shared by the full simulator initialization and focused backend tests.
    void configure_address_decoder(const MemConfig& mem_config);

    // 当前地址分配器的地址生成方法
    bool init(const SysConfig& config);

    // 1-D Weight
    bool weight_malloc_1D();
    std::vector<uint32_t> weight_allocate_1D(std::vector<uint32_t> dims, uint32_t precision);  // Allocate a 1-D tensor and return its placement metadata.

    extern std::vector<std::vector<uint32_t>> rababg_bank_index;

    // 2-D Weight
    extern uint32_t tile_height;
    extern uint32_t tile_width;
    extern uint32_t pim_unit;

    extern uint32_t interleaved_banks_per_tile;
    extern uint32_t allocated_tiles_per_iteration;
    extern uint32_t interleaved_row_offset;
    extern std::vector<std::vector<std::vector<uint32_t>>> interleaved_bank_index;
    extern uint32_t burst_weight_row_unit;
    extern uint32_t burst_times_per_tile;
    extern uint32_t bank_allocated_columns_per_tile;
    extern uint32_t weight_interleave_columns;
    extern uint32_t weight_rows_per_bank_row;

    extern std::vector<uint32_t> weight_allocate_2D(std::vector<uint32_t> dims, uint32_t precision);
    extern std::vector<uint32_t> weight_2D_allocate_in_sequence(std::vector<uint32_t> dims, uint32_t precision);
    extern std::vector<uint32_t> weight_2D_allocate_in_IANUS(std::vector<uint32_t> dims, uint32_t precision);
    std::vector<uint32_t> weight_2D_allocate_in_DASH(std::vector<uint32_t> dims, uint32_t precision);

    // VCache allocation
    extern uint32_t KVCache_allocate_row;
    extern uint32_t parallel_KCache_head_per_channel;
    extern uint32_t parallel_VCache_head_per_channel;

    extern uint32_t kv_cache_entry_size;
    extern uint32_t max_active_reqs;
    extern uint32_t max_seq_len;
    extern uint32_t h;  // Q head count, kept for existing attention compute loops
    extern uint32_t h_q;
    extern uint32_t h_kv;
    extern uint32_t d_k;
    extern uint32_t kv_dim;
    extern uint32_t kv_group_size;
    uint32_t get_kv_head_index(uint32_t q_head_index);
    extern uint32_t kv_cache_entry_burst_num;
    extern uint32_t layers;
    // K Cache
    extern uint32_t KCache_interleaved_banks_per_head;
    extern uint32_t allocated_KCache_head_per_iteration;
    extern uint32_t KCache_rows_per_bank_row;
    extern uint32_t burst_times_per_KCache_row;
    extern uint32_t KCache_rows_per_allocation;
    extern std::vector<std::vector<std::vector<uint32_t>>> KCache_interleaved_bank_index;
    std::vector<std::vector<uint32_t>> kcache_allocate(std::vector<uint32_t> dims, uint32_t precision);
    // V Cache
    extern uint32_t VCache_interleaved_banks_per_head;
    extern uint32_t allocated_VCache_head_per_iteration;
    extern uint32_t VCache_columns_per_bank;
    extern uint32_t VCache_burst_row_unit;
    extern uint32_t VCache_rows_per_allocation;
    extern uint32_t VCache_row_units_per_allocation;
    extern uint32_t VCache_rows_per_bank_row;
    extern std::vector<std::vector<std::vector<uint32_t>>> VCache_interleaved_bank_index;
    std::vector<std::vector<uint32_t>> vcache_allocate(std::vector<uint32_t> dims, uint32_t precision);
    uint32_t kvcache_append(std::vector<std::vector<uint32_t>>* allocated_kvcache_rows, TensorType tensor_type);  // 在现有的分配行的基础上，构建新行，之后再完成进一步的计算
    // Activation
    bool activation_malloc();
    bool activation_refresh();

    std::vector<uint32_t> activation_allocate(std::vector<uint32_t> dims, uint32_t precision);
    std::vector<uint32_t> activation_allocate_in_sequence(std::vector<uint32_t> dims, uint32_t precision);

    // Address gen based on index
    addr_type make_address_by_index(uint32_t rank_index, uint32_t bankgroup_index, uint32_t bank_index, uint32_t row_index, uint32_t column_index, uint32_t channel_index);
    addr_type make_address_by_index_with_shift(uint32_t rank_index, uint32_t bankgroup_index, uint32_t bank_index, uint32_t row_index, uint32_t column_index, uint32_t channel_index);

    addr_type weight_address_allocate(uint32_t size);

    addr_type activation_address_allocate(uint32_t size);
    addr_type kvcache_address_allocate(uint32_t size);

    addr_type make_address(uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, uint32_t row, uint32_t column, uint32_t channel = 0);
    addr_type make_address();
    addr_type make_address_with_channel(uint32_t channel, uint32_t inner_row_loop_index, uint32_t middle_row_loop_index, uint32_t outer_row_loop_index, uint32_t row, uint32_t col);

    addr_type add_channel_index(addr_type addr, uint32_t channel_index);

    uint32_t get_channel_index(addr_type addr);
    uint32_t get_rank_index(addr_type addr);
    uint32_t get_bankgroup_index(addr_type addr);
    uint32_t get_bank_index(addr_type addr);
    uint32_t get_row_index(addr_type addr);
    uint32_t get_col_index(addr_type addr);

    uint32_t get_data_offset(addr_type addr, uint32_t offset_size); // get addr offset based on the set data size

    bool weight_allocate(std::vector<uint32_t> dims);

    bool malloc_kvcache_space();

    addr_type addr_align(addr_type addr);  // Align an address for the legacy sequential allocator.
    addr_type addr_row_align(addr_type addr);
    addr_type get_next_aligned_addr();

    bool check_addrs(const std::vector<addr_type>& addrs);

    bool allocate_in_sequence(uint64_t size);

    bool activation_allocate_in_sequence(uint64_t size);

    bool column_align();
    bool row_align();

    uint32_t get_precision(TensorType tensor_type);

    addr_type get_sequence_address(uint64_t size, uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, uint32_t row, uint32_t column);
    addr_type get_sequence_address(uint64_t burst_offset, uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, std::vector<uint32_t> allocated_rows, uint32_t column);

    addr_type get_sequence_address_pim(uint64_t burst_offset, uint32_t row, uint32_t column);
    addr_type get_sequence_address_pim(uint64_t burst_offset, std::vector<uint32_t> allocated_rows, uint32_t column);

    addr_type get_sequence_address_ianus(uint64_t burst_offset, uint32_t row_loop_outer, uint32_t row_loop_middle, uint32_t row_loop_inner, uint32_t row, uint32_t column);


    /*
    // Virtual Memory & Address Hashing
    extern bool use_virtual_memory;
    extern uint64_t virtual_alloc_ptr;
    extern std::map<uint64_t, uint64_t> page_table;
    extern std::map<uint32_t, uint32_t> bank_alloc_ptr;

    uint64_t allocate_virtual(uint64_t size);
    uint64_t translate_address(uint64_t vaddr);
    uint64_t get_physical_addr_hashed(uint64_t vpn); // Helper for hashing
    */
    void cleanup();
}

namespace PIMHashAddressing {

struct AddressGroup {
    addr_type aligned_start_address;
    uint32_t burst_count;
};

// Group consecutive burst-addressed PIM accesses by the DRAM hash granularity.
// The returned start address remains in DRAM burst-address units.
std::vector<AddressGroup> group_comp_addresses(
    const std::vector<addr_type>& addresses,
    uint32_t hash_granularity_bytes,
    uint32_t dram_burst_size_bytes);

}  // namespace PIMHashAddressing


namespace PIM_Parameters {
    extern bool dual_bank;
    extern uint32_t PU_num_per_channel;
    extern uint32_t global_input_buffer_size;
    extern uint32_t output_buffer_size;

    // power parameters
    extern double pim_static_power_per_pu;
    extern double pim_dynamic_power_per_pu_comp;
    extern double pim_buffer_static_power_per_kb;
    extern double pim_buffer_dynamic_power_per_bit;

    bool init(const SysConfig& config);
}



enum class Color { RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, DEFAULT};

enum class Opcode {
    MOVIN,
    MOVOUT,
    MOVOUT_POOL,
    GEMM_PRELOAD,
    GEMM,
    GEMM_WRITE,
    GEMV,
    GELU,
    SILU,
    SOFTMAX,
    ADD,
    MUL,
    BAR,   // 用于实现同步
    COMP,
    IM2COL,
    LAYERNORM,
    RMSNORM,
    ROPE,
    DATA_CONVERT,
    PIM_HEADER,
    PIM_GWRITE,
    PIM_COMP,
    PIM_COMP_HASH,
    PIM_READRES,
    PIM_COMPS_READRES,
    DUMMY,
    SIZE
};

struct Tile;


// Instruction
struct Instruction {
    Opcode opcode;
    cycle_type start_cycle;
    cycle_type finish_cycle;
    std::string id;
    std::vector<std::string> dependent_ids;
    std::string dest_id;
    addr_type dest_addr;
    uint32_t size;
    std::vector<addr_type> src_addrs;
    int spad_id;
    int accum_spad_id;
    uint32_t operand_id = 0;
    addr_type base_addr;

    // for load store instruction operations
    uint32_t tensor_id;

    // for matrix multiplication systolic array utilization
    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;

    bool src_from_accum = false;
    bool valid = true;
    bool skip = false;

    bool per_ch_inst = false;
    bool is_pim_inst = false;

    std::weak_ptr<Tile> parent_tile;

    std::string inst_information;
    // Preserve semantic memory range information for trace/debug.
    addr_type logical_start_addr = 0;
    uint32_t exec_len_bytes = 0;

    std::string repr();
    std::string print_optype();
};


// for Sub-batch interleaving
enum class Stage {Single_test, Multi_test, Init, Prefill, Decode, NPU_Decode, Finish};
enum class StagePlatform { SA, PIM, SIZE };
bool is_llama_model_name(const std::string& model_name);
std::string stageToString(Stage stage);
std::string stagePlatformToString(StagePlatform sp);

inline std::string stageToString(Stage stage) {
    static const std::map<Stage, std::string> stageMap = {{Stage::Finish, "Finish"},
        {Stage::Prefill, "Prefill"}, {Stage::Decode, "Decode"},
        {Stage::NPU_Decode, "NPU_Decode"},
        {Stage::Single_test, "Single_test"},
        {Stage::Multi_test, "Multi_test"},
    };

    auto it = stageMap.find(stage);
    return (it != stageMap.end()) ? it->second : "unknown";
}

inline std::string stagePlatformToString(StagePlatform sp) {
    static const std::map<StagePlatform, std::string> spMap = {
        {StagePlatform::SA, "SA"},
        {StagePlatform::PIM, "PIM"},
    };

    auto it = spMap.find(sp);
    return (it != spMap.end()) ? it->second : "unknown";
}


struct Tile {
    enum class Status {INITIALIZED, RUNNING, FINISH, BAR, EMPTY,};
    Status status = Status::EMPTY;
    std::string optype;
    uint32_t operation_id;
    uint32_t batch;

    // assume several GEMMs
    std::vector<uint32_t> batches;
    uint32_t head_index;
    // assume N,K @ K,M
    uint32_t N;
    uint32_t K;
    uint32_t M;

    TileStat stat;
    std::deque<Instruction> instructions;
    bool accum;
    bool skip;
    int spad_id;
    int accum_spad_id;

    int core_id = -1;  // Initial the core index

    // initialized when Tile moves into core.
    // count up when MOVIN op exists,
    // populate accurate memory request when load instruction is decoded
    uint32_t remaining_loads = 0;
    // computation instruction count
    uint32_t remaining_computes = 0;
    // count up when MOVOUT op exists,
    // count up for the compute instruction
    // populate accurate memory request when store instruction is decoded
    uint32_t remaining_accum_io = 0;

    bool pim_tile = false;
    uint32_t remain_pim_gwrite = 0;
    uint32_t remain_pim_comp = 0;
    uint32_t remain_pim_readers = 0;

    StagePlatform stage_platform = StagePlatform::SA;  // SA program / PIM program (for sub-batch interleaving)

    // With compile-time Tile Pruning enabled, the operation compiler emits a
    // lightweight descriptor first.  Only tiles retained by the Scheduler's
    // warmup/sample/tail plan materialize their instruction/address streams.
    bool deferred_compile = false;
    std::function<void(Tile&)> materializer;

    void materialize() {
        if (!deferred_compile) {
            return;
        }
        const int assigned_core = core_id;
        const StagePlatform assigned_stage = stage_platform;
        auto compile = std::move(materializer);
        deferred_compile = false;
        if (!compile) {
            throw std::runtime_error(
                "deferred Tile is missing its materializer");
        }
        compile(*this);
        core_id = assigned_core;
        stage_platform = assigned_stage;
        deferred_compile = false;
        materializer = {};
    }
    
    // PIM Bandwidth Stats
    cycle_type pim_start_cycle = 0;
    cycle_type pim_finish_cycle = 0;
    uint64_t pim_inst_count = 0;

    std::string repr();
};

// Additive logical work removed by Proportional Tile Pruning.  Timing and
// stall fields are deliberately excluded because they require sampled timing
// models rather than static inspection of the skipped instruction stream.
struct ProportionalWorkloadStat {
    uint64_t tiles = 0;
    uint64_t memory_reads = 0;
    uint64_t memory_writes = 0;
    uint64_t memory_read_bytes = 0;
    uint64_t memory_write_bytes = 0;
    uint64_t num_calculation = 0;

    uint64_t pim_pheader = 0;
    uint64_t pim_gwrite = 0;
    uint64_t pim_comp = 0;
    uint64_t pim_readres = 0;

    uint64_t gemm = 0;
    uint64_t gemv = 0;
    uint64_t layernorm = 0;
    uint64_t rmsnorm = 0;
    uint64_t rope = 0;
    uint64_t softmax = 0;
    uint64_t add = 0;
    uint64_t mul = 0;
    uint64_t gelu = 0;
    uint64_t silu = 0;
    uint64_t im2col = 0;
    uint64_t dummy = 0;

    // Per-channel logical request counts. Each entry corresponds to one
    // memory transaction of dram_burst_size bytes.
    std::vector<uint64_t> channel_memory_reads;
    std::vector<uint64_t> channel_memory_writes;
    std::vector<uint64_t> channel_pim_pheader;
    std::vector<uint64_t> channel_pim_gwrite;
    std::vector<uint64_t> channel_pim_comp;
    std::vector<uint64_t> channel_pim_readres;

    ProportionalWorkloadStat& operator+=(const ProportionalWorkloadStat& rhs) {
        tiles += rhs.tiles;
        memory_reads += rhs.memory_reads;
        memory_writes += rhs.memory_writes;
        memory_read_bytes += rhs.memory_read_bytes;
        memory_write_bytes += rhs.memory_write_bytes;
        num_calculation += rhs.num_calculation;
        pim_pheader += rhs.pim_pheader;
        pim_gwrite += rhs.pim_gwrite;
        pim_comp += rhs.pim_comp;
        pim_readres += rhs.pim_readres;
        gemm += rhs.gemm;
        gemv += rhs.gemv;
        layernorm += rhs.layernorm;
        rmsnorm += rhs.rmsnorm;
        rope += rhs.rope;
        softmax += rhs.softmax;
        add += rhs.add;
        mul += rhs.mul;
        gelu += rhs.gelu;
        silu += rhs.silu;
        im2col += rhs.im2col;
        dummy += rhs.dummy;
        const auto merge_channels = [](std::vector<uint64_t>& dst,
                                       const std::vector<uint64_t>& src) {
            if (dst.size() < src.size()) {
                dst.resize(src.size(), 0);
            }
            for (size_t channel = 0; channel < src.size(); ++channel) {
                dst[channel] += src[channel];
            }
        };
        merge_channels(channel_memory_reads, rhs.channel_memory_reads);
        merge_channels(channel_memory_writes, rhs.channel_memory_writes);
        merge_channels(channel_pim_pheader, rhs.channel_pim_pheader);
        merge_channels(channel_pim_gwrite, rhs.channel_pim_gwrite);
        merge_channels(channel_pim_comp, rhs.channel_pim_comp);
        merge_channels(channel_pim_readres, rhs.channel_pim_readres);
        return *this;
    }
};

// Per-operation DRAM state sampled by Decode Pruning. Request/command traffic
// is compensated separately by ProportionalWorkloadStat; this structure holds
// only row-state, refresh, PIM-state, and elapsed-cycle information.
struct DecodePruningDramState {
    uint64_t samples = 0;
    std::vector<std::unordered_map<std::string, uint64_t>> counters;
    std::vector<std::vector<uint64_t>> rank_active_cycles;
    std::vector<std::vector<uint64_t>> sref_cycles;
    std::vector<std::vector<uint64_t>> pim_rank_active_cycles;
};

enum class MemoryAccessType { READ, WRITE, GWRITE, COMP, READRES, P_HEADER, COMPS_READRES, COMP_HASH, SIZE };

std::string memAccessTypeString(MemoryAccessType type);
std::string opcodeTypeString(Opcode opcode);


typedef struct MemoryAccess {
    static int req_count;
    static int pre_req_count;

    uint32_t id;

    // Original logical address before two-level mapping.
    addr_type logical_dram_address;
    addr_type dram_address;
    addr_type spad_address;
    uint64_t size;
    // Preserve logical access range information for trace/debug.
    addr_type logical_start_addr;
    uint32_t exec_len_bytes;
    MemoryAccessType req_type;
    bool request;
    uint32_t core_id;
    uint32_t mem_id;
    cycle_type start_cycle;
    cycle_type dram_enter_cycle;
    cycle_type dram_finish_cycle;

    cycle_type sample_cycle;    // 严格CDC下，这条请求被dram边沿采样到的周期

    int buffer_id;
    std::vector<uint8_t> data;
    bool data_ready;

    static std::vector<std::unique_ptr<MemoryAccess>> from_instruction(Instruction &inst, uint32_t id,
                                                        uint32_t size, MemoryAccessType req_type,
                                                        bool request, uint32_t core_id,
                                                        cycle_type start_cycle, int buffer_id,
                                                        StagePlatform stage_platform);

    static std::vector<std::unique_ptr<MemoryAccess>> gen_trace_from_instruction(Instruction &inst, uint32_t id,
                                                        uint32_t size, MemoryAccessType req_type,
                                                        bool request, uint32_t core_id,
                                                        cycle_type start_cycle, int buffer_id,
                                                        StagePlatform stage_platform);

    static std::vector<std::unique_ptr<MemoryAccess>> gen_pim_trace_from_instruction(Instruction &inst, uint32_t id,
                                                        uint32_t size, MemoryAccessType req_type,
                                                        bool request, uint32_t core_id,
                                                        cycle_type start_cycle, int buffer_id,
                                                        StagePlatform stage_platform);


    std::weak_ptr<Tile> parent_tile;
    // SA program / PIM program (for sub-batch interleaving)
    StagePlatform stage_platform;

    // Set only while the request is owned by a Core MemoryAccessOwner. All
    // transport queues and DRAM models hold non-owning pointers.
    static constexpr size_t unowned_slot = static_cast<size_t>(-1);
    size_t owner_slot;

    static void log_count() {
        spdlog::info("total pre req count {} / memory request count {}", pre_req_count, req_count);
    }

    std::unique_ptr<MemoryAccess> clone() const {
        auto result = std::unique_ptr<MemoryAccess>(new MemoryAccess(*this));
        result->owner_slot = unowned_slot;
        return result;
    }

} MemoryAccess;


// Owns every live request created by one Core. Requests keep a stable heap
// address while raw observer pointers move through Core, ICNT, and DRAM
// queues. Reusing vacant slots keeps ownership overhead proportional to the
// peak number of in-flight requests rather than the total simulation count.
class MemoryAccessOwner {
public:
    MemoryAccess* adopt(std::unique_ptr<MemoryAccess> access) {
        assert(access != nullptr);

        size_t slot;
        if (_free_slots.empty()) {
            slot = _slots.size();
            _slots.push_back(nullptr);
        }
        else {
            slot = _free_slots.back();
            _free_slots.pop_back();
        }

        access->owner_slot = slot;
        MemoryAccess* observer = access.get();
        _slots[slot] = std::move(access);
        _outstanding++;
        return observer;
    }

    void release(MemoryAccess* access) {
        assert(access != nullptr);
        const size_t slot = access->owner_slot;
        assert(slot != MemoryAccess::unowned_slot);
        assert(slot < _slots.size());
        assert(_slots[slot].get() == access);

        access->owner_slot = MemoryAccess::unowned_slot;
        _slots[slot].reset();
        _free_slots.push_back(slot);
        assert(_outstanding > 0);
        _outstanding--;
    }

    size_t outstanding() const { return _outstanding; }

private:
    std::vector<std::unique_ptr<MemoryAccess>> _slots;
    std::vector<size_t> _free_slots;
    size_t _outstanding = 0;
};




// The NeuPIM Sub-batch Interleaving
inline uint32_t generate_id() {
    static uint32_t id_counter{0};
    return id_counter++;
}

inline uint32_t generate_mem_access_id() {
    static uint32_t id_counter{0};
    return id_counter++;
}

class BTensor;

typedef struct {
    // client to scheduler.
    uint32_t id;
    cycle_type arrival_cycle;    // time spent on client == arrival time to scheduler
    cycle_type completed_cycle;  // return time to client

    // request demand
    uint32_t input_size;   // input sequence length
    uint32_t output_size;  // # tokens to generate

    // request status
    bool is_initiated;   // whether initialization phase is done
    uint32_t generated;  // # tokens generated
    // mapped channel
    int channel;

    std::vector<Ptr<BTensor>> K_cache;
    std::vector<Ptr<BTensor>> V_cache;

} InferRequest;


// Singleton helper that keeps at most one instance during program execution.
template <typename T>
class Singleton {
protected:
    static T *instance;

public:
    static T *GetInstance() {
        if (instance == nullptr) instance = new T();

        return instance;
    }
    static void Delete() { delete instance; }
};
template <typename T>
T *Singleton<T>::instance = nullptr;

// 下方是对于当前仿真器程序的执行过程中所使用的各种数据与指令相关内容进行定义
std::string to_hex(uint32_t input);

template <typename... Args>
std::string name_gen(Args... args) {
    std::vector<std::string> strs = {args...};
    assert(!strs.empty());
    std::string ret;
    for (auto &str : strs) {
        ret += str + ".";
    }
    ret.resize(ret.size() - 1);
    return ret;
}



void print_backtrace();
void ast(bool cond);
template <typename T>
std::vector<T> slice(std::vector<T> &inp, int start, int end) {
    if (end <= -1) end = inp.size() + (end + 1);
    return std::vector<T>(inp.begin() + start, inp.begin() + end);
}

std::unique_ptr<MemoryAccess> TransToMemoryAccess(
    Instruction &inst, uint32_t size, uint32_t core_id,
    cycle_type start_cycle, int buffer_id, StagePlatform stage_platform);

inline int MemoryAccess::req_count = 0;
inline int MemoryAccess::pre_req_count = 0;

/**
 * Logger takes vector of StatClass and filename
 * Expects fname without extension. (without .tsv)
 * StatClass needs following methods
 *   static std::string get_columns(): log names of the column separated with tab
 *   std::string repr(): log stats separated with tab
 *   to write stat in a single line.
 */
namespace Logger {
    template <typename StatClass>
    void log(std::vector<StatClass> stats, std::string fname) {
        fname += ".tsv";
        std::ofstream ofile(fname);
        if (!ofile.is_open()) {
            assert(0);
        }
        ofile << StatClass::get_columns();
        for (auto stat : stats) {
            ofile << stat.repr();
        }
        ofile.close();
    }
};  // namespace Logger


// The operation for execution
enum class Ops {
    RMSNorm,
    LayerNorm,  // start of the FFN and attn layer
    GEMM,     // QKV generation, attention score calculation, rescoring value,
    GEMM_Att,
    // projection, FFN * 2
    Split,
    GEMV,
    // Mask,
    Softmax,
    Add,  // for Residual Connection
    Mul,
    Gelu,
    SiLU,
    Reshape,
    Transpose,
    Concat,
    /* PIM fused operation */
    PIM_GEMV,
    PIM_GEMV_QKT,
    PIM_GEMV_SV,

    GEMV_Softmax, // L
    GEMV_Add,     // A

    GEMV_Att,
    PIM_GEMV_Att,
    DataConvert,
};

// Event Driven DRAM
enum class EventType {
    READ,
    WRITE,
    GWRITE,
    COMP,
    READRES,
    P_HEADER,
    COMPS_READRES,
    COMP_HASH,
    SIZE
};

enum class PIMRowCommand {
    NONE,
    PRECHARGE,
    ACTIVATE
};


struct Event {
    MemoryAccess* original_req = nullptr;

    EventType event_type;

    addr_type dram_address;
    addr_type spad_address;

    uint32_t channel_index;
    uint32_t rank_index;
    uint32_t bankgroup_index;
    uint32_t bank_index;
    uint32_t row_index;
    uint32_t column_index;

    std::weak_ptr<Tile> parent_tile;
    StagePlatform stage_platform;

    uint32_t         id;
    uint64_t         size;
    MemoryAccessType req_type;
    uint32_t         core_id;
    uint32_t         mem_id;
    int              buffer_id;

    cycle_type add_cycle;       // 添加时间
    cycle_type complete_cycle;  // 完成时间

    bool        need_precharge;
    cycle_type  precharge_cycle;
    bool        need_activate;
    cycle_type  activate_cycle;
    cycle_type  execute_cycle;
    PIMRowCommand pending_pim_row_command = PIMRowCommand::NONE;
    cycle_type  pim_precharge_cycle = 0;
    cycle_type  pim_activate_cycle = 0;
    cycle_type  pim_row_ready_cycle = 0;

    bool        processed;
    bool        response_only = false;
    bool        merge_open = true;
    bool        activated_for_access = false;
    uint64_t    merge_group_id = 0;
    std::vector<MemoryAccess*> merged_original_reqs;
};


namespace EventDrivenParams {
    // Activate Timing Parameters
    extern int activate_to_activate;
    extern int activate_to_activate_l;
    extern int activate_to_activate_s;
    extern int activate_to_precharge;
    extern int activate_to_read, activate_to_write;
    // Precharge Timing Parameters
    extern int precharge_to_activate;      // Delay from PRECHARGE to ACTIVATE.
    extern int precharge_to_precharge;

    // Refresh Timing Parameters
    extern int refresh_to_activate;

    extern int read_to_read_l;
    extern int read_to_read_s;
    extern int read_to_read_o;
    extern int read_to_write;
    extern int read_to_write_o;
    extern int read_to_precharge;

    extern int write_to_read_l;
    extern int write_to_read_s;
    extern int write_to_read_o;
    extern int write_to_write_l;
    extern int write_to_write_s;
    extern int write_to_write_o;
    extern int write_to_precharge;

    extern int burst_cycle;
    extern int read_delay;
    extern int write_delay;

    bool init(const SysConfig& config);

}







#endif
