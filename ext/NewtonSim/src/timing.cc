#include "timing.h"

#include <algorithm>
#include <utility>

#include "../../../src/common_function.hpp"

namespace dramsim3 {

Timing::Timing(const Config &config)
    : same_bank(static_cast<int>(CommandType::SIZE)),
      other_banks_same_bankgroup(static_cast<int>(CommandType::SIZE)),
      other_bankgroups_same_rank(static_cast<int>(CommandType::SIZE)),
      other_ranks(static_cast<int>(CommandType::SIZE)),
      same_rank(static_cast<int>(CommandType::SIZE)),
      same_channel(static_cast<int>(CommandType::SIZE)){
    int read_to_read_l = std::max(config.burst_cycle, config.tCCD_L);
    int read_to_read_s = std::max(config.burst_cycle, config.tCCD_S);
    int read_to_read_o = config.burst_cycle + config.tRTRS;
    int read_to_write = config.RL + config.burst_cycle - config.WL + config.tRTRS;
    int read_to_write_o = config.read_delay + config.burst_cycle + config.tRTRS - config.write_delay;
    int read_to_precharge = config.AL + config.tRTP;
    int readp_to_act = config.AL + config.burst_cycle + config.tRTP + config.tRP;

    int write_to_read_l = config.write_delay + config.tWTR_L;
    int write_to_read_s = config.write_delay + config.tWTR_S;
    int write_to_read_o = config.write_delay + config.burst_cycle + config.tRTRS - config.read_delay;
    int write_to_write_l = std::max(config.burst_cycle, config.tCCD_L);
    int write_to_write_s = std::max(config.burst_cycle, config.tCCD_S);
    int write_to_write_o = config.burst_cycle;
    int write_to_precharge = config.WL + config.burst_cycle + config.tWR;

    int precharge_to_activate = config.tRP;
    int precharge_to_precharge = config.tPPD;
    int read_to_activate = read_to_precharge + precharge_to_activate;
    int write_to_activate = write_to_precharge + precharge_to_activate;

    int activate_to_activate = config.tRC;
    int activate_to_activate_l = config.tRRD_L;
    int activate_to_activate_s = config.tRRD_S;
    int activate_to_precharge = config.tRAS;
    int activate_to_read, activate_to_write;
    if (config.IsGDDR() || config.IsHBM()) {
        activate_to_read = config.tRCDRD;
        activate_to_write = config.tRCDWR;
    } else {
        activate_to_read = config.tRCD - config.AL;
        activate_to_write = config.tRCD - config.AL;
    }
    int activate_to_refresh = config.tRC; // need to precharge before ref, so it's tRC

    // todo: deal with different refresh rate
    int refresh_to_refresh = config.tREFI; // refresh intervals (per rank level)
    int refresh_to_activate = config.tRFC; // tRFC is defined as ref to act
    int refresh_to_activate_bank = config.tRFCb;

    int self_refresh_entry_to_exit = config.tCKESR;
    int self_refresh_exit = config.tXS;
    // int powerdown_to_exit = config.tCKE;
    // int powerdown_exit = config.tXP;

    if (config.bankgroups == 1) {
        // for a bankgroup can be disabled, in that case
        // the value of tXXX_S should be used instead of tXXX_L
        // (because now the device is running at a lower freq)
        // we overwrite the following values so that we don't have
        // to change the assignement of the vectors
        read_to_read_l = std::max(config.burst_cycle, config.tCCD_S);
        write_to_read_l = config.write_delay + config.tWTR_S;
        write_to_write_l = std::max(config.burst_cycle, config.tCCD_S);
        activate_to_activate_l = config.tRRD_S;
    }

    // PIM Operation Latency Configuration
    // pim buffer operations
    int pheader_latency = config.p_header_delay;
    int gwrite_latency = config.gwrite_delay;
    int reader_latency = config.readers_delay;
    int pim_latency_per_burst = 0;  // 当前未定义，实际数值需要基于PU计算的吞吐率计算获得

    // same bank
    int pim_act_to_act_same_bk = activate_to_activate;
    int act_to_pim_act_same_bk = activate_to_activate;
    int read_to_pim_read_same_bk = read_to_read_l;
    int write_to_pim_read_same_bk = write_to_read_l;

    // other banks in same bankgroup
    int read_to_pim_read_same_bg = read_to_read_l;
    int write_to_pim_read_same_bg = write_to_read_l;

    // other bankgroups
    int read_to_pim_read_other_bg = read_to_read_s;
    int write_to_pim_read_other_bg = write_to_read_s;

    // other rank
    int read_to_pim_read_other_rank = read_to_read_s;
    int write_to_pim_read_other_rank = write_to_read_s;

    int gact_to_act_same_bg = activate_to_activate;

    // same rank
    int pim_to_precharge = read_to_precharge;

    pim_precharge_latency = 0;
    pim_activate_latency = 0;
    burst_cycle = config.burst_cycle;

    pim_precharge_to_activate = precharge_to_activate;
    pim_activate_to_read = activate_to_read;

    if (config.enable_dual_buffer) {
        // same bank
        pim_act_to_act_same_bk = activate_to_activate_l;
        act_to_pim_act_same_bk = activate_to_activate_l;
        read_to_pim_read_same_bk = 0;  // read_to_read_s;
        write_to_pim_read_same_bk = 0; // write_to_read_s;
        // same bankgroup
        read_to_pim_read_same_bg = 0;  // read_to_read_s;
        write_to_pim_read_same_bg = 0; // write_to_read_s;
        gact_to_act_same_bg = activate_to_activate_l;
        // same rank
        pim_to_precharge = 0;
    }

    // command READ
    same_bank[static_cast<int>(CommandType::READ)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_l},
        {CommandType::WRITE, read_to_write},
        {CommandType::READ_PRECHARGE, read_to_read_l},
        {CommandType::WRITE_PRECHARGE, read_to_write},
        {CommandType::PRECHARGE, read_to_precharge},
        // PIM Commands
        {CommandType::P_HEADER, pheader_latency},
        {CommandType::GWRITE, read_to_write},
        {CommandType::COMP, read_to_pim_read_same_bk},
        {CommandType::COMP_HASH, read_to_pim_read_same_bk},
        {CommandType::READRES, read_to_read_l},
        {CommandType::G_PRE, read_to_precharge},
        {CommandType::PIM_PRECHARGE, read_to_precharge},
    };

    other_banks_same_bankgroup[static_cast<int>(CommandType::READ)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_l},
        {CommandType::WRITE, read_to_write},
        {CommandType::READ_PRECHARGE, read_to_read_l},
        {CommandType::WRITE_PRECHARGE, read_to_write},
        // PIM Commands
        {CommandType::P_HEADER, pheader_latency},
        {CommandType::GWRITE, read_to_read_l},
        {CommandType::COMP, read_to_pim_read_same_bg},
        {CommandType::COMP_HASH, read_to_pim_read_same_bg},
        {CommandType::READRES, read_to_read_l},
    };

    other_bankgroups_same_rank[static_cast<int>(CommandType::READ)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_s},
            {CommandType::WRITE, read_to_write},
            {CommandType::READ_PRECHARGE, read_to_read_s},
            {CommandType::WRITE_PRECHARGE, read_to_write},
            // PIM Commands
            {CommandType::P_HEADER, pheader_latency},
            {CommandType::GWRITE, read_to_read_l},
            {CommandType::COMP, read_to_pim_read_other_bg},
            {CommandType::COMP_HASH, read_to_pim_read_other_bg},
            {CommandType::READRES, read_to_read_l},
        };

    other_ranks[static_cast<int>(CommandType::READ)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_o},
        {CommandType::WRITE, read_to_write_o},
        {CommandType::READ_PRECHARGE, read_to_read_o},
        {CommandType::WRITE_PRECHARGE, read_to_write_o},
        // PIM Commands
        {CommandType::P_HEADER, pheader_latency},
        {CommandType::GWRITE, read_to_read_l},
        {CommandType::COMP, read_to_pim_read_other_rank},
        {CommandType::COMP_HASH, read_to_pim_read_other_rank},
        {CommandType::READRES, read_to_read_l},
    };

    // command WRITE
    same_bank[static_cast<int>(CommandType::WRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, write_to_read_l},
        {CommandType::WRITE, write_to_write_l},
        {CommandType::READ_PRECHARGE, write_to_read_l},
        {CommandType::WRITE_PRECHARGE, write_to_write_l},
        {CommandType::PRECHARGE, write_to_precharge},
        // PIM Commands
        {CommandType::P_HEADER, pheader_latency},
        {CommandType::GWRITE, write_to_write_l},
        {CommandType::COMP, write_to_pim_read_same_bk},
        {CommandType::COMP_HASH, write_to_pim_read_same_bk},
        {CommandType::READRES, write_to_read_l},
        {CommandType::G_PRE, write_to_precharge},
        {CommandType::PIM_PRECHARGE, write_to_precharge},
    };
    other_banks_same_bankgroup[static_cast<int>(CommandType::WRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, write_to_read_l},
            {CommandType::WRITE, write_to_write_l},
            {CommandType::READ_PRECHARGE, write_to_read_l},
            {CommandType::WRITE_PRECHARGE, write_to_write_l},
            // PIM Commands
            {CommandType::P_HEADER, pheader_latency},
            {CommandType::GWRITE, write_to_write_l},
            {CommandType::COMP, write_to_pim_read_same_bg},
            {CommandType::COMP_HASH, write_to_pim_read_same_bg},
            {CommandType::READRES, write_to_read_l},
        };
    other_bankgroups_same_rank[static_cast<int>(CommandType::WRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, write_to_read_s},
            {CommandType::WRITE, write_to_write_s},
            {CommandType::READ_PRECHARGE, write_to_read_s},
            {CommandType::WRITE_PRECHARGE, write_to_write_s},
            // PIM Commands
            {CommandType::P_HEADER, pheader_latency},
            {CommandType::GWRITE, write_to_write_l},
            {CommandType::COMP, write_to_pim_read_other_bg},
            {CommandType::COMP_HASH, write_to_pim_read_other_bg},
            {CommandType::READRES, write_to_read_l},
        };
    other_ranks[static_cast<int>(CommandType::WRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, write_to_read_o},
        {CommandType::WRITE, write_to_write_o},
        {CommandType::READ_PRECHARGE, write_to_read_o},
        {CommandType::WRITE_PRECHARGE, write_to_write_o},
        // PIM Commands
        {CommandType::P_HEADER, pheader_latency},
        {CommandType::GWRITE, write_to_write_l},
        {CommandType::COMP, write_to_pim_read_other_rank},
        {CommandType::COMP_HASH, write_to_pim_read_other_rank},
        {CommandType::READRES, write_to_read_l},
    };

    // command READ_PRECHARGE
    same_bank[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::ACTIVATE, readp_to_act},
                                                 {CommandType::REFRESH, read_to_activate},
                                                 {CommandType::REFRESH_BANK, read_to_activate},
                                                 {CommandType::SREF_ENTER, read_to_activate}};
    other_banks_same_bankgroup[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, read_to_read_l},
                                                 {CommandType::WRITE, read_to_write},
                                                 {CommandType::READ_PRECHARGE, read_to_read_l},
                                                 {CommandType::WRITE_PRECHARGE, read_to_write}};
    other_bankgroups_same_rank[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, read_to_read_s},
                                                 {CommandType::WRITE, read_to_write},
                                                 {CommandType::READ_PRECHARGE, read_to_read_s},
                                                 {CommandType::WRITE_PRECHARGE, read_to_write}};
    other_ranks[static_cast<int>(CommandType::READ_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, read_to_read_o},
                                                 {CommandType::WRITE, read_to_write_o},
                                                 {CommandType::READ_PRECHARGE, read_to_read_o},
                                                 {CommandType::WRITE_PRECHARGE, read_to_write_o}};

    // command WRITE_PRECHARGE
    same_bank[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::ACTIVATE, write_to_activate},
                                                 {CommandType::REFRESH, write_to_activate},
                                                 {CommandType::REFRESH_BANK, write_to_activate},
                                                 {CommandType::SREF_ENTER, write_to_activate}};
    other_banks_same_bankgroup[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, write_to_read_l},
                                                 {CommandType::WRITE, write_to_write_l},
                                                 {CommandType::READ_PRECHARGE, write_to_read_l},
                                                 {CommandType::WRITE_PRECHARGE, write_to_write_l}};
    other_bankgroups_same_rank[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, write_to_read_s},
                                                 {CommandType::WRITE, write_to_write_s},
                                                 {CommandType::READ_PRECHARGE, write_to_read_s},
                                                 {CommandType::WRITE_PRECHARGE, write_to_write_s}};
    other_ranks[static_cast<int>(CommandType::WRITE_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::READ, write_to_read_o},
                                                 {CommandType::WRITE, write_to_write_o},
                                                 {CommandType::READ_PRECHARGE, write_to_read_o},
                                                 {CommandType::WRITE_PRECHARGE, write_to_write_o}};

    // command ACTIVATE
    same_bank[static_cast<int>(CommandType::ACTIVATE)] =
        std::vector<std::pair<CommandType, int>>{{CommandType::ACTIVATE, activate_to_activate},
                                                 {CommandType::READ, activate_to_read},
                                                 {CommandType::WRITE, activate_to_write},
                                                 {CommandType::READ_PRECHARGE, activate_to_read},
                                                 {CommandType::WRITE_PRECHARGE, activate_to_write},
                                                 {CommandType::PRECHARGE, activate_to_precharge},
                                                // PIM Commands
                                                {CommandType::P_HEADER, pheader_latency},
                                                {CommandType::COMP, activate_to_read},
                                                {CommandType::COMP_HASH, activate_to_read},
                                                {CommandType::PIM_ACTIVE, act_to_pim_act_same_bk},
                                                 {CommandType::G_ACT, act_to_pim_act_same_bk}};


    other_banks_same_bankgroup[static_cast<int>(CommandType::ACTIVATE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, activate_to_activate_l},
            {CommandType::REFRESH_BANK, activate_to_refresh},
            {CommandType::PIM_ACTIVE, act_to_pim_act_same_bk},
            {CommandType::G_ACT, activate_to_activate_l},
        };

    other_bankgroups_same_rank[static_cast<int>(CommandType::ACTIVATE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, activate_to_activate_s},
            {CommandType::REFRESH_BANK, activate_to_refresh},
            {CommandType::G_ACT, activate_to_activate_s},
        };

    // command PRECHARGE
    same_bank[static_cast<int>(CommandType::PRECHARGE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, precharge_to_activate},
        {CommandType::REFRESH, precharge_to_activate},
        {CommandType::REFRESH_BANK, precharge_to_activate},
        {CommandType::SREF_ENTER, precharge_to_activate},
    };

    // for those who need tPPD
    if (config.IsGDDR() || config.protocol == DRAMProtocol::LPDDR4 || config.protocol == DRAMProtocol::LPDDR5) {
        other_banks_same_bankgroup[static_cast<int>(CommandType::PRECHARGE)] =
            std::vector<std::pair<CommandType, int>>{
                {CommandType::PRECHARGE, precharge_to_precharge},
            };

        other_bankgroups_same_rank[static_cast<int>(CommandType::PRECHARGE)] =
            std::vector<std::pair<CommandType, int>>{
                {CommandType::PRECHARGE, precharge_to_precharge},
            };
    }

    // command REFRESH_BANK
    same_rank[static_cast<int>(CommandType::REFRESH_BANK)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, refresh_to_activate_bank},
            {CommandType::REFRESH, refresh_to_activate_bank},
            {CommandType::REFRESH_BANK, refresh_to_activate_bank},
            {CommandType::SREF_ENTER, refresh_to_activate_bank}};

    other_banks_same_bankgroup[static_cast<int>(CommandType::REFRESH_BANK)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, refresh_to_activate},
            {CommandType::REFRESH_BANK, refresh_to_refresh},
        };

    other_bankgroups_same_rank[static_cast<int>(CommandType::REFRESH_BANK)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, refresh_to_activate},
            {CommandType::REFRESH_BANK, refresh_to_refresh},
        };

    // REFRESH, SREF_ENTER and SREF_EXIT are isued to the entire
    // rank  command REFRESH
    same_rank[static_cast<int>(CommandType::REFRESH)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, refresh_to_activate},
        {CommandType::REFRESH, refresh_to_activate},
        {CommandType::SREF_ENTER, refresh_to_activate},
        {CommandType::G_ACT, refresh_to_activate},
    };

    // command SREF_ENTER
    // todo: add power down commands
    same_rank[static_cast<int>(CommandType::SREF_ENTER)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::SREF_EXIT, self_refresh_entry_to_exit}};

    // command SREF_EXIT
    same_rank[static_cast<int>(CommandType::SREF_EXIT)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, self_refresh_exit},
        {CommandType::REFRESH, self_refresh_exit},
        {CommandType::REFRESH_BANK, self_refresh_exit},
        {CommandType::SREF_ENTER, self_refresh_exit},
        {CommandType::G_ACT, self_refresh_exit} // >>> gsheo
    };

    // command PIM_PRECHARGE
    same_bank[static_cast<int>(CommandType::PIM_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::REFRESH, precharge_to_activate},
            {CommandType::G_ACT, precharge_to_activate},
        };

    // command GWRITE
    same_bank[static_cast<int>(CommandType::GWRITE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, gwrite_latency},
        {CommandType::WRITE, gwrite_latency},
        {CommandType::PRECHARGE, gwrite_latency},
        {CommandType::COMP, gwrite_latency},          // for double buffer
        {CommandType::COMP_HASH, gwrite_latency},
    };

    other_banks_same_bankgroup[static_cast<int>(CommandType::GWRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_l},          {CommandType::WRITE, read_to_write},
            {CommandType::GWRITE, gwrite_latency},        {CommandType::COMP, gwrite_latency}, {CommandType::COMP_HASH, gwrite_latency},
        };

    other_bankgroups_same_rank[static_cast<int>(CommandType::GWRITE)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::READ, read_to_read_s},
            {CommandType::WRITE, read_to_write},
            {CommandType::GWRITE, gwrite_latency},
            {CommandType::COMP, gwrite_latency},
            {CommandType::COMP_HASH, gwrite_latency},
        };

    // command G_ACT
    other_banks_same_bankgroup[static_cast<int>(CommandType::G_ACT)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::ACTIVATE, gact_to_act_same_bg},
            {CommandType::G_ACT, activate_to_activate},
            {CommandType::COMP, activate_to_read},
            {CommandType::COMP_HASH, activate_to_read},
            {CommandType::READRES, activate_to_read},
            {CommandType::PIM_PRECHARGE, activate_to_precharge},
        };

    same_rank[static_cast<int>(CommandType::G_ACT)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, config.tFAW},
        {CommandType::G_ACT, config.tFAW},
        {CommandType::COMP, config.tRCDRD},
        {CommandType::COMP_HASH, config.tRCDRD},
    };

    // command G_PRE
    same_bank[static_cast<int>(CommandType::G_PRE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, precharge_to_activate},
        {CommandType::REFRESH, precharge_to_activate},
        {CommandType::REFRESH_BANK, precharge_to_activate},
        {CommandType::SREF_ENTER, precharge_to_activate},
    };

    other_banks_same_bankgroup[static_cast<int>(CommandType::G_PRE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::ACTIVATE, precharge_to_activate},
        {CommandType::REFRESH, precharge_to_activate},
        {CommandType::REFRESH_BANK, precharge_to_activate},
        {CommandType::SREF_ENTER, precharge_to_activate},
        {CommandType::PRECHARGE, precharge_to_precharge},
    };

    other_bankgroups_same_rank[static_cast<int>(CommandType::PRECHARGE)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::PRECHARGE, precharge_to_precharge},
        {CommandType::G_PRE, precharge_to_precharge},
    };

    // command COMP
    same_rank[static_cast<int>(CommandType::COMP)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_s},
        {CommandType::WRITE, read_to_write},
        {CommandType::COMP, read_to_read_l},
        {CommandType::COMP, read_to_read_l},
        {CommandType::PIM_PRECHARGE, read_to_precharge},
        {CommandType::PRECHARGE, pim_to_precharge},
        {CommandType::G_PRE, pim_to_precharge},
    };

    // command READRES
    same_rank[static_cast<int>(CommandType::READRES)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_s},
        {CommandType::WRITE, read_to_write},
        {CommandType::GWRITE, read_to_write},
        {CommandType::READRES, read_to_read_s},
    };

    other_ranks[static_cast<int>(CommandType::READRES)] = std::vector<std::pair<CommandType, int>>{
        {CommandType::READ, read_to_read_o},
        {CommandType::WRITE, read_to_write_o},
        {CommandType::GWRITE, read_to_write},
        {CommandType::READRES, read_to_read_s},
    };

    // PIM Command Interval, same channel timing of PIM commands are defined here
    int GWRITE_latency = burst_cycle;
    int READERS_latency = burst_cycle;

    int COMP_latency_dual_bank = MyAddressAllocator::BL_num_per_row * burst_cycle; // Bank一行数据被读取, 再读一遍

    same_channel[static_cast<int>(CommandType::P_HEADER)] =
        std::vector<std::pair<CommandType, int>>{
            {CommandType::P_HEADER, pheader_latency},
            {CommandType::GWRITE, pheader_latency},
            {CommandType::COMP, pheader_latency},
            {CommandType::COMP_HASH, pheader_latency},
            {CommandType::READRES, pheader_latency},
    };

    same_channel[static_cast<int>(CommandType::GWRITE)] =
        std::vector<std::pair<CommandType, int>>{
                {CommandType::P_HEADER, config.burst_cycle},
                {CommandType::GWRITE, GWRITE_latency},
                {CommandType::COMP, 1},
                {CommandType::COMP_HASH, 1},
                {CommandType::READRES, write_to_read_l},  // bus direction exchange
        };

    same_channel[static_cast<int>(CommandType::COMP)] =
        std::vector<std::pair<CommandType, int>>{
                    {CommandType::P_HEADER, 1},
                    {CommandType::GWRITE, 1},
                    {CommandType::COMP, std::max(read_to_read_l, pim_latency_per_burst)},
                    {CommandType::COMP_HASH, std::max(read_to_read_l, pim_latency_per_burst)},
                    {CommandType::READRES, 1},
                    {CommandType::PIM_PRECHARGE, read_to_precharge},  // 在comp 之后 precharge
        };


    uint32_t burst_times_per_pim_hash = MyAddressAllocator::AddrGranularity_Hash_Bytes / MyAddressAllocator::dram_burst_size;

    same_channel[static_cast<int>(CommandType::COMP_HASH)] =
        std::vector<std::pair<CommandType, int>>{
                        {CommandType::P_HEADER, burst_times_per_pim_hash * burst_cycle+ 1},
                        {CommandType::GWRITE, burst_times_per_pim_hash * burst_cycle + 1},
                        {CommandType::COMP, burst_times_per_pim_hash * read_to_read_l + 1},
                        {CommandType::COMP_HASH, burst_times_per_pim_hash * read_to_read_l + 1},
                        {CommandType::READRES, burst_times_per_pim_hash * burst_cycle + 1},
                        {CommandType::PRECHARGE, burst_times_per_pim_hash * burst_cycle + read_to_precharge},
        };

    same_channel[static_cast<int>(CommandType::READRES)] =
        std::vector<std::pair<CommandType, int>>{
                {CommandType::P_HEADER, READERS_latency},
                {CommandType::GWRITE, write_to_read_l},
                {CommandType::COMP, 1},
                {CommandType::COMP_HASH, 1},
                {CommandType::READRES, READERS_latency},
        };

    // Dual Bank pipeline PIM operation
    // 仅有PIM Precharge，PIM Activate会由于dual bank的设置发生约束时间的变化
    if (PIM_Parameters::dual_bank) {
        if (config.IsGDDR() || config.protocol == DRAMProtocol::LPDDR4 || config.protocol == DRAMProtocol::LPDDR5) {
            pim_precharge_latency = precharge_to_precharge * MyAddressAllocator::ranks * MyAddressAllocator::bankgroups * MyAddressAllocator::banks / 2;
        }
        else {
            pim_precharge_latency = MyAddressAllocator::ranks * MyAddressAllocator::bankgroups * MyAddressAllocator::banks / 2;
        }

        if (config.IsGDDR()) {
            pim_activate_latency = config.t32AW * std::ceil(static_cast<double>(MyAddressAllocator::bankgroups * MyAddressAllocator::banks / 2) / 32);
        }
        else {
            pim_activate_latency = config.tFAW * std::ceil(static_cast<double>(MyAddressAllocator::bankgroups * MyAddressAllocator::banks / 2 - 1) / 4);
        }
        // Activate interleaved on banks of different Bankgroups
        // pim_activate_latency = std::max(pim_activate_latency, static_cast<int>(activate_to_activate_s * (MyAddressAllocator::ranks * MyAddressAllocator::bankgroups * MyAddressAllocator::banks / 2 - 1)));

        // In dual bank mode, precharge time is half
        // 还需要完成进一步的改进
        same_channel[static_cast<int>(CommandType::PIM_PRECHARGE)] =
            std::vector<std::pair<CommandType, int>>{
                    {CommandType::PIM_PRECHARGE, pim_precharge_latency},
                    {CommandType::PIM_ACTIVE, pim_precharge_to_activate},
                };

        same_channel[static_cast<int>(CommandType::PIM_ACTIVE)] =
            std::vector<std::pair<CommandType, int>>{
                    {CommandType::PIM_PRECHARGE, std::max(pim_activate_latency, activate_to_precharge)},
                    {CommandType::COMP, std::max(pim_activate_latency + pim_activate_to_read, COMP_latency_dual_bank - pim_precharge_to_activate)},
                    {CommandType::COMP_HASH, std::max(pim_activate_latency + pim_activate_to_read, COMP_latency_dual_bank - pim_precharge_to_activate)},
                    {CommandType::PIM_ACTIVE, std::max(pim_activate_latency, 0)},  // latency between two Activation
                };
    }
    else {
        if (config.IsGDDR() || config.protocol == DRAMProtocol::LPDDR4 || config.protocol == DRAMProtocol::LPDDR5) {
            pim_precharge_latency = precharge_to_precharge * MyAddressAllocator::ranks * MyAddressAllocator::bankgroups * MyAddressAllocator::banks;
        }
        else {
            pim_precharge_latency = MyAddressAllocator::ranks * MyAddressAllocator::bankgroups * MyAddressAllocator::banks; // Precharge all banks of channel
        }

        /*
        if (config.IsGDDR()) {
            pim_activate_latency = config.t32AW * std::ceil(static_cast<double>(MyAddressAllocator::bankgroups * MyAddressAllocator::banks) / 32 - 1) + (MyAddressAllocator::bankgroups * MyAddressAllocator::banks) % 32;
        }
        else {
            pim_activate_latency = config.tFAW * std::ceil(static_cast<double>(MyAddressAllocator::bankgroups * MyAddressAllocator::banks) / 4 - 1) + 4;
        }
        */
        pim_activate_latency = config.tFAW * std::ceil(static_cast<double>(MyAddressAllocator::bankgroups * MyAddressAllocator::banks) / 4 - 1) + 4;

        // Activate interleaved on banks of different Bankgroups
        pim_activate_latency = std::max(pim_activate_latency,
            static_cast<int>(activate_to_activate_s * (MyAddressAllocator::bankgroups * MyAddressAllocator::banks)));

        same_channel[static_cast<int>(CommandType::PIM_PRECHARGE)] =
        std::vector<std::pair<CommandType, int>>{
                {CommandType::PIM_PRECHARGE, pim_precharge_latency},
                {CommandType::PIM_ACTIVE, std::max(pim_precharge_latency, pim_precharge_to_activate)},
            };

        same_channel[static_cast<int>(CommandType::PIM_ACTIVE)] =
            std::vector<std::pair<CommandType, int>>{
                {CommandType::PIM_PRECHARGE, std::max(pim_activate_latency, activate_to_precharge)},
                {CommandType::COMP, pim_activate_latency + pim_activate_to_read - 1}, // Activate to comp
                {CommandType::COMP_HASH, pim_activate_latency + pim_activate_to_read - 1},
                {CommandType::PIM_ACTIVE, pim_activate_latency + config.tFAW},
            };
    }
}


void Timing::update_pim_act_timing_for_dual_bank(uint32_t comps_per_pim_row) {
    // update the timing constrain of the latency of PIM_Active to comp
    same_channel[static_cast<int>(CommandType::PIM_ACTIVE)][1].second =
        std::max((pim_activate_latency + pim_activate_to_read - 1), static_cast<int>(comps_per_pim_row * burst_cycle) - pim_precharge_to_activate - 1) ;
}

//
} // namespace dramsim3
