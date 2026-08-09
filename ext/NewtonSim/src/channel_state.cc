#include "channel_state.h"

#include <iterator>

#include "../../../src/DRAM/Dram.h"
#include "spdlog/spdlog.h"

namespace dramsim3 {
ChannelState::ChannelState(int channel_id, const Config &config, Timing &timing):
    rank_idle_cycles(config.ranks, 0),
    channel_id_(channel_id),
    config_(config),
    timing_(timing),
    rank_is_sref_(config.ranks, false),
    four_aw_(config_.ranks, std::vector<uint64_t>()),
    thirty_two_aw_(config_.ranks, std::vector<uint64_t>()) {
    comp_overhead_flag_ = true;

    bank_states_.reserve(config_.ranks);
    for (auto i = 0; i < config_.ranks; i++) {
        auto rank_states = std::vector<std::vector<BankState>>();
        rank_states.reserve(config_.bankgroups);
        for (auto j = 0; j < config_.bankgroups; j++) {
            auto bg_states = std::vector<BankState>(config_.banks_per_group,BankState(config_.enable_dual_buffer));
            rank_states.push_back(bg_states);
        }
        bank_states_.push_back(rank_states);
    }
}

// gsheo: just for all idle cycle count purpose.
bool ChannelState::IsAllBankIdleInRank(int rank) const {
    for (int j = 0; j < config_.bankgroups; j++) {
        for (int k = 0; k < config_.banks_per_group; k++) {
            // gsheo: IsRowOpen -> IsUsed
            if (bank_states_[rank][j][k].IsUsed()) {
                return false;
            }
        }
    }
    return true;
}

bool ChannelState::IsPIMIdleInRank(int rank) const {
    for (int j = 0; j < config_.bankgroups; j++) {
        for (int k = 0; k < config_.banks_per_group; k++) {
            if (bank_states_[rank][j][k].IsPIMUsed()) {
                return false;
            }
        }
    }
    return true;
}

// gsheo: not-used in original dramsim3
bool ChannelState::IsRWPendingOnRef(const Command &cmd) const {
    int rank = cmd.Rank();
    int bankgroup = cmd.Bankgroup();
    int bank = cmd.Bank();
    return (IsRowOpen(rank, bankgroup, bank) && RowHitCount(rank, bankgroup, bank) == 0 &&
            bank_states_[rank][bankgroup][bank].OpenRow() == cmd.Row());
}

/* gsheo: BankNeedRefresh is not used */
// refresh_policy: hbm, ddr4 - RANK_LEVEL_STAGGERED
// -> BankNeedRef(~, ~, ~, true) is not used
void ChannelState::BankNeedRefresh(int rank, int bankgroup, int bank, bool need) {
    if (need) {
        Address addr = Address(-1, rank, bankgroup, bank, -1, -1);
        refresh_q_.emplace_back(CommandType::REFRESH_BANK, addr, -1);
    } else {
        for (auto it = refresh_q_.begin(); it != refresh_q_.end(); it++) {
            if (it->Rank() == rank && it->Bankgroup() == bankgroup && it->Bank() == bank) {
                refresh_q_.erase(it);
                break;
            }
        }
    }
    return;
}

// Called by Refresh::InsertRefresh with need=true
// Called by ChannelState::UpdateState(cmd=REFRESH) with need=false
void ChannelState::RankNeedRefresh(int rank, bool need) {
    if (need) {
        Address addr = Address(-1, rank, -1, -1, -1, -1);
        refresh_q_.emplace_back(CommandType::REFRESH, addr, -1);
    } else {
        // Remove corresponding Rank target refresh command while rotating refresh_queue
        for (auto it = refresh_q_.begin(); it != refresh_q_.end(); it++) {
            if (it->Rank() == rank) {
                refresh_q_.erase(it);
                break;
            }
        }
    }
    return;
}

bool ChannelState::IsOpenTargetBanks(const Command &cmd) {
    int target_row = cmd.Row();
    if (cmd.IsPIMComp()) {
        int cur;
        // Check all banks open row
        for (auto i = 0; i < config_.ranks; i++) {
            for (auto j = 0; j < config_.bankgroups; j++) {
                for (auto k = 0; k < config_.banks_per_group; k++) {
                    cur = bank_states_[i][j][k].PIMOpenRow();
                    if (cur != target_row) return false;
                }
            }
        }
        return true;
    }
    else if (cmd.IsGwrite()) {
        auto bank_state = bank_states_[cmd.Rank()][cmd.Bankgroup()][cmd.Bank()];
        if (!bank_state.IsRowOpen()) return false;

        return target_row == bank_state.OpenRow();
    }
    return true;
}

bool ChannelState::CheckAllBanksSamePIMOpenRow(int reserved_row) {
    int pim_open_row;
    int cur;
    int cur_pim;
    for (auto i = 0; i < config_.ranks; i++) {
        for (auto j = 0; j < config_.bankgroups; j++) {
            for (auto k = 0; k < config_.banks_per_group; k++) {
                cur = bank_states_[i][j][k].OpenRow();
                cur_pim = bank_states_[i][j][k].PIMOpenRow();
                if (i + j + k == 0) pim_open_row = cur_pim;
                if (config_.enable_dual_buffer && cur == reserved_row) {
                    PrintError("Reserved row already opened in dram row buffer!");
                } else if (pim_open_row != cur_pim) {
                    PrintAllBankStates();
                    PrintError("PIM open row mismatch, pim_open_row:", pim_open_row,
                               ", cur_pim_open:", cur_pim);
                    return false;
                } else
                    continue;
            }
        }
    }
    pim_open_row_ = pim_open_row;
    return true;
}

Command ChannelState::GetReadyCommand(const Command &cmd, uint64_t clk) const {
    Command ready_cmd = Command();

    if (cmd.IsPIMHeader() && cmd.for_gwrite) {
        PrintError("PIM header for gwrite is deprecated!");
    }
    if (cmd.IsChannelCMD()) {  // P_HEADER, COMP, READRES, COMPS_READRES
        int num_ready = 0;
        int num_total_banks = config_.ranks * config_.bankgroups * config_.banks_per_group;
        int num_ready_gact = 0;
        PrintAllBankStates();
        for (auto i = 0; i < config_.ranks; i++) {
            for (auto j = 0; j < config_.bankgroups; j++) {
                for (auto k = 0; k < config_.banks_per_group; k++) {
                    ready_cmd = bank_states_[i][j][k].GetReadyCommand(cmd, clk);  // 需要检查每一个Bank_state的工作情况，从而选择性执行后续计算，其实除了COMP指令以外，其他的操作不需要
                    if (!ready_cmd.IsValid()) continue;
                    if (ready_cmd.cmd_type != cmd.cmd_type) {
                        /*
                        if (ready_cmd.cmd_type == CommandType::PIM_ACTIVE || ready_cmd.cmd_type == CommandType::PIM_PRECHARGE) {
                            spdlog::info("The {} command is required for {} command execution, for row {}", ready_cmd.CommandTypeString(), cmd.CommandTypeString(), ready_cmd.Row());
                        }
                        */
                        return ready_cmd;
                    }
                    else {
                        num_ready++;
                    }
                }
            }
        }
        if (num_ready == num_total_banks) {  // 所有的Bank满足时序要求，PIM指令ready
            return ready_cmd;
        } else {
            return Command();
        }
    }

    if (cmd.IsRankCMD()) {  // REFRESH, SREF_ENTER, SREF_EXIT
        int num_ready = 0;
        for (auto j = 0; j < config_.bankgroups; j++) {
            for (auto k = 0; k < config_.banks_per_group; k++) {
                ready_cmd = bank_states_[cmd.Rank()][j][k].GetReadyCommand(cmd, clk);
                if (!ready_cmd.IsValid()) {  // Not ready
                    continue;
                }
                if (ready_cmd.cmd_type != cmd.cmd_type) {  // likely PRECHARGE
                    Address new_addr = Address(-1, cmd.Rank(), j, k, -1, -1);
                    ready_cmd.addr = new_addr;
                    return ready_cmd;
                } else {
                    num_ready++;
                }
            }
        }
        // All bank ready
        if (num_ready == config_.banks) {
            return ready_cmd;
        } else {
            return Command();
        }
    }

    if (cmd.cmd_type == CommandType::G_ACT) {
        int num_ready = 0;
        // PrintDebug("get ready_cmd for G_ACT");

        for (auto k = 0; k < config_.banks_per_group; k++) {
            ready_cmd = bank_states_[cmd.Rank()][cmd.Bankgroup()][k].GetReadyCommand(cmd, clk);
            if (!ready_cmd.IsValid()) {  // Not ready
                continue;
            }
            if (ready_cmd.cmd_type != cmd.cmd_type) {
                if (ready_cmd.IsPIMPrecharge()) {
                    Address new_addr = Address(-1, cmd.Rank(), cmd.Bankgroup(), k, -1, -1);
                    ready_cmd.addr = new_addr;
                    PrintInfo("PIM precharge");
                    return ready_cmd;
                } else {
                    if (!config_.enable_dual_buffer &&
                        ready_cmd.cmd_type == CommandType::PRECHARGE) {
                        Address new_addr = Address(-1, cmd.Rank(), cmd.Bankgroup(), k, -1, -1);
                        ready_cmd.addr = new_addr;
                        return ready_cmd;
                    }
                    PrintError("Invalid command type", ready_cmd.CommandTypeString());
                }
            } else {
                if (!ActivationWindowOk(ready_cmd.Rank(), clk)) return Command();
                num_ready++;
            }
        }

        // All bankgroup ready
        if (num_ready == config_.banks_per_group) {
            return ready_cmd;
        } else {
            // PrintInfo("Not ready for G_ACT ,#ready:" +
            //                            std::to_string(num_ready));
            // std::cout << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
            //           << " bank:" << cmd.Bank() << " ";
            // PrintAllBankStates();
            return Command();
        }
    }
    else {
        // READ, WRITE, PWRITE, COMP_HASH
        ready_cmd = bank_states_[cmd.Rank()][cmd.Bankgroup()][cmd.Bank()].GetReadyCommand(cmd, clk);

        /*
        if (ready_cmd.cmd_type != CommandType::SIZE) {
            spdlog::info("Channel {} Check the bank state of Rank {}, Bankgroup {}, Bank {} Row {}, Column {} for {} command at dram cycle {}, return {}", channel_id_, cmd.Rank(), cmd.Bankgroup(), cmd.Bank(), cmd.Row(), cmd.Column() , cmd.CommandTypeString(), clk, ready_cmd.CommandTypeString());
        }
        */

        if (!ready_cmd.IsValid()) {
            return Command();
        }
        if (ready_cmd.cmd_type == CommandType::ACTIVATE) {
            if (!ActivationWindowOk(ready_cmd.Rank(), clk)) {
                return Command();
            }
        }
        return ready_cmd;
    }
}

int ChannelState::EstimatePIMOperationLatency(const Command &cmd, uint64_t clk) {
    bool pheader_for_gemv = cmd.IsPIMHeader() && !cmd.for_gwrite;
    assert(cmd.IsGwrite() || pheader_for_gemv);

    int target_row = cmd.Row();

    int precharge_latency = config_.AL + config_.tRTP;  // read_to_precharge
    int latency = 0;
    if (cmd.IsGwrite()) {
        if (IsRowOpen(cmd.Rank(), cmd.Bankgroup(), cmd.Bank())) {
            if (OpenRow(cmd.Rank(), cmd.Bankgroup(), cmd.Bank()) != target_row) {
                latency += precharge_latency;
                latency += config_.tRP        // precharge_to_act
                           + config_.tRCDRD;  // act_to_read
            }
        } else {
            latency += config_.tRP        // precharge_to_act
                       + config_.tRCDRD;  // act_to_read
        }
        latency += config_.gwrite_delay;
        return latency;

    } else {
        int num_gact = 0;
        int num_precharge = 0;
        int num_comps = std::max(cmd.num_comps, 6);  // pipeline filling time: 6
        int num_readres = cmd.num_readres;

        for (auto i = 0; i < config_.ranks; i++) {
            for (auto j = 0; j < config_.bankgroups; j++) {
                int need_precharge = 0;
                int proper_open = 0;
                for (auto k = 0; k < config_.banks_per_group; k++) {
                    int pim_open_row = bank_states_[i][j][k].PIMOpenRow();
                    if (pim_open_row == -1) {
                        // need to activate this row
                    } else if (pim_open_row == target_row) {
                        proper_open++;
                    } else {
                        // need precharge this row
                        need_precharge++;
                    }
                }
                if (proper_open == config_.banks_per_group) {
                    // lucky ! we don't need to G_ACT for this group
                } else {
                    // need G_ACT and precharge
                    num_precharge += need_precharge;
                    num_gact++;
                }
            }
        }
        latency += num_precharge * precharge_latency;
        latency += config_.tFAW * (num_gact / 2 - 1) + config_.tRCDRD;

        latency += std::max(config_.burst_cycle, config_.tCCD_S) * num_comps;
        latency += std::max(config_.burst_cycle, config_.tCCD_L) * (num_readres - 1);
        return latency;
    }
}

void ChannelState::PrintAllBankStates() const {
    if (!LOGGING_CONFIG::PIMSIM_LOGGING_DEBUG) return;
    if (channel_id_ != 0) return;
    std::cout << std::endl
              << ColorString(Color::GREEN) << "==== PIM States (ch:" << channel_id_
              << ") ====" << std::endl;

    for (auto j = 0; j < config_.bankgroups; j++) {
        std::cout << "[BankGroup " << j << "] ";
        for (auto k = 0; k < config_.banks_per_group; k++) {
            std::string bs = bank_states_[0][j][k].PIMStateToString();
            std::cout << std::setw(6) << bs << " ";
        }
        std::cout << "| ";
        for (auto k = 0; k < config_.banks_per_group; k++) {
            int pim_open_row = bank_states_[0][j][k].PIMOpenRow();
            std::cout << std::setw(4) << pim_open_row << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "===============" << ColorString(Color::RESET) << std::endl << std::endl;
}

void ChannelState::UpdateState(const Command &cmd) {
    if (cmd.IsChannelCMD()) {  // All the PIM Command
        for (auto i = 0; i < config_.ranks; i++) {
            for (auto j = 0; j < config_.bankgroups; j++) {
                for (auto k = 0; k < config_.banks_per_group; k++) {
                    bank_states_[i][j][k].UpdateState(cmd);
                }
            }
        }
    }
    else if (cmd.IsRankCMD()) {  // REFRESH, SREF_ENTER, SREF_EXIT
        for (auto j = 0; j < config_.bankgroups; j++) {
            for (auto k = 0; k < config_.banks_per_group; k++) {
                bank_states_[cmd.Rank()][j][k].UpdateState(cmd);
            }
        }
        if (cmd.IsRefresh()) {  // gsheo: REFRESH
            RankNeedRefresh(cmd.Rank(), false);
        } else if (cmd.cmd_type == CommandType::SREF_ENTER) {
            rank_is_sref_[cmd.Rank()] = true;
        } else if (cmd.cmd_type == CommandType::SREF_EXIT) {
            rank_is_sref_[cmd.Rank()] = false;
        }
    }
    else if (cmd.cmd_type == CommandType::G_ACT or cmd.cmd_type == CommandType::G_PRE) {
        for (uint32_t k = 0; k < config_.banks_per_group; k++) {
            bank_states_[cmd.Rank()][cmd.Bankgroup()][k].UpdateState(cmd);  // update state of an bankgroup
        }
    }
    else {  // The other operation for the ccorresponding bank
        bank_states_[cmd.Rank()][cmd.Bankgroup()][cmd.Bank()].UpdateState(cmd);
        if (cmd.IsRefresh()) {
            BankNeedRefresh(cmd.Rank(), cmd.Bankgroup(), cmd.Bank(), false);
        }
    }
}

void ChannelState::UpdateTiming(const Command &cmd, uint64_t clk) {
    switch (cmd.cmd_type) {
        case CommandType::ACTIVATE:
            UpdateActivationTimes(cmd.Rank(), clk);
        case CommandType::READ:
        case CommandType::READ_PRECHARGE:
        case CommandType::WRITE:
        case CommandType::WRITE_PRECHARGE:
        case CommandType::PRECHARGE:
        case CommandType::REFRESH_BANK:
            // todo - simulator speed? - Speciazlize which of the below
            // functions to call depending on the command type
            // Same Bank
            UpdateSameBankTiming(cmd.addr, timing_.same_bank[static_cast<int>(cmd.cmd_type)], clk);
            // Same Bankgroup other banks
            UpdateOtherBanksSameBankgroupTiming(cmd.addr, timing_.other_banks_same_bankgroup[static_cast<int>(cmd.cmd_type)], clk);
            // Other bankgroups
            UpdateOtherBankgroupsSameRankTiming(cmd.addr, timing_.other_bankgroups_same_rank[static_cast<int>(cmd.cmd_type)], clk);
            // Other ranks
            UpdateOtherRanksTiming(cmd.addr, timing_.other_ranks[static_cast<int>(cmd.cmd_type)], clk);
            break;
        case CommandType::REFRESH:
        case CommandType::SREF_ENTER:
        case CommandType::SREF_EXIT:
            UpdateSameRankTiming(cmd.addr, timing_.same_rank[static_cast<int>(cmd.cmd_type)], clk);
            break;
        // For PIM Command
        case CommandType::P_HEADER:
        case CommandType::GWRITE:
        case CommandType::COMP:
        case CommandType::COMP_HASH:
        case CommandType::READRES:
        case CommandType::PIM_PRECHARGE:
            UpdateTimingForPIM(cmd, timing_.same_channel[static_cast<int>(cmd.cmd_type)], clk); // PIM更新基于Same Channel实现
            break;
        case CommandType::PIM_ACTIVE:
            if (PIM_Parameters::dual_bank) {
                timing_.update_pim_act_timing_for_dual_bank(comps_per_pim_row_);
            }
            UpdateTimingForPIM(cmd, timing_.same_channel[static_cast<int>(cmd.cmd_type)], clk); // PIM更新基于Same Channel实现
            break;
        case CommandType::G_ACT:
        case CommandType::G_PRE:
            UpdateSameBankgroupTiming(cmd.addr, timing_.other_banks_same_bankgroup[static_cast<int>(cmd.cmd_type)], clk);
            UpdateSameRankTiming(cmd.addr, timing_.same_rank[static_cast<int>(cmd.cmd_type)], clk);
            break;
        default:
            AbruptExit(__FILE__, __LINE__);
    }
    // if (cmd.IsPIMCommand())
    //     PrintDebug("UpdateTiming Done");
    return;
}

void ChannelState::UpdateChannelTiming(const Command &cmd, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    int comp_to_comp = std::max(config_.burst_cycle, config_.tCCD_L);
    int num_comps = std::max(cmd.num_comps, 6);
    // adder tree filling time
    // PrintImportant("(UpdateChannelTiming) time:", clk + comps_readres_delay, "clk:", clk);
    if (cmd.IsPIMCommand()) {
        spdlog::info("PIM Command {} is executed in channel {} at cycle {}, Update Timing", cmd.CommandTypeString(), channel_id_, clk);
    }

    for (uint32_t i = 0; i < config_.ranks; i++) {
        for (uint32_t j = 0; j < config_.bankgroups; j++) {
            for (uint32_t k = 0; k < config_.banks_per_group; k++) {
                BankState &cur_bank_state = bank_states_[i][j][k];
                for (auto cmd_timing : cmd_timing_list) {
                    cur_bank_state.UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
                }
                // cur_bank_state.UpdateTiming(CommandType::COMPS_READRES, clk + comps_readres_delay);
            }
        }
    }
}

void ChannelState::UpdateSameBankTiming(const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    for (auto cmd_timing : cmd_timing_list) {
        bank_states_[addr.rank][addr.bankgroup][addr.bank].UpdateTiming(cmd_timing.first,clk + cmd_timing.second);
    }
}

void ChannelState::UpdateSameBankgroupTiming(const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    for (uint32_t k = 0; k < config_.banks_per_group; k++) {
        for (auto cmd_timing : cmd_timing_list) {
            bank_states_[addr.rank][addr.bankgroup][k].UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
        }
    }
}

void ChannelState::UpdateOtherBanksSameBankgroupTiming(const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    for (uint32_t k = 0; k < config_.banks_per_group; k++) {
        if (k != addr.bank) {
            for (auto cmd_timing : cmd_timing_list) {
                bank_states_[addr.rank][addr.bankgroup][k].UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
            }
        }
    }
}

void ChannelState::UpdateOtherBankgroupsSameRankTiming(const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    for (uint32_t j = 0; j < config_.bankgroups; j++) {
        if (j != addr.bankgroup) {
            for (uint32_t k = 0; k < config_.banks_per_group; k++) {
                for (auto cmd_timing : cmd_timing_list) {
                    bank_states_[addr.rank][j][k].UpdateTiming(cmd_timing.first,clk + cmd_timing.second);
                }
            }
        }
    }
}

void ChannelState::UpdateOtherRanksTiming(const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    for (uint32_t i = 0; i < config_.ranks; i++) {
        if (i != addr.rank) {
            for (uint32_t j = 0; j < config_.bankgroups; j++) {
                for (uint32_t k = 0; k < config_.banks_per_group; k++) {
                    for (auto cmd_timing : cmd_timing_list) {
                        bank_states_[i][j][k].UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
                    }
                }
            }
        }
    }
}

// Now use for only COMP
void ChannelState::UpdateTimingForPIM(const Command &cmd, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    // int pipeline_filling_time = config_.tCCD_S * 6;
    // spdlog::info("Update Timing of channel {} for PIM Command {}", channel_id_, cmd.CommandTypeString());

    if (cmd.cmd_type == CommandType::COMP_HASH) {
        // spdlog::info("Update timing for COMP HASH to the corresponding bank");
        BankState &cur_bank_state = bank_states_[cmd.addr.rank][cmd.addr.bankgroup][cmd.addr.bank];
        for (auto cmd_timing : cmd_timing_list) {
            cur_bank_state.UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
        }
    }
    else {
        for (auto i = 0; i < config_.ranks; i++) {
            for (auto j = 0; j < config_.bankgroups; j++) {
                for (auto k = 0; k < config_.banks_per_group; k++) {
                    BankState &cur_bank_state = bank_states_[i][j][k];
                    for (auto cmd_timing : cmd_timing_list) {
                        cur_bank_state.UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
                    }
                }
            }
        }
    }
}

void ChannelState::UpdateSameRankTiming(const Address &addr, const std::vector<std::pair<CommandType, int>> &cmd_timing_list, uint64_t clk) {
    for (auto j = 0; j < config_.bankgroups; j++) {
        for (auto k = 0; k < config_.banks_per_group; k++) {
            for (auto cmd_timing : cmd_timing_list) {
                // if (cmd_timing.first == CommandType::G_ACT) {
                //     std::cout << "timing update for G_ACT, clk:";
                //     std::cout << clk << " added:" << cmd_timing.second
                //               << std::endl;
                // }
                bank_states_[addr.rank][j][k].UpdateTiming(cmd_timing.first, clk + cmd_timing.second);
            }
        }
    }
}

void ChannelState::UpdateTimingAndStates(const Command &cmd, uint64_t clk) {
    UpdateState(cmd);
    UpdateTiming(cmd, clk);
}

bool ChannelState::ActivationWindowOk(int rank, uint64_t curr_time) const {
    bool tfaw_ok = IsFAWReady(rank, curr_time);
    if (config_.IsGDDR()) {
        if (!tfaw_ok)
            return false;
        else
            return Is32AWReady(rank, curr_time);
    }
    return tfaw_ok;
}

void ChannelState::UpdateActivationTimes(int rank, uint64_t curr_time) {
    if (!four_aw_[rank].empty() && curr_time >= four_aw_[rank][0]) {
        four_aw_[rank].erase(four_aw_[rank].begin());
    }
    four_aw_[rank].push_back(curr_time + config_.tFAW);
    if (config_.IsGDDR()) {
        if (!thirty_two_aw_[rank].empty() && curr_time >= thirty_two_aw_[rank][0]) {
            thirty_two_aw_[rank].erase(thirty_two_aw_[rank].begin());
        }
        thirty_two_aw_[rank].push_back(curr_time + config_.t32AW);
    }
}

bool ChannelState::IsFAWReady(int rank, uint64_t curr_time) const {
    if (!four_aw_[rank].empty()) {
        if (curr_time < four_aw_[rank][0] && four_aw_[rank].size() >= 4) {
            return false;
        }
    }
    return true;
}

// GDDR only
bool ChannelState::Is32AWReady(int rank, uint64_t curr_time) const {
    if (!thirty_two_aw_[rank].empty()) {
        if (curr_time < thirty_two_aw_[rank][0] && thirty_two_aw_[rank].size() >= 32) {
            return false;
        }
    }
    return true;
}

void ChannelState::UpdateCompsPerPimRow(uint32_t comps_per_pim_row) {
    comps_per_pim_row_ = comps_per_pim_row;
}



}  // namespace dramsim3
