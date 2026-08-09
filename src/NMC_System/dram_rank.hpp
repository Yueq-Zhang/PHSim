#ifndef DRAM_RANK_HPP
#define DRAM_RANK_HPP

#include "dram_bankgroup.hpp"
#include <fstream>
#include <sstream>

// 全局日志文件（静态，只初始化一次）
static std::ofstream g_bank_state_log;
static bool g_bank_log_open = false;

inline void init_bank_state_log() {
    if (!g_bank_log_open) {
        g_bank_state_log.open("bank_state_refresh.txt", std::ios::out | std::ios::trunc);
        g_bank_log_open = true;
        // 写表头
        g_bank_state_log
            << "event_type,"
            << "channel_id,"
            << "rank_id,"
            << "bankgroup_id,"
            << "bank_id,"
            << "refresh_cycle,"
            << "bank_state_before,"
            << "bank_state_after,"
            << "open_row_before,"
            << "open_row_after,"
            << "to_do_activate_before,"
            << "to_do_activate_after,"
            << "to_do_read_before,"
            << "to_do_read_after,"
            << "to_do_precharge_before,"
            << "to_do_precharge_after,"
            << "latest_trans_time\n";
        g_bank_state_log.flush();
    }
}

// ===== Refresh日志：dram_rank内部独立定义 =====
inline std::ofstream& get_rank_refresh_log() {
    static std::ofstream f("refresh_diag.txt",
                           std::ios::out | std::ios::trunc);
    static bool header_written = false;
    if (!header_written) {
        f << "source,rank_id,triggered_proc_cycle,"
             "refresh_exec_at,unblock_at,next_refresh_cycle\n";
        f.flush();
        header_written = true;
    }
    return f;
}


inline std::ofstream& get_newton_refresh_log() {
    static std::ofstream f("newton_refresh.txt", std::ios::out | std::ios::trunc);
    static bool header = false;
    if (!header) {
        f << "rank_id,refresh_count,refresh_cycle,next_refresh_cycle\n";
        f.flush();
        header = true;
    }
    return f;
}

class DRAMRank {
public:
    DRAMRank(MemConfig& MemConfig, int Channel_id,int Rank_id);
    ~DRAMRank();

    // uint64_t IssueTransactionToBankGroup(std::shared_ptr<Memory_Transaction> trans);
    uint64_t RankRefreshReady();
    void RankRefresh(uint64_t rank_refresh_cycle);
    void EnterSelfRefresh();
    void ExitSelfRefresh();

    void IssueActivate(std::shared_ptr<Event> event);
    void IssuePrecharge(std::shared_ptr<Event> event);
    void IssueWrite(std::shared_ptr<Event> event);
    void IssueRead(std::shared_ptr<Event> event);
    void IssuePIM(std::shared_ptr<Event> event);


    MemConfig& config_;
    int channel_id;
    int rank_id;

    bool in_self_refresh = false;

    std::vector<DRAMBankGroup*> dram_bankgroups;
    // Rank refresh relevant
    bool rank_refresh;

    uint64_t next_refresh_cycle = 0;
    int refresh_count = 0;
    int current_trans_refresh_count = 0;
    int rank_refresh_interval_ = 0;
    int rank_refresh_offset_ = 0;

    // Activation window relevant
    std::vector<std::pair<int, uint64_t>> rank_activate_recoder_;

    std::deque<cycle_type> activate_cycle_recorder;

    uint32_t to_precharge = 0;
    uint32_t to_activate = 0;
    uint64_t to_do_read = 0;
    uint64_t to_do_write = 0;

};

inline DRAMRank::DRAMRank(MemConfig& MemConfig, int Channel_id, int Rank_id): config_(MemConfig), channel_id(Channel_id), rank_id(Rank_id) {
    for (auto i = 0; i < config_.bankgroups; i++) {
        dram_bankgroups.emplace_back(new DRAMBankGroup(config_, &rank_activate_recoder_, rank_id, i));
    }
    // Logic for Rank-Level Refresh
    if (config_.refresh_policy == RefreshPolicy::RANK_LEVEL_SIMULTANEOUS) {
        rank_refresh = true;
        rank_refresh_interval_ = config_.tREFI;
    }
    else if (config_.refresh_policy == RefreshPolicy::RANK_LEVEL_STAGGERED) {
        rank_refresh = true;
        rank_refresh_interval_ = config_.tREFI;
        rank_refresh_offset_ = rank_refresh_interval_/config_.ranks * (Rank_id + 1);
    }
    else {
        rank_refresh = false;
        rank_refresh_interval_ = 0;
        rank_refresh_offset_ = 0;
        next_refresh_cycle = 0;
    }
    next_refresh_cycle = rank_refresh_offset_;

}

inline DRAMRank::~DRAMRank() {

}

inline void DRAMRank::IssuePrecharge(std::shared_ptr<Event> event) {
    if (config_.IsGDDR() || config_.protocol == DRAMProtocol::LPDDR4 || config_.protocol == DRAMProtocol::LPDDR5) {
        for (auto bankgroup : dram_bankgroups) {
            if (bankgroup->bankgroup_id == event->bankgroup_index) {  // same bankgroup other banks
                auto bank = bankgroup->dram_banks[event->bank_index];
                bank->Precharge(event);
                bankgroup->SameBankgroupPrecharge(event);
            }
            else {  // other bankgroup
                to_precharge = event->precharge_cycle + EventDrivenParams::precharge_to_precharge;
            }
        }
    }
    dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_activate = event->precharge_cycle + EventDrivenParams::precharge_to_activate;
}


inline void DRAMRank::IssueActivate(std::shared_ptr<Event> event) {
    // update Activation time window, for tFAW and tFAW32
    if (config_.IsGDDR()) {  // To store 32 event time
        if (activate_cycle_recorder.size() >= 32) {
            activate_cycle_recorder.pop_front();
        }
    }
    else {  // To store 4 event time
        if (activate_cycle_recorder.size() >= 4) {
            activate_cycle_recorder.pop_front();
        }
    }
    activate_cycle_recorder.push_back(event->activate_cycle);

    // 之后基于Event操作的位置更新Rank中的所有Bankgroup、以及Bank的延时
    for (auto bankgroup : dram_bankgroups) {
        if (bankgroup->bankgroup_id == event->bankgroup_index) {   // The Operation BankGroup
            auto bank = bankgroup->dram_banks[event->bank_index]; // The operation bank
            bank->Activate(event);
            bankgroup->SameBankgroupActivate(event);   // 操作Bank所在的bankgroup
        }
        else { // 这里是针对其他的bankgroup
            bankgroup->OtherBankgroupActivate(event); // 其他的bankgroup
        }
    }
    event->execute_cycle = event->activate_cycle + 1;
}



inline void DRAMRank::IssueRead(std::shared_ptr<Event> event) {

    for (auto bankgroup : dram_bankgroups) {
        if (bankgroup->bankgroup_id == event->bankgroup_index) {   // The Operation BankGroup
            bankgroup->dram_banks[event->bank_index]->Read(event); // The operation bank
            bankgroup->SameBankgroupRead(event);   // 操作Bank所在的bankgroup
        }
        else { // 这里是针对其他的bankgroup
            bankgroup->OtherBankgroupRead(event); // 其他的bankgroup
        }
    }
}


inline void DRAMRank::IssueWrite(std::shared_ptr<Event> event) {
    for (auto bankgroup : dram_bankgroups) {
        if (bankgroup->bankgroup_id == event->bankgroup_index) {   // The Operation BankGroup
            bankgroup->dram_banks[event->bank_index]->Write(event); // The operation bank
            bankgroup->SameBankgroupRead(event);   // 操作Bank所在的bankgroup
        }
        else { // 这里是针对其他的bankgroup
            bankgroup->OtherBankgroupRead(event); // 其他的bankgroup
        }
    }
}


inline void DRAMRank::IssuePIM(std::shared_ptr<Event> event) {


}

/*
inline uint64_t DRAMRank::IssueTransactionToBankGroup(std::shared_ptr<Memory_Transaction> trans) {
    // 基于当前Rank的操作当前事件的传播进行延时判定， 例如 在Rank中可能出现的情况是整个Rank处于刷新，或者refresh操作过程中，需要等待解除该操作过程
    if (rank_refresh) {
        if (trans->processing_cycle >= next_refresh_cycle) {
            current_trans_refresh_count = (trans->processing_cycle - rank_refresh_offset_) / rank_refresh_interval_ + 1;
            // spdlog::critical("Current Transaction issue time {} > Next Rank Refresh time {}, the {}th refresh is required for Rank {}", trans->processing_cycle, next_refresh_cycle, current_trans_refresh_count, rank_id);
            uint64_t rank_refresh_time = RankRefreshReady();

            // ===== 打印EventDriven实际触发refresh =====
            auto& log = get_rank_refresh_log();
            log << "EVENT_ACTUAL,"
                << rank_id                      << ","
                << trans->processing_cycle      << ","  // 触发时的processing_cycle
                << rank_refresh_time            << ","  // refresh实际执行时间
                << rank_refresh_time + config_.tRFC << ","  // bank unblock时间
                << next_refresh_cycle           << "\n";
            log.flush();
            // ==========================================

            RankRefresh(rank_refresh_time);
        }
    }

    uint64_t trans_processing_cycle = 0;
    if (trans->address.row != dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->open_row) {
        trans_processing_cycle = dram_bankgroups[trans->address.bankgroup]->IssueTransactionToBank(trans);  // Rank向BankGroup分发命令时，允许先进行precharge或者激活操作
        if (trans->is_write) {
            to_do_read = trans_processing_cycle + config_.other_bankgroup_write_to_read_latency;
            to_do_write = trans_processing_cycle + config_.other_bankgroup_write_to_write_latency;
        }
        else {
            to_do_read = trans_processing_cycle + config_.other_bankgroup_read_to_read_latency;
            to_do_write = trans_processing_cycle + config_.other_bankgroup_read_to_write_latency;
        }
    }
    else {
        if (trans->is_write) {
            if (trans->processing_cycle < to_do_write) {
                trans->processing_cycle = to_do_write;
                // std::cout << "The Write of address = " << trans->addr <<" is blocked until " << to_do_write << std::endl;
            }
            trans_processing_cycle = dram_bankgroups[trans->address.bankgroup]->IssueTransactionToBank(trans);
            to_do_read = trans_processing_cycle + config_.other_bankgroup_write_to_read_latency;
            to_do_write = trans_processing_cycle + config_.other_bankgroup_write_to_write_latency;
        }
        else {
            if (trans->processing_cycle < to_do_read) {
                trans->processing_cycle = to_do_read;
                // std::cout << "The Read of address = " << trans->addr <<" is blocked until " << to_do_read << std::endl;
            }
            trans_processing_cycle = dram_bankgroups[trans->address.bankgroup]->IssueTransactionToBank(trans);
            to_do_read = trans_processing_cycle + config_.other_bankgroup_read_to_read_latency;
            to_do_write = trans_processing_cycle + config_.other_bankgroup_read_to_write_latency;
        }
    }
    return trans_processing_cycle;
}
*/

inline uint64_t DRAMRank::RankRefreshReady() {

    uint64_t refresh_time = (current_trans_refresh_count - 1) * rank_refresh_interval_ + rank_refresh_offset_;
    if (current_trans_refresh_count - refresh_count == 1) {
        uint64_t to_refresh_time = 0;


        // 检查当前Rank中所有Bank的状态使其达到需求
        for (auto bankgroup : dram_bankgroups) {
            for (auto bank : bankgroup->dram_banks) {
                if (bank->bank_state == BankState::OPEN) {
                    // 如果bank_state处于开启状态，需要发送precharge的指令，同时检查延迟，最低为0，即当前周期就可以
                    bank->bank_state = BankState::CLOSED;
                    bank->open_row = -1;
                    if (refresh_time < bank->to_do_precharge) {
                        refresh_time = bank->to_do_precharge;
                    }
                    // refresh_time = refresh_time > bank->to_do_precharge ? refresh_time : bank->to_do_precharge;
                    refresh_time ++;
                    to_refresh_time = config_.tRP;
                }
                else {  // bank->bank_state == BankState::close
                    to_refresh_time = refresh_time + to_refresh_time < bank->latest_trans_time + config_.tRP ? bank->latest_trans_time + config_.tRP - refresh_time : to_refresh_time ;   // precharge to refresh
                }
            }
        }
        refresh_time += to_refresh_time;
    }

    else if (current_trans_refresh_count <= refresh_count) {
        throw std::runtime_error("Rank Refresh Error !!! ");
    }
    refresh_count = current_trans_refresh_count;
    next_refresh_cycle = refresh_count * rank_refresh_interval_ + rank_refresh_offset_;

    // spdlog::critical("At {}, a Rank_Refresh operation is executed for rank {}, the next refresh time is {}", refresh_time - 1, rank_id, next_refresh_cycle);
    return refresh_time - 1;
}


inline void DRAMRank::RankRefresh(uint64_t rank_refresh_cycle) {
    spdlog::critical("A Rank Refresh Operation is executed for Rank {} at time {}", rank_id, rank_refresh_cycle);

    for (int bg_idx = 0; bg_idx < static_cast<int>(dram_bankgroups.size()); ++bg_idx) {
        auto* bankgroup = dram_bankgroups[bg_idx];
        for (int bk_idx = 0; bk_idx < static_cast<int>(bankgroup->dram_banks.size()); ++bk_idx) {
            auto* bank = bankgroup->dram_banks[bk_idx];
            /*bank->bank_state = BankState::CLOSED;
            bank->to_do_activate = rank_refresh_cycle + config_.tRFC;*/
            // 必须重置的字段
            // ===== 记录更新前的状态 =====
            std::string state_before = (bank->bank_state == BankState::OPEN)
                                      ? "OPEN" : "CLOSED";
            int open_row_before       = bank->open_row;
            uint64_t activate_before  = bank->to_do_activate;
            uint64_t read_before      = bank->to_do_read;
            uint64_t pre_before       = bank->to_do_precharge;
            uint64_t latest_trans     = bank->latest_trans_time;

            // ===== 执行状态更新 =====
            bank->bank_state    = BankState::CLOSED;
            bank->open_row      = -1;
            bank->open_row_exec_event_count = 0;
            bank->pending_precharge = false;
            bank->pending_activate = false;
            bank->row_hit_count = 0;
            bank->to_do_activate  = rank_refresh_cycle + config_.tRFC;
            bank->to_do_precharge = 0;
            bank->to_do_read      = 0;
            bank->to_do_write     = 0;

            // ===== 写入日志 =====
            g_bank_state_log
                << "RANK_REFRESH,"
                << channel_id    << ","
                << rank_id       << ","
                << bg_idx        << ","
                << bk_idx        << ","
                << rank_refresh_cycle << ","
                << state_before  << ","
                << "CLOSED,"                      // state_after 固定是CLOSED
                << open_row_before << ","
                << -1            << ","           // open_row_after 固定是-1
                << activate_before << ","
                << (rank_refresh_cycle + config_.tRFC) << ","
                << read_before   << ","
                << 0             << ","           // to_do_read_after = 0
                << pre_before    << ","
                << 0             << ","           // to_do_precharge_after = 0
                << latest_trans  << "\n";
        }
    }
    g_bank_state_log.flush();
}


inline void DRAMRank::EnterSelfRefresh() {
    in_self_refresh = true;
}


inline void DRAMRank::ExitSelfRefresh() {
    in_self_refresh = false;
}




#endif
