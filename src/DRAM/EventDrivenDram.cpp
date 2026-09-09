#include <fstream>
#include <stdexcept>
#include <string>
#include "EventDrivenDram.h"

#include "../../ext/NewtonSim/src/configuration.h"
#include "DramDataContainer.h"
#include "NewtonSim/src/configuration.h"

#if ENABLE_DRAM_ALIGNMENT_TRACE
#define ED_TRACE_TRANSACTION(...) trace_transaction(__VA_ARGS__)
#define ED_TRACE_COMMAND(...) trace_issued_command(__VA_ARGS__)
#define ED_TRACE_TIME_ADVANCE(...) trace_time_advance(__VA_ARGS__)
#define ED_TRACE_ROW_DECISION(...) trace_event_row_decision(__VA_ARGS__)
#else
#define ED_TRACE_TRANSACTION(...) do { } while (0)
#define ED_TRACE_COMMAND(...) do { } while (0)
#define ED_TRACE_TIME_ADVANCE(...) do { } while (0)
#define ED_TRACE_ROW_DECISION(...) do { } while (0)
#endif

namespace {
enum class PimCmdSlot : size_t {
    P_HEADER = 0,
    GWRITE = 1,
    COMP = 2,
    COMP_HASH = 3,
    READRES = 4,
};

bool is_pim_access_type(MemoryAccessType type) {
    return type == MemoryAccessType::P_HEADER ||
           type == MemoryAccessType::GWRITE ||
           type == MemoryAccessType::COMP ||
           type == MemoryAccessType::COMP_HASH ||
           type == MemoryAccessType::READRES ||
           type == MemoryAccessType::COMPS_READRES;
}

bool is_pim_row_buffer_access(MemoryAccessType type) {
    return type == MemoryAccessType::COMP ||
           type == MemoryAccessType::COMPS_READRES;
}

bool is_pim_normal_row_access(MemoryAccessType type) {
    return type == MemoryAccessType::COMP_HASH;
}

bool is_pim_channel_access(MemoryAccessType type) {
    return type == MemoryAccessType::P_HEADER ||
           type == MemoryAccessType::GWRITE ||
           type == MemoryAccessType::READRES;
}

PimCmdSlot pim_slot(MemoryAccessType type) {
    switch (type) {
        case MemoryAccessType::P_HEADER:
            return PimCmdSlot::P_HEADER;
        case MemoryAccessType::GWRITE:
            return PimCmdSlot::GWRITE;
        case MemoryAccessType::COMP:
            return PimCmdSlot::COMP;
        case MemoryAccessType::COMP_HASH:
            return PimCmdSlot::COMP_HASH;
        case MemoryAccessType::READRES:
        case MemoryAccessType::COMPS_READRES:
            return PimCmdSlot::READRES;
        default:
            throw std::invalid_argument(
                "Unsupported PIM memory access type: " +
                std::to_string(static_cast<int>(type)));
    }
}

size_t pim_slot_index(PimCmdSlot slot) {
    return static_cast<size_t>(slot);
}

void update_pim_ready(DRAMBank* bank, PimCmdSlot slot, uint64_t ready_cycle) {
    auto& cur = bank->pim_cmd_ready[pim_slot_index(slot)];
    cur = std::max(cur, ready_cycle);
}

uint64_t pim_same_channel_delay(PimCmdSlot from, PimCmdSlot to, const MemConfig& config) {
    const uint64_t burst_cycle = config.burst_cycle;
    const uint64_t read_to_read_l = EventDrivenParams::read_to_read_l;
    const uint64_t write_to_read_l = EventDrivenParams::write_to_read_l;
    const uint64_t burst_times_per_pim_hash =
        MyAddressAllocator::dram_burst_size == 0
            ? 0
            : MyAddressAllocator::AddrGranularity_Hash_Bytes / MyAddressAllocator::dram_burst_size;
    const uint64_t pim_hash_cmd_gap = std::max<uint64_t>(1, read_to_read_l + burst_cycle + 6);

    switch (from) {
        case PimCmdSlot::P_HEADER:
            return 1;
        case PimCmdSlot::GWRITE:
            switch (to) {
                case PimCmdSlot::P_HEADER:
                case PimCmdSlot::GWRITE:
                    return burst_cycle;
                case PimCmdSlot::COMP:
                case PimCmdSlot::COMP_HASH:
                    return 1;
                case PimCmdSlot::READRES:
                    return write_to_read_l;
            }
            break;
        case PimCmdSlot::COMP:
            switch (to) {
                case PimCmdSlot::P_HEADER:
                case PimCmdSlot::GWRITE:
                case PimCmdSlot::READRES:
                    return 1;
                case PimCmdSlot::COMP:
                case PimCmdSlot::COMP_HASH:
                    return read_to_read_l;
            }
            break;
        case PimCmdSlot::COMP_HASH:
            switch (to) {
                case PimCmdSlot::P_HEADER:
                case PimCmdSlot::GWRITE:
                case PimCmdSlot::READRES:
                    return burst_times_per_pim_hash * burst_cycle + pim_hash_cmd_gap;
                case PimCmdSlot::COMP:
                case PimCmdSlot::COMP_HASH:
                    return burst_times_per_pim_hash * read_to_read_l + pim_hash_cmd_gap;
            }
            break;
        case PimCmdSlot::READRES:
            switch (to) {
                case PimCmdSlot::P_HEADER:
                case PimCmdSlot::READRES:
                    return burst_cycle;
                case PimCmdSlot::GWRITE:
                    return write_to_read_l;
                case PimCmdSlot::COMP:
                case PimCmdSlot::COMP_HASH:
                    return 1;
            }
            break;
    }
    throw std::logic_error(
        "Unsupported PIM timing transition from slot " +
        std::to_string(pim_slot_index(from)) + " to slot " +
        std::to_string(pim_slot_index(to)));
}

void update_pim_ready_after_cmd(DRAMBank* bank, PimCmdSlot from, uint64_t issue_cycle, const MemConfig& config) {
    update_pim_ready(bank, PimCmdSlot::P_HEADER, issue_cycle + pim_same_channel_delay(from, PimCmdSlot::P_HEADER, config));
    update_pim_ready(bank, PimCmdSlot::GWRITE, issue_cycle + pim_same_channel_delay(from, PimCmdSlot::GWRITE, config));
    update_pim_ready(bank, PimCmdSlot::COMP, issue_cycle + pim_same_channel_delay(from, PimCmdSlot::COMP, config));
    update_pim_ready(bank, PimCmdSlot::COMP_HASH, issue_cycle + pim_same_channel_delay(from, PimCmdSlot::COMP_HASH, config));
    update_pim_ready(bank, PimCmdSlot::READRES, issue_cycle + pim_same_channel_delay(from, PimCmdSlot::READRES, config));
}

void update_pim_ready_after_read(DRAMBank* bank, uint64_t issue_cycle, uint64_t comp_delay, uint64_t gwrite_delay) {
    update_pim_ready(bank, PimCmdSlot::P_HEADER, issue_cycle + 1);
    update_pim_ready(bank, PimCmdSlot::GWRITE, issue_cycle + gwrite_delay);
    update_pim_ready(bank, PimCmdSlot::COMP, issue_cycle + comp_delay);
    update_pim_ready(bank, PimCmdSlot::COMP_HASH, issue_cycle + comp_delay);
    update_pim_ready(bank, PimCmdSlot::READRES, issue_cycle + EventDrivenParams::read_to_read_l);
}

void update_pim_ready_after_write(DRAMBank* bank, uint64_t issue_cycle, uint64_t comp_delay, uint64_t gwrite_delay) {
    update_pim_ready(bank, PimCmdSlot::P_HEADER, issue_cycle + 1);
    update_pim_ready(bank, PimCmdSlot::GWRITE, issue_cycle + gwrite_delay);
    update_pim_ready(bank, PimCmdSlot::COMP, issue_cycle + comp_delay);
    update_pim_ready(bank, PimCmdSlot::COMP_HASH, issue_cycle + comp_delay);
    update_pim_ready(bank, PimCmdSlot::READRES, issue_cycle + EventDrivenParams::write_to_read_l);
}

uint64_t pim_precharge_latency(const MemConfig& config) {
    const uint64_t total_banks = MyAddressAllocator::ranks *
                                 MyAddressAllocator::bankgroups *
                                 MyAddressAllocator::banks;
    uint64_t banks_to_precharge = total_banks;
    if (config.dual_bank) {
        banks_to_precharge = std::max<uint64_t>(1, total_banks / 2);
    }
    if (config.IsGDDR() || config.protocol == DRAMProtocol::LPDDR4 || config.protocol == DRAMProtocol::LPDDR5) {
        return static_cast<uint64_t>(EventDrivenParams::precharge_to_precharge) * banks_to_precharge;
    }
    return banks_to_precharge;
}

uint64_t pim_activate_latency(const MemConfig& config) {
    const uint64_t banks_per_channel = MyAddressAllocator::bankgroups * MyAddressAllocator::banks;
    if (config.dual_bank) {
        const uint64_t active_banks = std::max<uint64_t>(1, banks_per_channel / 2);
        if (config.IsGDDR()) {
            return static_cast<uint64_t>(config.t32AW) *
                   static_cast<uint64_t>(std::ceil(static_cast<double>(active_banks) / 32.0));
        }
        const double faw_groups = std::ceil(static_cast<double>(active_banks > 0 ? active_banks - 1 : 0) / 4.0);
        return static_cast<uint64_t>(config.tFAW * faw_groups);
    }

    const double faw_groups = std::ceil(static_cast<double>(banks_per_channel) / 4.0 - 1.0);
    auto faw_latency = static_cast<uint64_t>(config.tFAW * faw_groups + 4);
    uint64_t stagger_latency = static_cast<uint64_t>(EventDrivenParams::activate_to_activate_s) * banks_per_channel;
    return std::max(faw_latency, stagger_latency);
}

uint64_t pim_activate_to_compute_delay(const MemConfig& config) {
    const uint64_t activate_latency = pim_activate_latency(config);
    const uint64_t activate_to_read = EventDrivenParams::activate_to_read;
    if (config.dual_bank) {
        const uint64_t comp_latency_dual_bank = MyAddressAllocator::BL_num_per_row * config.burst_cycle;
        const uint64_t after_activate = activate_latency + activate_to_read;
        const uint64_t after_dual_precharge =
            comp_latency_dual_bank > static_cast<uint64_t>(EventDrivenParams::precharge_to_activate)
                ? comp_latency_dual_bank - static_cast<uint64_t>(EventDrivenParams::precharge_to_activate)
                : 0;
        return std::max(after_activate, after_dual_precharge);
    }
    return activate_latency + activate_to_read - 1;
}

uint64_t get_pim_row_ready_cycle(const DRAMChannel* channel,
                                 const std::shared_ptr<Event>& event,
                                 const MemConfig& config,
                                 uint64_t candidate_cycle) {
    event->pending_pim_row_command = PIMRowCommand::NONE;
    if (!is_pim_row_buffer_access(event->req_type)) {
        return candidate_cycle;
    }

    candidate_cycle = std::max(
        candidate_cycle, event->pim_row_ready_cycle);
    const int target_row = static_cast<int>(event->row_index);
    bool all_banks_on_target_row = true;
    bool any_pim_row_open = false;
    uint64_t precharge_ready = candidate_cycle;
    uint64_t activate_ready = candidate_cycle;
    for (const auto* rank : channel->dram_ranks) {
        for (const auto* bankgroup : rank->dram_bankgroups) {
            for (const auto* bank : bankgroup->dram_banks) {
                // CycleAccurate uses one bank row state for both ordinary
                // DRAM and PIM commands.  Until the first PIM activation,
                // inherit the ordinary open row so a preceding READ/WRITE
                // row is not mistaken for a closed PIM bank.
                const int effective_open_row =
                    bank->pim_open_row != -1
                        ? bank->pim_open_row
                        : bank->open_row;
                all_banks_on_target_row &=
                    effective_open_row == target_row;
                any_pim_row_open |= effective_open_row != -1;
                precharge_ready =
                    std::max({
                        precharge_ready,
                        bank->to_do_pim_precharge,
                        bank->to_do_precharge
                    });
                activate_ready =
                    std::max({
                        activate_ready,
                        bank->to_do_pim_activate
                    });
            }
        }
    }
    if (all_banks_on_target_row) {
        return candidate_cycle;
    }

    if (any_pim_row_open) {
        const uint64_t precharge_issue = precharge_ready;
        event->pending_pim_row_command = PIMRowCommand::PRECHARGE;
        event->pim_precharge_cycle = precharge_issue;
        return precharge_issue;
    }
    event->pending_pim_row_command = PIMRowCommand::ACTIVATE;
    event->pim_activate_cycle = activate_ready;
    return activate_ready;
}

void update_pim_row_state_after_compute(DRAMChannel* channel,
                                        const std::shared_ptr<Event>& event,
                                        const MemConfig& config) {
    if (!is_pim_row_buffer_access(event->req_type)) {
        return;
    }

    const int target_row = static_cast<int>(event->row_index);
    const uint64_t activate_to_compute =
        pim_activate_to_compute_delay(config);
    const uint64_t activate_issue_cycle =
        event->execute_cycle >= activate_to_compute
            ? event->execute_cycle - activate_to_compute
            : 0;
    for (auto* rank : channel->dram_ranks) {
        for (auto* bankgroup : rank->dram_bankgroups) {
            for (auto* bank : bankgroup->dram_banks) {
                const int effective_open_row =
                    bank->pim_open_row != -1
                        ? bank->pim_open_row
                        : bank->open_row;
                const bool row_changed = effective_open_row != target_row;
                bank->pim_open_row = target_row;
                // A channel-wide PIM_ACTIVE opens the same row in the
                // CycleAccurate bank state.  Mirror it in the ordinary ED
                // row state as well so subsequent REF and READ/WRITE
                // scheduling observe the same open row.
                bank->open_row = target_row;
                // CycleAccurate constrains COMP -> PIM_PRECHARGE by
                // read_to_precharge.  The channel-wide activation issue
                // latency must not be re-anchored to every subsequent COMP.
                bank->to_do_pim_precharge =
                    event->execute_cycle + EventDrivenParams::read_to_precharge;
                if (row_changed) {
                    bank->to_do_pim_activate = std::max(
                        bank->to_do_pim_activate,
                        activate_issue_cycle + pim_activate_latency(config) +
                            config.tFAW);
                }
            }
        }
    }
}

uint64_t get_pim_ready_cycle(const DRAMChannel* channel, const std::shared_ptr<Event>& event) {
    const auto slot = pim_slot(event->req_type);
    if (event->req_type == MemoryAccessType::COMP_HASH) {
        auto* bank = channel->dram_ranks[event->rank_index]
                         ->dram_bankgroups[event->bankgroup_index]
                         ->dram_banks[event->bank_index];
        return bank->pim_cmd_ready[pim_slot_index(slot)];
    }

    uint64_t ready_cycle = 0;
    for (auto* rank : channel->dram_ranks) {
        for (auto* bankgroup : rank->dram_bankgroups) {
            for (auto* bank : bankgroup->dram_banks) {
                ready_cycle = std::max(ready_cycle, bank->pim_cmd_ready[pim_slot_index(slot)]);
            }
        }
    }
    return ready_cycle;
}

void update_pim_timing_after_pim(DRAMChannel* channel, const std::shared_ptr<Event>& event) {
    const auto from = pim_slot(event->req_type);
    if (event->req_type == MemoryAccessType::COMP_HASH) {
        auto* bank = channel->dram_ranks[event->rank_index]
                         ->dram_bankgroups[event->bankgroup_index]
                         ->dram_banks[event->bank_index];
        update_pim_ready_after_cmd(bank, from, event->execute_cycle, channel->config_);
        return;
    }

    for (auto* rank : channel->dram_ranks) {
        for (auto* bankgroup : rank->dram_bankgroups) {
            for (auto* bank : bankgroup->dram_banks) {
                update_pim_ready_after_cmd(bank, from, event->execute_cycle, channel->config_);
            }
        }
    }
}

void update_pim_timing_after_normal(DRAMChannel* channel, const std::shared_ptr<Event>& event, bool is_write) {
    for (auto* rank : channel->dram_ranks) {
        for (auto* bankgroup : rank->dram_bankgroups) {
            for (auto* bank : bankgroup->dram_banks) {
                const bool same_rank = static_cast<uint32_t>(rank->rank_id) == event->rank_index;
                const bool same_bankgroup = same_rank && static_cast<uint32_t>(bankgroup->bankgroup_id) == event->bankgroup_index;
                const bool same_bank = same_bankgroup && static_cast<uint32_t>(bank->bank_id) == event->bank_index;

                uint64_t comp_delay = 0;
                uint64_t gwrite_delay = 0;
                if (is_write) {
                    comp_delay = (same_bank || same_bankgroup) ? EventDrivenParams::write_to_read_l
                                                               : EventDrivenParams::write_to_read_s;
                    gwrite_delay = EventDrivenParams::write_to_write_l;
                    update_pim_ready_after_write(bank, event->execute_cycle, comp_delay, gwrite_delay);
                } else {
                    comp_delay = (same_bank || same_bankgroup) ? EventDrivenParams::read_to_read_l
                                                               : EventDrivenParams::read_to_read_s;
                    gwrite_delay = same_bank ? EventDrivenParams::read_to_write
                                             : EventDrivenParams::read_to_read_l;
                    update_pim_ready_after_read(bank, event->execute_cycle, comp_delay, gwrite_delay);
                }
            }
        }
    }
}

}

DRAMBank::DRAMBank(MemConfig& MemConfig, std::vector<std::pair<int, uint64_t>>* Rank_Activate_Recorder, int Rank_id, int BankGroup_id, int Bank_id) :
    config_(MemConfig),
    rank_activate_recorder(Rank_Activate_Recorder),
    rank_id(Rank_id),
    bankgroup_id(BankGroup_id),
    bank_id(Bank_id),
    open_row(-1),
    open_row_exec_event_count(0),
    pending_precharge(false),
    pending_activate(false),   //
    row_buf_policy_(MemConfig.row_buf_policy == "CLOSE_PAGE" ? RowBufPolicy::CLOSE_PAGE : RowBufPolicy::OPEN_PAGE) {
        if (config_.refresh_policy == RefreshPolicy::BANK_LEVEL_STAGGERED) {
            bank_refresh = true;
            refresh_interval_ = config_.tREFIb * config_.ranks * config_.bankgroups * config_.banks_per_group;
            refresh_offset_ = config_.tREFIb * (Rank_id * config_.bankgroups * config_.banks_per_group +  Bank_id * config_.bankgroups + BankGroup_id);
    }

    next_bank_refresh_time = refresh_interval_ + refresh_offset_;

    to_do_precharge = 0;
    to_do_activate = 0;
    to_do_read = 0;
    to_do_write = 0;
    pim_cmd_ready.fill(0);
    pim_open_row = -1;
    to_do_pim_precharge = 0;
    to_do_pim_activate = 0;
}

DRAMBank::~DRAMBank() {

}


void DRAMBank::PushPendingSwitchRow(std::shared_ptr<Event> event) {
    pending_activate_row.insert(std::make_pair(event->row_index, event));
    event->activate_cycle = std::max(event->activate_cycle, to_do_activate);
    spdlog::debug("Pending activate for event address {} and the activate cycle is {}", event->dram_address, event->activate_cycle);
}


void DRAMBank::Activate(std::shared_ptr<Event> event) {
    assert(open_row_exec_event_count == 0 && pending_activate == true);
    pending_activate = false;

    event->need_activate = false;
    open_row_exec_event_count = 1;
    to_do_read = event->activate_cycle + EventDrivenParams::activate_to_read;
    to_do_write = event->activate_cycle + EventDrivenParams::activate_to_write;
    to_do_activate = event->activate_cycle + EventDrivenParams::activate_to_activate;
    to_do_precharge = event->activate_cycle + EventDrivenParams::activate_to_precharge;
    open_row = event->row_index;
}


void DRAMBank::Precharge(std::shared_ptr<Event> event) {
    assert(open_row_exec_event_count == 0);
    assert(open_row != -1);
    assert(pending_activate == false);
    pending_precharge = false;
    pending_activate = true;
    event->need_precharge = false;
    event->need_activate = true;
    to_do_activate = event->precharge_cycle + EventDrivenParams::precharge_to_activate;
    open_row = -1;
}


void DRAMBank::Read(std::shared_ptr<Event> event) {
    assert(event->row_index == open_row);
    open_row_exec_event_count--;
    assert(open_row_exec_event_count>=0);
    to_do_read = event->execute_cycle + EventDrivenParams::read_to_read_l;
    to_do_write = event->execute_cycle + EventDrivenParams::read_to_write;
    // PRE must satisfy both read-to-precharge and the ACT-to-PRE (tRAS)
    // constraint established when the row was opened.
    to_do_precharge = std::max(to_do_precharge,
                               event->execute_cycle + EventDrivenParams::read_to_precharge);
}


void DRAMBank::Write(std::shared_ptr<Event> event) {
    if (event->row_index != static_cast<uint32_t>(open_row)) {
        spdlog::critical(
            "EventDriven WRITE row-state mismatch: address={}, "
            "rank={}, bankgroup={}, bank={}, event_row={}, open_row={}, "
            "open_row_exec_event_count={}, pending_precharge={}, "
            "pending_activate={}",
            event->dram_address, rank_id, bankgroup_id, bank_id,
            event->row_index, open_row, open_row_exec_event_count,
            pending_precharge, pending_activate);
    }
    assert(event->row_index == open_row);
    open_row_exec_event_count--;
    assert(open_row_exec_event_count>=0);
    to_do_read = event->execute_cycle + EventDrivenParams::write_to_read_l;
    to_do_write = event->execute_cycle + EventDrivenParams::write_to_write_l;
    // Do not erase a still-later tRAS constraint with write-to-precharge.
    to_do_precharge = std::max(to_do_precharge,
                               event->execute_cycle + EventDrivenParams::write_to_precharge);
}


void DRAMBank::PIM(std::shared_ptr<Event> event) {
    // TODO::PIM Operation is not support
}


void DRAMBank::AddPendingPrechargeEvent(std::shared_ptr<Event> event) {
    const bool row_has_pending_events = pending_precharge_event.find(event->row_index) != pending_precharge_event.end();
    const bool row_already_ordered = std::find(pending_precharge_order_queue.begin(), pending_precharge_order_queue.end(), event->row_index) != pending_precharge_order_queue.end();

    if (!row_has_pending_events && !row_already_ordered) {
        pending_precharge_order_queue.push_back(event->row_index);
    }
    pending_precharge_event[event->row_index].push_back(event);
}


DRAMBankGroup::DRAMBankGroup(MemConfig& MemConfig, std::vector<std::pair<int, uint64_t>>* Rank_Activate_Recorder, int Rank_id, int BankGroup_id): config_(MemConfig), rank_activate_recorder(Rank_Activate_Recorder), rank_id(Rank_id), bankgroup_id(BankGroup_id) {
    to_do_read = 0;
    to_do_write = 0;
    to_do_activate = 0;
    to_do_precharge = 0;
    for (auto i = 0; i < config_.banks_per_group; i++) {
        dram_banks.push_back(new DRAMBank(config_, rank_activate_recorder, rank_id, bankgroup_id, i));
    }
}


DRAMBankGroup::~DRAMBankGroup() {
    for (auto* bank : dram_banks) {
        delete bank;
    }
}


void DRAMBankGroup::SameBankgroupPrecharge(std::shared_ptr<Event> event) {
    to_do_precharge = event->precharge_cycle + EventDrivenParams::precharge_to_precharge;
}


void DRAMBankGroup::SameBankgroupActivate(std::shared_ptr<Event> event) {
    to_do_activate = event->activate_cycle + EventDrivenParams::activate_to_activate_l;
}


void DRAMBankGroup::OtherBankgroupActivate(std::shared_ptr<Event> event) {
    to_do_activate = event->activate_cycle + EventDrivenParams::activate_to_activate_s;
}


void DRAMBankGroup::SameBankgroupRead(std::shared_ptr<Event> event) {
    to_do_read = event->execute_cycle + EventDrivenParams::read_to_read_l;
    to_do_write = event->execute_cycle + EventDrivenParams::read_to_write;
}

void DRAMBankGroup::OtherBankgroupRead(std::shared_ptr<Event> event) {
    to_do_read = event->execute_cycle + EventDrivenParams::read_to_read_s;
    to_do_write = event->execute_cycle + EventDrivenParams::read_to_write;
}

void DRAMBankGroup::SameBankgroupWrite(std::shared_ptr<Event> event) {
    to_do_read = event->execute_cycle + EventDrivenParams::write_to_read_l;
    to_do_write = event->execute_cycle + EventDrivenParams::write_to_write_l;
}

void DRAMBankGroup::OtherBankgroupWrite(std::shared_ptr<Event> event) {
    to_do_read = event->execute_cycle + EventDrivenParams::write_to_read_s;
    to_do_write = event->execute_cycle + EventDrivenParams::write_to_write_s;
}

/////////////////////////////////////////////////////////////
// DRAMRank constructor
/////////////////////////////////////////////////////////////

DRAMRank::DRAMRank(MemConfig& MemConfig, int Channel_id, int Rank_id): config_(MemConfig), channel_id(Channel_id), rank_id(Rank_id) {
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

DRAMRank::~DRAMRank() {
    for (auto* bankgroup : dram_bankgroups) {
        delete bankgroup;
    }
}

void DRAMRank::IssuePrecharge(std::shared_ptr<Event> event) {
    auto* target_bankgroup = dram_bankgroups[event->bankgroup_index];
    auto* target_bank = target_bankgroup->dram_banks[event->bank_index];

    // The target-bank PRE state transition is required for every protocol.
    // tPPD propagation to other banks is a separate protocol-specific rule.
    target_bank->Precharge(event);

    if (config_.IsGDDR() || config_.protocol == DRAMProtocol::LPDDR4 || config_.protocol == DRAMProtocol::LPDDR5) {
        for (auto bankgroup : dram_bankgroups) {
            if (bankgroup->bankgroup_id == event->bankgroup_index) {  // same bankgroup other banks
                bankgroup->SameBankgroupPrecharge(event);
            }
            else {  // other bankgroup
                to_precharge = event->precharge_cycle + EventDrivenParams::precharge_to_precharge;
            }
        }
    }
    target_bank->to_do_activate = event->precharge_cycle + EventDrivenParams::precharge_to_activate;
}


void DRAMRank::IssueActivate(std::shared_ptr<Event> event) {
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

    for (auto bankgroup : dram_bankgroups) {
        if (bankgroup->bankgroup_id == event->bankgroup_index) {   // The Operation BankGroup
            auto bank = bankgroup->dram_banks[event->bank_index]; // The operation bank
            bank->Activate(event);
            bankgroup->SameBankgroupActivate(event);
        }
        else {
            bankgroup->OtherBankgroupActivate(event);
        }
    }
    event->execute_cycle = event->activate_cycle + 1;
}


void DRAMRank::IssueRead(std::shared_ptr<Event> event) {
    for (auto bankgroup : dram_bankgroups) {
        if (bankgroup->bankgroup_id == event->bankgroup_index) {   // The Operation BankGroup
            bankgroup->dram_banks[event->bank_index]->Read(event); // The operation bank
            bankgroup->SameBankgroupRead(event);
        }
        else {
            bankgroup->OtherBankgroupRead(event);
        }
    }
}


void DRAMRank::IssueWrite(std::shared_ptr<Event> event) {
    for (auto bankgroup : dram_bankgroups) {
        if (bankgroup->bankgroup_id == event->bankgroup_index) {   // The Operation BankGroup
            bankgroup->dram_banks[event->bank_index]->Write(event); // The operation bank
            bankgroup->SameBankgroupWrite(event);
        }
        else {
            bankgroup->OtherBankgroupWrite(event);
        }
    }
}


void DRAMRank::IssuePIM(std::shared_ptr<Event> event) {

}


uint64_t DRAMRank::RankRefreshReady(
    std::vector<RefreshPrechargeRecord>* precharge_records) {
    uint64_t refresh_time = (current_trans_refresh_count - 1) * rank_refresh_interval_ + rank_refresh_offset_;
    if (current_trans_refresh_count - refresh_count == 1) {
        uint64_t to_refresh_time = 0;
        for (auto bankgroup : dram_bankgroups) {
            for (auto bank : bankgroup->dram_banks) {
                const int effective_open_row =
                    bank->pim_open_row != -1
                        ? bank->pim_open_row
                        : bank->open_row;
                if (effective_open_row != -1) {
                    refresh_time = std::max({
                        refresh_time,
                        bank->to_do_precharge,
                        bank->to_do_pim_precharge
                    });
                    if (precharge_records != nullptr) {
                        precharge_records->push_back({
                            refresh_time,
                            static_cast<uint32_t>(rank_id),
                            static_cast<uint32_t>(bankgroup->bankgroup_id),
                            static_cast<uint32_t>(bank->bank_id),
                            static_cast<uint32_t>(effective_open_row)});
                    }
                    bank->open_row = -1;
                    bank->pim_open_row = -1;
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

void DRAMRank::RankRefresh(uint64_t rank_refresh_cycle) {
    // spdlog::critical("A Rank Refresh Operation is executed for Rank {} at time {}", rank_id, rank_refresh_cycle);
    const uint64_t refresh_done_cycle =
        rank_refresh_cycle + config_.tRFC;
    for (auto bankgroup : dram_bankgroups) {
        for (auto bank : bankgroup->dram_banks) {
            bank->open_row      = -1;
            bank->pim_open_row  = -1;
            bank->open_row_exec_event_count = 0;
            bank->pending_precharge = false;
            bank->pending_activate = false;
            bank->to_do_activate = refresh_done_cycle;
            // NewtonSim's current CA timing constrains REF -> ordinary ACT by
            // tRFC, but it does not add REF -> PIM_ACTIVE to the same-rank
            // timing table.  Closing the PIM row is therefore sufficient:
            // the next COMP will regenerate the PIM_ACTIVE latency without
            // incorrectly charging tRFC to the PIM path.
        }
    }
}

void DRAMRank::EnterSelfRefresh() {
    in_self_refresh = true;
}


void DRAMRank::ExitSelfRefresh() {
    in_self_refresh = false;
}

DRAMChannel::DRAMChannel(MemConfig& MemConfig, int Channel_id) :config_(MemConfig), channel_id(Channel_id),is_unified_queue_(config_.unified_queue), write_draining_(0), clk_(0) {
    for (auto rank_id = 0; rank_id < config_.ranks; rank_id++) {
        dram_ranks.emplace_back(new DRAMRank(config_, channel_id, rank_id));
    }
    const int bank_queue_count = config_.ranks * config_.bankgroups * config_.banks_per_group;
    physical_rr_next_key = bank_queue_count > 1 ? 1 : 0;
}

DRAMChannel::~DRAMChannel() {
    for (auto* rank : dram_ranks) {
        delete rank;
    }
}

void DRAMChannel::GetPrechargeCycle(std::shared_ptr<Event> event) {
    auto* rank = dram_ranks[event->rank_index];
    auto* bankgroup = rank->dram_bankgroups[event->bankgroup_index];
    auto* bank = bankgroup->dram_banks[event->bank_index];
    // PendingPrecharge() and the post-execute row-switch path already seed
    // precharge_cycle with the first command cycle (producer_cycle + 1).
    // Advancing it again here made every EventDriven PRE one cycle late.
    event->precharge_cycle = std::max({event->precharge_cycle,
                                      static_cast<cycle_type>(rank->to_precharge),
                                      static_cast<cycle_type>(bankgroup->to_do_precharge),
                                      static_cast<cycle_type>(bank->to_do_precharge)});
    if (event->precharge_cycle < earliest_queue.second) {
        earliest_queue = {2, event->precharge_cycle};
    }
}

void DRAMChannel::IssuePrechargeEvent(std::shared_ptr<Event> event) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    dram_ranks[event->rank_index]->IssuePrecharge(event);
    AdvancePhysicalRoundRobin(event);
    activate_queue.push_back(event);
    if (activate_queue.size() == 1) {
        GetActivateCycle(activate_queue.front());
    }
    precharge_queue.pop_front();
    if (!precharge_queue.empty()) {
        GetPrechargeCycle(precharge_queue.front());
    }
    RefreshEarliestQueue();
}

void DRAMChannel::GetActivateCycle(std::shared_ptr<Event> event) {
    event->activate_cycle = std::max({event->activate_cycle,
        dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_activate,
        dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_activate});

    const auto& activate_cycles = dram_ranks[event->rank_index]->activate_cycle_recorder;
    if (activate_cycles.size() >= 4) {
        size_t idx = activate_cycles.size() - 4;
        event->activate_cycle = std::max(event->activate_cycle, activate_cycles[idx] + config_.tFAW);
    }

    if (config_.IsGDDR() && activate_cycles.size() >= 32) {
        size_t idx = activate_cycles.size() - 32;
        event->activate_cycle = std::max(event->activate_cycle, activate_cycles[idx] + config_.t32AW);
    }

    if (event->activate_cycle < earliest_queue.second) {
        earliest_queue = {1, event->activate_cycle};
    }
}

// Issue an ACTIVATE event and append it to the execution queue.
void DRAMChannel::IssueActivateEvent(std::shared_ptr<Event> event) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    dram_ranks[event->rank_index]->IssueActivate(event);
    AdvancePhysicalRoundRobin(event);
    execute_queue.push_back(event);
    if (execute_queue.size() == 1) {
        RefreshExecutionSchedulerHeadCycles();
    }
    auto operation_bank = dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];
    if (!operation_bank->pending_precharge_order_queue.empty() && event->row_index == operation_bank->pending_precharge_order_queue.front()) {
        operation_bank->pending_precharge_order_queue.pop_front();
        auto it = operation_bank->pending_precharge_event.find(event->row_index);
        if (it != operation_bank->pending_precharge_event.end()) {
            for (auto same_row_event: it->second) {
                execute_queue.push_back(same_row_event);
                operation_bank->open_row_exec_event_count++;
            }
            operation_bank->pending_precharge_event.erase(it);
        }
    }
    activate_queue.pop_front();
    if (!activate_queue.empty()) {
        GetActivateCycle(activate_queue.front());
    }
    RefreshEarliestQueue();
}


void DRAMChannel::GetExecuteCycle(std::shared_ptr<Event> event) {
    bool is_write = event->req_type == MemoryAccessType::WRITE;
    bool is_pim = is_pim_access_type(event->req_type);
    if (is_pim && !is_pim_normal_row_access(event->req_type)) {
        // A PIM command cannot begin its implicit PRE/ACT sequence before
        // the transaction itself reaches DRAM.  The previous increment-only
        // form could reuse an old bank-ready timestamp and effectively start
        // a row switch before event->add_cycle.
        event->execute_cycle =
            std::max(event->execute_cycle, event->add_cycle + 1);
    }
    else if (is_write) {
        event->execute_cycle = std::max({event->add_cycle + 1,
                                dram_ranks[event->rank_index]->to_do_write,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_write,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_write});
    }
    else {
        event->execute_cycle = std::max({event->add_cycle + 1,
                                dram_ranks[event->rank_index]->to_do_read,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_read,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_read});
    }

    if (is_pim) {
        if (is_pim_channel_access(event->req_type)) {
            event->execute_cycle = std::max({
                event->execute_cycle,
                dram_ranks[event->rank_index]->to_do_read,
                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_read,
                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_read
            });
        }
        event->execute_cycle = std::max(event->execute_cycle, get_pim_ready_cycle(this, event));
        if (is_pim_row_buffer_access(event->req_type)) {
            event->execute_cycle =
                get_pim_row_ready_cycle(this, event, config_,
                                        event->execute_cycle);
        }
    }

    if (event->execute_cycle < earliest_queue.second) {
        earliest_queue = {0, event->execute_cycle};
    }
}

void DRAMChannel::RefreshExecutionSchedulerHeadCycles() {
#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
    execute_queue.set_arbitration_state(physical_rr_next_key,
                                        NextPhysicalArbitrationCycle());
    while (!execute_queue.empty()) {
        auto event = execute_queue.front();
        if (event == nullptr || execute_queue.has_refreshed_execute_cycle(event)) {
            break;
        }
        GetExecuteCycle(event);
        execute_queue.mark_refreshed_execute_cycle(event);
    }
#else
    if (!execute_queue.empty()) {
        GetExecuteCycle(execute_queue.front());
    }
#endif
}

int DRAMChannel::PhysicalBankKey(const std::shared_ptr<Event>& event) const {
    if (event == nullptr) {
        return 0;
    }
    return (static_cast<int>(event->rank_index) * config_.bankgroups +
            static_cast<int>(event->bankgroup_index)) *
               config_.banks_per_group +
           static_cast<int>(event->bank_index);
}

cycle_type DRAMChannel::NextPhysicalArbitrationCycle() const {
    if (last_physical_command_cycle == std::numeric_limits<cycle_type>::max()) {
        return 0;
    }
    return last_physical_command_cycle + 1;
}

void DRAMChannel::AdvancePhysicalRoundRobin(const std::shared_ptr<Event>& event) {
    const int bank_queue_count = config_.ranks * config_.bankgroups * config_.banks_per_group;
    if (event == nullptr || bank_queue_count <= 0) {
        return;
    }
    physical_rr_next_key = (PhysicalBankKey(event) + 1) % bank_queue_count;
}

void DRAMChannel::IssuePIMPrechargeEvent(
    const std::shared_ptr<Event>& event, cycle_type issue_cycle) {
    assert(event != nullptr);
    assert(event->pending_pim_row_command == PIMRowCommand::PRECHARGE);
    const cycle_type activate_ready = issue_cycle + std::max<uint64_t>(
        pim_precharge_latency(config_),
        static_cast<uint64_t>(EventDrivenParams::precharge_to_activate));
    for (auto* rank : dram_ranks) {
        for (auto* bankgroup : rank->dram_bankgroups) {
            for (auto* bank : bankgroup->dram_banks) {
                bank->pim_open_row = -1;
                bank->open_row = -1;
                bank->to_do_pim_activate = std::max(
                    bank->to_do_pim_activate, activate_ready);
            }
        }
    }
    event->pending_pim_row_command = PIMRowCommand::NONE;
#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
    execute_queue.invalidate_refreshed_execute_cycles();
#endif
    RefreshEarliestQueue();
}

void DRAMChannel::IssuePIMActivateEvent(
    const std::shared_ptr<Event>& event, cycle_type issue_cycle) {
    assert(event != nullptr);
    assert(event->pending_pim_row_command == PIMRowCommand::ACTIVATE);
    const int target_row = static_cast<int>(event->row_index);
    for (auto* rank : dram_ranks) {
        for (auto* bankgroup : rank->dram_bankgroups) {
            for (auto* bank : bankgroup->dram_banks) {
                bank->pim_open_row = target_row;
                bank->open_row = target_row;
            }
        }
    }
    event->pim_row_ready_cycle =
        issue_cycle + pim_activate_to_compute_delay(config_);
    event->pending_pim_row_command = PIMRowCommand::NONE;
#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
    execute_queue.invalidate_refreshed_execute_cycles();
#endif
    RefreshEarliestQueue();
}


void DRAMChannel::IssueExecuteEvent(std::shared_ptr<Event> event, cycle_type issue_cycle) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    // Remove the event while its scheduler key still has the value used by
    // front().  Mutating execute_cycle first can make pop_front() select a
    // different event in the priority round-robin queue.
    execute_queue.pop_front();
    event->execute_cycle = issue_cycle;
    AdvancePhysicalRoundRobin(event);
    bool is_write = event->req_type == MemoryAccessType::WRITE;
    bool is_pim = is_pim_access_type(event->req_type);
    if (!is_pim) {
        // Close the merge generation exactly when the physical READ/WRITE
        // command is issued.
        event->merge_open = false;
    }
    if (is_write) {
        for (auto rank : dram_ranks) {
            if (rank->rank_id == event->rank_index) {
                rank->IssueWrite(event);
            }
            else {
                rank->to_do_read = event->execute_cycle + EventDrivenParams::write_to_read_o;
                rank->to_do_write = event->execute_cycle + EventDrivenParams::write_to_write_o;
            }
        }
        event->complete_cycle = event->execute_cycle + config_.write_delay;
        update_pim_timing_after_normal(this, event, true);
    }
    else if (!is_pim || is_pim_normal_row_access(event->req_type)) {  // Read
        for (auto rank : dram_ranks) {
            if (rank->rank_id == event->rank_index) {
                rank->IssueRead(event);
            }
            else {
                rank->to_do_read = event->execute_cycle + EventDrivenParams::read_to_read_o;
                rank->to_do_write = event->execute_cycle + EventDrivenParams::read_to_write_o;
            }
        }
        event->complete_cycle = event->execute_cycle + config_.read_delay;
        if (!is_pim) {
            update_pim_timing_after_normal(this, event, false);
        }
    }

    if (is_pim) {
        if (is_pim_row_buffer_access(event->req_type)) {
            update_pim_row_state_after_compute(this, event, config_);
        }
        switch (event->req_type) {
            case MemoryAccessType::P_HEADER:
                event->complete_cycle = event->execute_cycle + 1;
                break;
            case MemoryAccessType::GWRITE:
                event->complete_cycle = event->execute_cycle + config_.burst_cycle;
                break;
            case MemoryAccessType::COMP:
            case MemoryAccessType::COMP_HASH:
                event->complete_cycle = event->execute_cycle + config_.read_delay;
                break;
            case MemoryAccessType::READRES:
            case MemoryAccessType::COMPS_READRES:
                event->complete_cycle = event->execute_cycle + config_.burst_cycle;
                break;
            default:
                break;
        }
        update_pim_timing_after_pim(this, event);
    }

#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
    // READ/WRITE/PIM commands update rank-, bankgroup-, and bank-level timing
    // constraints.  Cached head readiness from before this command is stale.
    execute_queue.invalidate_refreshed_execute_cycles();
#endif

    // process the pending precharge event
    auto operation_bank = dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];
    if (operation_bank->open_row_exec_event_count == 0 &&
        !operation_bank->pending_precharge &&
        !operation_bank->pending_activate &&
        !operation_bank->pending_precharge_order_queue.empty()) {
        auto operation_row = operation_bank->pending_precharge_order_queue.front();
        auto it = operation_bank->pending_precharge_event.find(operation_row);
        if (it != operation_bank->pending_precharge_event.end() && !it->second.empty()) {
            auto pending_event = it->second.front();
            it->second.pop_front();
            if (it->second.empty()) {
                operation_bank->pending_precharge_event.erase(it);
            }
            pending_event->need_precharge = true;
            pending_event->precharge_cycle = event->execute_cycle + 1;
            operation_bank->pending_precharge = true;
            precharge_queue.push_back(pending_event);
            if (precharge_queue.size() == 1) {
                GetPrechargeCycle(precharge_queue.front());
            }
        }
    }

    return_queue.push_back(event);
    spdlog::debug("(EventDriven DRAM) Event {} is Push back to return queue at cycle {}, current return queue size is {}",
        event->dram_address, event->execute_cycle, return_queue.size());

    if (!execute_queue.empty()) {
        RefreshExecutionSchedulerHeadCycles();
    }
    RefreshEarliestQueue();
}


cycle_type DRAMChannel::GetNextRankRefreshCycle() const {
    cycle_type next_cycle = std::numeric_limits<cycle_type>::max();
    for (auto rank : dram_ranks) {
        if (rank->rank_refresh) {
            next_cycle = std::min(next_cycle, static_cast<cycle_type>(rank->next_refresh_cycle));
        }
    }
    return next_cycle;
}


cycle_type DRAMChannel::IssueRankRefresh(cycle_type refresh_cycle) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    refresh_precharge_records.clear();
    cycle_type actual_refresh_cycle = refresh_cycle;
    for (auto rank : dram_ranks) {
        if (!rank->rank_refresh || rank->next_refresh_cycle > refresh_cycle) {
            continue;
        }
        rank->current_trans_refresh_count =
            (refresh_cycle - rank->rank_refresh_offset_) / rank->rank_refresh_interval_ + 1;
        const auto rank_refresh_time = rank->RankRefreshReady(
            &refresh_precharge_records);
        actual_refresh_cycle = std::max(
            actual_refresh_cycle,
            static_cast<cycle_type>(rank_refresh_time));
        spdlog::debug("(EventDriven DRAM) Rank Refresh at cycle {} for Channel {}, Rank {}, actual refresh cycle {}, next refresh {}",
            refresh_cycle, channel_id, rank->rank_id, rank_refresh_time, rank->next_refresh_cycle);
        rank->RankRefresh(rank_refresh_time);
    }
    RefreshEarliestQueue();
    return actual_refresh_cycle;
}


void DRAMChannel::RefreshEarliestQueue() {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};

    const int bank_queue_count = config_.ranks * config_.bankgroups * config_.banks_per_group;
    const cycle_type arbitration_cycle = NextPhysicalArbitrationCycle();
    auto rr_distance = [&](const std::shared_ptr<Event>& event) {
        if (bank_queue_count <= 0) {
            return 0;
        }
        return (PhysicalBankKey(event) - physical_rr_next_key + bank_queue_count) %
               bank_queue_count;
    };

    if (!execute_queue.empty()) {
#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
        RefreshExecutionSchedulerHeadCycles();
#endif
    }

    auto select_deque_head = [&](auto& queue, auto refresh_cycle, auto get_cycle) {
        if (queue.empty()) {
            return;
        }
        cycle_type minimum_cycle = std::numeric_limits<cycle_type>::max();
        bool ready_exists = false;
        auto selected = queue.end();
        int selected_distance = std::numeric_limits<int>::max();
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            auto& event = *it;
            refresh_cycle(event);
            const cycle_type cycle = get_cycle(event);
            const int distance = rr_distance(event);
            if (cycle <= arbitration_cycle) {
                if (!ready_exists || distance < selected_distance) {
                    selected = it;
                    selected_distance = distance;
                }
                ready_exists = true;
                continue;
            }

            if (!ready_exists &&
                (selected == queue.end() || cycle < minimum_cycle ||
                 (cycle == minimum_cycle && distance < selected_distance))) {
                selected = it;
                minimum_cycle = cycle;
                selected_distance = distance;
            }
        }
        if (selected != queue.end() && selected != queue.begin()) {
            std::rotate(queue.begin(), selected, std::next(selected));
        }
    };

    if (!activate_queue.empty()) {
        select_deque_head(
            activate_queue,
            [&](const std::shared_ptr<Event>& event) { GetActivateCycle(event); },
            [](const std::shared_ptr<Event>& event) { return event->activate_cycle; });
    }
    if (!precharge_queue.empty()) {
        select_deque_head(
            precharge_queue,
            [&](const std::shared_ptr<Event>& event) { GetPrechargeCycle(event); },
            [](const std::shared_ptr<Event>& event) { return event->precharge_cycle; });
    }

    // Get*Cycle() updates earliest_queue as a side effect.  Rebuild the final
    // choice here so all command classes participate in the same per-bank RR
    // arbitration, matching CycleAccurate's PER_BANK queue scan.
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    int selected_queue_type = -1;
    cycle_type selected_cycle = std::numeric_limits<cycle_type>::max();
    int selected_distance = std::numeric_limits<int>::max();
    bool ready_exists = false;
    auto consider_candidate = [&](int queue_type, cycle_type cycle,
                                  const std::shared_ptr<Event>& event) {
        if (event == nullptr) {
            return;
        }
        const int distance = rr_distance(event);
        if (cycle <= arbitration_cycle) {
            if (!ready_exists || distance < selected_distance) {
                selected_queue_type = queue_type;
                selected_cycle = cycle;
                selected_distance = distance;
            }
            ready_exists = true;
            return;
        }
        if (!ready_exists &&
            (selected_queue_type == -1 || cycle < selected_cycle ||
             (cycle == selected_cycle && distance < selected_distance))) {
            selected_queue_type = queue_type;
            selected_cycle = cycle;
            selected_distance = distance;
        }
    };
    if (!execute_queue.empty()) {
        auto event = execute_queue.front();
        consider_candidate(0, event == nullptr ? 0 : event->execute_cycle, event);
    }
    if (!activate_queue.empty()) {
        const auto& event = activate_queue.front();
        consider_candidate(1, event->activate_cycle, event);
    }
    if (!precharge_queue.empty()) {
        const auto& event = precharge_queue.front();
        consider_candidate(2, event->precharge_cycle, event);
    }
    if (selected_queue_type != -1) {
        earliest_queue = {selected_queue_type, selected_cycle};
    }
    if (execute_queue.empty() && activate_queue.empty() && precharge_queue.empty()) {
        const auto next_rank_refresh = GetNextRankRefreshCycle();
        if (next_rank_refresh < earliest_queue.second) {
            earliest_queue = {3, next_rank_refresh};
        }
    }
}


MemorySystem::MemorySystem(const MemConfig& mem_config, int mem_id): config_(mem_config), memory_id_(mem_id), last_transaction_clk_(0){
    dram_channels.reserve(config_.channels);
    for (auto i = 0; i < config_.channels; i++) {
        dram_channels.emplace_back(new DRAMChannel(config_, i));
    }
    spdlog::info("EventDriven DRAM initialized with {} channels",
                 config_.channels);
}


MemorySystem::~MemorySystem() {
    for (auto & dram_channel : dram_channels) {
        delete dram_channel;
    }
    spdlog::info("EventDriven DRAM memory system destroyed");
}


int MemorySystem::GetChannel(uint64_t hex_addr) const {
    return MyAddressAllocator::get_channel_index(hex_addr);
}

static void trace_event_row_decision(const std::shared_ptr<Event>& event,
                                     const DRAMBank* bank,
                                     cycle_type decision_cycle,
                                     const char* decision) {
    if (!Config::system_config.record_dram_completion_trace ||
        event == nullptr || bank == nullptr) {
        return;
    }
    constexpr uint64_t kRowDecisionTraceLimit = 40000;
    static std::ofstream trace;
    static bool opened = false;
    static uint64_t sequence = 0;
    if (sequence >= kRowDecisionTraceLimit) {
        return;
    }
    if (!opened) {
        const std::string path = Config::system_config.log_dir +
                                 "/dram_row_decision_eventdriven.csv";
        trace.open(path,
                   std::ofstream::out | std::ofstream::trunc);
        if (!trace.is_open()) {
            throw std::runtime_error(
                "Cannot open EventDriven row-decision trace: " + path);
        }
        opened = true;
        trace << "decision_sequence,channel,add_cycle,decision_cycle,address,type,rank,bankgroup,bank,target_row,open_row,open_row_exec_count,pending_precharge,pending_activate,pending_row_count,decision\n";
        if (!trace.good()) {
            throw std::runtime_error(
                "Failed to write EventDriven row-decision trace header: " +
                path);
        }
    }
    trace << sequence++ << ',' << event->channel_index << ','
          << event->add_cycle << ',' << decision_cycle << ','
          << event->dram_address << ','
          << memAccessTypeString(event->req_type) << ','
          << event->rank_index << ',' << event->bankgroup_index << ','
          << event->bank_index << ',' << event->row_index << ','
          << bank->open_row << ',' << bank->open_row_exec_event_count << ','
          << bank->pending_precharge << ',' << bank->pending_activate << ','
          << bank->pending_precharge_order_queue.size() << ','
          << decision << '\n';
    if (!trace.good()) {
        throw std::runtime_error(
            "Failed to write EventDriven row-decision trace");
    }
}


void MemorySystem::CheckRowSwitch(std::shared_ptr<Event> event) {
    // get the operation bank of event and check row switch;
    // if bank close page, Activate the Bank row
    // if row buffer hit, push into execute queue and open_row_exec_event_count++
    // if row buffer miss, if open_row_exec_event_count == 0 directly precharge; else if open_row_exec_event_count != 0,
    // buffer into pending_precharge_order_queue, and pending_precharge_event,
    // after execution, open_row_exec_event_count = 0, queue not empty, fetch the event for precharge and activate
    auto operation_bank = dram_channels[event->channel_index]->dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];
    const cycle_type decision_cycle =
        dram_channels[event->channel_index]->_dram_cycle;
    auto current_row= operation_bank->open_row;

    if (current_row == -1) {   // row page close, event in activate_queue
        if (operation_bank->pending_precharge_order_queue.empty() &&
            operation_bank->pending_precharge == false &&
            operation_bank->pending_activate == false){
            ED_TRACE_ROW_DECISION(event, operation_bank, decision_cycle, "CLOSED_ACTIVATE");
            PendingActivate(event);
            operation_bank->pending_activate = true;
            spdlog::debug("(EventDriven DRAM) At {} Cycle input event {} are added into the Activate queue, current Activate queue size is {}",
                event->add_cycle, event->dram_address, dram_channels[event->channel_index]->activate_queue.size());
            }
        else {
            ED_TRACE_ROW_DECISION(event, operation_bank, decision_cycle, "CLOSED_DEFER");
            operation_bank->AddPendingPrechargeEvent(event);
            spdlog::debug("(EventDriven DRAM) At {} Cycle input event {} are added into the Pending Precharge queue of channel {}, Rank {}, Bankgroup {}, Bank {}, Row {} for current pending precharge row is {}",
            event->add_cycle, event->dram_address, event->channel_index, event->rank_index, event->bankgroup_index, event->bank_index, event->row_index, operation_bank->pending_precharge_order_queue.front());
        }
    }
    else if (current_row != event->row_index){  // need row switch
        if (operation_bank->open_row_exec_event_count == 0 &&
            operation_bank->pending_precharge_order_queue.empty() &&
            operation_bank->pending_precharge == false) {
            ED_TRACE_ROW_DECISION(event, operation_bank, decision_cycle, "CONFLICT_PRECHARGE");
            // Non-ASCII comment replaced to avoid encoding issues.
            PendingPrecharge(event);
            spdlog::debug("(EventDriven DRAM) At {} Cycle input event {} are added into the Precharge queue, current Precharge queue size is {}",
                event->add_cycle, event->dram_address, dram_channels[event->channel_index]->precharge_queue.size());
        }
        else {
            ED_TRACE_ROW_DECISION(event, operation_bank, decision_cycle, "CONFLICT_DEFER");
            // Non-ASCII comment replaced to avoid encoding issues.
            operation_bank->AddPendingPrechargeEvent(event);
            spdlog::debug("(EventDriven DRAM) At {} Cycle input event {} are added into the Pending Precharge queue of channel {}, Rank {}, Bankgroup {}, Bank {}, Row {} for current open_row {} with execution event count = {}",
                event->add_cycle, event->dram_address, event->channel_index, event->rank_index, event->bankgroup_index, event->bank_index, event->row_index, operation_bank->open_row, operation_bank->open_row_exec_event_count);
        }
      // dram_channels[event->channel_index]->activate_queue.push_back(event);
    }
    else {
        if (operation_bank->pending_precharge || !operation_bank->pending_precharge_order_queue.empty()) {
            ED_TRACE_ROW_DECISION(event, operation_bank, decision_cycle, "HIT_DEFER");
            operation_bank->AddPendingPrechargeEvent(event);
            spdlog::debug("(EventDriven DRAM) At {} Cycle input event {} are added into the Pending Precharge queue of channel {}, Rank {}, Bankgroup {}, Bank {}, Row {} for pending bank row switch",
                event->add_cycle, event->dram_address, event->channel_index, event->rank_index, event->bankgroup_index, event->bank_index, event->row_index);
        }
        else {
            ED_TRACE_ROW_DECISION(event, operation_bank, decision_cycle, "HIT_EXECUTE");
            PendingExecuted(event);
            operation_bank->open_row_exec_event_count++;
            spdlog::debug("(EventDriven DRAM) At {} Cycle input event {} are added into the Executed queue, row buffer hit, current Executed queue size is {}",
                event->add_cycle, event->dram_address, dram_channels[event->channel_index]->execute_queue.size());
        }
    }
}

void MemorySystem::CheckPIMRowSwitch(std::shared_ptr<Event> event) {
    if (!is_pim_row_buffer_access(event->req_type)) {
        PendingExecuted(event);
        return;
    }
    // Row state can change while this request waits behind earlier FIFO PIM
    // commands.  Calculate the channel-wide PRE/ACT timing only when the
    // request reaches the execute head in GetExecuteCycle(); doing it here as
    // well charges the same row transition twice.
    PendingExecuted(event);
}


void MemorySystem::PendingPrecharge(std::shared_ptr<Event> event) {
    event->need_precharge = true;
    event->precharge_cycle = event->add_cycle + 1;
    dram_channels[event->channel_index]->precharge_queue.push_back(event);
    auto operation_bank = dram_channels[event->channel_index]->dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];
    operation_bank->pending_precharge = true;
    // operation_bank->pending_precharge_event[event->row_index].push_back(event);
    if (dram_channels[event->channel_index]->precharge_queue.size() == 1) {
        dram_channels[event->channel_index]->GetPrechargeCycle(dram_channels[event->channel_index]->precharge_queue.front());
        spdlog::debug("(EventDriven DRAM) Get the Precharge time of the first precharge event with address {} of channel {} at {}",
            event->dram_address, event->channel_index, event->precharge_cycle);
    }
}

void MemorySystem::PendingActivate(std::shared_ptr<Event> event) {
    event->need_activate = true;
    event->activate_cycle = event->add_cycle + 1;
    dram_channels[event->channel_index]->activate_queue.push_back(event);
    if (dram_channels[event->channel_index]->activate_queue.size() == 1) {
        dram_channels[event->channel_index]->GetActivateCycle(dram_channels[event->channel_index]->activate_queue.front()); // Non-ASCII comment replaced to avoid encoding issues.
        spdlog::debug("(EventDriven DRAM) Get the Activation time of the first activate event with address {} of channel {} at {}",
            event->dram_address, event->channel_index,  event->activate_cycle);
    }
}

void MemorySystem::PendingExecuted(std::shared_ptr<Event> event) {
    event->execute_cycle = event->execute_cycle + 1;
    dram_channels[event->channel_index]->execute_queue.push_back(event);
    if (dram_channels[event->channel_index]->execute_queue.size() == 1) {
        dram_channels[event->channel_index]->RefreshExecutionSchedulerHeadCycles(); // Non-ASCII comment replaced to avoid encoding issues.
        spdlog::debug("(EventDriven DRAM) Get the Execution time of the first execute event with address {} of channel {} at {}",
            event->dram_address, event->channel_index, event->execute_cycle);
    }
}

ResponseQueue::ResponseQueue(int Size) : Size(Size), NumReserved(0) {}

bool ResponseQueue::isAvailable() const {
    return NumReserved + OutputQueue.size() < Size;
}

bool ResponseQueue::isAvailable(uint32_t count) const {
    return NumReserved + OutputQueue.size() + count - 1 < Size;
}

void ResponseQueue::reserve() {
    if (NumReserved > Size) {
        spdlog::debug("current ResponseQueue reserve {} trace", NumReserved);
    }
    else if (NumReserved == Size) {
        spdlog::debug("current ResponseQueue fill with {} trace, while a new trace push", Size);
    }
    NumReserved++;
}

void ResponseQueue::push(MemoryAccess* original_req) {
    OutputQueue.push_back(original_req);
    assert(NumReserved > 0);
    NumReserved--;
    spdlog::debug("(EventDriven DRAM) Response queue push, current ReservedNumber is {} ", NumReserved);
}

bool ResponseQueue::isEmpty() const { return OutputQueue.empty(); }

void ResponseQueue::pop() {
    OutputQueue.pop_front();
}

MemoryAccess* ResponseQueue::top() const { return OutputQueue.front(); }


//////////////////////////////////////////
// EventDrivenDram constructor
//////////////////////////////////////////
EventDrivenDram::EventDrivenDram(const SysConfig& config,
                                 DramDataContainer* data_container)
    : dramsim3_config_(std::make_unique<dramsim3::Config>(
          config.memory_config_path_, config.output_path_)),
      _data_container(data_container) {
    // PIM completion callback
    std::function<void(uint64_t)> pim_callback = [&](uint64_t addr) {
        auto channel_index = MyAddressAllocator::get_channel_index(addr);
        auto it = _pending_pim_events[channel_index].find(addr);
        assert(_pending_pim_events[channel_index].count(addr)>0);
        auto memory_req = it->second->original_req;
        memory_req->dram_finish_cycle = it->second->complete_cycle;
        response_event_queues_[channel_index].push(memory_req);    // push Memory Req
        _pending_pim_events[channel_index].erase(it);              // pop
        spdlog::debug("(EventDriven DRAM) write_callback of address {} at cycle {}", addr, memory_req->dram_finish_cycle);
    };

    EventDrivenParams::init(config);

    pim_callback_ = pim_callback;

    _pending_read_events.resize(config.dram_channels);
    _pending_write_events.resize(config.dram_channels);
    _pending_pim_events.resize(config.dram_channels);
    _open_read_merge_events.resize(config.dram_channels);
    _open_write_merge_events.resize(config.dram_channels);

    _read_queue.resize(config.dram_channels);
    _write_buffer.resize(config.dram_channels);
    _pim_queue.resize(config.dram_channels);

    transaction_queue_size = dramsim3_config_->trans_queue_size;

    // response queue for each channel
    for (int ch = 0; ch < config.dram_channels; ++ch) {
        response_event_queues_.emplace_back(ResponseQueue(transaction_queue_size));
    }

    _stats.resize(config.dram_channels);
    _stats_last_cycle.resize(config.dram_channels, 0);
    transaction_arrival_sequence_.resize(config.dram_channels, 0);
    for (auto &channel_stats : _stats) {
        channel_stats.all_bank_idle_cycles.resize(config.mem_config.ranks, 0);
        channel_stats.rank_active_cycles.resize(config.mem_config.ranks, 0);
        channel_stats.estimated_all_bank_idle_cycles.resize(
            config.mem_config.ranks, 0);
        channel_stats.estimated_rank_active_cycles.resize(
            config.mem_config.ranks, 0);
        channel_stats.pim_all_bank_idle_cycles.resize(
            config.mem_config.ranks, 0);
        channel_stats.pim_rank_active_cycles.resize(
            config.mem_config.ranks, 0);
        channel_stats.estimated_pim_all_bank_idle_cycles.resize(
            config.mem_config.ranks, 0);
        channel_stats.estimated_pim_rank_active_cycles.resize(
            config.mem_config.ranks, 0);
        channel_stats.time_compensation_idle_baseline.resize(
            config.mem_config.ranks, 0);
        channel_stats.time_compensation_active_baseline.resize(
            config.mem_config.ranks, 0);
    }
#if ENABLE_DRAM_ALIGNMENT_TRACE
    if (config.record_dram_completion_trace) {
        const auto open_trace = [](std::ofstream& stream,
                                   const std::string& path) {
            stream.open(path, std::ofstream::out | std::ofstream::trunc);
            if (!stream.is_open()) {
                throw std::runtime_error(
                    "Cannot open EventDriven diagnostic trace: " + path);
            }
        };
        open_trace(command_trace_, Config::system_config.log_dir +
                                       "/dram_command_eventdriven.csv");
        open_trace(time_advance_trace_, Config::system_config.log_dir +
                                            "/dram_time_advance_eventdriven.csv");
        open_trace(transaction_trace_, Config::system_config.log_dir +
                                           "/dram_transaction_eventdriven.csv");
        command_trace_ << "command_sequence,channel,cycle,command,address,rank,bankgroup,bank,row,col,open_row_before,source\n";
        time_advance_trace_ << "advance_sequence,channel,source,from_cycle,to_cycle,delta,earliest_type,earliest_cycle\n";
        transaction_trace_ << "sequence,arrival_sequence,channel,cycle,action,type,address,detail\n";
        if (!command_trace_.good() || !time_advance_trace_.good() ||
            !transaction_trace_.good()) {
            throw std::runtime_error(
                "Failed to write EventDriven diagnostic trace headers");
        }
    }
#endif

    _push_valid.resize(config.dram_channels, true);
    _pop_valid.resize(config.dram_channels, true);

    _write_draining.resize(config.dram_channels, 0);
    _last_transaction_schedule_cycle.resize(
        config.dram_channels, std::numeric_limits<cycle_type>::max());
    _last_idle_schedule_check_cycle.resize(
        config.dram_channels, std::numeric_limits<cycle_type>::max());
    // Start due so every channel performs one full scheduling pass before the
    // fast path can skip idle ICNT ticks.
    _next_wakeup_cycle.resize(config.dram_channels, 0);
    _next_return_completion_cycle.resize(
        config.dram_channels, std::numeric_limits<cycle_type>::max());
    _rw_dependency_lock.resize(config.dram_channels, false);
    _rw_dependency_addr.resize(config.dram_channels, 0);

    _memsys = std::make_unique<MemorySystem>(config.mem_config, 0);

    spdlog::info("========== EventDriven Config ==========");
    spdlog::info("[EVENT_CONFIG] tRCD={}, tRCDRD={}, AL={}",
                 _memsys->config_.tRCD,
                 _memsys->config_.tRCDRD,
                 _memsys->config_.AL);
    spdlog::info("[EVENT_CONFIG] activate_read={}",
                 _memsys->config_.activate_read);
    spdlog::info("[EVENT_CONFIG] read_delay={}, RL={}, burst_cycle={}",
                 _memsys->config_.read_delay,
                 _memsys->config_.RL,
                 _memsys->config_.burst_cycle);
    spdlog::info("[EVENT_CONFIG] tRAS={}, tRP={}, tRC={}",
                 _memsys->config_.tRAS,
                 _memsys->config_.tRP,
                 _memsys->config_.tRC);
    spdlog::info("[EVENT_CONFIG] row_buf_policy={}", _memsys->config_.row_buf_policy);
    spdlog::info("========================================");

    spdlog::debug("EventDrivenDram initialized");
    spdlog::debug("  channels   = {}", _memsys->config_.channels);
    spdlog::debug("  shift_bits = {}", _memsys->config_.shift_bits);
    spdlog::debug("  ch_pos     = {}", _memsys->config_.ch_pos);
    spdlog::debug("  ch_mask    = {:#x}", _memsys->config_.ch_mask);
}

EventDrivenDram::~EventDrivenDram() = default;

void EventDrivenDram::advance_cycle(cycle_type, size_t, size_t, double) {
    // ED advances channels directly to the timestamp of the next observable
    // request or completion. A simulator DRAM tick therefore has no local
    // state to advance.
}

void EventDrivenDram::synchronize_cycles(cycle_type) {
    // ED uses the Simulator's absolute DRAM cycle when requests are prepared.
}

void EventDrivenDram::prepare_request(MemoryAccess* request,
                                      cycle_type current_cycle) {
    if (request != nullptr) {
        request->dram_enter_cycle = current_cycle;
    }
}

void EventDrivenDram::schedule_pending_work(uint32_t cid,
                                            cycle_type current_cycle) {
    schedule_pending_operation(cid, current_cycle);
}


Event EventDrivenDram::generate_event_from_req(MemoryAccess *req) {
    Event pending_event;

    req->request = false;
    pending_event.original_req = req;

    pending_event.event_type = static_cast<EventType>(int(req->req_type));
    pending_event.dram_address = req->dram_address;
    pending_event.spad_address = req->spad_address;

    pending_event.channel_index = MyAddressAllocator::get_channel_index(pending_event.dram_address);
    pending_event.rank_index = MyAddressAllocator::get_rank_index(pending_event.dram_address);
    pending_event.bankgroup_index = MyAddressAllocator::get_bankgroup_index(pending_event.dram_address);
    pending_event.bank_index = MyAddressAllocator::get_bank_index(pending_event.dram_address);
    pending_event.row_index = MyAddressAllocator::get_row_index(pending_event.dram_address);
    pending_event.column_index = MyAddressAllocator::get_col_index(pending_event.dram_address);

    pending_event.parent_tile = req->parent_tile;
    pending_event.stage_platform = req->stage_platform;

    pending_event.id = req->id;
    pending_event.core_id = req->core_id;
    pending_event.mem_id = req->mem_id;
    pending_event.buffer_id = req->buffer_id;

    pending_event.req_type = req->req_type;
    pending_event.size = req->size;

    pending_event.add_cycle = req->dram_enter_cycle;
    pending_event.complete_cycle = req->dram_enter_cycle;

    pending_event.need_precharge = false;
    pending_event.need_activate = false;
    pending_event.precharge_cycle = 0;
    pending_event.activate_cycle = 0;

    pending_event.execute_cycle = 0;
    pending_event.processed = false;

    return pending_event;
}


void EventDrivenDram::push(uint32_t cid, MemoryAccess *req) {
    if (req == nullptr) {
        return;
    }
    if (req->req_type == MemoryAccessType::READ) {
        _stats[cid].num_read_requests++;
    } else if (req->req_type == MemoryAccessType::WRITE) {
        _stats[cid].num_write_requests++;
    } else {
        _stats[cid].num_pim_requests++;
    }
    // A push changes the channel queues even when the DRAM cycle number is
    // unchanged, so the next idle check in this cycle must not be suppressed.
    _last_idle_schedule_check_cycle[cid] =
        std::numeric_limits<cycle_type>::max();
    _memsys->dram_channels[cid]->_pre_dram_cycle = _memsys->dram_channels[cid]->_dram_cycle;
    _memsys->dram_channels[cid]->_dram_cycle = std::max(_memsys->dram_channels[cid]->_dram_cycle, req->dram_enter_cycle);

    spdlog::debug("----------- EventDriven DRAM Cycle Updated of Channel {} from {} to {} by {} MemoryAccess Push  -----------",
    cid, _memsys->dram_channels[cid]->_pre_dram_cycle, _memsys->dram_channels[cid]->_dram_cycle, req->dram_address);
    process_finish_event(cid, _memsys->dram_channels[cid]->_dram_cycle);

    // Generate Event
    const addr_type addr = req->dram_address;
    auto event = std::make_shared<Event>(generate_event_from_req(req));
            ED_TRACE_TRANSACTION(cid, event->add_cycle, "ARRIVE", event);
    response_event_queues_[cid].reserve();
    if (req->req_type == MemoryAccessType::WRITE) {
        std::shared_ptr<Event> open_event;
        auto open_it = _open_write_merge_events[cid].find(addr);
        if (open_it != _open_write_merge_events[cid].end()) {
            open_event = open_it->second.lock();
        }
        if (open_event != nullptr && open_event->merge_open) {
            _stats[cid].num_write_merges++;
            _stats[cid].num_write_buf_hits++;
            ED_TRACE_TRANSACTION(cid, event->add_cycle, "MERGED_WRITE", event);
            spdlog::debug("Write merge for address {} into group {}",
                          addr, open_event->merge_group_id);
        }
        else {
            event->merge_group_id = next_merge_group_id_++;
            event->merge_open = true;
            _pending_write_events[cid].insert(std::make_pair(addr, event));
            _write_buffer[cid].push_back(event);
            _open_write_merge_events[cid][addr] = event;
        }

        auto response_event = std::make_shared<Event>(*event);
        // The physical write event remains in the write buffer after the
        // logical response returns to the Core. Only the response event may
        // retain this borrowed pointer; otherwise the physical event would
        // keep a dangling pointer after Core releases the request.
        event->original_req = nullptr;
        response_event->response_only = true;
        response_event->complete_cycle = _memsys->dram_channels[cid]->_dram_cycle + 1;
        _memsys->dram_channels[cid]->return_queue.push_back(response_event);
        note_return_completion(cid, response_event);
    }
    else if (req->req_type == MemoryAccessType::READ) {
        std::shared_ptr<Event> open_write;
        auto write_it = _open_write_merge_events[cid].find(addr);
        if (write_it != _open_write_merge_events[cid].end()) {
            open_write = write_it->second.lock();
        }

        if (open_write != nullptr && open_write->merge_open) {
            // Match Cycle Accurate write-buffer forwarding. Each logical read
            // receives its own one-cycle response and is not a DRAM command.
            event->response_only = true;
            ED_TRACE_TRANSACTION(cid, event->add_cycle, "FORWARDED_READ", event);
            event->merge_open = false;
            event->merged_original_reqs.push_back(req);
            event->complete_cycle = _memsys->dram_channels[cid]->_dram_cycle + 1;
            _memsys->dram_channels[cid]->return_queue.push_back(event);
            note_return_completion(cid, event);
        }
        else {
            std::shared_ptr<Event> open_read;
            auto read_it = _open_read_merge_events[cid].find(addr);
            if (read_it != _open_read_merge_events[cid].end()) {
                open_read = read_it->second.lock();
            }
            if (open_read != nullptr && open_read->merge_open) {
#if ENABLE_DRAM_ALIGNMENT_TRACE
                TracingPendingReadMap::record_merge(
                    addr, cid, event->add_cycle, open_read->add_cycle,
                    open_read->merged_original_reqs.size());
#endif
                open_read->merged_original_reqs.push_back(req);
                _stats[cid].num_read_merges++;
                ED_TRACE_TRANSACTION(cid, event->add_cycle, "MERGED_READ", event);
                spdlog::debug("Read merge for address {} into group {}",
                              addr, open_read->merge_group_id);
            }
            else {
                event->merge_group_id = next_merge_group_id_++;
                event->merge_open = true;
                event->merged_original_reqs.push_back(req);
                _pending_read_events[cid].insert(std::make_pair(addr, event));
                _read_queue[cid].push_back(event);
                _open_read_merge_events[cid][addr] = event;
            }
        }
    }
    else {  // PIM Request
        _pending_pim_events[cid].insert(std::make_pair(addr, event));
        _pim_queue[cid].push_back(event);
        // spdlog::debug("A PIM transaction {} is insert into pending_pim_queue and pim_queue_ of channel {}", trans.TransactionTypeString(), channel_id_);
    }
    schedule_pending_event_transaction(cid);
    recompute_next_wakeup(cid, _memsys->dram_channels[cid]->_dram_cycle);
}

void EventDrivenDram::pop(uint32_t cid) {
    auto* response_transaction = response_event_queues_[cid].top();
    response_event_queues_[cid].pop();
    spdlog::debug("(EventDriven DRAM) ResponseQueue : At Cycle {}, channel {} pop transaction {}, {} exist",
        _memsys->dram_channels[cid]->_dram_cycle, cid, response_transaction->dram_address, response_event_queues_[cid].NumReserved);
}

MemoryAccess *EventDrivenDram::top(uint32_t cid) {
    auto* memory_response = response_event_queues_[cid].top();
    if (_data_container != nullptr) {
        _data_container->apply_response(memory_response);
    }
    return memory_response;
}

bool EventDrivenDram::is_full(uint32_t cid, MemoryAccess *req) {
    process_finish_event(cid, req->dram_enter_cycle);
    bool return_queue_available = response_event_queues_[cid].isAvailable(1);
    bool input_queue_available;
    if (req->req_type == MemoryAccessType::WRITE) {
        input_queue_available = _write_buffer[cid].size() < transaction_queue_size;
    }
    else {
        input_queue_available = _read_queue[cid].size() < transaction_queue_size;
    }

    bool is_full = !(return_queue_available && input_queue_available);
    return is_full;
}

bool EventDrivenDram::is_empty(uint32_t cid) {
    bool empty = response_event_queues_[cid].isEmpty();
    return empty;
}

uint32_t EventDrivenDram::get_channel_id(MemoryAccess *access) {
    return _memsys->GetChannel(access->dram_address);
}

void EventDrivenDram::trace_issued_command(
    uint32_t cid, cycle_type cycle, const char* command,
    const std::shared_ptr<Event>& event, const char* source) {
    constexpr uint64_t kCommandTraceLimit = 40000;
    if (!command_trace_.is_open() || event == nullptr ||
        command_trace_sequence_ >= kCommandTraceLimit) {
        return;
    }
    auto* bank = _memsys->dram_channels[cid]
                     ->dram_ranks[event->rank_index]
                     ->dram_bankgroups[event->bankgroup_index]
                     ->dram_banks[event->bank_index];
    command_trace_ << command_trace_sequence_++ << ',' << cid << ','
                   << cycle << ',' << command << ',' << event->dram_address
                   << ',' << event->rank_index << ',' << event->bankgroup_index
                   << ',' << event->bank_index << ',' << event->row_index
                   << ',' << event->column_index << ',' << bank->open_row
                   << ',' << source << '\n';
}

cycle_type EventDrivenDram::issue_rank_refresh_with_precharges(
    uint32_t cid, cycle_type requested_cycle, const char* source) {
    auto* channel = _memsys->dram_channels[cid];
    const cycle_type actual_refresh_cycle =
        channel->IssueRankRefresh(requested_cycle);

    constexpr uint64_t kCommandTraceLimit = 40000;
    for (const auto& record : channel->refresh_precharge_records) {
        accumulate_background_cycles(cid, record.cycle);
        _stats[cid].num_pre_cmds++;
        if (command_trace_.is_open() &&
            command_trace_sequence_ < kCommandTraceLimit) {
            command_trace_ << command_trace_sequence_++ << ',' << cid << ','
                           << record.cycle << ",PRE,0," << record.rank << ','
                           << record.bankgroup << ',' << record.bank << ','
                           << record.row << ",-1," << record.row
                           << ",refresh_" << source << '\n';
        }
    }

    accumulate_background_cycles(cid, actual_refresh_cycle);
    _stats[cid].num_ref_cmds++;
    if (command_trace_.is_open() &&
        command_trace_sequence_ < kCommandTraceLimit) {
        command_trace_ << command_trace_sequence_++ << ',' << cid << ','
                       << actual_refresh_cycle
                       << ",REF,0,-1,-1,-1,-1,-1,-1,refresh_"
                       << source << '\n';
    }
    spdlog::debug(
        "(EventDriven DRAM) Rank Refresh requested at cycle {} for Channel {}, issued at cycle {} after {} explicit PRE commands",
        requested_cycle, cid, actual_refresh_cycle,
        channel->refresh_precharge_records.size());
    channel->RecordPhysicalCommandCycle(actual_refresh_cycle);
    return actual_refresh_cycle;
}

void EventDrivenDram::trace_time_advance(
    uint32_t cid, cycle_type from_cycle, cycle_type to_cycle,
    int earliest_type, cycle_type earliest_cycle, const char* source) {
    constexpr uint64_t kTimeAdvanceTraceLimit = 40000;
    if (!time_advance_trace_.is_open() ||
        time_advance_trace_sequence_ >= kTimeAdvanceTraceLimit) {
        return;
    }
    const cycle_type delta = to_cycle >= from_cycle ? to_cycle - from_cycle : 0;
    time_advance_trace_ << time_advance_trace_sequence_++ << ',' << cid << ','
                        << source << ',' << from_cycle << ',' << to_cycle << ','
                        << delta << ',' << earliest_type << ','
                        << earliest_cycle << '\n';
}

void EventDrivenDram::trace_transaction(
    uint32_t cid, cycle_type cycle, const char* action,
    const std::shared_ptr<Event>& event, const char* detail) {
    constexpr uint64_t kTransactionTraceLimit = 10000;
    if (!transaction_trace_.is_open() || event == nullptr ||
        transaction_trace_sequence_ >= kTransactionTraceLimit) {
        return;
    }
    const bool is_arrival = std::string(action) == "ARRIVE";
    transaction_trace_ << transaction_trace_sequence_++ << ',';
    if (is_arrival) {
        transaction_trace_ << transaction_arrival_sequence_[cid]++;
    }
    transaction_trace_ << ',' << cid << ',' << cycle << ',' << action << ','
                       << memAccessTypeString(event->req_type) << ','
                       << event->dram_address << ',' << detail << '\n';
}


void EventDrivenDram::erase_pending_event(
    std::multimap<uint64_t, std::shared_ptr<Event>>& pending,
    uint64_t addr,
    const std::shared_ptr<Event>& event) {
    auto range = pending.equal_range(addr);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == event) {
            pending.erase(it);
            return;
        }
    }
    assert(false && "completed event is missing from its pending generation");
}


void EventDrivenDram::note_return_completion(
    uint32_t cid, const std::shared_ptr<Event>& event) {
    if (event != nullptr) {
        _next_return_completion_cycle[cid] = std::min(
            _next_return_completion_cycle[cid], event->complete_cycle);
    }
}


void EventDrivenDram::recompute_return_completion(uint32_t cid) {
    cycle_type next_cycle = std::numeric_limits<cycle_type>::max();
    for (const auto& event : _memsys->dram_channels[cid]->return_queue) {
        if (event != nullptr) {
            next_cycle = std::min(next_cycle, event->complete_cycle);
        }
    }
    _next_return_completion_cycle[cid] = next_cycle;
}


void EventDrivenDram::process_finish_event(uint32_t cid, cycle_type current_cycle) {
    auto& return_queue = _memsys->dram_channels[cid]->return_queue;
    if (return_queue.empty() ||
        current_cycle <= _next_return_completion_cycle[cid]) {
        return;
    }
    auto it = return_queue.begin();
    bool removed_event = false;
    while (it != return_queue.end()) {
        auto finish_event = *it;
        if (finish_event == nullptr) {
            break;
        }
        if (current_cycle <= finish_event->complete_cycle) {
            ++it;
            continue;
        }
        if (finish_event->req_type == MemoryAccessType::WRITE) {
            if (finish_event->response_only) {
                ED_TRACE_TRANSACTION(cid, finish_event->complete_cycle,
                                  "LOGICAL_COMPLETE", finish_event);
                record_completed_request(cid, finish_event->req_type);
                auto memory_req = finish_event->original_req;
                memory_req->dram_finish_cycle = finish_event->complete_cycle;
                response_event_queues_[cid].push(memory_req);
                spdlog::debug("*** (EventDriven DRAM) write response of address {} for channel {} at cycle {}",
                    finish_event->dram_address, cid, memory_req->dram_finish_cycle);
            }
            else {
                ED_TRACE_TRANSACTION(cid, finish_event->complete_cycle,
                                  "PHYSICAL_COMPLETE", finish_event);
                erase_pending_event(_pending_write_events[cid],
                                    finish_event->dram_address,
                                    finish_event);
                auto open_it = _open_write_merge_events[cid].find(
                    finish_event->dram_address);
                if (open_it != _open_write_merge_events[cid].end() &&
                    open_it->second.lock() == finish_event) {
                    _open_write_merge_events[cid].erase(open_it);
                }
            }
            it = return_queue.erase(it);
            removed_event = true;
        }
        else if (finish_event->req_type == MemoryAccessType::READ) {
            ED_TRACE_TRANSACTION(cid, finish_event->complete_cycle,
                              "COMPLETE", finish_event);
            record_completed_request(cid, finish_event->req_type);
            assert(!finish_event->merged_original_reqs.empty());
            for (auto* memory_req : finish_event->merged_original_reqs) {
                memory_req->dram_finish_cycle = finish_event->complete_cycle;
                response_event_queues_[cid].push(memory_req);
            }
            _stats[cid].num_logical_reads_done +=
                finish_event->merged_original_reqs.size();

            if (!finish_event->response_only) {
                erase_pending_event(_pending_read_events[cid],
                                    finish_event->dram_address,
                                    finish_event);
                auto open_it = _open_read_merge_events[cid].find(
                    finish_event->dram_address);
                if (open_it != _open_read_merge_events[cid].end() &&
                    open_it->second.lock() == finish_event) {
                    _open_read_merge_events[cid].erase(open_it);
                }
            }
            it = return_queue.erase(it);
            removed_event = true;
            assert(finish_event.use_count() == 1);
        }
        else if (finish_event->req_type == MemoryAccessType::P_HEADER ||
                 finish_event->req_type == MemoryAccessType::GWRITE ||
                 finish_event->req_type == MemoryAccessType::COMP ||
                 finish_event->req_type == MemoryAccessType::COMP_HASH ||
                 finish_event->req_type == MemoryAccessType::READRES ||
                 finish_event->req_type == MemoryAccessType::COMPS_READRES){
            record_completed_request(cid, finish_event->req_type);
            pim_callback_(finish_event->dram_address);
            it = return_queue.erase(it);
            removed_event = true;
            assert(finish_event.use_count() == 1);
                 }
        else {
            break;
        }
    }
    if (removed_event) {
        recompute_return_completion(cid);
    }
}


void EventDrivenDram::schedule_pending_event_transaction(uint32_t cid) {
    auto* channel = _memsys->dram_channels[cid];
    channel->RefreshEarliestQueue();
    ED_TRACE_TIME_ADVANCE(
        cid, _memsys->dram_channels[cid]->_pre_dram_cycle,
        _memsys->dram_channels[cid]->_dram_cycle,
        _memsys->dram_channels[cid]->earliest_queue.first,
        _memsys->dram_channels[cid]->earliest_queue.second, "push");
    cycle_type operation_cycle = channel->_pre_dram_cycle + 1;
    operation_cycle = std::max(operation_cycle, channel->earliest_queue.second);
    if (!channel->config_.enable_hbm_dual_cmd &&
        channel->last_physical_command_cycle !=
            std::numeric_limits<cycle_type>::max()) {
        operation_cycle = std::max(
            operation_cycle, channel->last_physical_command_cycle + 1);
    }
    while (operation_cycle <= channel->_dram_cycle) {
        std::shared_ptr<Event> issue_event;
        switch (_memsys->dram_channels[cid]->earliest_queue.first) {
            case 0:
                issue_event = _memsys->dram_channels[cid]->execute_queue.front();
                if (issue_event->pending_pim_row_command ==
                    PIMRowCommand::PRECHARGE) {
                    issue_event->pim_precharge_cycle = operation_cycle;
                    accumulate_background_cycles(cid, operation_cycle);
                    _stats[cid].num_pim_precharge_cmds++;
                    ED_TRACE_COMMAND(
                        cid, operation_cycle, "PIM_PRECHARGE",
                        issue_event, "push");
                    channel->IssuePIMPrechargeEvent(
                        issue_event, operation_cycle);
                    break;
                }
                if (issue_event->pending_pim_row_command ==
                    PIMRowCommand::ACTIVATE) {
                    issue_event->pim_activate_cycle = operation_cycle;
                    accumulate_background_cycles(cid, operation_cycle);
                    _stats[cid].num_pim_activate_cmds++;
                    ED_TRACE_COMMAND(
                        cid, operation_cycle, "PIM_ACTIVE",
                        issue_event, "push");
                    channel->IssuePIMActivateEvent(
                        issue_event, operation_cycle);
                    break;
                }
                spdlog::debug("(EventDriven DRAM) Event {} is Executed at cycle {} for Channel {}, Rank {}, Bankgroup {}, Bank {} Row {} Column {}, tick by push",
                    issue_event->dram_address, operation_cycle, issue_event->channel_index, issue_event->rank_index, issue_event->bankgroup_index, issue_event->bank_index,issue_event->row_index, issue_event->column_index);
                accumulate_background_cycles(cid, operation_cycle);
                record_execute_command(cid, issue_event);
                ED_TRACE_COMMAND(
                    cid, operation_cycle,
                    issue_event->req_type == MemoryAccessType::READ ? "READ" :
                    issue_event->req_type == MemoryAccessType::WRITE ? "WRITE" : "PIM",
                    issue_event, "push");
                _memsys->dram_channels[cid]->IssueExecuteEvent(issue_event, operation_cycle);
                note_return_completion(cid, issue_event);
                break;
            case 1:
                issue_event = _memsys->dram_channels[cid]->activate_queue.front();
                issue_event->activate_cycle = operation_cycle;
                spdlog::debug("(EventDriven DRAM) Event {} is Activated at cycle {} for Channel {}, Rank {}, Bankgroup {}, Bank {} Row {}, tick by push",
                    issue_event->dram_address, operation_cycle, issue_event->channel_index, issue_event->rank_index, issue_event->bankgroup_index, issue_event->bank_index,issue_event->row_index);
                accumulate_background_cycles(cid, operation_cycle);
                _stats[cid].num_act_cmds++;
                issue_event->activated_for_access = true;
                ED_TRACE_COMMAND(cid, operation_cycle, "ACT", issue_event, "push");
                _memsys->dram_channels[cid]->IssueActivateEvent(issue_event);
                break;
            case 2:
                issue_event = _memsys->dram_channels[cid]->precharge_queue.front();
                issue_event->precharge_cycle = operation_cycle;
                spdlog::debug("(EventDriven DRAM) Event {} is Precharged at cycle {} for Channel {}, Rank {}, Bankgroup {}, Bank {} Row {}, tick by push",
                    issue_event->dram_address, operation_cycle, issue_event->channel_index, issue_event->rank_index, issue_event->bankgroup_index, issue_event->bank_index, issue_event->row_index);
                accumulate_background_cycles(cid, operation_cycle);
                _stats[cid].num_pre_cmds++;
                _stats[cid].num_ondemand_pres++;
                ED_TRACE_COMMAND(cid, operation_cycle, "PRE", issue_event, "push");
                _memsys->dram_channels[cid]->IssuePrechargeEvent(issue_event);
                break;
            case 3:
                operation_cycle = issue_rank_refresh_with_precharges(
                    cid, operation_cycle, "push");
                break;
            default:
                throw std::runtime_error("not support");
        }
        channel->RecordPhysicalCommandCycle(operation_cycle);
        operation_cycle = std::max(operation_cycle, channel->earliest_queue.second);
        if (!channel->config_.enable_hbm_dual_cmd) {
            operation_cycle = std::max(
                operation_cycle, channel->last_physical_command_cycle + 1);
        }
    }

    // CycleAccurate enters refresh-waiting mode as soon as tREFI expires and
    // stops issuing normal commands from the affected rank.  EventDriven has
    // already materialized transactions into ACT/PRE/execute queues, so those
    // commands must drain before the rank state can be refreshed safely.  Do
    // not refill the physical queues while a refresh is due; once they become
    // empty, RefreshEarliestQueue() selects the overdue refresh immediately.
    if (channel->GetNextRankRefreshCycle() <= channel->_dram_cycle) {
        return;
    }

    const cycle_type transaction_cycle = _memsys->dram_channels[cid]->_dram_cycle;
    if (_last_transaction_schedule_cycle[cid] == transaction_cycle) {
        return;
    }

    const bool physical_queue_empty = channel->execute_queue.empty() &&
                                      channel->activate_queue.empty() &&
                                      channel->precharge_queue.empty();
    if (!_rw_dependency_lock[cid] && _write_draining[cid] == 0) {
        const size_t write_size = _write_buffer[cid].size();
        const auto write_capacity = static_cast<size_t>(dramsim3_config_->trans_queue_size);
        if ((write_capacity > 0 && write_size >= write_capacity)
            || (write_size > 8 && physical_queue_empty)
            || (write_size > 0 && !_pending_pim_events[cid].empty())) {
            _write_draining[cid] = static_cast<int>(write_size);
        }
    }
    QueueClass queue_to_schedule;
    if (_rw_dependency_lock[cid]) {
         queue_to_schedule = QueueClass::READ_Q;
    }
    else if (_write_draining[cid] > 0) {
         queue_to_schedule = QueueClass::WRITE_Q;
    }
    else if (!_pim_queue[cid].empty() && _read_queue[cid].empty() &&
             _write_buffer[cid].empty() && physical_queue_empty) {
         // PIM PRE/ACT commands change the row state of every bank in the
         // channel.  They must not overtake already-materialized physical
         // READ/WRITE commands, regardless of the batch scheduling mode.
         queue_to_schedule = QueueClass::PIM_Q;
    }
    else {
        queue_to_schedule = QueueClass::READ_Q;
    }

    const bool queue_has_work =
        (queue_to_schedule == QueueClass::READ_Q && !_read_queue[cid].empty()) ||
        (queue_to_schedule == QueueClass::WRITE_Q &&
         !_write_buffer[cid].empty() && !_pending_write_events[cid].empty()) ||
        (queue_to_schedule == QueueClass::PIM_Q && !_pim_queue[cid].empty());
    if (!queue_has_work) {
        return;
    }
    _last_transaction_schedule_cycle[cid] = transaction_cycle;

    switch (queue_to_schedule) {
        case QueueClass::READ_Q:
            issue_pending_read_event(cid);
            break;
        case QueueClass::WRITE_Q:
            issue_pending_write_event(cid);
            break;
        case QueueClass::PIM_Q:
            issue_pending_pim_event(cid);
            break;
        default:
            throw std::logic_error(
                "Unknown EventDriven DRAM queue class on channel " +
                std::to_string(cid));
    }
}

void EventDrivenDram::recompute_next_wakeup(uint32_t cid,
                                            cycle_type current_cycle) {
    auto* channel = _memsys->dram_channels[cid];
    cycle_type next_cycle = std::numeric_limits<cycle_type>::max();

    if (channel->earliest_queue.first != -1) {
        next_cycle = channel->earliest_queue.second;
    }

    if (!channel->return_queue.empty()) {
        const cycle_type complete_cycle =
            _next_return_completion_cycle[cid];
        if (complete_cycle != std::numeric_limits<cycle_type>::max()) {
            next_cycle = std::min(next_cycle, complete_cycle + 1);
        }
    }

    // When no physical command is materialized, a logical transaction still
    // needs another scheduling opportunity. Existing behavior admits at most
    // one such transaction per ICNT scheduling cycle.
    const bool has_logical_work = !_read_queue[cid].empty() ||
                                  !_write_buffer[cid].empty() ||
                                  !_pim_queue[cid].empty();
    if (channel->earliest_queue.first == -1 && has_logical_work) {
        const cycle_type next_transaction_cycle =
            current_cycle == std::numeric_limits<cycle_type>::max()
                ? current_cycle
                : current_cycle + 1;
        next_cycle = std::min(next_cycle, next_transaction_cycle);
    }

    _next_wakeup_cycle[cid] = next_cycle;
}


void EventDrivenDram::schedule_pending_operation(uint32_t cid, cycle_type current_cycle) {
    auto* channel = _memsys->dram_channels[cid];
    if (_last_idle_schedule_check_cycle[cid] == current_cycle) {
        return;
    }
    _last_idle_schedule_check_cycle[cid] = current_cycle;
    if (current_cycle < _next_wakeup_cycle[cid]) {
        // RefreshEarliestQueue() is intentionally retained here. Besides
        // calculating readiness, it reproduces the baseline scheduler's
        // per-tick queue normalization and therefore preserves merge/issue
        // ordering. We still avoid the full return-queue completion scan when
        // the cached completion minimum proves that nothing can finish yet.
        channel->RefreshEarliestQueue();
        if (channel->earliest_queue.first == -1 ||
            current_cycle < channel->earliest_queue.second) {
            return;
        }
    }
    process_finish_event(cid, current_cycle);
    channel->RefreshEarliestQueue();
    ED_TRACE_TIME_ADVANCE(
        cid, _memsys->dram_channels[cid]->_dram_cycle, current_cycle,
        _memsys->dram_channels[cid]->earliest_queue.first,
        _memsys->dram_channels[cid]->earliest_queue.second, "idle");
    if (current_cycle < _memsys->dram_channels[cid]->earliest_queue.second &&
        _memsys->dram_channels[cid]->earliest_queue.first != -1) {
        recompute_next_wakeup(cid, current_cycle);
        return;
    }
    else {
        _memsys->dram_channels[cid]->_pre_dram_cycle = _memsys->dram_channels[cid]->_dram_cycle;
        _memsys->dram_channels[cid]->_dram_cycle = current_cycle;
        cycle_type operation_cycle = channel->_pre_dram_cycle + 1;
        operation_cycle = std::max(operation_cycle, channel->earliest_queue.second);
        if (!channel->config_.enable_hbm_dual_cmd &&
            channel->last_physical_command_cycle !=
                std::numeric_limits<cycle_type>::max()) {
            operation_cycle = std::max(
                operation_cycle, channel->last_physical_command_cycle + 1);
        }
        while (operation_cycle <= channel->_dram_cycle) {
            std::shared_ptr<Event> issue_event;
            switch (_memsys->dram_channels[cid]->earliest_queue.first) {
                case 0:
                    issue_event = _memsys->dram_channels[cid]->execute_queue.front();
                    if (issue_event->pending_pim_row_command ==
                        PIMRowCommand::PRECHARGE) {
                        issue_event->pim_precharge_cycle = operation_cycle;
                        accumulate_background_cycles(cid, operation_cycle);
                        _stats[cid].num_pim_precharge_cmds++;
                        ED_TRACE_COMMAND(
                            cid, operation_cycle, "PIM_PRECHARGE",
                            issue_event, "idle");
                        channel->IssuePIMPrechargeEvent(
                            issue_event, operation_cycle);
                        break;
                    }
                    if (issue_event->pending_pim_row_command ==
                        PIMRowCommand::ACTIVATE) {
                        issue_event->pim_activate_cycle = operation_cycle;
                        accumulate_background_cycles(cid, operation_cycle);
                        _stats[cid].num_pim_activate_cmds++;
                        ED_TRACE_COMMAND(
                            cid, operation_cycle, "PIM_ACTIVE",
                            issue_event, "idle");
                        channel->IssuePIMActivateEvent(
                            issue_event, operation_cycle);
                        break;
                    }
                    spdlog::debug("(EventDriven DRAM) Event {} is Executed at cycle {} for Channel {}, Rank {}, Bankgroup {}, Bank {} Row {} Column {}, tick without push",
                        issue_event->dram_address, operation_cycle, issue_event->channel_index, issue_event->rank_index, issue_event->bankgroup_index, issue_event->bank_index,issue_event->row_index, issue_event->column_index);
                    accumulate_background_cycles(cid, operation_cycle);
                    record_execute_command(cid, issue_event);
                    ED_TRACE_COMMAND(
                        cid, operation_cycle,
                        issue_event->req_type == MemoryAccessType::READ ? "READ" :
                        issue_event->req_type == MemoryAccessType::WRITE ? "WRITE" : "PIM",
                        issue_event, "idle");
                    _memsys->dram_channels[cid]->IssueExecuteEvent(issue_event, operation_cycle);
                    note_return_completion(cid, issue_event);
                    break;
                case 1:
                    issue_event = _memsys->dram_channels[cid]->activate_queue.front();
                    issue_event->activate_cycle = operation_cycle;
                    spdlog::debug("(EventDriven DRAM) Event {} is Activated at cycle {} for Channel {}, Rank {}, Bankgroup {}, Bank {} Row {}, tick without push",
                        issue_event->dram_address, operation_cycle, issue_event->channel_index, issue_event->rank_index, issue_event->bankgroup_index, issue_event->bank_index,issue_event->row_index);
                    accumulate_background_cycles(cid, operation_cycle);
                    _stats[cid].num_act_cmds++;
                    issue_event->activated_for_access = true;
                    ED_TRACE_COMMAND(cid, operation_cycle, "ACT", issue_event, "idle");
                    _memsys->dram_channels[cid]->IssueActivateEvent(issue_event);
                    break;
                case 2:
                    issue_event = _memsys->dram_channels[cid]->precharge_queue.front();
                    issue_event->precharge_cycle = operation_cycle;
                    spdlog::debug("(EventDriven DRAM) Event {} is Precharged at cycle {} for Channel {}, Rank {}, Bankgroup {}, Bank {}, tick without push",
                        issue_event->dram_address, operation_cycle, issue_event->channel_index, issue_event->rank_index, issue_event->bankgroup_index, issue_event->bank_index);
                    accumulate_background_cycles(cid, operation_cycle);
                    _stats[cid].num_pre_cmds++;
                    _stats[cid].num_ondemand_pres++;
                    ED_TRACE_COMMAND(cid, operation_cycle, "PRE", issue_event, "idle");
                    _memsys->dram_channels[cid]->IssuePrechargeEvent(issue_event);
                    break;
                case 3:
                    operation_cycle = issue_rank_refresh_with_precharges(
                        cid, operation_cycle, "idle");
                    break;
                default:
                    throw std::runtime_error("not support");
            }
            channel->RecordPhysicalCommandCycle(operation_cycle);
            operation_cycle = std::max(operation_cycle, channel->earliest_queue.second);
            if (!channel->config_.enable_hbm_dual_cmd) {
                operation_cycle = std::max(
                    operation_cycle, channel->last_physical_command_cycle + 1);
            }
        }
        process_finish_event(cid, current_cycle);
    }

    // Preserve refresh priority while draining physical commands that were
    // materialized before tREFI expired.  Without this barrier, one new
    // transaction is inserted on every scheduling pass and refresh can be
    // postponed indefinitely under a sustained HBM request stream.
    if (channel->GetNextRankRefreshCycle() <= channel->_dram_cycle) {
        recompute_next_wakeup(cid, current_cycle);
        return;
    }

    if (_last_transaction_schedule_cycle[cid] == current_cycle) {
        recompute_next_wakeup(cid, current_cycle);
        return;
    }

    const bool physical_queue_empty = channel->execute_queue.empty() &&
                                      channel->activate_queue.empty() &&
                                      channel->precharge_queue.empty();
    if (!_rw_dependency_lock[cid] && _write_draining[cid] == 0) {
        const size_t write_size = _write_buffer[cid].size();
        const auto write_capacity = static_cast<size_t>(dramsim3_config_->trans_queue_size);
        if ((write_capacity > 0 && write_size >= write_capacity)
            || (write_size > 8 && physical_queue_empty)
            || (write_size > 0 && !_pending_pim_events[cid].empty())
            || (write_size > 0 && _read_queue[cid].empty())) {
            _write_draining[cid] = static_cast<int>(write_size);
        }
    }

    QueueClass queue_to_schedule;
    if (_rw_dependency_lock[cid]) {
        queue_to_schedule = QueueClass::READ_Q;
    }
    else if (_write_draining[cid] > 0) {
        queue_to_schedule = QueueClass::WRITE_Q;
    }
    else if (!_pim_queue[cid].empty() && _read_queue[cid].empty() &&
             _write_buffer[cid].empty() && physical_queue_empty) {
        // A channel-wide PIM row transition cannot overlap normal commands
        // whose row-hit decision was made against the previous bank state.
        queue_to_schedule = QueueClass::PIM_Q;
    }
    else {
        queue_to_schedule = QueueClass::READ_Q;
    }

    const bool queue_has_work =
        (queue_to_schedule == QueueClass::READ_Q && !_read_queue[cid].empty()) ||
        (queue_to_schedule == QueueClass::WRITE_Q &&
         !_write_buffer[cid].empty() && !_pending_write_events[cid].empty()) ||
        (queue_to_schedule == QueueClass::PIM_Q && !_pim_queue[cid].empty());
    if (queue_has_work) {
        _last_transaction_schedule_cycle[cid] = current_cycle;
        switch (queue_to_schedule) {
            case QueueClass::READ_Q:
                issue_pending_read_event(cid);
                break;
            case QueueClass::WRITE_Q:
                issue_pending_write_event(cid);
                break;
            case QueueClass::PIM_Q:
                issue_pending_pim_event(cid);
                break;
            default:
                throw std::logic_error(
                    "Unknown EventDriven DRAM queue class on channel " +
                    std::to_string(cid));
        }
    }
    recompute_next_wakeup(cid, current_cycle);
}


void EventDrivenDram::issue_pending_read_event(uint32_t cid) {
    if (_read_queue[cid].empty()) {
        return;
    }
    auto read_event = _read_queue[cid].front();
    if (read_event->processed) {
        throw std::logic_error(
            "EventDriven DRAM attempted to process read event twice on channel " +
            std::to_string(cid) + ", address " +
            std::to_string(read_event->dram_address));
    }
    ED_TRACE_TRANSACTION(cid, _memsys->dram_channels[cid]->_dram_cycle,
                      "TO_ROW_STATE", read_event);
    _memsys->CheckRowSwitch(read_event);
    read_event->processed = true;
    _read_queue[cid].pop_front();
}


void EventDrivenDram::issue_pending_write_event(uint32_t cid) {

    if (_write_buffer[cid].empty() || _pending_write_events[cid].empty()) {
        _write_draining[cid] = 0;
        return;
    }

    if (!_write_buffer[cid].empty() && _write_draining[cid] > 0) {
        auto write_event = _write_buffer[cid].front();
        if (write_event->processed) {
            throw std::logic_error(
                "EventDriven DRAM attempted to process write event twice on channel " +
                std::to_string(cid) + ", address " +
                std::to_string(write_event->dram_address));
        }
        ED_TRACE_TRANSACTION(cid, _memsys->dram_channels[cid]->_dram_cycle,
                          "TO_ROW_STATE", write_event);
        _memsys->CheckRowSwitch(write_event);
        write_event->processed = true;
        _write_buffer[cid].pop_front();
        _write_draining[cid] -= 1; // Drain write buffer
    }
    if (_write_draining[cid] < 0) {
        _write_draining[cid] = 0;
    }
    if (_write_buffer[cid].empty()) {
        _write_draining[cid] = 0;
    }
}


void EventDrivenDram::issue_pending_pim_event(uint32_t cid) {
    if (_pending_pim_events[cid].empty()) {
        return;
    }
    if (!_pim_queue[cid].empty()) {
        auto pim_event = _pim_queue[cid].front();
        if (pim_event->processed) {
            throw std::logic_error(
                "EventDriven DRAM attempted to process PIM event twice on channel " +
                std::to_string(cid) + ", address " +
                std::to_string(pim_event->dram_address));
        }
        // The physical PIM command queue changes from empty/non-empty at this
        // point.  Settle the preceding interval using the old queue state;
        // otherwise the next event-driven jump retroactively labels that
        // whole interval as PIM work.
        accumulate_background_cycles(
            cid, _memsys->dram_channels[cid]->_dram_cycle);
        if (is_pim_normal_row_access(pim_event->req_type)) {  // COMP_HASH for Each Bank
            _memsys->CheckRowSwitch(pim_event);
        }
        else {  // For COMP for All Bank
            _memsys->CheckPIMRowSwitch(pim_event);
        }
        pim_event->processed = true;
        _pim_queue[cid].pop_front();
    }
}


bool EventDrivenDram::running() {
    for (int cid = 0; cid < MyAddressAllocator::dram_channels; cid++) {
        const auto *channel = _memsys->dram_channels[cid];
        const auto has_event = [](const std::shared_ptr<Event>& event) {
            return event != nullptr;
        };
#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
        const bool execute_has_work = channel->execute_queue.any_of(has_event);
#else
        const bool execute_has_work =
            std::any_of(channel->execute_queue.begin(), channel->execute_queue.end(), has_event);
#endif
        const bool activate_has_work =
            std::any_of(channel->activate_queue.begin(), channel->activate_queue.end(), has_event);
        const bool precharge_has_work =
            std::any_of(channel->precharge_queue.begin(), channel->precharge_queue.end(), has_event);
        const bool return_has_work =
            std::any_of(channel->return_queue.begin(),
                        channel->return_queue.end(), has_event);
        if (!_pending_read_events[cid].empty() ||
            !_pending_write_events[cid].empty() ||
            !_pending_pim_events[cid].empty() ||
            !response_event_queues_[cid].isEmpty() ||
            response_event_queues_[cid].NumReserved > 0 ||
            !_read_queue[cid].empty() ||
            !_write_buffer[cid].empty() ||
            !_pim_queue[cid].empty() ||
            _write_draining[cid] > 0 ||
            execute_has_work ||
            activate_has_work ||
            precharge_has_work ||
            return_has_work) {
            return true;
        }
    }
    return false;
}


void EventDrivenDram::accumulate_background_cycles(uint32_t cid, cycle_type until_cycle) {
    if (cid >= _stats.size() || until_cycle <= _stats_last_cycle[cid]) {
        return;
    }
    const uint64_t delta = until_cycle - _stats_last_cycle[cid];
    _stats[cid].num_cycles += delta;
    auto *channel = _memsys->dram_channels[cid];
    const auto is_queued_pim = [](const std::shared_ptr<Event>& event) {
        return event != nullptr && is_pim_access_type(event->req_type);
    };
#ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
    const bool execute_has_pim =
        channel->execute_queue.any_of(is_queued_pim);
#else
    const bool execute_has_pim =
        std::any_of(channel->execute_queue.begin(),
                    channel->execute_queue.end(), is_queued_pim);
#endif
    const bool activate_has_pim =
        std::any_of(channel->activate_queue.begin(),
                    channel->activate_queue.end(), is_queued_pim);
    const bool precharge_has_pim =
        std::any_of(channel->precharge_queue.begin(),
                    channel->precharge_queue.end(), is_queued_pim);
    // Cycle Accurate defines pim_cycles as cycles for which its PIM command
    // queue is non-empty.  A pending response is no longer queued work, so do
    // not extend PIM time through the command's completion latency.
    if (execute_has_pim || activate_has_pim || precharge_has_pim) {
        _stats[cid].pim_cycles += delta;
    }

    for (size_t rank_idx = 0; rank_idx < channel->dram_ranks.size(); ++rank_idx) {
        bool all_idle = true;
        bool pim_all_idle = true;
        auto *rank = channel->dram_ranks[rank_idx];
        for (auto *bankgroup : rank->dram_bankgroups) {
            for (auto *bank : bankgroup->dram_banks) {
                if (bank->open_row != -1) {
                    all_idle = false;
                }
                if (bank->pim_open_row != -1) {
                    pim_all_idle = false;
                }
            }
        }
        if (all_idle) {
            _stats[cid].all_bank_idle_cycles[rank_idx] += delta;
        } else {
            _stats[cid].rank_active_cycles[rank_idx] += delta;
        }
        if (pim_all_idle) {
            _stats[cid].pim_all_bank_idle_cycles[rank_idx] += delta;
        } else {
            _stats[cid].pim_rank_active_cycles[rank_idx] += delta;
        }
    }
    _stats_last_cycle[cid] = until_cycle;
}


void EventDrivenDram::record_execute_command(uint32_t cid, const std::shared_ptr<Event>& event) {
    if (cid >= _stats.size() || event == nullptr) {
        return;
    }
    switch (event->req_type) {
        case MemoryAccessType::READ:
            _stats[cid].num_read_cmds++;
            if (!event->activated_for_access) {
                _stats[cid].num_read_row_hits++;
            }
            break;
        case MemoryAccessType::WRITE:
            _stats[cid].num_write_cmds++;
            if (!event->activated_for_access) {
                _stats[cid].num_write_row_hits++;
            }
            break;
        case MemoryAccessType::P_HEADER:
            _stats[cid].num_pheader_cmds++;
            _stats[cid].num_pim_cmds++;
            break;
        case MemoryAccessType::GWRITE:
            _stats[cid].num_gwrite_cmds++;
            _stats[cid].num_pim_cmds++;
            break;
        case MemoryAccessType::COMP:
        case MemoryAccessType::COMP_HASH:
            _stats[cid].num_comp_cmds++;
            _stats[cid].num_pim_cmds++;
            break;
        case MemoryAccessType::READRES:
        case MemoryAccessType::COMPS_READRES:
            _stats[cid].num_readres_cmds++;
            _stats[cid].num_pim_cmds++;
            break;
        default:
            break;
    }
}

void EventDrivenDram::record_completed_request(uint32_t cid, MemoryAccessType req_type) {
    if (cid >= _stats.size()) {
        return;
    }
    if (req_type == MemoryAccessType::READ) {
        _stats[cid].num_reads_done++;
    } else if (req_type == MemoryAccessType::WRITE) {
        _stats[cid].num_writes_done++;
    } else {
        _stats[cid].num_pim_done++;
    }
}

void EventDrivenDram::write_event_driven_stats(bool print_to_log) const {
    const std::string output_prefix = Config::system_config.log_dir + "/eventdrivendram";
    std::ofstream json_out(output_prefix + ".json", std::ofstream::out);
    std::ofstream txt_out(output_prefix + ".txt", std::ofstream::out);
    if (!json_out.is_open() || !txt_out.is_open()) {
        throw std::runtime_error(
            "EventDriven DRAM statistics output path is not writable: " +
            output_prefix);
    }

    json_out << "{";
    for (size_t cid = 0; cid < _stats.size(); ++cid) {
        const auto &s = _stats[cid];
        const uint64_t logical_read_cmds = s.num_read_cmds + s.estimated_read_cmds;
        const uint64_t logical_write_cmds = s.num_write_cmds + s.estimated_write_cmds;
        const uint64_t logical_reads_done =
            s.num_logical_reads_done + s.estimated_reads_done;
        const uint64_t logical_writes_done =
            s.num_writes_done + s.estimated_writes_done;
        const uint64_t logical_pim_done =
            s.num_pim_done + s.estimated_pim_done;
        const uint64_t logical_act_cmds = s.num_act_cmds + s.estimated_act_cmds;
        const uint64_t logical_pre_cmds = s.num_pre_cmds + s.estimated_pre_cmds;
        const uint64_t logical_write_buf_hits =
            s.num_write_buf_hits + s.estimated_write_buf_hits;
        const uint64_t logical_read_row_hits =
            s.num_read_row_hits + s.estimated_read_row_hits;
        const uint64_t logical_write_row_hits =
            s.num_write_row_hits + s.estimated_write_row_hits;
        const uint64_t logical_ondemand_pres =
            s.num_ondemand_pres + s.estimated_ondemand_pres;
        const uint64_t logical_gwrite_cmds = s.num_gwrite_cmds + s.estimated_gwrite_cmds;
        const uint64_t logical_comp_cmds = s.num_comp_cmds + s.estimated_comp_cmds;
        const uint64_t logical_readres_cmds = s.num_readres_cmds + s.estimated_readres_cmds;
        const uint64_t logical_pim_cmds =
            s.num_pim_cmds + s.estimated_pim_cmds;
        const uint64_t logical_pim_activate_cmds =
            s.num_pim_activate_cmds + s.estimated_pim_activate_cmds;
        const uint64_t logical_pim_precharge_cmds =
            s.num_pim_precharge_cmds + s.estimated_pim_precharge_cmds;
        // EventDriven advances each channel on the absolute simulator
        // timeline when the next event is processed. Consequently cycles,
        // refresh commands, and rank-state cycles already include the clock
        // jump made for a pruned region. The estimated time counters are
        // diagnostics only; adding them here would compensate the same
        // interval twice.
        const uint64_t logical_ref_cmds = s.num_ref_cmds;
        const uint64_t logical_refb_cmds = s.num_refb_cmds;
        const uint64_t logical_cycles = s.num_cycles;
        const double avg_memory_bandwidth = s.num_cycles == 0
            ? 0.0
            : static_cast<double>(
                static_cast<long double>(
                    logical_reads_done + logical_writes_done) *
                dramsim3_config_->request_size_bytes /
                (static_cast<long double>(s.num_cycles) *
                 dramsim3_config_->tCK));
        const double act_energy = logical_act_cmds * dramsim3_config_->act_energy_inc;
        const double read_energy = logical_read_cmds * dramsim3_config_->read_energy_inc;
        const double write_energy = logical_write_cmds * dramsim3_config_->write_energy_inc;
        const double ref_energy = logical_ref_cmds * dramsim3_config_->ref_energy_inc;
        const double refb_energy = logical_refb_cmds * dramsim3_config_->refb_energy_inc;
        const double gwrite_energy = logical_gwrite_cmds * dramsim3_config_->gwrite_energy_inc;
        const double comp_energy = logical_comp_cmds * dramsim3_config_->comp_energy_inc;
        const double readres_energy = logical_readres_cmds * dramsim3_config_->readres_energy_inc;

        std::vector<uint64_t> logical_rank_active_cycles(
            s.rank_active_cycles.size(), 0);
        std::vector<uint64_t> logical_all_bank_idle_cycles(
            s.all_bank_idle_cycles.size(), 0);
        std::vector<uint64_t> logical_pim_rank_active_cycles(
            s.pim_rank_active_cycles.size(), 0);
        std::vector<uint64_t> logical_pim_all_bank_idle_cycles(
            s.pim_all_bank_idle_cycles.size(), 0);
        double background_energy = 0.0;
        for (size_t rank = 0; rank < s.all_bank_idle_cycles.size(); ++rank) {
            const uint64_t estimated_active =
                s.estimated_rank_active_cycles[rank];
            const uint64_t estimated_idle =
                s.estimated_all_bank_idle_cycles[rank];
            const uint64_t estimated_state_cycles =
                estimated_active + estimated_idle;

            // The absolute ED clock jump is already included in num_cycles.
            // With no events in the skipped interval, that jump is charged to
            // the currently observed rank state (normally all-bank-idle).
            // Replace those recorded state cycles with the sampled active/idle
            // distribution instead of adding another copy of the interval.
            const uint64_t replace_idle = std::min<uint64_t>(
                estimated_state_cycles, s.all_bank_idle_cycles[rank]);
            const uint64_t replace_active = std::min<uint64_t>(
                estimated_state_cycles - replace_idle,
                s.rank_active_cycles[rank]);
            logical_rank_active_cycles[rank] =
                s.rank_active_cycles[rank] - replace_active +
                estimated_active;
            logical_all_bank_idle_cycles[rank] =
                s.all_bank_idle_cycles[rank] - replace_idle +
                estimated_idle;

            background_energy +=
                logical_rank_active_cycles[rank] *
                dramsim3_config_->act_stb_energy_inc;
            background_energy +=
                logical_all_bank_idle_cycles[rank] *
                dramsim3_config_->pre_stb_energy_inc;

            const uint64_t estimated_pim_active =
                s.estimated_pim_rank_active_cycles[rank];
            const uint64_t estimated_pim_idle =
                s.estimated_pim_all_bank_idle_cycles[rank];
            const uint64_t estimated_pim_state_cycles =
                estimated_pim_active + estimated_pim_idle;
            const uint64_t replace_pim_idle = std::min<uint64_t>(
                estimated_pim_state_cycles,
                s.pim_all_bank_idle_cycles[rank]);
            const uint64_t replace_pim_active = std::min<uint64_t>(
                estimated_pim_state_cycles - replace_pim_idle,
                s.pim_rank_active_cycles[rank]);
            logical_pim_rank_active_cycles[rank] =
                s.pim_rank_active_cycles[rank] - replace_pim_active +
                estimated_pim_active;
            logical_pim_all_bank_idle_cycles[rank] =
                s.pim_all_bank_idle_cycles[rank] - replace_pim_idle +
                estimated_pim_idle;
        }

        const double logical_time =
            static_cast<double>(logical_cycles) * dramsim3_config_->tCK;
        const double pim_command_time =
            dramsim3_config_->burst_cycle * dramsim3_config_->tCK;
        const double pu_idle_time = std::max(
            0.0, logical_time -
                logical_comp_cmds * pim_command_time);
        const double pim_buffer_idle_time = std::max(
            0.0, logical_time -
                logical_pim_cmds * pim_command_time);
        const double pim_background_energy =
            pu_idle_time * dramsim3_config_->pu_static_energy_inc +
            pim_buffer_idle_time *
                dramsim3_config_->pim_buffer_static_energy_inc;
        const double pim_dynamic_energy =
            gwrite_energy + comp_energy + readres_energy;
        const double total_pim_energy =
            pim_background_energy + pim_dynamic_energy;

        const double total_energy = act_energy + read_energy + write_energy +
            ref_energy + refb_energy + background_energy + total_pim_energy;
        const double average_power = logical_cycles == 0 ? 0.0 : total_energy / logical_cycles;

        if (cid != 0) {
            json_out << ",\n";
        }
        json_out << "\"" << cid << "\":{";
        json_out << "\"channel\":" << cid;
        json_out << ",\"num_cycles\":" << s.num_cycles;
        json_out << ",\"estimated_cycles\":" << s.estimated_cycles;
        json_out << ",\"logical_cycles\":" << logical_cycles;
        json_out << ",\"num_reads_done\":" << logical_reads_done;
        json_out << ",\"num_writes_done\":" << logical_writes_done;
        json_out << ",\"measured_num_reads_done\":" << s.num_logical_reads_done;
        json_out << ",\"measured_num_writes_done\":" << s.num_writes_done;
        json_out << ",\"physical_num_reads_done\":" << s.num_reads_done;
        json_out << ",\"estimated_num_reads_done\":" << s.estimated_reads_done;
        json_out << ",\"estimated_num_writes_done\":" << s.estimated_writes_done;
        json_out << ",\"num_pim_done\":" << s.num_pim_done;
        json_out << ",\"measured_num_pim_done\":" << s.num_pim_done;
        json_out << ",\"estimated_num_pim_done\":"
                 << s.estimated_pim_done;
        json_out << ",\"logical_num_pim_done\":"
                 << logical_pim_done;
        json_out << ",\"num_logical_reads_done\":" << logical_reads_done;
        json_out << ",\"num_read_merges\":" << s.num_read_merges;
        json_out << ",\"num_write_merges\":" << s.num_write_merges;
        json_out << ",\"num_read_cmds\":" << s.num_read_cmds;
        json_out << ",\"num_write_cmds\":" << s.num_write_cmds;
        json_out << ",\"num_act_cmds\":" << s.num_act_cmds;
        json_out << ",\"num_pre_cmds\":" << s.num_pre_cmds;
        json_out << ",\"num_write_buf_hits\":" << logical_write_buf_hits;
        json_out << ",\"num_read_row_hits\":" << logical_read_row_hits;
        json_out << ",\"num_write_row_hits\":" << logical_write_row_hits;
        json_out << ",\"num_ondemand_pres\":" << logical_ondemand_pres;
        json_out << ",\"measured_num_write_buf_hits\":" << s.num_write_buf_hits;
        json_out << ",\"measured_num_read_row_hits\":" << s.num_read_row_hits;
        json_out << ",\"measured_num_write_row_hits\":" << s.num_write_row_hits;
        json_out << ",\"measured_num_ondemand_pres\":" << s.num_ondemand_pres;
        json_out << ",\"num_ref_cmds\":" << s.num_ref_cmds;
        json_out << ",\"num_refb_cmds\":" << s.num_refb_cmds;
        json_out << ",\"num_pheader_cmds\":" << s.num_pheader_cmds;
        json_out << ",\"num_gwrite_cmds\":" << s.num_gwrite_cmds;
        json_out << ",\"num_comp_cmds\":" << s.num_comp_cmds;
        json_out << ",\"num_readres_cmds\":" << s.num_readres_cmds;
        json_out << ",\"num_pim_cmds\":" << s.num_pim_cmds;
        json_out << ",\"num_pim_activate_cmds\":"
                 << s.num_pim_activate_cmds;
        json_out << ",\"num_pim_precharge_cmds\":"
                 << s.num_pim_precharge_cmds;
        json_out << ",\"estimated_read_cmds\":" << s.estimated_read_cmds;
        json_out << ",\"estimated_write_cmds\":" << s.estimated_write_cmds;
        json_out << ",\"estimated_act_cmds\":" << s.estimated_act_cmds;
        json_out << ",\"estimated_pre_cmds\":" << s.estimated_pre_cmds;
        json_out << ",\"estimated_num_write_buf_hits\":"
                 << s.estimated_write_buf_hits;
        json_out << ",\"estimated_num_read_row_hits\":"
                 << s.estimated_read_row_hits;
        json_out << ",\"estimated_num_write_row_hits\":"
                 << s.estimated_write_row_hits;
        json_out << ",\"estimated_num_ondemand_pres\":"
                 << s.estimated_ondemand_pres;
        json_out << ",\"estimated_pheader_cmds\":" << s.estimated_pheader_cmds;
        json_out << ",\"estimated_gwrite_cmds\":" << s.estimated_gwrite_cmds;
        json_out << ",\"estimated_comp_cmds\":" << s.estimated_comp_cmds;
        json_out << ",\"estimated_readres_cmds\":" << s.estimated_readres_cmds;
        json_out << ",\"estimated_pim_cmds\":" << s.estimated_pim_cmds;
        json_out << ",\"estimated_pim_activate_cmds\":"
                 << s.estimated_pim_activate_cmds;
        json_out << ",\"estimated_pim_precharge_cmds\":"
                 << s.estimated_pim_precharge_cmds;
        json_out << ",\"pim_cycles\":" << s.pim_cycles;
        json_out << ",\"estimated_pim_cycles\":"
                 << s.estimated_pim_cycles;
        json_out << ",\"logical_pim_cycles\":"
                 << s.pim_cycles + s.estimated_pim_cycles;
        json_out << ",\"logical_read_cmds\":" << logical_read_cmds;
        json_out << ",\"logical_write_cmds\":" << logical_write_cmds;
        json_out << ",\"logical_act_cmds\":" << logical_act_cmds;
        json_out << ",\"logical_pre_cmds\":" << logical_pre_cmds;
        json_out << ",\"logical_pheader_cmds\":"
                 << s.num_pheader_cmds + s.estimated_pheader_cmds;
        json_out << ",\"logical_gwrite_cmds\":" << logical_gwrite_cmds;
        json_out << ",\"logical_comp_cmds\":" << logical_comp_cmds;
        json_out << ",\"logical_readres_cmds\":" << logical_readres_cmds;
        json_out << ",\"logical_pim_cmds\":"
                 << s.num_pim_cmds + s.estimated_pim_cmds;
        json_out << ",\"logical_pim_activate_cmds\":"
                 << logical_pim_activate_cmds;
        json_out << ",\"logical_pim_precharge_cmds\":"
                 << logical_pim_precharge_cmds;
        json_out << ",\"estimated_ref_cmds\":" << s.estimated_ref_cmds;
        json_out << ",\"estimated_refb_cmds\":" << s.estimated_refb_cmds;
        json_out << ",\"logical_ref_cmds\":" << logical_ref_cmds;
        json_out << ",\"logical_refb_cmds\":" << logical_refb_cmds;
        json_out << ",\"num_read_requests\":" << s.num_read_requests;
        json_out << ",\"num_write_requests\":" << s.num_write_requests;
        json_out << ",\"num_pim_requests\":" << s.num_pim_requests;
        json_out << ",\"estimated_read_requests\":" << s.estimated_read_requests;
        json_out << ",\"estimated_write_requests\":" << s.estimated_write_requests;
        json_out << ",\"estimated_pim_requests\":" << s.estimated_pim_requests;
        json_out << ",\"logical_read_requests\":"
                 << s.num_read_requests + s.estimated_read_requests;
        json_out << ",\"logical_write_requests\":"
                 << s.num_write_requests + s.estimated_write_requests;
        json_out << ",\"logical_pim_requests\":"
                 << s.num_pim_requests + s.estimated_pim_requests;
        json_out << ",\"act_energy\":" << act_energy;
        json_out << ",\"read_energy\":" << read_energy;
        json_out << ",\"write_energy\":" << write_energy;
        json_out << ",\"ref_energy\":" << ref_energy;
        json_out << ",\"refb_energy\":" << refb_energy;
        json_out << ",\"gwrite_energy\":" << gwrite_energy;
        json_out << ",\"comp_energy\":" << comp_energy;
        json_out << ",\"readres_energy\":" << readres_energy;
        json_out << ",\"background_energy\":" << background_energy;
        json_out << ",\"pim_background_energy\":" << pim_background_energy;
        json_out << ",\"pim_dynamic_energy\":" << pim_dynamic_energy;
        json_out << ",\"total_pim_energy\":" << total_pim_energy;
        json_out << ",\"total_energy\":" << total_energy;
        json_out << ",\"average_power\":" << average_power;
        json_out << ",\"avg_memory_bandwidth\":" << avg_memory_bandwidth;
        json_out << ",\"all_bank_idle_cycles\":{";
        for (size_t rank = 0; rank < s.all_bank_idle_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":" << s.all_bank_idle_cycles[rank];
        }
        json_out << "},\"rank_active_cycles\":{";
        for (size_t rank = 0; rank < s.rank_active_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":" << s.rank_active_cycles[rank];
        }
        json_out << "},\"logical_all_bank_idle_cycles\":{";
        for (size_t rank = 0; rank < logical_all_bank_idle_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << logical_all_bank_idle_cycles[rank];
        }
        json_out << "},\"logical_rank_active_cycles\":{";
        for (size_t rank = 0; rank < logical_rank_active_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << logical_rank_active_cycles[rank];
        }
        json_out << "},\"estimated_all_bank_idle_cycles\":{";
        for (size_t rank = 0;
             rank < s.estimated_all_bank_idle_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << s.estimated_all_bank_idle_cycles[rank];
        }
        json_out << "},\"estimated_rank_active_cycles\":{";
        for (size_t rank = 0;
             rank < s.estimated_rank_active_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << s.estimated_rank_active_cycles[rank];
        }
        json_out << "},\"pim_all_bank_idle_cycles\":{";
        for (size_t rank = 0; rank < s.pim_all_bank_idle_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << s.pim_all_bank_idle_cycles[rank];
        }
        json_out << "},\"pim_rank_active_cycles\":{";
        for (size_t rank = 0; rank < s.pim_rank_active_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << s.pim_rank_active_cycles[rank];
        }
        json_out << "},\"logical_pim_all_bank_idle_cycles\":{";
        for (size_t rank = 0;
             rank < logical_pim_all_bank_idle_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << logical_pim_all_bank_idle_cycles[rank];
        }
        json_out << "},\"logical_pim_rank_active_cycles\":{";
        for (size_t rank = 0;
             rank < logical_pim_rank_active_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << logical_pim_rank_active_cycles[rank];
        }
        json_out << "},\"estimated_pim_all_bank_idle_cycles\":{";
        for (size_t rank = 0;
             rank < s.estimated_pim_all_bank_idle_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << s.estimated_pim_all_bank_idle_cycles[rank];
        }
        json_out << "},\"estimated_pim_rank_active_cycles\":{";
        for (size_t rank = 0;
             rank < s.estimated_pim_rank_active_cycles.size(); ++rank) {
            if (rank != 0) json_out << ",";
            json_out << "\"" << rank << "\":"
                     << s.estimated_pim_rank_active_cycles[rank];
        }
        json_out << "}}";

        txt_out << "###########################################\n";
        txt_out << "## EventDriven Statistics of Channel " << cid << "\n";
        txt_out << "###########################################\n";
        txt_out << "num_cycles = " << s.num_cycles << "\n";
        txt_out << "estimated_cycles = " << s.estimated_cycles << "\n";
        txt_out << "logical_cycles = " << logical_cycles << "\n";
        txt_out << "pim_cycles = " << s.pim_cycles << "\n";
        txt_out << "estimated_pim_cycles = " << s.estimated_pim_cycles << "\n";
        txt_out << "logical_pim_cycles = "
                << s.pim_cycles + s.estimated_pim_cycles << "\n";
        txt_out << "num_reads_done = " << logical_reads_done << "\n";
        txt_out << "num_writes_done = " << logical_writes_done << "\n";
        txt_out << "measured_num_reads_done = " << s.num_logical_reads_done << "\n";
        txt_out << "measured_num_writes_done = " << s.num_writes_done << "\n";
        txt_out << "physical_num_reads_done = " << s.num_reads_done << "\n";
        txt_out << "estimated_num_reads_done = " << s.estimated_reads_done << "\n";
        txt_out << "estimated_num_writes_done = " << s.estimated_writes_done << "\n";
        txt_out << "num_pim_done = " << s.num_pim_done << "\n";
        txt_out << "measured_num_pim_done = "
                << s.num_pim_done << "\n";
        txt_out << "estimated_num_pim_done = "
                << s.estimated_pim_done << "\n";
        txt_out << "logical_num_pim_done = "
                << logical_pim_done << "\n";
        txt_out << "num_logical_reads_done = " << logical_reads_done << "\n";
        txt_out << "num_read_merges = " << s.num_read_merges << "\n";
        txt_out << "num_write_merges = " << s.num_write_merges << "\n";
        txt_out << "num_read_cmds = " << s.num_read_cmds << "\n";
        txt_out << "num_write_cmds = " << s.num_write_cmds << "\n";
        txt_out << "num_act_cmds = " << s.num_act_cmds << "\n";
        txt_out << "num_pre_cmds = " << s.num_pre_cmds << "\n";
        txt_out << "num_write_buf_hits = " << logical_write_buf_hits << "\n";
        txt_out << "num_read_row_hits = " << logical_read_row_hits << "\n";
        txt_out << "num_write_row_hits = " << logical_write_row_hits << "\n";
        txt_out << "num_ondemand_pres = " << logical_ondemand_pres << "\n";
        txt_out << "measured_num_write_buf_hits = " << s.num_write_buf_hits << "\n";
        txt_out << "measured_num_read_row_hits = " << s.num_read_row_hits << "\n";
        txt_out << "measured_num_write_row_hits = " << s.num_write_row_hits << "\n";
        txt_out << "measured_num_ondemand_pres = " << s.num_ondemand_pres << "\n";
        txt_out << "num_ref_cmds = " << s.num_ref_cmds << "\n";
        txt_out << "num_refb_cmds = " << s.num_refb_cmds << "\n";
        txt_out << "num_pheader_cmds = " << s.num_pheader_cmds << "\n";
        txt_out << "num_gwrite_cmds = " << s.num_gwrite_cmds << "\n";
        txt_out << "num_comp_cmds = " << s.num_comp_cmds << "\n";
        txt_out << "num_readres_cmds = " << s.num_readres_cmds << "\n";
        txt_out << "num_pim_cmds = " << s.num_pim_cmds << "\n";
        txt_out << "num_pim_activate_cmds = "
                << s.num_pim_activate_cmds << "\n";
        txt_out << "num_pim_precharge_cmds = "
                << s.num_pim_precharge_cmds << "\n";
        txt_out << "estimated_read_cmds = " << s.estimated_read_cmds << "\n";
        txt_out << "estimated_write_cmds = " << s.estimated_write_cmds << "\n";
        txt_out << "estimated_act_cmds = " << s.estimated_act_cmds << "\n";
        txt_out << "estimated_pre_cmds = " << s.estimated_pre_cmds << "\n";
        txt_out << "estimated_num_write_buf_hits = "
                << s.estimated_write_buf_hits << "\n";
        txt_out << "estimated_num_read_row_hits = "
                << s.estimated_read_row_hits << "\n";
        txt_out << "estimated_num_write_row_hits = "
                << s.estimated_write_row_hits << "\n";
        txt_out << "estimated_num_ondemand_pres = "
                << s.estimated_ondemand_pres << "\n";
        txt_out << "logical_read_cmds = " << logical_read_cmds << "\n";
        txt_out << "logical_write_cmds = " << logical_write_cmds << "\n";
        txt_out << "logical_act_cmds = " << logical_act_cmds << "\n";
        txt_out << "logical_pre_cmds = " << logical_pre_cmds << "\n";
        txt_out << "logical_pim_activate_cmds = "
                << logical_pim_activate_cmds << "\n";
        txt_out << "logical_pim_precharge_cmds = "
                << logical_pim_precharge_cmds << "\n";
        txt_out << "num_read_requests = " << s.num_read_requests << "\n";
        txt_out << "num_write_requests = " << s.num_write_requests << "\n";
        txt_out << "num_pim_requests = " << s.num_pim_requests << "\n";
        txt_out << "estimated_read_requests = " << s.estimated_read_requests << "\n";
        txt_out << "estimated_write_requests = " << s.estimated_write_requests << "\n";
        txt_out << "estimated_pim_requests = " << s.estimated_pim_requests << "\n";
        txt_out << "logical_read_requests = "
                << s.num_read_requests + s.estimated_read_requests << "\n";
        txt_out << "logical_write_requests = "
                << s.num_write_requests + s.estimated_write_requests << "\n";
        txt_out << "logical_pim_requests = "
                << s.num_pim_requests + s.estimated_pim_requests << "\n";
        txt_out << "act_energy = " << act_energy << "\n";
        txt_out << "read_energy = " << read_energy << "\n";
        txt_out << "write_energy = " << write_energy << "\n";
        txt_out << "ref_energy = " << ref_energy << "\n";
        txt_out << "refb_energy = " << refb_energy << "\n";
        txt_out << "gwrite_energy = " << gwrite_energy << "\n";
        txt_out << "comp_energy = " << comp_energy << "\n";
        txt_out << "readres_energy = " << readres_energy << "\n";
        txt_out << "background_energy = " << background_energy << "\n";
        txt_out << "pim_background_energy = " << pim_background_energy << "\n";
        txt_out << "pim_dynamic_energy = " << pim_dynamic_energy << "\n";
        txt_out << "total_pim_energy = " << total_pim_energy << "\n";
        txt_out << "total_energy = " << total_energy << "\n";
        txt_out << "average_power = " << average_power << "\n";
        txt_out << "avg_memory_bandwidth = " << avg_memory_bandwidth << "\n";

        if (print_to_log) {
            spdlog::info("###########################################");
            spdlog::info("## EventDriven Statistics of Channel {}", cid);
            spdlog::info("###########################################");
            spdlog::info("num_cycles = {}", s.num_cycles);
            spdlog::info("logical_cycles = {} (estimated {})", logical_cycles,
                         s.estimated_cycles);
            spdlog::info(
                "num_reads_done = {} (measured logical {}, physical {}, estimated {})",
                logical_reads_done, s.num_logical_reads_done,
                s.num_reads_done, s.estimated_reads_done);
            spdlog::info(
                "num_writes_done = {} (measured {}, estimated {})",
                logical_writes_done, s.num_writes_done,
                s.estimated_writes_done);
            spdlog::info(
                "num_pim_done = {} (measured {}, estimated {})",
                logical_pim_done, s.num_pim_done,
                s.estimated_pim_done);
            spdlog::info("num_logical_reads_done = {}", logical_reads_done);
            spdlog::info("num_read_merges = {}", s.num_read_merges);
            spdlog::info("num_write_merges = {}", s.num_write_merges);
            spdlog::info("num_read_cmds = {}", s.num_read_cmds);
            spdlog::info("num_write_cmds = {}", s.num_write_cmds);
            spdlog::info("num_act_cmds = {}", s.num_act_cmds);
            spdlog::info("num_pre_cmds = {}", s.num_pre_cmds);
            spdlog::info(
                "logical write-buffer/read-row/write-row/on-demand-PRE = {}/{}/{}/{} (estimated {}/{}/{}/{})",
                logical_write_buf_hits, logical_read_row_hits,
                logical_write_row_hits, logical_ondemand_pres,
                s.estimated_write_buf_hits, s.estimated_read_row_hits,
                s.estimated_write_row_hits, s.estimated_ondemand_pres);
            spdlog::info("num_ref_cmds = {}", s.num_ref_cmds);
            spdlog::info("num_refb_cmds = {}", s.num_refb_cmds);
            spdlog::info("num_pheader_cmds = {}", s.num_pheader_cmds);
            spdlog::info("num_gwrite_cmds = {}", s.num_gwrite_cmds);
            spdlog::info("num_comp_cmds = {}", s.num_comp_cmds);
            spdlog::info("num_readres_cmds = {}", s.num_readres_cmds);
            spdlog::info("num_pim_cmds = {}", s.num_pim_cmds);
            spdlog::info(
                "PIM_ACTIVE/PIM_PRECHARGE commands = {}/{} "
                "(logical {}/{})",
                s.num_pim_activate_cmds, s.num_pim_precharge_cmds,
                logical_pim_activate_cmds,
                logical_pim_precharge_cmds);
            spdlog::info(
                "logical commands read/write/act/pre = {}/{}/{}/{} (estimated {}/{}/{}/{})",
                logical_read_cmds, logical_write_cmds, logical_act_cmds,
                logical_pre_cmds, s.estimated_read_cmds,
                s.estimated_write_cmds, s.estimated_act_cmds,
                s.estimated_pre_cmds);
            spdlog::info(
                "logical requests read/write/pim = {}/{}/{} (estimated {}/{}/{})",
                s.num_read_requests + s.estimated_read_requests,
                s.num_write_requests + s.estimated_write_requests,
                s.num_pim_requests + s.estimated_pim_requests,
                s.estimated_read_requests, s.estimated_write_requests,
                s.estimated_pim_requests);
            spdlog::info("act_energy = {}", act_energy);
            spdlog::info("read_energy = {}", read_energy);
            spdlog::info("write_energy = {}", write_energy);
            spdlog::info("ref_energy = {}", ref_energy);
            spdlog::info("refb_energy = {}", refb_energy);
            spdlog::info("gwrite_energy = {}", gwrite_energy);
            spdlog::info("comp_energy = {}", comp_energy);
            spdlog::info("readres_energy = {}", readres_energy);
            spdlog::info("background_energy = {}", background_energy);
            spdlog::info("pim_background_energy = {}",
                         pim_background_energy);
            spdlog::info("pim_dynamic_energy = {}", pim_dynamic_energy);
            spdlog::info("total_pim_energy = {}", total_pim_energy);
            spdlog::info("total_energy = {}", total_energy);
            spdlog::info("average_power = {}", average_power);
            spdlog::info("avg_memory_bandwidth = {}", avg_memory_bandwidth);
        }
    }
    json_out << "}";
    if (print_to_log) {
        spdlog::info("EventDriven DRAM stats written to {}.txt and {}.json", output_prefix, output_prefix);
    }
}

void EventDrivenDram::print_stat() {
    for (uint32_t cid = 0; cid < _stats.size(); ++cid) {
        accumulate_background_cycles(cid, _memsys->dram_channels[cid]->_dram_cycle);
    }
    write_event_driven_stats(true);  // TODO::

    const auto &mem_config = _memsys->config_;
    const int num_channels = mem_config.channels;
    const int pu_num = mem_config.PU_num;
    const int total_pus = num_channels * pu_num;

    const double total_buffer_size_kb =
        (num_channels * mem_config.input_buffer_size +
         num_channels * mem_config.output_buffer_size * pu_num) / 1024.0;
    const double total_pu_area = total_pus * mem_config.pim_pu_area;
    const double total_ctrl_area = total_pus * mem_config.pim_controller_area_overhead;
    const double total_buffer_area = total_buffer_size_kb * mem_config.pim_buffer_area_per_kb;

    const double bus_width_bytes = mem_config.bus_width / 8.0;
    const double bandwidth_per_channel_gbs = bus_width_bytes * (2.0 / mem_config.tCK);
    const double total_bandwidth_gbs = bandwidth_per_channel_gbs * num_channels;
    const double total_hybrid_bonding_area =
        total_bandwidth_gbs * mem_config.hybrid_bonding_bw_area_ratio;
    const double total_area =
        total_pu_area + total_ctrl_area + total_buffer_area + total_hybrid_bonding_area;

    spdlog::info("PIM Area Overhead Analysis:");
    spdlog::info("PIM Structure: {} channel, {} pus per channel, with {}KB global input buffer and {}KB output buffer",
        num_channels, pu_num, mem_config.input_buffer_size / 1024.0, mem_config.output_buffer_size / 1024.0);
    spdlog::info("Total PIM Units: {}", total_pus);
    spdlog::info("Total PU Area: {:.4f} mm2", total_pu_area);
    spdlog::info("Total Controller Area: {:.4f} mm2", total_ctrl_area);
    spdlog::info("Total Buffer Area: {:.4f} mm2", total_buffer_area);
    spdlog::info("Total Hybrid Bonding Area: {:.4f} mm2", total_hybrid_bonding_area);
    spdlog::info("Total Area Overhead: {:.4f} mm2", total_area);
}

void EventDrivenDram::apply_estimated_workload(
    const ProportionalWorkloadStat& workload,
    const std::vector<uint64_t>* write_command_override) {
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        const auto channel_value = [channel](const std::vector<uint64_t>& values) {
            return channel < values.size() ? values[channel] : 0ULL;
        };
        _stats[channel].estimated_read_requests +=
            channel_value(workload.channel_memory_reads);
        _stats[channel].estimated_write_requests +=
            channel_value(workload.channel_memory_writes);
        _stats[channel].estimated_reads_done +=
            channel_value(workload.channel_memory_reads);
        _stats[channel].estimated_writes_done +=
            channel_value(workload.channel_memory_writes);
        _stats[channel].estimated_pim_requests +=
            channel_value(workload.channel_pim_pheader) +
            channel_value(workload.channel_pim_gwrite) +
            channel_value(workload.channel_pim_comp) +
            channel_value(workload.channel_pim_readres);
        _stats[channel].estimated_pim_done +=
            channel_value(workload.channel_pim_pheader) +
            channel_value(workload.channel_pim_gwrite) +
            channel_value(workload.channel_pim_comp) +
            channel_value(workload.channel_pim_readres);

        const auto& base = _command_sampling_baseline[channel];
        auto& stat = _stats[channel];
        const uint64_t sampled_reads = stat.num_read_requests - base.num_read_requests;
        const uint64_t sampled_writes = stat.num_write_requests - base.num_write_requests;
        const uint64_t sampled_pim = stat.num_pim_requests - base.num_pim_requests;
        const uint64_t sampled_total = sampled_reads + sampled_writes + sampled_pim;
        const uint64_t skipped_reads = channel_value(workload.channel_memory_reads);
        const uint64_t skipped_writes = channel_value(workload.channel_memory_writes);
        const uint64_t skipped_pheader = channel_value(workload.channel_pim_pheader);
        const uint64_t skipped_gwrite = channel_value(workload.channel_pim_gwrite);
        const uint64_t skipped_comp = channel_value(workload.channel_pim_comp);
        const uint64_t skipped_readres = channel_value(workload.channel_pim_readres);
        const uint64_t skipped_pim = skipped_pheader + skipped_gwrite +
                                     skipped_comp + skipped_readres;
        const uint64_t skipped_total = skipped_reads + skipped_writes + skipped_pim;
        const auto scale = [](uint64_t measured, uint64_t skipped,
                              uint64_t sampled) -> uint64_t {
            if (measured == 0 || skipped == 0 || sampled == 0) return 0;
            return static_cast<uint64_t>(std::llround(
                static_cast<long double>(measured) * skipped / sampled));
        };
        const auto weighted_scale = [&](uint64_t warmup_commands,
                                        uint64_t warmup_requests,
                                        uint64_t sample_commands,
                                        uint64_t sample_requests,
                                        uint64_t skipped_requests) -> uint64_t {
            if (!_command_has_warmup_sample) {
                return scale(sample_commands, skipped_requests, sample_requests);
            }
            long double weighted_rate = 0.0L;
            long double available_weight = 0.0L;
            if (warmup_requests > 0 && _command_warmup_weight > 0.0) {
                weighted_rate += static_cast<long double>(_command_warmup_weight) *
                    warmup_commands / warmup_requests;
                available_weight += _command_warmup_weight;
            }
            const double sample_weight = 1.0 - _command_warmup_weight;
            if (sample_requests > 0 && sample_weight > 0.0) {
                weighted_rate += static_cast<long double>(sample_weight) *
                    sample_commands / sample_requests;
                available_weight += sample_weight;
            }
            if (skipped_requests == 0 || available_weight == 0.0L) return 0;
            return static_cast<uint64_t>(std::llround(
                weighted_rate * skipped_requests / available_weight));
        };
        const auto combined_scale = [](uint64_t warmup_commands,
                                       uint64_t warmup_requests,
                                       uint64_t sample_commands,
                                       uint64_t sample_requests,
                                       uint64_t skipped_requests) -> uint64_t {
            const uint64_t combined_requests =
                warmup_requests + sample_requests;
            if (skipped_requests == 0 || combined_requests == 0) return 0;
            return static_cast<uint64_t>(std::llround(
                static_cast<long double>(warmup_commands + sample_commands) *
                skipped_requests / combined_requests));
        };
        uint64_t warmup_reads = 0;
        uint64_t warmup_writes = 0;
        uint64_t warmup_pim = 0;
        uint64_t warmup_total = 0;
        uint64_t warmup_read_cmds = 0;
        uint64_t warmup_write_cmds = 0;
        uint64_t warmup_act_cmds = 0;
        uint64_t warmup_pre_cmds = 0;
        uint64_t warmup_write_buf_hits = 0;
        uint64_t warmup_read_row_hits = 0;
        uint64_t warmup_write_row_hits = 0;
        uint64_t warmup_ondemand_pres = 0;
        uint64_t warmup_pim_activate_cmds = 0;
        uint64_t warmup_pim_precharge_cmds = 0;
        if (_command_has_warmup_sample) {
            const auto& warmup_base = _command_warmup_baseline[channel];
            const auto& warmup_end = _command_warmup_end[channel];
            warmup_reads = warmup_end.num_read_requests -
                           warmup_base.num_read_requests;
            warmup_writes = warmup_end.num_write_requests -
                            warmup_base.num_write_requests;
            warmup_pim = warmup_end.num_pim_requests -
                         warmup_base.num_pim_requests;
            warmup_total = warmup_reads + warmup_writes + warmup_pim;
            warmup_read_cmds = warmup_end.num_read_cmds -
                               warmup_base.num_read_cmds;
            warmup_write_cmds = warmup_end.num_write_cmds -
                                warmup_base.num_write_cmds;
            warmup_act_cmds = warmup_end.num_act_cmds -
                              warmup_base.num_act_cmds;
            warmup_pre_cmds = warmup_end.num_pre_cmds -
                              warmup_base.num_pre_cmds;
            warmup_write_buf_hits = warmup_end.num_write_buf_hits -
                                    warmup_base.num_write_buf_hits;
            warmup_read_row_hits = warmup_end.num_read_row_hits -
                                   warmup_base.num_read_row_hits;
            warmup_write_row_hits = warmup_end.num_write_row_hits -
                                    warmup_base.num_write_row_hits;
            warmup_ondemand_pres = warmup_end.num_ondemand_pres -
                                   warmup_base.num_ondemand_pres;
            warmup_pim_activate_cmds =
                warmup_end.num_pim_activate_cmds -
                warmup_base.num_pim_activate_cmds;
            warmup_pim_precharge_cmds =
                warmup_end.num_pim_precharge_cmds -
                warmup_base.num_pim_precharge_cmds;
        }
        uint64_t estimated_reads = _command_has_warmup_sample
            ? combined_scale(
                  warmup_read_cmds, warmup_reads,
                  stat.num_read_cmds - base.num_read_cmds, sampled_reads,
                  skipped_reads)
            : weighted_scale(
                  warmup_read_cmds, warmup_reads,
                  stat.num_read_cmds - base.num_read_cmds, sampled_reads,
                  skipped_reads);
        uint64_t estimated_writes = _command_has_warmup_sample
            ? combined_scale(
                  warmup_write_cmds, warmup_writes,
                  stat.num_write_cmds - base.num_write_cmds, sampled_writes,
                  skipped_writes)
            : weighted_scale(
                  warmup_write_cmds, warmup_writes,
                  stat.num_write_cmds - base.num_write_cmds, sampled_writes,
                  skipped_writes);
        if (write_command_override != nullptr &&
            channel < write_command_override->size()) {
            estimated_writes =
                write_command_override->at(channel);
        }
        const uint64_t estimated_acts = weighted_scale(
            warmup_act_cmds, warmup_total,
            stat.num_act_cmds - base.num_act_cmds, sampled_total,
            skipped_total);
        const uint64_t estimated_pres = weighted_scale(
            warmup_pre_cmds, warmup_total,
            stat.num_pre_cmds - base.num_pre_cmds, sampled_total,
            skipped_total);
        const uint64_t estimated_write_buf_hits =
            _command_has_warmup_sample
                ? combined_scale(
                      warmup_write_buf_hits, warmup_writes,
                      stat.num_write_buf_hits - base.num_write_buf_hits,
                      sampled_writes, skipped_writes)
                : weighted_scale(
                      warmup_write_buf_hits, warmup_writes,
                      stat.num_write_buf_hits - base.num_write_buf_hits,
                      sampled_writes, skipped_writes);
        const uint64_t estimated_read_row_hits = weighted_scale(
            warmup_read_row_hits, warmup_reads,
            stat.num_read_row_hits - base.num_read_row_hits,
            sampled_reads, skipped_reads);
        const uint64_t estimated_write_row_hits =
            _command_has_warmup_sample
                ? combined_scale(
                      warmup_write_row_hits, warmup_writes,
                      stat.num_write_row_hits - base.num_write_row_hits,
                      sampled_writes, skipped_writes)
                : weighted_scale(
                      warmup_write_row_hits, warmup_writes,
                      stat.num_write_row_hits - base.num_write_row_hits,
                      sampled_writes, skipped_writes);
        const uint64_t estimated_ondemand_pres = weighted_scale(
            warmup_ondemand_pres, warmup_total,
            stat.num_ondemand_pres - base.num_ondemand_pres,
            sampled_total, skipped_total);
        const uint64_t estimated_pim_activate_cmds = weighted_scale(
            warmup_pim_activate_cmds, warmup_pim,
            stat.num_pim_activate_cmds - base.num_pim_activate_cmds,
            sampled_pim, skipped_pim);
        const uint64_t estimated_pim_precharge_cmds = weighted_scale(
            warmup_pim_precharge_cmds, warmup_pim,
            stat.num_pim_precharge_cmds - base.num_pim_precharge_cmds,
            sampled_pim, skipped_pim);
        // Decode Pruning skips a complete operation and therefore starts this
        // command window with no physical commands to sample. Primary
        // MOVIN/MOVOUT and PIM requests have a one-to-one command mapping, so
        // preserve their exact logical counts in that case. Locality-dependent
        // ACT/PRE and hit counters are replayed separately from the sampled
        // DRAM-state template.
        if (skipped_reads > 0 && sampled_reads == 0) {
            estimated_reads = skipped_reads;
        }
        if (write_command_override == nullptr &&
            skipped_writes > 0 && sampled_writes == 0) {
            estimated_writes = skipped_writes;
        }
        stat.estimated_read_cmds += estimated_reads;
        stat.estimated_write_cmds += estimated_writes;
        stat.estimated_act_cmds += estimated_acts;
        stat.estimated_pre_cmds += estimated_pres;
        stat.estimated_write_buf_hits += estimated_write_buf_hits;
        stat.estimated_read_row_hits += estimated_read_row_hits;
        stat.estimated_write_row_hits += estimated_write_row_hits;
        stat.estimated_ondemand_pres += estimated_ondemand_pres;
        stat.estimated_pim_activate_cmds +=
            estimated_pim_activate_cmds;
        stat.estimated_pim_precharge_cmds +=
            estimated_pim_precharge_cmds;
        const auto primary_pim_commands =
            [&scale](uint64_t measured_commands,
                     uint64_t skipped_commands) -> uint64_t {
                return measured_commands == 0
                    ? skipped_commands
                    : scale(
                          measured_commands, skipped_commands,
                          measured_commands);
            };
        stat.estimated_pheader_cmds += primary_pim_commands(
            stat.num_pheader_cmds - base.num_pheader_cmds,
            skipped_pheader);
        stat.estimated_gwrite_cmds += primary_pim_commands(
            stat.num_gwrite_cmds - base.num_gwrite_cmds,
            skipped_gwrite);
        stat.estimated_comp_cmds += primary_pim_commands(
            stat.num_comp_cmds - base.num_comp_cmds,
            skipped_comp);
        stat.estimated_readres_cmds += primary_pim_commands(
            stat.num_readres_cmds - base.num_readres_cmds,
            skipped_readres);
        stat.estimated_pim_cmds += sampled_pim == 0
            ? skipped_pim
            : scale(
                  stat.num_pim_cmds - base.num_pim_cmds,
                  skipped_pim, sampled_pim);
        spdlog::info(
            "EventDriven DRAM CH[{}] command compensation: warmup/sample requests {}/{}, weights {:.3f}/{:.3f}, skipped requests {}, estimated read/write/act/pre {}/{}/{}/{}",
            channel, warmup_total, sampled_total,
            _command_has_warmup_sample ? _command_warmup_weight : 0.0,
            _command_has_warmup_sample ? 1.0 - _command_warmup_weight : 1.0,
            skipped_total, estimated_reads, estimated_writes,
            estimated_acts, estimated_pres);
        spdlog::info(
            "EventDriven DRAM CH[{}] locality compensation: write-buffer/read-row/write-row/on-demand-PRE {}/{}/{}/{}",
            channel, estimated_write_buf_hits, estimated_read_row_hits,
            estimated_write_row_hits, estimated_ondemand_pres);
    }
}

double EventDrivenDram::get_avg_bw_util() {
    uint64_t logical_requests = 0;
    cycle_type current_cycle = 0;
    for (const auto& stat : _stats) {
        logical_requests +=
            stat.num_read_requests + stat.estimated_read_requests +
            stat.num_write_requests + stat.estimated_write_requests +
            stat.num_pim_requests + stat.estimated_pim_requests;
        current_cycle = std::max<cycle_type>(current_cycle, stat.num_cycles);
    }
    const uint64_t stage_requests =
        logical_requests >= _stage_request_baseline
            ? logical_requests - _stage_request_baseline : 0;
    const cycle_type stage_cycles =
        current_cycle >= _stage_cycle_baseline
            ? current_cycle - _stage_cycle_baseline : 0;
    const double utilization =
        stage_cycles == 0 || _stats.empty()
            ? 0.0
            : static_cast<double>(
                static_cast<long double>(stage_requests) *
                _memsys->config_.burst_cycle / _stats.size() /
                stage_cycles * 100.0L);
    _stage_request_baseline = logical_requests;
    _stage_cycle_baseline = current_cycle;
    return utilization;
}

uint64_t EventDrivenDram::get_avg_pim_cycle() {
    uint64_t logical_pim_cycles = 0;
    for (const auto& stat : _stats) {
        logical_pim_cycles += stat.pim_cycles + stat.estimated_pim_cycles;
    }
    const uint64_t stage_pim_cycles =
        logical_pim_cycles >= _stage_pim_cycle_baseline
            ? logical_pim_cycles - _stage_pim_cycle_baseline : 0;
    _stage_pim_cycle_baseline = logical_pim_cycles;
    return _stats.empty() ? 0 : stage_pim_cycles / _stats.size();
}

void EventDrivenDram::reset_pim_cycle() {
    _stage_pim_cycle_baseline = 0;
    for (const auto& stat : _stats) {
        _stage_pim_cycle_baseline +=
            stat.pim_cycles + stat.estimated_pim_cycles;
    }
}

void EventDrivenDram::begin_proportional_command_sampling() {
    _command_sampling_baseline = _stats;
    _command_warmup_baseline = _stats;
    _command_warmup_end.clear();
    _command_has_warmup_sample = false;
}

void EventDrivenDram::begin_decode_pruning_state_sample(
    const std::string& operation) {
    _decode_pruning_state_operation = operation;
    auto& baseline = _decode_pruning_state_baseline;
    baseline = DecodePruningDramState{};
    baseline.counters.resize(_stats.size());
    baseline.rank_active_cycles.resize(_stats.size());
    baseline.sref_cycles.resize(_stats.size());
    baseline.pim_rank_active_cycles.resize(_stats.size());
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        const auto& stat = _stats[channel];
        auto& counters = baseline.counters[channel];
        counters["num_act_cmds"] = stat.num_act_cmds;
        counters["num_pre_cmds"] = stat.num_pre_cmds;
        counters["num_write_buf_hits"] = stat.num_write_buf_hits;
        counters["num_read_row_hits"] = stat.num_read_row_hits;
        counters["num_write_row_hits"] = stat.num_write_row_hits;
        counters["num_ondemand_pres"] = stat.num_ondemand_pres;
        counters["num_pim_activate_cmds"] =
            stat.num_pim_activate_cmds;
        counters["num_pim_precharge_cmds"] =
            stat.num_pim_precharge_cmds;
        counters["num_ref_cmds"] = stat.num_ref_cmds;
        counters["num_refb_cmds"] = stat.num_refb_cmds;
        counters["num_write_requests"] = stat.num_write_requests;
        counters["num_write_cmds"] = stat.num_write_cmds;
        counters["pim_cycles"] = stat.pim_cycles;
        counters["num_cycles"] = stat.num_cycles;
        baseline.rank_active_cycles[channel] =
            stat.rank_active_cycles;
        baseline.sref_cycles[channel].resize(
            stat.rank_active_cycles.size(), 0);
        baseline.pim_rank_active_cycles[channel] =
            stat.pim_rank_active_cycles;
    }
}

void EventDrivenDram::finish_decode_pruning_state_sample(
    const std::string& operation, uint32_t required_samples) {
    assert(operation == _decode_pruning_state_operation);
    assert(required_samples > 0);
    auto& accumulator =
        _decode_pruning_state_accumulators[operation];
    if (accumulator.counters.empty()) {
        accumulator.counters.resize(_stats.size());
        accumulator.rank_active_cycles.resize(_stats.size());
        accumulator.sref_cycles.resize(_stats.size());
        accumulator.pim_rank_active_cycles.resize(_stats.size());
        for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
            const size_t ranks =
                _stats[channel].rank_active_cycles.size();
            accumulator.rank_active_cycles[channel].resize(ranks, 0);
            accumulator.sref_cycles[channel].resize(ranks, 0);
            accumulator.pim_rank_active_cycles[channel].resize(ranks, 0);
        }
    }
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        const auto& stat = _stats[channel];
        const std::unordered_map<std::string, uint64_t> current = {
            {"num_act_cmds", stat.num_act_cmds},
            {"num_pre_cmds", stat.num_pre_cmds},
            {"num_write_buf_hits", stat.num_write_buf_hits},
            {"num_read_row_hits", stat.num_read_row_hits},
            {"num_write_row_hits", stat.num_write_row_hits},
            {"num_ondemand_pres", stat.num_ondemand_pres},
            {"num_pim_activate_cmds", stat.num_pim_activate_cmds},
            {"num_pim_precharge_cmds", stat.num_pim_precharge_cmds},
            {"num_ref_cmds", stat.num_ref_cmds},
            {"num_refb_cmds", stat.num_refb_cmds},
            {"num_write_requests", stat.num_write_requests},
            {"num_write_cmds", stat.num_write_cmds},
            {"pim_cycles", stat.pim_cycles},
            {"num_cycles", stat.num_cycles}};
        for (const auto& [counter, value] : current) {
            const uint64_t baseline =
                _decode_pruning_state_baseline
                    .counters[channel].at(counter);
            accumulator.counters[channel][counter] +=
                value >= baseline ? value - baseline : 0;
        }
        for (size_t rank = 0;
             rank < stat.rank_active_cycles.size(); ++rank) {
            const auto add_delta = [](uint64_t& sum, uint64_t value,
                                      uint64_t baseline) {
                sum += value >= baseline ? value - baseline : 0;
            };
            add_delta(
                accumulator.rank_active_cycles[channel][rank],
                stat.rank_active_cycles[rank],
                _decode_pruning_state_baseline
                    .rank_active_cycles[channel][rank]);
            add_delta(
                accumulator.pim_rank_active_cycles[channel][rank],
                stat.pim_rank_active_cycles[rank],
                _decode_pruning_state_baseline
                    .pim_rank_active_cycles[channel][rank]);
        }
    }
    ++accumulator.samples;
    if (accumulator.samples < required_samples) return;

    DecodePruningDramState result = accumulator;
    const auto average = [&accumulator](uint64_t value) {
        return static_cast<uint64_t>(std::llround(
            static_cast<long double>(value) / accumulator.samples));
    };
    result.samples = accumulator.samples;
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        for (auto& [counter, value] : result.counters[channel]) {
            value = average(value);
        }
        for (size_t rank = 0;
             rank < result.rank_active_cycles[channel].size(); ++rank) {
            result.rank_active_cycles[channel][rank] =
                average(result.rank_active_cycles[channel][rank]);
            result.pim_rank_active_cycles[channel][rank] =
                average(result.pim_rank_active_cycles[channel][rank]);
        }
    }
    _decode_pruning_state_templates[operation] = std::move(result);
}

std::vector<uint64_t>
EventDrivenDram::decode_pruning_sampled_write_requests(
    const std::string& operation) const {
    std::vector<uint64_t> writes(_stats.size(), 0);
    const auto template_it =
        _decode_pruning_state_templates.find(operation);
    if (template_it != _decode_pruning_state_templates.end()) {
        for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
            writes[channel] =
                template_it->second.counters[channel].at(
                    "num_write_requests");
        }
        return writes;
    }
    assert(operation == _decode_pruning_state_operation);
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        const uint64_t current = _stats[channel].num_write_requests;
        const uint64_t baseline =
            _decode_pruning_state_baseline.counters[channel].at(
                "num_write_requests");
        writes[channel] =
            current >= baseline ? current - baseline : 0;
    }
    return writes;
}

std::vector<uint64_t>
EventDrivenDram::decode_pruning_sampled_write_commands(
    const std::string& operation) const {
    std::vector<uint64_t> commands(_stats.size(), 0);
    const auto template_it =
        _decode_pruning_state_templates.find(operation);
    if (template_it != _decode_pruning_state_templates.end()) {
        for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
            commands[channel] =
                template_it->second.counters[channel].at(
                    "num_write_cmds");
        }
        return commands;
    }
    assert(operation == _decode_pruning_state_operation);
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        const uint64_t current = _stats[channel].num_write_cmds;
        const uint64_t baseline =
            _decode_pruning_state_baseline.counters[channel].at(
                "num_write_cmds");
        commands[channel] =
            current >= baseline ? current - baseline : 0;
    }
    return commands;
}

void EventDrivenDram::apply_decode_pruning_state(
    const std::string& operation, cycle_type skipped_dram_cycles) {
    const auto template_it =
        _decode_pruning_state_templates.find(operation);
    assert(template_it != _decode_pruning_state_templates.end());
    const auto& state = template_it->second;
    const auto scale = [skipped_dram_cycles](
                           uint64_t value,
                           uint64_t sampled_cycles) -> uint64_t {
        if (value == 0 || sampled_cycles == 0 ||
            skipped_dram_cycles == 0) {
            return 0;
        }
        return static_cast<uint64_t>(std::llround(
            static_cast<long double>(value) * skipped_dram_cycles /
            sampled_cycles));
    };
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        auto& stat = _stats[channel];
        const auto& counters = state.counters[channel];
        const uint64_t sampled_cycles = counters.at("num_cycles");
        const bool npu_attention_projection =
            counters.at("pim_cycles") == 0 &&
            (operation.find(".attn.QGen") != std::string::npos ||
             operation.find(".attn.KGen") != std::string::npos ||
             operation.find(".attn.VGen") != std::string::npos ||
             operation.find(".attn.proj") != std::string::npos);
        if (npu_attention_projection) {
            const uint64_t assumed_write_cmds =
                counters.at("num_write_requests");
            assert(stat.estimated_write_cmds >= assumed_write_cmds);
            stat.estimated_write_cmds -= assumed_write_cmds;
            stat.estimated_write_cmds +=
                counters.at("num_write_cmds");
        }
        stat.estimated_act_cmds += counters.at("num_act_cmds");
        stat.estimated_pre_cmds += counters.at("num_pre_cmds");
        stat.estimated_write_buf_hits +=
            counters.at("num_write_buf_hits");
        stat.estimated_read_row_hits +=
            counters.at("num_read_row_hits");
        stat.estimated_write_row_hits +=
            counters.at("num_write_row_hits");
        stat.estimated_ondemand_pres +=
            counters.at("num_ondemand_pres");
        stat.estimated_pim_activate_cmds +=
            counters.at("num_pim_activate_cmds");
        stat.estimated_pim_precharge_cmds +=
            counters.at("num_pim_precharge_cmds");
        stat.estimated_pim_cycles +=
            scale(counters.at("pim_cycles"), sampled_cycles);
        stat.estimated_cycles += skipped_dram_cycles;
        for (size_t rank = 0;
             rank < stat.rank_active_cycles.size(); ++rank) {
            uint64_t active = scale(
                state.rank_active_cycles[channel][rank],
                sampled_cycles);
            active = std::min<uint64_t>(
                active, skipped_dram_cycles);
            stat.estimated_rank_active_cycles[rank] += active;
            stat.estimated_all_bank_idle_cycles[rank] +=
                skipped_dram_cycles - active;

            uint64_t pim_active = scale(
                state.pim_rank_active_cycles[channel][rank],
                sampled_cycles);
            pim_active = std::min<uint64_t>(
                pim_active, skipped_dram_cycles);
            stat.estimated_pim_rank_active_cycles[rank] += pim_active;
            stat.estimated_pim_all_bank_idle_cycles[rank] +=
                skipped_dram_cycles - pim_active;
        }
        spdlog::info(
            "Decode Pruning EventDriven DRAM CH[{}] state replay for {}: "
            "ACT/PRE {}/{}, PIM cycles {}, elapsed {}",
            channel, operation, counters.at("num_act_cmds"),
            counters.at("num_pre_cmds"),
            scale(counters.at("pim_cycles"), sampled_cycles),
            skipped_dram_cycles);
    }
}

void EventDrivenDram::mark_proportional_command_warmup_complete(
    double warmup_weight) {
    assert(warmup_weight >= 0.0 && warmup_weight <= 1.0);
    _command_warmup_weight = warmup_weight;
    _command_warmup_end = _stats;
    _command_sampling_baseline = _stats;
    _command_has_warmup_sample = true;
    spdlog::info(
        "Marked EventDriven proportional DRAM command warmup window with warmup/sample weights {:.3f}/{:.3f}",
        _command_warmup_weight, 1.0 - _command_warmup_weight);
}

void EventDrivenDram::apply_estimated_time(cycle_type skipped_dram_cycles,
                                           bool physical_tail_pending) {
    if (physical_tail_pending) return;
    if (skipped_dram_cycles == 0) return;
    const auto scale = [skipped_dram_cycles](uint64_t measured,
                                             uint64_t sampled) -> uint64_t {
        if (measured == 0 || sampled == 0) return 0;
        return static_cast<uint64_t>(std::llround(
            static_cast<long double>(measured) * skipped_dram_cycles /
            sampled));
    };
    for (uint32_t channel = 0; channel < _stats.size(); ++channel) {
        auto& stat = _stats[channel];
        const auto& base = _command_sampling_baseline[channel];
        const uint64_t sampled_cycles = stat.num_cycles - base.num_cycles;
        const uint64_t sampled_pim_cycles =
            stat.pim_cycles - base.pim_cycles;
        stat.estimated_pim_cycles +=
            scale(sampled_pim_cycles, sampled_cycles);
        stat.estimated_cycles += skipped_dram_cycles;
        stat.time_compensation_cycle_baseline = stat.num_cycles;
        stat.time_compensation_ref_baseline = stat.num_ref_cmds;
        stat.time_compensation_refb_baseline = stat.num_refb_cmds;
        stat.estimated_ref_cmds += scale(
            stat.num_ref_cmds - base.num_ref_cmds, sampled_cycles);
        stat.estimated_refb_cmds += scale(
            stat.num_refb_cmds - base.num_refb_cmds, sampled_cycles);
        for (size_t rank = 0; rank < stat.rank_active_cycles.size(); ++rank) {
            stat.time_compensation_active_baseline[rank] =
                stat.rank_active_cycles[rank];
            stat.time_compensation_idle_baseline[rank] =
                stat.all_bank_idle_cycles[rank];
            const uint64_t sampled_active =
                stat.rank_active_cycles[rank] - base.rank_active_cycles[rank];
            uint64_t estimated_active = scale(sampled_active, sampled_cycles);
            estimated_active = std::min<uint64_t>(estimated_active,
                                                  skipped_dram_cycles);
            stat.estimated_rank_active_cycles[rank] += estimated_active;
            stat.estimated_all_bank_idle_cycles[rank] +=
                skipped_dram_cycles - estimated_active;
        }
        spdlog::info(
            "EventDriven DRAM CH[{}] time compensation: sampled cycles {}, skipped cycles {}, estimated REF/REFB {}/{}",
            channel, sampled_cycles, skipped_dram_cycles,
            stat.estimated_ref_cmds, stat.estimated_refb_cmds);
    }
}
