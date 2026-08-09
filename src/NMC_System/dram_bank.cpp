#include "dram_bank.hpp"

#include <memory>
#include <algorithm>


DRAMBank::DRAMBank(MemConfig& MemConfig, std::vector<std::pair<int, uint64_t>>* Rank_Activate_Recorder, int Rank_id, int BankGroup_id, int Bank_id) :
    config_(MemConfig),
    rank_activate_recorder(Rank_Activate_Recorder),
    rank_id(Rank_id),
    bankgroup_id(BankGroup_id),
    bank_id(Bank_id),
    bank_state(BankState::CLOSED),
    open_row(-1),
    open_row_exec_event_count(0),
    pending_precharge(false),
    pending_activate(false),   //
    row_hit_count(0),
    row_miss_count(0),
    row_buf_policy_(MemConfig.row_buf_policy == "CLOSE_PAGE" ? RowBufPolicy::CLOSE_PAGE : RowBufPolicy::OPEN_PAGE)
    {
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

}

DRAMBank::~DRAMBank() {

}
/*
uint64_t DRAMBank::ProcessTransaction(std::shared_ptr<Memory_Transaction> trans) {

    // spdlog::trace("A {} Transaction is processed in Rank : {}, BankGroup: {}, Bank: {}", trans->is_write ? "Write" : "Read", rank_id, bankgroup_id, bank_id);

    bool is_hit = (bank_state == BankState::OPEN && trans->address.row == open_row);

    spdlog::info("[DEBUG_BANK] Addr:{:#10x} | Bank:{} | State:{} | OpenRow:{} | TargetRow:{} | Result:{}",
                 trans->addr, bank_id,
                 (bank_state == BankState::OPEN ? "OPEN" : "CLOSED"),
                 open_row, trans->address.row,
                 (is_hit ? "ROW_HIT" : "ROW_MISS"));

    // ===== 追踪第一条trace在Bank级的处理 =====
    static int bank_process_count = 0;
    if (bank_process_count == 0 && trans->addr == 0x0) {
        spdlog::warn("========== BANK ProcessTransaction (FIRST TRACE) ==========");
        spdlog::warn("[BANK_PROCESS] addr={:#x}, bank_state={}, open_row={}",
                     trans->addr, bank_state == BankState::CLOSED ? "CLOSED" : "OPEN", open_row);
        spdlog::warn("[BANK_PROCESS] processing_cycle={}", trans->processing_cycle);
        bank_process_count++;
    }

    // 首先判断是否需要refresh, 进行refresh的话，已经包括了执行refresh的时间以及refresh之后precharge的时间
    if (bank_refresh) {
        if (trans->processing_cycle >= next_bank_refresh_time) {
            unsigned int current_trans_bank_refresh_count = (trans->processing_cycle - refresh_offset_) / refresh_interval_;
            latest_refresh_time = BankRefresh(current_trans_bank_refresh_count);
            //latest_refresh_time = latest_refresh_time > rank_activate_recorder->end()->second + config_.tRC ?latest_refresh_time : rank_activate_recorder->end()->second + config_.tRC; // Influence of Activate to Bank_Refresh

            if (!rank_activate_recorder->empty()) {
                auto last_act = rank_activate_recorder->back().second;
                latest_refresh_time = std::max(latest_refresh_time, last_act + config_.tRC);
            }

            bank_state = BankState::CLOSED; // after refresh, update the bank state
            to_do_activate = latest_refresh_time + config_.tRFCb;

            bank_refresh_count = current_trans_bank_refresh_count;
            next_bank_refresh_time = (bank_refresh_count + 1) * refresh_interval_ + refresh_offset_;
        }
    }

    // Transaction的执行过程，如果是关闭状态，需要添加一个activate操作
    if (bank_state == BankState::CLOSED) {
        // do_activate
        trans->processing_cycle = trans->processing_cycle > to_do_activate ? trans->processing_cycle : to_do_activate;  // Transaction的时间
        Activate(trans);
        if (trans->is_write) {
            trans->processing_cycle = trans->processing_cycle > to_do_write ? trans->processing_cycle : to_do_write ; // 激活后执行读写操作
            Write(trans);  // 一个activate的时间 + activate之后读写的时间
        }
        else {
            trans->processing_cycle = trans->processing_cycle > to_do_read ? trans->processing_cycle : to_do_read; // 激活后执行读写操作
            Read(trans);  // 一个activate的时间 + activate之后读写的时间

            // ===== 记录READ完成时间 =====
            if (bank_process_count == 1 && trans->addr == 0x0) {
                spdlog::warn("[BANK_PROCESS] After Read(), complete_cycle={}", trans->complete_cycle);
                spdlog::warn("[BANK_PROCESS] processing_cycle={}, read_delay={}",
                             trans->processing_cycle, config_.read_delay);
                spdlog::warn("==========================================\n");
                bank_process_count++;
            }
        }
        open_row = trans->address.row;
        bank_state = BankState::OPEN;
    }
    else if (bank_state == BankState::OPEN) {
        // Row Buffer Hit or Row Buffer Miss
        if (trans->address.row == open_row) {
            if (trans->is_write) {
                row_hit_count++;
                Write(trans);
            }
            else {
                // row buffer hit read
                // std::cout << "Row Buffer Hit Read for channel :" << trans->address.channel << " Rank: " <<  trans->address.rank << " BankGroup: " <<  trans->address.bankgroup << " Bank: " << trans->address.bank << std::endl;
                Read(trans);
            }
        }
        else {
            RowSwitch(trans);
            if (trans->is_write) {
                // Row buffer miss write
                // std::cout << "Row Buffer Miss Write for channel :" << trans->address.channel << " Rank: " <<  trans->address.rank << " BankGroup: " <<  trans->address.bankgroup << " Bank: " << trans->address.bank << std::endl;
                Write(trans);
            }
            else {
                Read(trans);
            }
        }
    }

    return trans->processing_cycle;
}


int DRAMBank::BankRefresh(unsigned int current_refresh_count) {
    int refresh_execute_block_time = 0;
    uint64_t refresh_idle_time = (current_refresh_count-1) * refresh_interval_ + refresh_offset_;

    if (bank_state == BankState::OPEN && current_refresh_count-bank_refresh_count == 1) {
        // 如果Bank处于开启情况，需要首先完成precharge，发送过程满足Bank的当前时序要求, 最小为0。执行precharge关闭当前板卡行之后，在经过tRP才能进行refresh刷新操作。
        refresh_execute_block_time += std::max(static_cast<int>(to_do_precharge - refresh_idle_time), 0);    // do precharge
        refresh_execute_block_time ++;
        refresh_execute_block_time += config_.tRP;                                                                // precharge to activate(refresh)
    }
    else if (bank_state == BankState::CLOSED){
        refresh_execute_block_time += std::max(static_cast<int>(to_do_activate - refresh_idle_time), 0);  // Bank 处于关闭状态时，考虑Bank的Activate的时间限制与当前刷新的时间即可。
    }
    else {
        refresh_execute_block_time = 0;   // 两次刷新之间没有Transaction出现，在这种情况下Bank是稳定的
    }

    // spdlog::info(" Rank: {}, BankGroup {}, Bank {}, Refreshed at time {}", bank_id, bankgroup_id, bank_id, refresh_idle_time + refresh_execute_block_time);

    return refresh_idle_time + refresh_execute_block_time  ;
}
*/


void DRAMBank::PushPendingSwitchRow(std::shared_ptr<Event> event) {
    pending_activate_row.insert(std::make_pair(event->row_index, event));
    event->activate_cycle = std::max(event->activate_cycle, to_do_activate);
    spdlog::info("Pending activate for event address {} and the activate cycle is {}", event->dram_address, event->activate_cycle);
}


// ------------------------ 下方是可用的接口函数 -------------------------------------------
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


void DRAMBank::Precharge(std::shared_ptr<Event> event) { //TODO: 后面可以创建一个针对Refresh的Precharge, 或者对现有的Precharge进行修改
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
    to_do_read = event->execute_cycle + EventDrivenParams::read_to_read_l;
    to_do_write = event->execute_cycle + EventDrivenParams::read_to_write;
    to_do_precharge = event->execute_cycle + EventDrivenParams::read_to_precharge;
}


void DRAMBank::Write(std::shared_ptr<Event> event) {
    assert(event->row_index == open_row);
    open_row_exec_event_count--;
    to_do_read = event->execute_cycle + EventDrivenParams::write_to_read_l;
    to_do_write = event->execute_cycle + EventDrivenParams::write_to_write_l;
    to_do_precharge = event->execute_cycle + EventDrivenParams::write_to_precharge;
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
