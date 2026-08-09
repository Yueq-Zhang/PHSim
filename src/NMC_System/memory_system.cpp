
#include "memory_system.hpp"


MemorySystem::MemorySystem(const MemConfig& mem_config, int mem_id): config_(mem_config), memory_id_(mem_id), last_transaction_clk_(0){
    dram_channels.reserve(config_.channels);
    for (auto i = 0; i < config_.channels; i++) {
        dram_channels.emplace_back(new DRAMChannel(config_, i));
    }
    std::cout << "The Dram Channel with " << config_.channels << " channels is initialized" << std::endl;
}


MemorySystem::~MemorySystem() {
    for (auto & dram_channel : dram_channels) {
        delete dram_channel;
    }
    std::cout << "Memory System Destructor" << std::endl;
}


int MemorySystem::GetChannel(uint64_t hex_addr) const {
    // 原逻辑：直接按 MemConfig 位域从 hex_addr 中取 channel，不做 shift。
    // 这里保留旧逻辑作为注释，实际改为与 CA/allocator 使用同一套解码接口，
    // 避免 ED 外层 channel 选择与 DRAMChannel 内部 bank/row 解码口径不一致。
    /*
    // hex_addr >>= config_.shift_bits;
    return (hex_addr >> config_.ch_pos) & config_.ch_mask;
    */
    return MyAddressAllocator::get_channel_index(hex_addr);
}

/*
bool MemorySystem::AddTransaction(std::shared_ptr<Memory_Transaction> trans) {
    int channel_index = GetChannel(trans->addr);
    return dram_channels[channel_index]->AddTransaction(trans);
}
*/
/*
TransactionStatus MemorySystem::AddTransaction(std::shared_ptr<Memory_Transaction> trans) {
    int channel_index = GetChannel(trans->addr);
    return dram_channels[channel_index]->AddTransaction(trans);
}

TransactionStatus MemorySystem::AddTransaction(std::shared_ptr<Memory_Transaction> trans, uint32_t cid) {
    return dram_channels[cid]->AddTransaction(trans);
}

bool MemorySystem::AddPIMTransaction(std::shared_ptr<PIM_Transaction> trans) {
    // 在这里例化对应的PIM单元
    // 将相连的下层
    int channel_index = GetChannel(trans->addr);
    return dram_channels[channel_index]->AddPIMTransaction(trans);
}
*/
void MemorySystem::CheckRowSwitch(std::shared_ptr<Event> event) {
    // get the operation bank of event and check row switch;
    // if bank close page, Activate the Bank row
    // if row buffer hit, push into execute queue and open_row_exec_event_count++
    // if row buffer miss, if open_row_exec_event_count == 0 directly precharge; else if open_row_exec_event_count != 0,
    // buffer into pending_precharge_order_queue, and pending_precharge_event,
    // after execution, open_row_exec_event_count = 0, queue not empty, fetch the event for precharge and activate
    auto operation_bank = dram_channels[event->channel_index]->dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];
    auto current_row= operation_bank->open_row;

    if (current_row == -1) {   // row page close, event in activate_queue
        if (operation_bank->pending_precharge_order_queue.empty() &&
            operation_bank->pending_precharge == false &&
            operation_bank->pending_activate == false){
            PendingActivate(event);
            operation_bank->pending_activate = true;
            spdlog::info("(EventDriven DRAM) At {} Cycle input event {} are added into the Activate queue, current Activate queue size is {}",
                event->add_cycle, event->dram_address, dram_channels[event->channel_index]->activate_queue.size());
            }
        else {
            operation_bank->AddPendingPrechargeEvent(event);
            spdlog::info("(EventDriven DRAM) At {} Cycle input event {} are added into the Pending Precharge queue of channel {}, Rank {}, Bankgroup {}, Bank {}, Row {} for current pending precharge row is {}",
            event->add_cycle, event->dram_address, event->channel_index, event->rank_index, event->bankgroup_index, event->bank_index, event->row_index, operation_bank->pending_precharge_order_queue.front());
        }
    }
    else if (current_row != event->row_index){  // need row switch
        if (operation_bank->open_row_exec_event_count == 0 &&
            operation_bank->pending_precharge_order_queue.empty() &&
            operation_bank->pending_precharge == false) {
            // 当前bank前没有event，可以直接换行
            PendingPrecharge(event);
            spdlog::info("(EventDriven DRAM) At {} Cycle input event {} are added into the Precharge queue, current Precharge queue size is {}",
                event->add_cycle, event->dram_address, dram_channels[event->channel_index]->precharge_queue.size());
        }
        else {
            // 当前bank存在前面的event, 先缓存起来，等后面处理
            operation_bank->AddPendingPrechargeEvent(event);
            spdlog::info("(EventDriven DRAM) At {} Cycle input event {} are added into the Pending Precharge queue of channel {}, Rank {}, Bankgroup {}, Bank {}, Row {} for current open_row {} with execution event count = {}",
                event->add_cycle, event->dram_address, event->channel_index, event->rank_index, event->bankgroup_index, event->bank_index, event->row_index, operation_bank->open_row, operation_bank->open_row_exec_event_count);
        }
      // dram_channels[event->channel_index]->activate_queue.push_back(event);
    }
    else {
        if (operation_bank->pending_precharge || !operation_bank->pending_precharge_order_queue.empty()) {
            operation_bank->AddPendingPrechargeEvent(event);
            spdlog::info("(EventDriven DRAM) At {} Cycle input event {} are added into the Pending Precharge queue of channel {}, Rank {}, Bankgroup {}, Bank {}, Row {} for pending bank row switch",
                event->add_cycle, event->dram_address, event->channel_index, event->rank_index, event->bankgroup_index, event->bank_index, event->row_index);
        }
        else {
            PendingExecuted(event);
            operation_bank->open_row_exec_event_count++;
            spdlog::info("(EventDriven DRAM) At {} Cycle input event {} are added into the Executed queue, row buffer hit, current Executed queue size is {}",
                event->add_cycle, event->dram_address, dram_channels[event->channel_index]->execute_queue.size());
        }
    }
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
        spdlog::info("(EventDriven DRAM) Get the Precharge time of the first precharge event with address {} of channel {} at {}",
            event->dram_address, event->channel_index, event->precharge_cycle);
    }
}

void MemorySystem::PendingActivate(std::shared_ptr<Event> event) {
    event->need_activate = true;
    event->activate_cycle = event->add_cycle + 1;
    dram_channels[event->channel_index]->activate_queue.push_back(event);
    if (dram_channels[event->channel_index]->activate_queue.size() == 1) {
        dram_channels[event->channel_index]->GetActivateCycle(dram_channels[event->channel_index]->activate_queue.front()); // 对于首个输入的Event进行处理
        spdlog::info("(EventDriven DRAM) Get the Activation time of the first activate event with address {} of channel {} at {}",
            event->dram_address, event->channel_index,  event->activate_cycle);
    }
}

void MemorySystem::PendingExecuted(std::shared_ptr<Event> event) {
    event->execute_cycle = event->execute_cycle + 1;
    dram_channels[event->channel_index]->execute_queue.push_back(event);
    if (dram_channels[event->channel_index]->execute_queue.size() == 1) {
        dram_channels[event->channel_index]->GetExecuteCycle(dram_channels[event->channel_index]->execute_queue.front()); // 模型对于首个输入的Event进行处理
        spdlog::info("(EventDriven DRAM) Get the Execution time of the first execute event with address {} of channel {} at {}",
            event->dram_address, event->channel_index, event->execute_cycle);
    }
}
