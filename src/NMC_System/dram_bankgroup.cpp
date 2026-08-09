#include "dram_bankgroup.hpp"



DRAMBankGroup::DRAMBankGroup(MemConfig& MemConfig, std::vector<std::pair<int, uint64_t>>* Rank_Activate_Recorder, int Rank_id, int BankGroup_id): config_(MemConfig), rank_activate_recorder(Rank_Activate_Recorder), rank_id(Rank_id), bankgroup_id(BankGroup_id) {
    to_do_read = 0;
    to_do_write = 0;
    to_do_activate = 0;
    to_do_precharge = 0;
    for (auto i = 0; i < config_.banks_per_group; i++) {
        dram_banks.push_back(new DRAMBank(config_, rank_activate_recorder, rank_id, bankgroup_id, i));
    }
}

/*
uint64_t DRAMBankGroup::IssueTransactionToBank(std::shared_ptr<Memory_Transaction> trans) {
    // 在BankGroup不产生阻塞的情况下，向对应的Bank发送
    uint64_t trans_processing_cycle = 0;
    if (trans->is_write) {
        trans->processing_cycle = trans->processing_cycle > to_do_write ? trans->processing_cycle : to_do_write;
        trans_processing_cycle = dram_banks[trans->address.bank]->ProcessTransaction(trans);
        to_do_read = trans_processing_cycle + config_.same_bankgroup_read_to_read_latency;
        to_do_write = trans_processing_cycle + config_.same_bankgroup_read_to_write_latency;
    }
    else {
        trans->processing_cycle = trans->processing_cycle > to_do_read ? trans->processing_cycle : to_do_read;
        trans_processing_cycle = dram_banks[trans->address.bank]->ProcessTransaction(trans);
        to_do_read = trans_processing_cycle + config_.same_bankgroup_write_to_read_latency;
        to_do_write = trans_processing_cycle + config_.same_bankgroup_write_to_write_latency;
    }
    return trans_processing_cycle;

    uint64_t trans_processing_cycle = 0;
    if (trans->is_write) {
        trans->processing_cycle = std::max(trans->processing_cycle, to_do_write);
        trans_processing_cycle = dram_banks[trans->address.bank]->ProcessTransaction(trans);
        // 写完之后，更新写到读和写到写的约束
        to_do_read = trans_processing_cycle + config_.same_bankgroup_write_to_read_latency;
        to_do_write = trans_processing_cycle + config_.same_bankgroup_write_to_write_latency;
    }
    else {
        trans->processing_cycle = std::max(trans->processing_cycle, to_do_read);
        trans_processing_cycle = dram_banks[trans->address.bank]->ProcessTransaction(trans);
        // 读完之后，更新读到读和读到写的约束
        to_do_read = trans_processing_cycle + config_.same_bankgroup_read_to_read_latency;
        to_do_write = trans_processing_cycle + config_.same_bankgroup_read_to_write_latency;
    }
    return trans_processing_cycle;
}
*/

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
