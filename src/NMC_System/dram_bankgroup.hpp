#ifndef DRAM_BANKGROUP_HPP
#define DRAM_BANKGROUP_HPP

#include "dram_bank.hpp"

class DRAMBankGroup {
public:
    DRAMBankGroup(MemConfig& MemConfig, std::vector<std::pair<int, uint64_t>>* Rank_Activate_Recorder, int Rank_id, int BankGroup_id);
    // uint64_t IssueTransactionToBank(std::shared_ptr<Memory_Transaction> trans);

    void SameBankgroupPrecharge(std::shared_ptr<Event> event);

    void SameBankgroupActivate(std::shared_ptr<Event> event);
    void OtherBankgroupActivate(std::shared_ptr<Event> event);

    void SameBankgroupRead(std::shared_ptr<Event> event);
    void OtherBankgroupRead(std::shared_ptr<Event> event);

    int rank_id, bankgroup_id;

    MemConfig& config_;
    std::vector<DRAMBank*> dram_banks;
    std::vector<std::pair<int, uint64_t>>* rank_activate_recorder;

    uint64_t to_do_read;
    uint64_t to_do_write;
    uint64_t to_do_precharge;
    uint64_t to_do_activate;



};


#endif
