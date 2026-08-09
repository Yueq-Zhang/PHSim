#ifndef MEMORY_SYSTEM_HPP
#define MEMORY_SYSTEM_HPP

// construct a memory system with the Memory Controller and Memory structure

#include <iostream>
#include "../common_function.hpp"
#include "dram_channel.hpp"
// #include "memory_transaction.hpp"
// #include "PIM_unit.hpp"

class MemorySystem {
public:
    MemorySystem(const MemConfig& mem_config, int mem_id);
    virtual ~MemorySystem();
    int GetChannel(uint64_t hex_addr) const; // 基于输入的transaction地址，将其输送至对应的channel中
    //bool AddTransaction(std::shared_ptr<Memory_Transaction> trans);      // 输入的transaction，对其进行处理，之后完成对于下一级的分发。
    // TransactionStatus AddTransaction(std::shared_ptr<Memory_Transaction> trans);
    // TransactionStatus AddTransaction(std::shared_ptr<Memory_Transaction> trans, uint32_t cid);
    // bool AddPIMTransaction(std::shared_ptr<PIM_Transaction> trans);

    void CheckRowSwitch(std::shared_ptr<Event> event);  // 检查是否需要完成换行, 同时将event放置在相应的位置处

    void PendingPrecharge(std::shared_ptr<Event> event);
    void PendingActivate(std::shared_ptr<Event> event);
    void PendingExecuted(std::shared_ptr<Event> event);

    // DRAM 整体架构的设计，包括channel，rank, bank_group, bank
    // 运行过程中，一个输入的transaction基于操作与地址信息进行解码，发送到对应的Channel Rank Bank 的Queue，直接更新bank的行为，按照层级更新行为，从controller开始，逐级向下传递，更新时间戳，最终得到一个事件完成的时间。
    // 因此只在被唤醒的情况下更新对应器件的内容，如果没有被唤醒，则不需要发生操作。刷新操作是自动进行的。
    MemConfig config_;
    std::vector<DRAMChannel*> dram_channels;  // DRAM controller for each channel
    int memory_id_;
    uint64_t last_transaction_clk_;
    // std::vector<std::shared_ptr<Memory_Transaction>> transaction_queue;
};

#endif