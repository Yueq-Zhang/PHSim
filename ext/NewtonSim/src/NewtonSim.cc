#include "NewtonSim.h"

#include "common.h"
#include "configuration.h"
#include "dram_system.h"
#include "../../../src/common_function.hpp"

namespace dramsim3 {
NewtonSim::NewtonSim(const std::string &config_file, const std::string &output_dir)
    : config_(new Config(config_file, output_dir)) {

    std::function<void(uint64_t)> read_callback = [&](uint64_t addr) {
        // PrintInfo("(NewtonSim) read_callback");
        int channel = GetChannel(addr);
        assert(pending_read_q_.count(addr) > 0);
        if (pending_read_q_.count(addr) == 0)
            exit(1);
        auto it = pending_read_q_.find(addr);       // search

        auto memory_req = (MemoryAccess *)it->second;
        memory_req->dram_finish_cycle = dram_system_->clk_;

        response_queues_[channel].push(memory_req); // push
        pending_read_q_.erase(it);                  // pop
        // spdlog::info("*** (Cycle Accurate DRAM) read_callback of address {} for channel {} at cycle {}",addr, channel, memory_req->dram_finish_cycle);
    };

    std::function<void(uint64_t)> write_callback = [&](uint64_t addr) {
        // PrintInfo("(NewtonSim) write_callback");
        int channel = GetChannel(addr);
        assert(pending_write_q_.count(addr) > 0); // assert does not work
        if (pending_write_q_.count(addr) == 0)
            exit(1);

        auto it = pending_write_q_.find(addr);      // search

        auto memory_req = (MemoryAccess *)it->second;
        memory_req->dram_finish_cycle = dram_system_->clk_;

        response_queues_[channel].push(it->second); // push
        pending_write_q_.erase(it);                 // pop
        // spdlog::info("*** (Cycle Accurate DRAM) write_callback of address {} for channel {} at cycle {}", addr, channel, memory_req->dram_finish_cycle);
    };

    std::function<void(uint64_t)> pim_callback = [&](uint64_t addr) {
        int channel = GetChannel(addr);
        assert(pending_pim_q_.count(addr) > 0);      // assert does not work
        if (pending_pim_q_.count(addr) == 0)
            exit(1);

        auto it = pending_pim_q_.find(addr);         // search
        auto memory_req = (MemoryAccess *)it->second;
        memory_req->dram_finish_cycle = dram_system_->clk_;
        response_queues_[channel].push(it->second);  // push
        pending_pim_q_.erase(it);                  // pop
    };

    dram_system_ = new JedecDRAMSystem(*config_, output_dir, read_callback, write_callback, pim_callback);

    // printf("Newtonsim: # of channel= %d\n", config_->channels);
    // # channels = # reponse queues,
    // size of response queue = trans_queue_size
    int res_q_size;

    if (config_->memory_type == MemoryType::DRAM) {
        res_q_size = config_->trans_queue_size * 3;
    } else
        res_q_size = config_->trans_queue_size; // original: config_->trans_queue_size
    // int res_q_size = config_->trans_queue_size * 3;
    for (int ch = 0; ch < config_->channels; ++ch) {
        response_queues_.push_back(ResponseQueue(res_q_size));
    }
    // for (int ch = 0; ch < config_->channels; ++ch) {
    //     response_queues_rd_.push_back(ResponseQueue(res_q_size));
    //     response_queues_wr_.push_back(ResponseQueue(res_q_size));
    //     response_queues_pim_.push_back(ResponseQueue(res_q_size));
    // }
}

uint64_t NewtonSim::GetAvgPIMCycles() { return dram_system_->GetAvgPIMCycles(); }
void NewtonSim::ResetPIMCycle() { dram_system_->ResetPIMCycle(); }
uint64_t NewtonSim::GetCounter(uint32_t channel,
                               const std::string& name) const {
    return dram_system_->GetCounter(channel, name);
}
void NewtonSim::AddCounter(uint32_t channel, const std::string& name,
                           uint64_t value) {
    dram_system_->AddCounter(channel, name, value);
}
uint64_t NewtonSim::GetVecCounter(uint32_t channel,
                                  const std::string& name, int pos) const {
    return dram_system_->GetVecCounter(channel, name, pos);
}
void NewtonSim::AddVecCounter(uint32_t channel, const std::string& name,
                              int pos, uint64_t value) {
    dram_system_->AddVecCounter(channel, name, pos, value);
}

NewtonSim::~NewtonSim() {
    // std::cout << "NewtonSim delete" << std::endl;
    delete (dram_system_);
    delete (config_);
}

void NewtonSim::ClockTick() {
    dram_system_->ClockTick();
}

double NewtonSim::GetTCK() const { return config_->tCK; }

int NewtonSim::GetBusBits() const { return config_->bus_width; }

int NewtonSim::GetBurstLength() const { return config_->BL; }

int NewtonSim::GetBurstCycle() const { return config_->burst_cycle; }

int NewtonSim::GetQueueSize() const {
    exit(-1);
    // unused method
    return config_->trans_queue_size;
}

int NewtonSim::GetChannel(uint64_t hex_addr) const { return dram_system_->GetChannel(hex_addr); };

uint64_t NewtonSim::MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col) {
    return config_->MakeAddress(channel, rank, bankgroup, bank, row, col);
}
uint64_t NewtonSim::EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps,
                                    int num_readres) {
    return config_->EncodePIMHeader(channel, row, for_gwrite, num_comps, num_readres);
}

bool NewtonSim::WillAcceptTransaction(uint64_t hex_addr, int req_type) const {
    TransactionType type = static_cast<TransactionType>(req_type);
    // check response queue size
    uint32_t channel = GetChannel(hex_addr);
    auto my_channel = MyAddressAllocator::get_channel_index(hex_addr);
    assert(channel == my_channel);
    bool available = response_queues_[channel].isAvailable(1);
    /*
    if (req_type == (int)TransactionType::P_HEADER)
        available = true;
    */
    // std::cout << "NewtonSim::WillAcceptTransaction(" << channel << ") : "
    //           << std::to_string(
    //                  available &&
    //                  dram_system_->WillAcceptTransaction(hex_addr, type))
    //           << std::endl;
    return available && dram_system_->WillAcceptTransaction(hex_addr, type);
}

bool NewtonSim::AddTransaction(uint64_t hex_addr, int req_type,
                               void *original_req, uint32_t core_id) {
    // Add the Memory Transaction to the corresponding DRAM channel and place it inside the response queue
    TransactionType type = static_cast<TransactionType>(req_type);
    int channel = GetChannel(hex_addr);
    income_req_cnt_++;

    if (type == TransactionType::P_HEADER || type == TransactionType::GWRITE || type == TransactionType::COMP || type == TransactionType::COMP_HASH || type == TransactionType::READRES) {
        response_queues_[channel].reserve();
        PushToPendingQueue(hex_addr, type, original_req);
        // spdlog::info("A PIM Transaction: {} is add to channel {} at DRAM system clk {}", TransactionTypeString(type), channel, dram_system_->clk_);
        return dram_system_->AddTransaction(hex_addr, type, core_id);
    }
    else {
        response_queues_[channel].reserve();
        PushToPendingQueue(hex_addr, type, original_req);
        return dram_system_->AddTransaction(hex_addr, type, core_id);
    }
}

uint64_t NewtonSim::GetCoreCommandCounter(
    uint32_t core_id, const std::string& name) const {
    return dram_system_->GetCoreCommandCounter(core_id, name);
}

void NewtonSim::PushToPendingQueue(uint64_t addr, TransactionType req_type, void *original_req) {
    switch (req_type) {
        case TransactionType::P_HEADER:
        case TransactionType::GWRITE:  // Write to the global input buffer.
        case TransactionType::COMP:  // comp has the data amount of one read burst for all bank in parallel
        case TransactionType::COMP_HASH:
        case TransactionType::READRES:
        case TransactionType::COMPS_READRES:
            pending_pim_q_.insert(std::make_pair(addr, original_req));
            break;
        case TransactionType::READ:
            pending_read_q_.insert(std::make_pair(addr, original_req));
            break;
        case TransactionType::WRITE:
            pending_write_q_.insert(std::make_pair(addr, original_req));
            break;
        default:
            break;
    }
    return;
}

bool NewtonSim::IsEmpty(uint32_t channel) const {
    bool empty = response_queues_[channel].isEmpty();
    // std::cout << "NewtonSim::IsEmpty(" << channel
    //           << ") : " << std::to_string(empty) << std::endl;
    return empty;
};
void *NewtonSim::Top(uint32_t channel) const {
    // printf("TOP channel= %d\n", channel);
    return response_queues_[channel].top();
};
void NewtonSim::Pop(uint32_t channel) {
    // auto finished_memory_transaction = (MemoryAccess *)response_queues_[channel].top();
    /*
    spdlog::info("Current {} memory Transaction Generate time is {}, Transaction enter DRAM time is {}, Transaction Finish time is {}",
        memAccessTypeString(finished_memory_transaction->req_type),finished_memory_transaction->start_cycle, finished_memory_transaction->dram_enter_cycle, finished_memory_transaction->dram_finish_cycle);
    */
    /*
    if (finished_memory_transaction->req_type != MemoryAccessType::WRITE && finished_memory_transaction->dram_finish_cycle > last_response_time) {
        assert(finished_memory_transaction->dram_finish_cycle - last_response_time >= config_->burst_cycle);
        // spdlog::info("Transaction interval is {}", finished_memory_transaction->dram_finish_cycle - last_response_time);
        last_response_time = finished_memory_transaction->dram_finish_cycle;
    }
    else if (finished_memory_transaction->req_type != MemoryAccessType::WRITE && finished_memory_transaction->dram_finish_cycle < last_response_time){
        // assert(0);
    }
    */
    /*
    if (finished_memory_transaction->req_type == MemoryAccessType::P_HEADER) {
        spdlog::info("PIM Memory Request P_HEADER finished");
    }
    else if (finished_memory_transaction->req_type == MemoryAccessType::GWRITE) {
        spdlog::info("PIM Memory Request GWRITE finished");
    }
    else if (finished_memory_transaction->req_type == MemoryAccessType::COMP) {
        spdlog::info("PIM Memory Request COMP finished");
    }
    else if (finished_memory_transaction->req_type == MemoryAccessType::READRES) {
        spdlog::info("PIM Memory Request READRES finished");
    }
    */
    /*
    spdlog::info("The Memory transaction is finished at {} in channel {} with address {}, added at dram cycle {}", finished_memory_transaction->dram_finish_cycle,
        channel, finished_memory_transaction->dram_address, finished_memory_transaction->dram_enter_cycle);

    assert(channel == MyAddressAllocator::get_channel_index(finished_memory_transaction->dram_address));

    spdlog::info("The finished address is for channel {}, Rank {}, BankGroup {}, Bank {}, Row {}, Column {}",
        channel,
        MyAddressAllocator::get_rank_index(finished_memory_transaction->dram_address),
        MyAddressAllocator::get_bankgroup_index(finished_memory_transaction->dram_address),
        MyAddressAllocator::get_bank_index(finished_memory_transaction->dram_address),
        MyAddressAllocator::get_row_index(finished_memory_transaction->dram_address),
        MyAddressAllocator::get_col_index(finished_memory_transaction->dram_address)
    );
    */
    auto response_transaction = (MemoryAccess *)response_queues_[channel].top();
    response_queues_[channel].pop();
    // spdlog::info("(CycleAccurate DRAM) ResponseQueue : At Cycle {}, channel {} pop transaction {}, {} exist",
    //    dram_system_->clk_, channel, response_transaction->dram_address,response_queues_[channel].NumReserved);
    outcome_req_cnt_++;
};

void NewtonSim::PrintStats() const { dram_system_->PrintStats(); }

void NewtonSim::ResetStats() { dram_system_->ResetStats(); }

NewtonSim::ResponseQueue::ResponseQueue(int Size) : Size(Size), NumReserved(0) {}

bool NewtonSim::ResponseQueue::isAvailable() const {
    return NumReserved + OutputQueue.size() < Size;
}

bool NewtonSim::ResponseQueue::isAvailable(uint32_t count) const {
    return NumReserved + OutputQueue.size() + count - 1 < Size;
}

void NewtonSim::ResponseQueue::reserve() {
    assert(NumReserved < Size);
    NumReserved++;
}

void NewtonSim::ResponseQueue::push(void *original_req) {
    // std::cout << "ResponseQueue::push" << std::endl;
    OutputQueue.push_back(original_req);
    assert(NumReserved > 0);
    NumReserved--;
    // spdlog::info("(Cycle Accurate DRAM) Response queue push, current ReservedNumber is {} ", NumReserved);
}

bool NewtonSim::ResponseQueue::isEmpty() const { return OutputQueue.empty(); }

void NewtonSim::ResponseQueue::pop() {
    auto it = OutputQueue.begin();
    OutputQueue.erase(it);
}
void *NewtonSim::ResponseQueue::top() const { return OutputQueue.front(); }

} // namespace dramsim3

void dramsim3::NewtonSim::receive_predicting_config(size_t unstable_length, size_t sample_length, double inst_ratio) {
    _unstable_length = unstable_length;
    _sample_length = sample_length;
    _inst_ratio = inst_ratio;
}
