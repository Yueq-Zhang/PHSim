#include "newton_controller.h"

#include <iomanip>
#include <iostream>
#include <limits>

#include "../../../src/common_function.hpp"
#include "../../../src/DRAM/Dram.h"


namespace dramsim3 {

NewtonController::NewtonController(int channel, const Config &config, Timing &timing):
    channel_id_(channel), clk_(0), config_(config), simple_stats_(config_, channel_id_), channel_state_(channel, config, timing),
    pim_cmd_queue_(channel_id_, config, channel_state_, simple_stats_), refresh_(config, channel_state_, simple_stats_),
    last_issue_clk_(0), row_buf_policy_(config.row_buf_policy == "CLOSE_PAGE" ? RowBufPolicy::CLOSE_PAGE : RowBufPolicy::OPEN_PAGE),
    last_trans_clk_(0),  write_draining_(0) {

    read_queue_.reserve(config_.trans_queue_size);
    write_buffer_.reserve(config_.trans_queue_size); // BUG: increase write_buffer size

    pim_queue_.reserve(config_.trans_queue_size);

#if ENABLE_DRAM_ALIGNMENT_TRACE
    read_merge_trace_.open(config_.output_dir + "read_merge_cycleaccurate_ch" +
                           std::to_string(channel_id_) + ".csv",
                           std::ofstream::out | std::ofstream::trunc);
    if (read_merge_trace_.is_open()) {
        read_merge_trace_ << "sequence,channel,merge_cycle,address,first_added_cycle,age_cycles,pending_before\n";
    }
    if (::Config::system_config.record_dram_completion_trace) {
        command_trace_.open(config_.output_dir + "dram_command_cycleaccurate_ch" +
                            std::to_string(channel_id_) + ".csv",
                            std::ofstream::out | std::ofstream::trunc);
        if (command_trace_.is_open()) {
            command_trace_ << "command_sequence,channel,cycle,command,address,rank,bankgroup,bank,row,col,open_row_before\n";
        }
        transaction_trace_.open(config_.output_dir + "dram_transaction_cycleaccurate_ch" +
                                std::to_string(channel_id_) + ".csv",
                                std::ofstream::out | std::ofstream::trunc);
        if (transaction_trace_.is_open()) {
            transaction_trace_ << "sequence,arrival_sequence,channel,cycle,action,type,address,detail\n";
        }
    }
#endif

    rw_dependency_lock_ = false;
    rw_dependency_addr_ = 0;

    comps_per_pim_row_ = 0;
}

// stat for pim utilization
void NewtonController::ResetPIMCycle() { pim_cmd_queue_.ResetPIMCycle(); }
uint64_t NewtonController::GetPIMCycle() { return pim_cmd_queue_.GetPIMCycle(); }

// - [x] handle pim command
std::pair<uint64_t, TransactionType> NewtonController::ReturnDoneTrans(uint64_t clk) {
    auto it = return_queue_.begin();
    while (it != return_queue_.end()) {
        if (clk >= it->complete_cycle) {
            if (it->is_write()) {
                simple_stats_.Increment("num_writes_done");
            } else if (it->is_read()) {
                PrintTransactionLog("ReturnDoneRead", channel_id_, clk_, *it);
                simple_stats_.Increment("num_reads_done");
                simple_stats_.AddValue("read_latency", clk_ - it->added_cycle);
            } else if (it->req_type == TransactionType::P_HEADER) {
               // spdlog::info("channel {}, P_HEADER done at time {}, p_header_latency {}", channel_id_, clk_, clk_ - it->added_cycle); // PrintInfo("cid:", channel_id_, "P_HEADER done, p_header_latency:", clk_ - it->added_cycle);
               // simple_stats_.AddValue("num_pheader_cmds", clk_ - it->added_cycle);
                simple_stats_.Increment("num_pheader_cmds");
                simple_stats_.Increment("num_pim_cmds");
            } else if (it->req_type == TransactionType::GWRITE) {
                // pim_cmd_queue_.FinishGwrite();
                // spdlog::info("channel {}, GWRITE done at time {}, gwrite_latency {}", channel_id_, clk_, clk_ - it->added_cycle); // PrintInfo("cid:", channel_id_, "GWRITE done, gwrite_latency:", clk_ - it->added_cycle);
                // simple_stats_.AddValue("gwrite_latency", clk_ - it->added_cycle);
                simple_stats_.Increment("num_pim_cmds");
            } else if (it->req_type == TransactionType::COMP || it->req_type == TransactionType::COMP_HASH) {
                // spdlog::info("channel {}, COMP done at time {}, comp_latency {}", channel_id_, clk_, clk_ - it->added_cycle); // PrintInfo("COMP done, cid:", channel_id_);
                simple_stats_.Increment("num_pim_cmds");
            } else if (it->req_type == TransactionType::READRES) {
                // spdlog::info("channel {}, READRES done at time {}, readers_latency {}", channel_id_, clk_, clk_ - it->added_cycle); //PrintInfo("READRES done, cid:", channel_id_); //PrintInfo("READRES done, cid:", channel_id_);
                simple_stats_.Increment("num_pim_cmds");
            }

            // Erase the computation result based on address, for PIM operation strictly follows the order, just remove the first operation
#if ENABLE_DRAM_ALIGNMENT_TRACE
            TraceTransaction("COMPLETE", *it);
#endif
            auto pair = std::make_pair(it->addr, it->req_type);
            it = return_queue_.erase(it);
            // spdlog::info("{} Transaction for address {} finished at time {}", it->TransactionTypeString(), it->addr, clk);
            return pair;
        }
        else {
            ++it;  //
        }
    }
    return std::make_pair(-1, TransactionType::SIZE);
}

void NewtonController::ClockTick() {
    // update refresh counter
    refresh_.ClockTick();
    bool cmd_issued = false;

    Command cmd;
    if (channel_state_.IsRefreshWaiting()) {
        PrintColor(Color::RED, "Refresh Waiting.., clk:", clk_);
        cmd = pim_cmd_queue_.FinishRefresh();
    }

    if (!cmd.IsValid()) {
        std::pair<int, int> refresh_slack = refresh_.GetRefreshSlack();
        cmd = pim_cmd_queue_.GetCommandToIssue(refresh_slack);
    }

    if (cmd.IsValid()) {
        IssueCommand(cmd);
        cmd_issued = true;
    }

    // power updates pt 1
    for (int i = 0; i < config_.ranks; i++) {
        if (channel_state_.IsRankSelfRefreshing(i)) {
            simple_stats_.IncrementVec("sref_cycles", i);
        } else {
            bool all_idle = channel_state_.IsAllBankIdleInRank(i);
            if (all_idle) {
                simple_stats_.IncrementVec("all_bank_idle_cycles", i);
                channel_state_.rank_idle_cycles[i] += 1;
            } else {
                simple_stats_.IncrementVec("rank_active_cycles", i);
                // reset
                channel_state_.rank_idle_cycles[i] = 0;
            }
        }
        if (config_.memory_type != MemoryType::DRAM) {
            if (channel_state_.IsPIMIdleInRank(i)) {
                simple_stats_.IncrementVec("pim_all_bank_idle_cycles", i);
            } else {
                simple_stats_.IncrementVec("pim_rank_active_cycles", i);
            }
        }
    }

    // power updates pt 2: move idle ranks into self-refresh mode to save power
    if (config_.enable_self_refresh && !cmd_issued) {
        for (auto i = 0; i < config_.ranks; i++) {
            if (channel_state_.IsRankSelfRefreshing(i)) {
                // wake up!
                if (!pim_cmd_queue_.rank_q_empty[i]) {
                    auto addr = Address();
                    addr.rank = i;
                    auto cmd = Command(CommandType::SREF_EXIT, addr, -1);
                    cmd = channel_state_.GetReadyCommand(cmd, clk_);
                    if (cmd.IsValid()) {
                        IssueCommand(cmd);
                        break;
                    }
                }
            } else {
                if (pim_cmd_queue_.rank_q_empty[i] && channel_state_.rank_idle_cycles[i] >= config_.sref_threshold) {
                    auto addr = Address();
                    addr.rank = i;
                    auto cmd = Command(CommandType::SREF_ENTER, addr, -1);
                    cmd = channel_state_.GetReadyCommand(cmd, clk_);
                    if (cmd.IsValid()) {
                        IssueCommand(cmd);
                        break;
                    }
                }
            }
        }
    }

    ScheduleTransaction();
    clk_++;
    pim_cmd_queue_.ClockTick();
    simple_stats_.Increment("num_cycles");

    //>>> gsheo: for debug (to fix infinite loop)
    int interval = 20;
    // channel_id_ == TROUBLE_CHANNEL
    if (clk_ % interval == 0 && LOGGING_CONFIG::STATUS_CHECK) {
        if (LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE) {
            if (channel_id_ != LOGGING_CONFIG::TROUBLE_CHANNEL)
                return;
        }
        PrintDebug("-------NewtonSim Status Check (cid:", channel_id_, ")-------");

        bool clean_related_read =
            read_queue_.empty() && pending_rd_q_.empty() && pim_cmd_queue_.QueueEmpty();
        bool clean_related_write =
            write_buffer_.empty() && pending_wr_q_.empty() && pim_cmd_queue_.QueueEmpty();
        bool clean_related_pim =
            read_queue_.empty() && pending_rd_q_.empty() && pim_cmd_queue_.QueueEmpty(-1);

        if (clean_related_read && clean_related_write && clean_related_pim) {
            PrintDebug("All queue empty!");
            return;
        }
        PrintDebug("clk:", clk_);
        PrintDebug("pim_queue.size:", pim_queue_.size());
        PrintDebug("pim_command_queue size:", pim_cmd_queue_.GetPIMQueueSize());
        PrintDebug("read_queue.size:", read_queue_.size());
        PrintDebug("write_buffer.size:", write_buffer_.size());
        PrintDebug("pending_rd_q_", pending_rd_q_.size());
        PrintDebug("pending_wr_q_", pending_wr_q_.size());
        PrintDebug("pending_pim_q_", pending_pim_q_.size());
        pim_cmd_queue_.PrintAllQueue();
        if (write_buffer_.empty() && !pending_wr_q_.empty() && pim_cmd_queue_.QueueEmpty()) {
            PrintDebug("Something wrong!!");
            PrintDebug("write_buffer is empty, but pending_wr_q_ is not empty");
            // show content:
            // for (auto it = pending_wr_q_.begin(); it != pending_wr_q_.end(); ++it) {
            //     std::cout << "addr:" << (*it).first
            //               << " => type:" << (*it).second.TransactionTypeString() << '\n';
            // }
        }
        if (read_queue_.empty() && !pending_rd_q_.empty() && pim_cmd_queue_.QueueEmpty()) {
            PrintDebug("Something wrong!!");
            PrintDebug("read_queue is empty, but pending_rd_q_ is not empty");
            // show content:
            // for (auto it = pending_rd_q_.begin(); it != pending_rd_q_.end(); ++it) {
            //     std::cout << "addr:" << (*it).first
            //               << " => type:" << (*it).second.TransactionTypeString() << '\n';
            // }
        }
    }
    // <<< gsheo

    // >>> gsheo: for debug (read nothing else, during pim computation)
    // if (clk_ == 22750) {
    //     LOGGING_CONFIG::STATUS_CHECK = true;
    //     LOGGING_CONFIG::PIMSIM_LOGGING = false;
    //     LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE = false;
    //     LOGGING_CONFIG::TROUBLE_CHANNEL = 0;
    // }
    // if (clk_ == 28340) {
    //     LOGGING_CONFIG::STATUS_CHECK = false;
    //     LOGGING_CONFIG::PIMSIM_LOGGING = false;
    //     LOGGING_CONFIG::LOGGING_ONLY_TROUBLE_ZONE = false;
    //     // exit(0);
    // }

    // <<< gsheo

    return;
}

bool NewtonController::WillAcceptTransaction(uint64_t hex_addr, TransactionType req_type) {
    bool is_write = req_type == TransactionType::WRITE;
    // bool is_read = req_type == TransactionType::READ;

    if (is_write) {
        return write_buffer_.size() < write_buffer_.capacity();
    } else {
        return read_queue_.size() < read_queue_.capacity();
    }
}

bool NewtonController::AddTransaction(Transaction trans) {
    trans.added_cycle = clk_;
#if ENABLE_DRAM_ALIGNMENT_TRACE
    TraceTransaction("ARRIVE", trans);
#endif
    simple_stats_.AddValue("interarrival_latency", clk_ - last_trans_clk_);
    last_trans_clk_ = clk_;

    if (trans.is_write()) {
        PrintTransactionLog("AddTransaction(WR)", channel_id_, clk_, trans);
        if (pending_wr_q_.count(trans.addr) == 0) { // can not merge writes
            pending_wr_q_.insert(std::make_pair(trans.addr, trans));
            write_buffer_.push_back(trans);
        }
        else {
            // The logical write still receives its one-cycle response, but no
            // additional physical WRITE transaction is created.
            simple_stats_.Increment("num_write_buf_hits");
#if ENABLE_DRAM_ALIGNMENT_TRACE
            TraceTransaction("MERGED_WRITE", trans);
#endif
        }
        trans.complete_cycle = clk_ + 1;
        return_queue_.push_back(trans);
        return true;
    }
    else if (trans.is_read()) {
        // if in write buffer, use the write buffer value
        // >> gsheo: debug
        PrintTransactionLog("AddTransaction(RD)", channel_id_, clk_, trans);
        // << debug
        if (pending_wr_q_.count(trans.addr) > 0) {
            trans.complete_cycle = clk_ + 1;
            return_queue_.push_back(trans);
#if ENABLE_DRAM_ALIGNMENT_TRACE
            TraceTransaction("FORWARDED_READ", trans);
#endif
            return true;
        }

        const auto pending_before = pending_rd_q_.count(trans.addr);
#if ENABLE_DRAM_ALIGNMENT_TRACE
        uint64_t first_added_cycle = clk_;
        if (pending_before > 0) {
            first_added_cycle = pending_rd_q_.equal_range(trans.addr).first->second.added_cycle;
        }
#endif
        pending_rd_q_.insert(std::make_pair(trans.addr, trans));
#if ENABLE_DRAM_ALIGNMENT_TRACE
        if (pending_before > 0 && read_merge_trace_.is_open()) {
            read_merge_trace_ << read_merge_sequence_++ << ','
                              << channel_id_ << ','
                              << clk_ << ','
                              << trans.addr << ','
                              << first_added_cycle << ','
                              << (clk_ - first_added_cycle) << ','
                              << pending_before << '\n';
        }
#endif
        if (pending_before > 0) {
#if ENABLE_DRAM_ALIGNMENT_TRACE
            TraceTransaction("MERGED_READ", trans);
#endif
        }
        if (pending_rd_q_.count(trans.addr) == 1) {
            read_queue_.push_back(trans);
        }
        return true;
    }
    else {
        pending_pim_q_.insert(std::make_pair(trans.addr, trans));   
        pim_queue_.push_back(trans);
        // spdlog::info("A PIM transaction {} is insert into pending_pim_queue and pim_queue_ of channel {}", trans.TransactionTypeString(), channel_id_);
        return true;
    }
}

void NewtonController::PrintTransactionQueue() const {
    auto queue = read_queue_;
    std::string commands = "";
    for (auto it = queue.begin(); it != queue.end(); ++it) {
        commands += it->TransactionTypeString() + " ";
    }
    PrintWarning("READ trans_q (cid:", channel_id_, "):", commands);

    commands = "";
    for (auto it = write_buffer_.begin(); it != write_buffer_.end(); ++it) {
        commands += it->TransactionTypeString() + " ";
    }
    PrintWarning("WRITE trans_q (cid:", channel_id_, "):", commands);
}

void NewtonController::ScheduleTransaction() {
    if (!rw_dependency_lock_ && write_draining_ == 0) {
        // we basically have a upper and lower threshold for write buffer
        if ((write_buffer_.size() >= write_buffer_.capacity()) ||
            (write_buffer_.size() > 8 && pim_cmd_queue_.QueueEmpty()) ||
            (write_buffer_.size() > 0 && pim_queue_.size() > 0)) {
            // write_buffer is full or there are transactions more than 8 and cmd_queue is empty
            write_draining_ = write_buffer_.size();
        }
    }

    // after all the read-write complete, execute PIM
    int pim_q_size = pim_cmd_queue_.GetPIMQueueSize();

    // Transaction queues
    enum QueueToSchedule {READ_Q, WRITE_BUFFER, PIM, SIZE };
    QueueToSchedule queue_to_schedule = SIZE;

    if (rw_dependency_lock_)
        queue_to_schedule = READ_Q;
    else if (write_draining_ > 0)
        queue_to_schedule = WRITE_BUFFER;
    else if (pim_queue_.size() > 0 && read_queue_.size() == 0 && write_buffer_.size() == 0) {
        queue_to_schedule = PIM;
    }
    else
        queue_to_schedule = READ_Q;

    assert(queue_to_schedule != SIZE);

    // Select the Schedule queue
    std::vector<Transaction> &queue = (queue_to_schedule == READ_Q) ? read_queue_ :
                                      (queue_to_schedule == PIM) ? pim_queue_ :
                                      write_buffer_;

    // std::vector<Transaction> &queue = queue_to_schedule == READ_Q ? read_queue_ : write_buffer_;
    // The transaction queue maybe the read PIM or write
    for (auto it = queue.begin(); it != queue.end(); it++) {
        auto cmd = TransToCommand(*it);
        // check if current command convert from transaction can be accepted
        bool will_accept_command = false;
        if (cmd.PIMQCommand()) { // if PIM Transaction, check the PIM Queue Capacity
            will_accept_command = pim_cmd_queue_.WillAcceptPIMCommand();
        }
        // int rank = cmd.IsGwrite() ? -1 : cmd.Rank();
        else {  // if not PIM Transaction, check the PIM Queue states
            will_accept_command = pim_cmd_queue_.WillAcceptCommand(cmd.Rank(), cmd.Bankgroup(), cmd.Bank());
        }

        if (will_accept_command) {
            if (cmd.IsWrite()) {
                // Enforce R->W dependency
                if (pending_rd_q_.count(it->addr) > 0) {
                    // if there is read transaction (it->addr),
                    // first push it
                    if (read_queue_.size() > 0) {
                        for (int i = 0; i < read_queue_.size(); i++) {
                            if (read_queue_[i].addr == it->addr) {
                                // PrintDebug("(ScheduleTransaction) R->W dependency:!", it->addr);
                                rw_dependency_lock_ = true;
                                rw_dependency_addr_ = it->addr;
                                break;
                            }
                        }
                    }
                    write_draining_ = 0;
                    break;
                }
                else if (pending_pim_q_.count(it->addr) > 0) {
                    auto pim_trans_ = pending_pim_q_.find(cmd.hex_addr);
                    if (pim_trans_->second.added_cycle < it->added_cycle) {
                        write_draining_ = 0;
                        break;
                    }
                }
                write_draining_ -= 1;
            }
            if (cmd.IsRead()) {
                if (rw_dependency_lock_ && rw_dependency_addr_ == cmd.hex_addr) {
                    // PrintDebug("(ScheduleTransaction) Solve R->W dependency:!", it->addr);
                    rw_dependency_addr_ = 0;
                    rw_dependency_lock_ = false;
                }
            }
            pim_cmd_queue_.AddCommand(cmd);
#if ENABLE_DRAM_ALIGNMENT_TRACE
            TraceTransaction("TO_COMMAND_QUEUE", *it,
                             cmd.CommandTypeString().c_str());
#endif
            // spdlog::info("(CycleAccurate DRAM) At cycle {}, A {} Transaction {} is convert to cmd, current transaction in the queue is {}",
            //     clk_, cmd.CommandTypeString(), cmd.hex_addr, queue.size());
            queue.erase(it);
            break;
        }
    }
}

void NewtonController::TraceTransaction(const char* action,
                                        const Transaction& trans,
                                        const char* detail) {
#if ENABLE_DRAM_ALIGNMENT_TRACE
    constexpr uint64_t kTransactionTraceLimit = 10000;
    if (!transaction_trace_.is_open() ||
        transaction_trace_sequence_ >= kTransactionTraceLimit) {
        return;
    }
    const bool is_arrival = std::string(action) == "ARRIVE";
    const uint64_t arrival_sequence = is_arrival
                                          ? transaction_arrival_sequence_++
                                          : std::numeric_limits<uint64_t>::max();
    transaction_trace_ << transaction_trace_sequence_++ << ',';
    if (is_arrival) {
        transaction_trace_ << arrival_sequence;
    }
    transaction_trace_ << ',' << channel_id_ << ',' << clk_ << ',' << action
                       << ',' << trans.TransactionTypeString() << ',' << trans.addr
                       << ',' << detail << '\n';
#else
    (void)action;
    (void)trans;
    (void)detail;
#endif
}

void NewtonController::IssueCommand(const Command &cmd) {
    // PrintControllerLog("IssueCommand", channel_id_, clk_, cmd);
    last_issue_clk_ = clk_;
#if ENABLE_DRAM_ALIGNMENT_TRACE
    constexpr uint64_t kCommandTraceLimit = 20000;
    if (command_trace_.is_open() && command_trace_sequence_ < kCommandTraceLimit) {
        int open_row_before = -2;
        if (cmd.Rank() >= 0 && cmd.Bankgroup() >= 0 && cmd.Bank() >= 0) {
            open_row_before = channel_state_.OpenRow(
                cmd.Rank(), cmd.Bankgroup(), cmd.Bank());
        }
        command_trace_ << command_trace_sequence_++ << ','
                       << channel_id_ << ',' << clk_ << ','
                       << cmd.CommandTypeString() << ',' << cmd.hex_addr << ','
                       << cmd.Rank() << ',' << cmd.Bankgroup() << ','
                       << cmd.Bank() << ',' << cmd.Row() << ','
                       << cmd.Column() << ',' << open_row_before << '\n';
    }
#endif
    // if read/write, update pending queue and return queue
    if (cmd.IsRead()) {
        auto num_reads = pending_rd_q_.count(cmd.hex_addr);
        if (num_reads == 0) {
            std::cerr << cmd.hex_addr << " not in read queue! " << std::endl;
            exit(1);
        }
        // if there are multiple reads pending return them all
        while (num_reads > 0) {
            auto it = pending_rd_q_.find(cmd.hex_addr);
            it->second.complete_cycle = clk_ + config_.read_delay;
            return_queue_.push_back(it->second);
            pending_rd_q_.erase(it);
            num_reads -= 1;
        }
    }
    else if (cmd.IsWrite()) {
        // there should be only 1 write to the same location at a time
        auto it = pending_wr_q_.find(cmd.hex_addr);
        if (it == pending_wr_q_.end()) {
            std::cerr << cmd.hex_addr << " not in write queue!" << std::endl;
            exit(1);
        }
        auto wr_lat = clk_ - it->second.added_cycle + config_.write_delay;
        simple_stats_.AddValue("write_latency", wr_lat);
        pending_wr_q_.erase(it);
    }
    else if (cmd.IsPIMHeader() || cmd.IsReadRes() || cmd.IsGwrite() || cmd.IsPIMComp()) {
        auto num_pending_pim_trans = pending_pim_q_.count(cmd.hex_addr); 
        if (num_pending_pim_trans == 0) {
            PrintError("cid:", channel_id_, "not in pending pim queue!", cmd.CommandTypeString(), "addr:", HexString(cmd.hex_addr));
        }
        auto it = pending_pim_q_.find(cmd.hex_addr);
        // readres delay == read delay
        if (cmd.IsPIMHeader())
            it->second.complete_cycle = clk_ + config_.p_header_delay;
        if (cmd.IsReadRes())
            it->second.complete_cycle = clk_ + config_.readers_delay;
        if (cmd.IsGwrite())
            it->second.complete_cycle = clk_ + config_.gwrite_delay;
        if (cmd.IsPIMComp())
            it->second.complete_cycle = clk_ + config_.comp_dealy;
        return_queue_.push_back(it->second);
        pending_pim_q_.erase(it);
    }
    else if (cmd.IsRefresh()) {
        /// spdlog::info("DRAM channel is refreshed");
    }
    /*
    else {
        throw std::runtime_error("UNKNOWN Command");
        // PrintInfo("START GEMV");
        // PrintError("you cannot issue PIM header command!");
        // return;
    }
    */
    /*
    if (cmd.cmd_type == CommandType::ACTIVATE) {
        spdlog::info("(CycleAccurate DRAM) with Channel {} issue an Activate command for address {} of Rank {}, Bankgroup {}, Bank {}, Row {} at cycle {}",
            channel_id_, cmd.hex_addr, cmd.Rank(), cmd.Bankgroup(), cmd.Bank(), cmd.Row(), clk_);
    }
    else if (cmd.cmd_type == CommandType::PRECHARGE) {
        spdlog::info("(CycleAccurate DRAM) with Channel {} issue an Precharge command for address {} of Rank {}, Bankgroup {}, Bank {} at cycle {}",
            channel_id_, cmd.hex_addr, cmd.Rank(), cmd.Bankgroup(), cmd.Bank(), clk_);
    }
    */
    // must update stats before states (for row hits)
    UpdateCommandStats(cmd);
    channel_state_.UpdateTimingAndStates(cmd, clk_);
}

// - [x] translate PIM transaction to command
Command NewtonController::TransToCommand(const Transaction &trans) {
    auto addr = config_.AddressMapping(trans.addr);
    CommandType cmd_type = CommandType::SIZE;
    if (row_buf_policy_ == RowBufPolicy::OPEN_PAGE) {
        switch (trans.req_type) {
        case TransactionType::READ:
            cmd_type = CommandType::READ;
            break;
        case TransactionType::WRITE:
            cmd_type = CommandType::WRITE;
            break;
        case TransactionType::GWRITE:
            cmd_type = CommandType::GWRITE;
            break;
        case TransactionType::COMP:
            /*
            addr.rank = -1;
            addr.bankgroup = -1;
            addr.bank = -1;
            */
            cmd_type = CommandType::COMP;
            break;
        case TransactionType::COMP_HASH:
            cmd_type = CommandType::COMP_HASH;
            // return Command(cmd_type, addr, trans.addr, trans.hash_comps);  // TODO: variable comp length
            break;
        case TransactionType::READRES:
            //return DecodePIMTransaction(trans);
            cmd_type = CommandType::READRES;
            break;
        case TransactionType::P_HEADER:
            cmd_type = CommandType::P_HEADER;
            break;
        default:
            break;
        }
    } else {
        cmd_type = trans.is_write() ? CommandType::WRITE_PRECHARGE : CommandType::READ_PRECHARGE;
    }
    assert(cmd_type != CommandType::SIZE);
    return Command(cmd_type, addr, trans.addr, trans.core_id);
}

Command NewtonController::DecodePIMTransaction(const Transaction &trans) {
    assert(trans.req_type == TransactionType::READRES);
    CommandType cmd_type = CommandType::READRES;

    auto addr = config_.AddressMapping(trans.addr);
    int num_comps = 0;

    num_comps += addr.rank * config_.bankgroups * config_.banks_per_group;
    num_comps += addr.bankgroup * config_.banks_per_group;
    num_comps += addr.bank;
    num_comps += 1;
    // we have only 5 bits usable,
    // encode (num_comps-1) to rabgba bit 

    addr.rank = -1;
    addr.bankgroup = -1;
    addr.bank = -1;
    bool is_last = addr.column == 1;

    // fix num_readres to 1
    return Command(cmd_type, addr, trans.addr, is_last, num_comps,
                   trans.core_id);
}

int NewtonController::QueueUsage() const { return pim_cmd_queue_.QueueUsage(); }

void NewtonController::PrintEpochStats() {
    simple_stats_.Increment("epoch_num");
    simple_stats_.PrintEpochStats();
}

void NewtonController::PrintFinalStats() {
    simple_stats_.PrintFinalStats();
}

void NewtonController::UpdateCommandStats(const Command &cmd) {
    const auto originating_core = [this, &cmd]() {
        const auto read_it = pending_rd_q_.find(cmd.hex_addr);
        if (read_it != pending_rd_q_.end()) return read_it->second.core_id;
        const auto write_it = pending_wr_q_.find(cmd.hex_addr);
        if (write_it != pending_wr_q_.end()) return write_it->second.core_id;
        const auto pim_it = pending_pim_q_.find(cmd.hex_addr);
        if (pim_it != pending_pim_q_.end()) return pim_it->second.core_id;
        return cmd.core_id;
    };
    switch (cmd.cmd_type) {
    case CommandType::READ:
    case CommandType::READ_PRECHARGE:
        simple_stats_.Increment("num_read_cmds");
        if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(), cmd.Bank()) != 0) {
            simple_stats_.Increment("num_read_row_hits");
        }
        break;
    case CommandType::WRITE:
    case CommandType::WRITE_PRECHARGE:
        simple_stats_.Increment("num_write_cmds");
        if (channel_state_.RowHitCount(cmd.Rank(), cmd.Bankgroup(), cmd.Bank()) != 0) {
            simple_stats_.Increment("num_write_row_hits");
        }
        break;
    case CommandType::ACTIVATE:
        simple_stats_.Increment("num_act_cmds");
        core_command_counters_[originating_core()]["num_act_cmds"]++;
        break;
    case CommandType::PRECHARGE:
        simple_stats_.Increment("num_pre_cmds");
        core_command_counters_[originating_core()]["num_pre_cmds"]++;
        break;
    case CommandType::REFRESH:
        simple_stats_.Increment("num_ref_cmds");
        break;
    case CommandType::REFRESH_BANK:
        simple_stats_.Increment("num_refb_cmds");
        break;
    case CommandType::SREF_ENTER:
        simple_stats_.Increment("num_srefe_cmds");
        break;
    case CommandType::SREF_EXIT:
        simple_stats_.Increment("num_srefx_cmds");
        break;
    case CommandType::GWRITE:
        simple_stats_.Increment("num_gwrite_cmds");
        break;
    case CommandType::G_ACT:
        simple_stats_.Increment("num_gact_cmds");
        break;
    case CommandType::COMP:
        simple_stats_.Increment("num_comp_cmds");
        comps_per_pim_row_++;
        break;
    case CommandType::COMP_HASH:
        simple_stats_.Increment("num_comp_cmds");
        break;
    case CommandType::READRES:
        simple_stats_.Increment("num_readres_cmds");
        break;
    case CommandType::PIM_PRECHARGE:
        simple_stats_.Increment("num_pim_precharge_cmds");
        if (comps_per_pim_row_ > 0) {
            // spdlog::info("current COMP number before PIM precharge is: {}", comps_per_pim_row_);
            // spdlog::info("Save for updating");
            channel_state_.UpdateCompsPerPimRow(comps_per_pim_row_);
            comps_per_pim_row_ = 0;
        }
        break;
    case CommandType::PIM_ACTIVE:
        simple_stats_.Increment("num_pim_activate_cmds");
        break;
    case CommandType::P_HEADER:
        simple_stats_.Increment("num_pim_precharge_cmds");
        break;
    default:
        PrintError(cmd.CommandTypeString());
        AbruptExit(__FILE__, __LINE__);
    }
}

uint64_t NewtonController::GetCoreCommandCounter(
    uint32_t core_id, const std::string& name) const {
    const auto core_it = core_command_counters_.find(core_id);
    if (core_it == core_command_counters_.end()) {
        return 0;
    }
    const auto counter_it = core_it->second.find(name);
    return counter_it == core_it->second.end() ? 0 : counter_it->second;
}




} // namespace dramsim3
