#pragma once

#ifndef ENABLE_DRAM_ALIGNMENT_TRACE
#define ENABLE_DRAM_ALIGNMENT_TRACE 0
#endif

#include "../DRAM/Dram.h"
#include "../DRAM/IDramBackend.h"
#include "../common_function.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE


class PriorityRoundRobinEventScheduler {
public:
    using EventPtr = std::shared_ptr<Event>;
    using DequeType = std::deque<EventPtr>;

    mutable std::mutex mutex_;
    std::map<int, DequeType> map_;
    // One execute-ready generation per physical-bank queue. This replaces the
    // hot-path unordered_set<Event*> while keeping invalidation O(1).
    std::vector<uint64_t> refreshed_key_epochs_;
    int current_key_ = 0;
    cycle_type arbitration_cycle_ = 0;
    uint64_t timing_epoch_ = 1;
    mutable int cached_front_key_ = -1;
    mutable bool cached_front_key_valid_ = false;

    PriorityRoundRobinEventScheduler() = default;

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_locked();
    }

    int get_key(const EventPtr& event) const {
        const bool is_pim =
            event->req_type == MemoryAccessType::P_HEADER ||
            event->req_type == MemoryAccessType::GWRITE ||
            event->req_type == MemoryAccessType::COMP ||
            event->req_type == MemoryAccessType::COMP_HASH ||
            event->req_type == MemoryAccessType::READRES ||
            event->req_type == MemoryAccessType::COMPS_READRES;
        if (is_pim) {
            // Channel-wide PIM commands have no bank-level scheduling choice.
            // Keep them in one deque so the execute scheduler preserves their
            // input order while still allowing timing-constrained pipelining.
            return MyAddressAllocator::ranks *
                   MyAddressAllocator::bankgroups *
                   MyAddressAllocator::banks;
        }
        const int rank_index = static_cast<int>(event->rank_index);
        const int bankgroup_index = static_cast<int>(event->bankgroup_index);
        const int bank_index = static_cast<int>(event->bank_index);
        return (rank_index * MyAddressAllocator::bankgroups + bankgroup_index) *
                   MyAddressAllocator::banks +
               bank_index;
    }

    void set_arbitration_state(int current_key, cycle_type arbitration_cycle) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_key_ == current_key && arbitration_cycle_ == arbitration_cycle) {
            return;
        }
        current_key_ = current_key;
        arbitration_cycle_ = arbitration_cycle;
        cached_front_key_valid_ = false;
    }

    void push_back(EventPtr event) {
        push_single_event(std::move(event));
    }

    void push_single_event(EventPtr event) {
        if (event == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const int key = get_key(event);
        if (static_cast<size_t>(key) >= refreshed_key_epochs_.size()) {
            refreshed_key_epochs_.resize(static_cast<size_t>(key) + 1, 0);
        }
        map_[key].push_back(std::move(event));
        cached_front_key_valid_ = false;
    }

    void push_multi_event(const std::vector<EventPtr>& events) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto event : events) {
            if (event == nullptr) {
                continue;
            }
            const int key = get_key(event);
            if (static_cast<size_t>(key) >= refreshed_key_epochs_.size()) {
                refreshed_key_epochs_.resize(static_cast<size_t>(key) + 1, 0);
            }
            map_[key].push_back(std::move(event));
        }
        cached_front_key_valid_ = false;
    }

    void push_multi_event(const std::deque<EventPtr>& events) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto event : events) {
            if (event == nullptr) {
                continue;
            }
            const int key = get_key(event);
            if (static_cast<size_t>(key) >= refreshed_key_epochs_.size()) {
                refreshed_key_epochs_.resize(static_cast<size_t>(key) + 1, 0);
            }
            map_[key].push_back(std::move(event));
        }
        cached_front_key_valid_ = false;
    }

    EventPtr front() const {
        std::lock_guard<std::mutex> lock(mutex_);
        const int key = select_front_key_locked();
        if (key < 0) {
            return nullptr;
        }
        return map_.at(key).front();
    }

    void pop_front() {
        std::lock_guard<std::mutex> lock(mutex_);
        const int key = select_front_key_locked();
        if (key < 0) {
            return;
        }

        auto it = map_.find(key);
        if (it == map_.end() || it->second.empty()) {
            return;
        }

        EventPtr event = it->second.front();
        if (static_cast<size_t>(key) < refreshed_key_epochs_.size()) {
            refreshed_key_epochs_[key] = 0;
        }
        it->second.pop_front();
        cached_front_key_valid_ = false;

        const int consumed_key = key;
        if (it->second.empty()) {
            map_.erase(it);
        }

        if (map_.empty()) {
            current_key_ = 0;
            return;
        }

        auto next_it = map_.upper_bound(consumed_key);
        if (next_it == map_.end()) {
            next_it = map_.begin();
        }
        current_key_ = next_it->first;
    }

    std::optional<EventPtr> pop() {
        EventPtr event = front();
        if (event == nullptr) {
            return std::nullopt;
        }
        pop_front();
        return event;
    }

    bool has_refreshed_execute_cycle(const EventPtr& event) const {
        if (event == nullptr) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const int key = get_key(event);
        return static_cast<size_t>(key) < refreshed_key_epochs_.size() &&
               refreshed_key_epochs_[key] == timing_epoch_;
    }

    void mark_refreshed_execute_cycle(const EventPtr& event) {
        if (event == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const int key = get_key(event);
        if (static_cast<size_t>(key) >= refreshed_key_epochs_.size()) {
            refreshed_key_epochs_.resize(static_cast<size_t>(key) + 1, 0);
        }
        refreshed_key_epochs_[key] = timing_epoch_;
        // GetExecuteCycle() may have changed which bank head wins arbitration.
        cached_front_key_valid_ = false;
    }

    void invalidate_refreshed_execute_cycles() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++timing_epoch_;
        if (timing_epoch_ == 0) {
            timing_epoch_ = 1;
            std::fill(refreshed_key_epochs_.begin(), refreshed_key_epochs_.end(), 0);
        }
        cached_front_key_valid_ = false;
    }

    template <typename Predicate>
    bool any_of(Predicate pred) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& pair : map_) {
            for (const auto& event : pair.second) {
                if (pred(event)) {
                    return true;
                }
            }
        }
        return false;
    }

private:
    size_t size_locked() const {
        size_t total_size = 0;
        for (const auto& pair : map_) {
            total_size += pair.second.size();
        }
        return total_size;
    }

    int select_front_key_locked() const {
        if (cached_front_key_valid_) {
            return cached_front_key_;
        }
        if (map_.empty()) {
            cached_front_key_ = -1;
            cached_front_key_valid_ = true;
            return -1;
        }

        cycle_type best_execute_cycle = std::numeric_limits<cycle_type>::max();
        bool ready_event_exists = false;
        for (const auto& pair : map_) {
            if (pair.second.empty() || pair.second.front() == nullptr) {
                continue;
            }
            const cycle_type execute_cycle = pair.second.front()->execute_cycle;
            if (execute_cycle <= arbitration_cycle_) {
                ready_event_exists = true;
            }
            best_execute_cycle = std::min(best_execute_cycle, execute_cycle);
        }
        if (best_execute_cycle == std::numeric_limits<cycle_type>::max()) {
            cached_front_key_ = -1;
            cached_front_key_valid_ = true;
            return -1;
        }

        auto it = map_.lower_bound(current_key_);
        if (it == map_.end()) {
            it = map_.begin();
        }

        for (size_t checked = 0; checked < map_.size(); ++checked) {
            if (!it->second.empty() && it->second.front() != nullptr) {
                const cycle_type execute_cycle = it->second.front()->execute_cycle;
                if ((ready_event_exists && execute_cycle <= arbitration_cycle_) ||
                    (!ready_event_exists && execute_cycle == best_execute_cycle)) {
                    cached_front_key_ = it->first;
                    cached_front_key_valid_ = true;
                    return cached_front_key_;
                }
            }
            ++it;
            if (it == map_.end()) {
                it = map_.begin();
            }
        }
        cached_front_key_ = -1;
        cached_front_key_valid_ = true;
        return cached_front_key_;
    }
};

// EventDrivenDram owns the small DRAM hierarchy it actually uses. These
// declarations were previously pulled from src/NMC_System, but keeping them
// here makes the event-driven model self-contained and avoids unused NMC code
enum class RowBufPolicy {OPEN_PAGE, CLOSE_PAGE};

class DRAMBank {
public:
    DRAMBank(MemConfig& mem_config, std::vector<std::pair<int, uint64_t>>* rank_activate_recorder,
             int rank_id, int bankgroup_id, int bank_id);
    ~DRAMBank();

    void Activate(std::shared_ptr<Event> event);
    void Precharge(std::shared_ptr<Event> event);
    void Read(std::shared_ptr<Event> event);
    void Write(std::shared_ptr<Event> event);
    void PIM(std::shared_ptr<Event> event);
    void AddPendingPrechargeEvent(std::shared_ptr<Event> event);
    void PushPendingSwitchRow(std::shared_ptr<Event> event);

    MemConfig& config_;
    std::vector<std::pair<int, uint64_t>>* rank_activate_recorder;
    int rank_id, bankgroup_id, bank_id;

    RowBufPolicy row_buf_policy_;
    int open_row;
    int open_row_exec_event_count;
    std::deque<int> pending_precharge_order_queue;
    std::unordered_map<int, std::deque<std::shared_ptr<Event>>> pending_precharge_event;
    bool pending_precharge;
    bool pending_activate;

    uint64_t latest_refresh_time = 0;
    uint64_t next_bank_refresh_time;
    bool bank_refresh = false;
    int bank_refresh_count = 0;
    int refresh_interval_ = 0;
    int refresh_offset_ = 0;

    uint64_t latest_trans_time = 0;
    uint64_t to_do_precharge;
    uint64_t to_do_activate;
    uint64_t to_do_read;
    uint64_t to_do_write;
    std::array<uint64_t, 5> pim_cmd_ready;
    int pim_open_row = -1;
    uint64_t to_do_pim_precharge = 0;
    uint64_t to_do_pim_activate = 0;

    std::multimap<uint32_t, std::shared_ptr<Event>> pending_activate_row;
};


class DRAMBankGroup {
public:
    DRAMBankGroup(MemConfig& mem_config, std::vector<std::pair<int, uint64_t>>* rank_activate_recorder, int rank_id, int bankgroup_id);
    ~DRAMBankGroup();

    void SameBankgroupPrecharge(std::shared_ptr<Event> event);
    void SameBankgroupActivate(std::shared_ptr<Event> event);
    void OtherBankgroupActivate(std::shared_ptr<Event> event);
    void SameBankgroupRead(std::shared_ptr<Event> event);
    void OtherBankgroupRead(std::shared_ptr<Event> event);
    void SameBankgroupWrite(std::shared_ptr<Event> event);
    void OtherBankgroupWrite(std::shared_ptr<Event> event);

    int rank_id, bankgroup_id;
    MemConfig& config_;
    std::vector<DRAMBank*> dram_banks;
    std::vector<std::pair<int, uint64_t>>* rank_activate_recorder;

    uint64_t to_do_read;
    uint64_t to_do_write;
    uint64_t to_do_precharge;
    uint64_t to_do_activate;
};

struct RefreshPrechargeRecord {
    cycle_type cycle = 0;
    uint32_t rank = 0;
    uint32_t bankgroup = 0;
    uint32_t bank = 0;
    uint32_t row = 0;
};


class DRAMRank {
public:
    DRAMRank(MemConfig& mem_config, int channel_id, int rank_id);
    ~DRAMRank();

    uint64_t RankRefreshReady(
        std::vector<RefreshPrechargeRecord>* precharge_records);
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

    bool rank_refresh;
    uint64_t next_refresh_cycle = 0;
    int refresh_count = 0;
    int current_trans_refresh_count = 0;
    int rank_refresh_interval_ = 0;
    int rank_refresh_offset_ = 0;

    std::vector<std::pair<int, uint64_t>> rank_activate_recoder_;
    std::deque<cycle_type> activate_cycle_recorder;

    uint32_t to_precharge = 0;
    uint32_t to_activate = 0;
    uint64_t to_do_read = 0;
    uint64_t to_do_write = 0;
};


class DRAMChannel {
public:
    DRAMChannel(MemConfig& mem_config, int channel_id);
    ~DRAMChannel();

    void ScheduleTransaction() {}
    void WriteBufferDrain() {}
    void ArrangeWriteBufferDrain() {}
    // void OtherRankTimingUpdate(uint64_t processing_cycle, bool is_write, int rank_id);

    void IssueActivateEvent(std::shared_ptr<Event> event);
    void IssuePrechargeEvent(std::shared_ptr<Event> event);
    void IssuePIMPrechargeEvent(
        const std::shared_ptr<Event>& event, cycle_type issue_cycle);
    void IssuePIMActivateEvent(
        const std::shared_ptr<Event>& event, cycle_type issue_cycle);
    void IssueExecuteEvent(std::shared_ptr<Event> event, cycle_type issue_cycle);
    cycle_type IssueRankRefresh(cycle_type refresh_cycle);
    cycle_type GetNextRankRefreshCycle() const;

    void GetPrechargeCycle(std::shared_ptr<Event> event);
    void GetActivateCycle(std::shared_ptr<Event> event);
    void GetExecuteCycle(std::shared_ptr<Event> event);
    void RefreshExecutionSchedulerHeadCycles();
    void EnqueueExecuteBankInterleaved(std::deque<std::shared_ptr<Event>> events);
    void RefreshEarliestQueue();
    int PhysicalBankKey(const std::shared_ptr<Event>& event) const;
    cycle_type NextPhysicalArbitrationCycle() const;
    void AdvancePhysicalRoundRobin(const std::shared_ptr<Event>& event);
    void RecordPhysicalCommandCycle(cycle_type cycle) {
        if (last_physical_command_cycle ==
                std::numeric_limits<cycle_type>::max() ||
            cycle > last_physical_command_cycle) {
            last_physical_command_cycle = cycle;
        }
    }

    MemConfig& config_;
    // ????execute queue???
    #ifdef USE_PRIORITY_ROUND_ROBIN_EXECUTE_QUEUE
    PriorityRoundRobinEventScheduler execute_queue;
    #else
    std::deque<std::shared_ptr<Event>> execute_queue;
    #endif

    std::deque<std::shared_ptr<Event>> activate_queue;
    std::deque<std::shared_ptr<Event>> precharge_queue;
    std::deque<std::shared_ptr<Event>> return_queue;
    std::unordered_map<uint32_t, std::deque<std::shared_ptr<Event>>> bank_row_buffer_conflict_event_queue;
    int channel_id;
    cycle_type _dram_cycle = 0;
    cycle_type _pre_dram_cycle = 0;
    cycle_type last_physical_command_cycle =
        std::numeric_limits<cycle_type>::max();
    int physical_rr_next_key = 1;
    std::vector<RefreshPrechargeRecord> refresh_precharge_records;
    std::pair<int, cycle_type> earliest_queue = {0, std::numeric_limits<uint64_t>::max()};

    bool is_unified_queue_;
    int write_draining_;
    std::vector<uint64_t> write_address_buffer_;
    std::vector<DRAMRank*> dram_ranks;
    uint64_t clk_;
};


class MemorySystem {
public:
    MemorySystem(const MemConfig& mem_config, int mem_id);
    virtual ~MemorySystem();

    int GetChannel(uint64_t hex_addr) const;
    void CheckRowSwitch(std::shared_ptr<Event> event);
    void CheckPIMRowSwitch(std::shared_ptr<Event> event);
    void PendingPrecharge(std::shared_ptr<Event> event);
    void PendingActivate(std::shared_ptr<Event> event);
    void PendingExecuted(std::shared_ptr<Event> event);

    MemConfig config_;
    std::vector<DRAMChannel*> dram_channels;
    int memory_id_;
    uint64_t last_transaction_clk_;
};


class ResponseQueue {
public:
    ResponseQueue(int Size);
    bool isAvailable() const;
    bool isAvailable(uint32_t count) const;
    bool isEmpty() const;
    void reserve();
    void push(MemoryAccess* original_req);
    MemoryAccess* top() const;
    void pop();

    const uint32_t Size;
    uint32_t NumReserved;
    std::deque<MemoryAccess*> OutputQueue;
};


// A std::multimap-compatible pending map with a shared read-merge trace helper.
// Scheduling semantics remain identical to std::multimap.
class TracingPendingReadMap
    : public std::multimap<uint64_t, std::shared_ptr<Event>> {
public:
    using Base = std::multimap<uint64_t, std::shared_ptr<Event>>;
    using EventPtr = std::shared_ptr<Event>;
    using iterator = Base::iterator;

    static void record_merge(uint64_t address,
                             uint32_t channel,
                             uint64_t merge_cycle,
                             uint64_t first_added_cycle,
                             uint64_t pending_before) {
#if ENABLE_DRAM_ALIGNMENT_TRACE
        static std::ofstream trace = [] {
            std::ofstream out(Config::system_config.log_dir +
                                  "/read_merge_eventdriven.csv",
                              std::ofstream::out | std::ofstream::trunc);
            if (out.is_open()) {
                out << "sequence,channel,merge_cycle,address,first_added_cycle,age_cycles,pending_before\n";
            }
            return out;
        }();
        static uint64_t sequence = 0;
        if (trace.is_open()) {
            const uint64_t age_cycles = merge_cycle >= first_added_cycle
                                            ? merge_cycle - first_added_cycle
                                            : 0;
            trace << sequence++ << ','
                  << channel << ','
                  << merge_cycle << ','
                  << address << ','
                  << first_added_cycle << ','
                  << age_cycles << ','
                  << pending_before << '\n';
        }
#else
        (void)address;
        (void)channel;
        (void)merge_cycle;
        (void)first_added_cycle;
        (void)pending_before;
#endif
    }

    iterator insert(std::pair<uint64_t, EventPtr> value) {
        return Base::insert(std::move(value));
    }
};


class DramDataContainer;

class EventDrivenDram : public IDramBackend {
public:
    enum class EDTransactionType {
        DRAM_READ,
        DRAM_WRITE,
        DRAM_PIM,
    };

    enum class QueueClass {
        READ_Q,
        WRITE_Q,
        PIM_Q
    };

    EventDrivenDram(const SysConfig& config,
                    DramDataContainer* data_container = nullptr);
    ~EventDrivenDram() override;
    void push(uint32_t cid, MemoryAccess *req) override;  // Push Memory Access
    void pop(uint32_t cid) override;

    MemoryAccess *top(uint32_t cid) override;
    bool is_full(uint32_t cid, MemoryAccess *req) override;
    bool is_empty(uint32_t cid) override;
    bool running() override;

    Event generate_event_from_req(MemoryAccess *req);
    void schedule_pending_event_transaction(uint32_t cid);
    void issue_pending_read_event(uint32_t cid);
    void issue_pending_write_event(uint32_t cid);
    void issue_pending_pim_event(uint32_t cid);

    void process_finish_event(uint32_t cid, cycle_type current_cycle);
    void schedule_pending_operation(uint32_t cid, cycle_type current_cycle);

    std::unique_ptr<dramsim3::Config> dramsim3_config_;
    DramDataContainer* _data_container;
    bool _serialize_pim_after_physical_rw;
    std::unique_ptr<MemorySystem> _memsys;
    std::function<void(uint64_t req_id)> pim_callback_;

    int transaction_queue_size;
    // pending event queue
    std::vector<TracingPendingReadMap> _pending_read_events;
    std::vector<std::multimap<uint64_t, std::shared_ptr<Event>>> _pending_write_events;
    std::vector<std::multimap<uint64_t, std::shared_ptr<Event>>> _pending_pim_events;

    // Keep only the currently mergeable generation per address.
    // Closed generations remain pending until their response returns.
    std::vector<std::unordered_map<addr_type, std::weak_ptr<Event>>> _open_read_merge_events;
    std::vector<std::unordered_map<addr_type, std::weak_ptr<Event>>> _open_write_merge_events;

    // operation queue
    std::vector<std::deque<std::shared_ptr<Event>>> _read_queue;
    std::vector<std::deque<std::shared_ptr<Event>>> _write_buffer;
    std::vector<std::deque<std::shared_ptr<Event>>> _pim_queue;

    std::vector<int> _write_draining;
    std::vector<cycle_type> _last_transaction_schedule_cycle;
    std::vector<cycle_type> _last_idle_schedule_check_cycle;
    // Earliest cycle at which an idle channel can make observable progress.
    // New request pushes bypass this cache and recompute it immediately.
    std::vector<cycle_type> _next_wakeup_cycle;
    std::vector<cycle_type> _next_return_completion_cycle;

    std::vector<ResponseQueue> response_event_queues_;

    uint32_t get_channel_id(MemoryAccess *access) override;
    double   get_avg_bw_util();
    uint64_t get_avg_pim_cycle();
    void     reset_pim_cycle();
    void     log(Stage)           {}
    void     print_stat()        ;
    void apply_estimated_workload(
        const ProportionalWorkloadStat& workload,
        const std::vector<uint64_t>* write_command_override = nullptr);
    void begin_proportional_command_sampling();
    void mark_proportional_command_warmup_complete(double warmup_weight);
    void apply_estimated_time(cycle_type skipped_dram_cycles);
    void begin_decode_pruning_state_sample(const std::string& operation);
    void finish_decode_pruning_state_sample(
        const std::string& operation, uint32_t required_samples);
    std::vector<uint64_t> decode_pruning_sampled_write_requests(
        const std::string& operation) const;
    std::vector<uint64_t> decode_pruning_sampled_write_commands(
        const std::string& operation) const;
    void apply_decode_pruning_state(
        const std::string& operation, cycle_type skipped_dram_cycles);

private:

    struct EventDrivenChannelStats {
        uint64_t num_cycles = 0;
        uint64_t num_reads_done = 0;
        uint64_t num_writes_done = 0;
        uint64_t num_pim_done = 0;
        uint64_t num_logical_reads_done = 0;
        uint64_t num_read_merges = 0;
        uint64_t num_write_merges = 0;
        uint64_t num_read_cmds = 0;
        uint64_t num_write_cmds = 0;
        uint64_t num_act_cmds = 0;
        uint64_t num_pre_cmds = 0;
        uint64_t num_write_buf_hits = 0;
        uint64_t num_read_row_hits = 0;
        uint64_t num_write_row_hits = 0;
        uint64_t num_ondemand_pres = 0;
        uint64_t num_ref_cmds = 0;
        uint64_t num_refb_cmds = 0;
        uint64_t num_pheader_cmds = 0;
        uint64_t num_gwrite_cmds = 0;
        uint64_t num_comp_cmds = 0;
        uint64_t num_readres_cmds = 0;
        uint64_t num_pim_cmds = 0;
        uint64_t num_pim_activate_cmds = 0;
        uint64_t num_pim_precharge_cmds = 0;
        uint64_t num_read_requests = 0;
        uint64_t num_write_requests = 0;
        uint64_t num_pim_requests = 0;
        uint64_t estimated_read_requests = 0;
        uint64_t estimated_write_requests = 0;
        uint64_t estimated_pim_requests = 0;
        uint64_t estimated_reads_done = 0;
        uint64_t estimated_writes_done = 0;
        uint64_t estimated_pim_done = 0;
        uint64_t estimated_read_cmds = 0;
        uint64_t estimated_write_cmds = 0;
        uint64_t estimated_act_cmds = 0;
        uint64_t estimated_pre_cmds = 0;
        uint64_t estimated_write_buf_hits = 0;
        uint64_t estimated_read_row_hits = 0;
        uint64_t estimated_write_row_hits = 0;
        uint64_t estimated_ondemand_pres = 0;
        uint64_t estimated_pheader_cmds = 0;
        uint64_t estimated_gwrite_cmds = 0;
        uint64_t estimated_comp_cmds = 0;
        uint64_t estimated_readres_cmds = 0;
        uint64_t estimated_pim_cmds = 0;
        uint64_t estimated_pim_activate_cmds = 0;
        uint64_t estimated_pim_precharge_cmds = 0;
        uint64_t pim_cycles = 0;
        uint64_t estimated_pim_cycles = 0;
        uint64_t estimated_cycles = 0;
        uint64_t estimated_ref_cmds = 0;
        uint64_t estimated_refb_cmds = 0;
        std::vector<uint64_t> all_bank_idle_cycles;
        std::vector<uint64_t> rank_active_cycles;
        std::vector<uint64_t> estimated_all_bank_idle_cycles;
        std::vector<uint64_t> estimated_rank_active_cycles;
        std::vector<uint64_t> pim_all_bank_idle_cycles;
        std::vector<uint64_t> pim_rank_active_cycles;
        std::vector<uint64_t> estimated_pim_all_bank_idle_cycles;
        std::vector<uint64_t> estimated_pim_rank_active_cycles;
        uint64_t time_compensation_cycle_baseline = 0;
        uint64_t time_compensation_ref_baseline = 0;
        uint64_t time_compensation_refb_baseline = 0;
        std::vector<uint64_t> time_compensation_idle_baseline;
        std::vector<uint64_t> time_compensation_active_baseline;
    };

    std::vector<EventDrivenChannelStats> _stats;
    std::vector<EventDrivenChannelStats> _command_sampling_baseline;
    std::vector<EventDrivenChannelStats> _command_warmup_baseline;
    std::vector<EventDrivenChannelStats> _command_warmup_end;
    DecodePruningDramState _decode_pruning_state_baseline;
    std::string _decode_pruning_state_operation;
    std::unordered_map<std::string, DecodePruningDramState>
        _decode_pruning_state_accumulators;
    std::unordered_map<std::string, DecodePruningDramState>
        _decode_pruning_state_templates;
    bool _command_has_warmup_sample = false;
    double _command_warmup_weight = 0.5;
    uint64_t _stage_request_baseline = 0;
    cycle_type _stage_cycle_baseline = 0;
    uint64_t _stage_pim_cycle_baseline = 0;
    std::vector<cycle_type> _stats_last_cycle;
    uint64_t next_merge_group_id_ = 1;
    std::ofstream command_trace_;
    std::ofstream time_advance_trace_;
    std::ofstream transaction_trace_;
    uint64_t command_trace_sequence_ = 0;
    uint64_t time_advance_trace_sequence_ = 0;
    uint64_t transaction_trace_sequence_ = 0;
    std::vector<uint64_t> transaction_arrival_sequence_;

    void accumulate_background_cycles(uint32_t cid, cycle_type until_cycle);
    void record_execute_command(uint32_t cid, const std::shared_ptr<Event>& event);
    void record_completed_request(uint32_t cid, MemoryAccessType req_type);
    void note_return_completion(
        uint32_t cid, const std::shared_ptr<Event>& event);
    void recompute_return_completion(uint32_t cid);
    void recompute_next_wakeup(uint32_t cid, cycle_type current_cycle);
    cycle_type issue_rank_refresh_with_precharges(
        uint32_t cid, cycle_type requested_cycle, const char* source);
    void write_event_driven_stats(bool print_to_log) const;
    void trace_issued_command(uint32_t cid, cycle_type cycle,
                              const char* command,
                              const std::shared_ptr<Event>& event,
                              const char* source);
    void trace_time_advance(uint32_t cid, cycle_type from_cycle,
                             cycle_type to_cycle, int earliest_type,
                             cycle_type earliest_cycle, const char* source);
    void trace_transaction(uint32_t cid, cycle_type cycle, const char* action,
                           const std::shared_ptr<Event>& event,
                           const char* detail = "");
    static void erase_pending_event(
        std::multimap<uint64_t, std::shared_ptr<Event>>& pending,
        uint64_t addr,
        const std::shared_ptr<Event>& event);

    std::vector<bool> _push_valid;
    std::vector<bool> _pop_valid;

    std::vector<bool> _rw_dependency_lock;
    std::vector<addr_type> _rw_dependency_addr;  // not used currently

};
