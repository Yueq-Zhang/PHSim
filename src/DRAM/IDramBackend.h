#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

struct MemoryAccess;
struct ProportionalWorkloadStat;
enum class Stage;

// Complete simulator-facing contract shared by the cycle-accurate and
// event-driven DRAM implementations. Backend-specific timing behavior stays
// inside each implementation so Simulator and Scheduler do not need parallel
// CA/ED control paths.
class IDramBackend {
   public:
    virtual ~IDramBackend() = default;

    // Runtime lifecycle and request transport.
    virtual bool running() = 0;
    virtual void advance_cycle(uint64_t current_cycle,
                               size_t unstable_length,
                               size_t sample_length,
                               double instruction_ratio) = 0;
    virtual void synchronize_cycles(uint64_t skipped_cycles) = 0;
    virtual void prepare_request(MemoryAccess* request,
                                 uint64_t current_cycle) = 0;
    virtual void schedule_pending_work(uint32_t cid,
                                       uint64_t current_cycle) = 0;
    virtual bool is_full(uint32_t cid, MemoryAccess* request) = 0;
    virtual void push(uint32_t cid, MemoryAccess* request) = 0;
    virtual bool is_empty(uint32_t cid) = 0;
    virtual MemoryAccess* top(uint32_t cid) = 0;
    virtual void pop(uint32_t cid) = 0;
    virtual uint32_t get_channel_id(MemoryAccess* request) = 0;

    // Stage statistics and accelerated-simulation accounting.
    virtual void print_stat() = 0;
    virtual double get_avg_bw_util() = 0;
    virtual uint64_t get_avg_pim_cycle() = 0;
    virtual void reset_pim_cycle() = 0;
    virtual void log(Stage stage) = 0;
    virtual void apply_estimated_workload(
        const ProportionalWorkloadStat& workload,
        const std::vector<uint64_t>* write_command_override = nullptr) = 0;
    virtual void begin_proportional_command_sampling() = 0;
    virtual void mark_proportional_command_warmup_complete(
        double warmup_weight) = 0;
    virtual void apply_estimated_time(uint64_t skipped_dram_cycles,
                                      bool physical_tail_pending) = 0;
    virtual void begin_decode_pruning_state_sample(
        const std::string& operation) = 0;
    virtual void finish_decode_pruning_state_sample(
        const std::string& operation, uint32_t required_samples) = 0;
    virtual std::vector<uint64_t> decode_pruning_sampled_write_requests(
        const std::string& operation) const = 0;
    virtual std::vector<uint64_t> decode_pruning_sampled_write_commands(
        const std::string& operation) const = 0;
    virtual void apply_decode_pruning_state(
        const std::string& operation,
        uint64_t skipped_dram_cycles) = 0;

    // Only the CA backend currently exposes per-core command attribution.
    virtual bool supports_core_command_counters() const noexcept {
        return false;
    }
    virtual uint64_t get_core_command_counter(
        uint32_t, const std::string&) const {
        throw std::logic_error(
            "The active DRAM backend does not expose per-core command counters");
    }
};
