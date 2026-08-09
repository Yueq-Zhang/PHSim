#ifndef DRAM_H
#define DRAM_H

#include <queue>
#include <unordered_map>
#include <utility>

#include "../common_function.hpp"
#include "../ext/NewtonSim/include/newtonsim/NewtonSim.h"

class Dram {
   public:
    virtual bool running() = 0;
    virtual void cycle() = 0;
    virtual bool is_full(uint32_t cid, MemoryAccess *request) = 0;
    virtual void push(uint32_t cid, MemoryAccess *request) = 0;
    virtual bool is_empty(uint32_t cid) = 0;
    virtual MemoryAccess *top(uint32_t cid) = 0;
    virtual void pop(uint32_t cid) = 0;
    virtual uint32_t get_channel_id(MemoryAccess *request) = 0;
    virtual void print_stat() {}
    addr_type get_addr_align() { return _addr_align; }

    virtual double get_avg_bw_util() = 0;
    virtual uint64_t get_avg_pim_cycle() = 0;
    virtual void reset_pim_cycle() = 0;
    virtual void log(Stage stage) = 0;

   protected:
    explicit Dram(const SysConfig& config) : _config(config) {}
    const SysConfig& _config;
    uint32_t _n_ch;
    cycle_type _cycles;
    addr_type _addr_align;
};

class PIM : public Dram {
   public:
    PIM(const SysConfig& config);
    ~PIM() = default;
    virtual bool running() override;
    virtual void cycle() override;
    virtual bool is_full(uint32_t cid, MemoryAccess *request) override;
    virtual void push(uint32_t cid, MemoryAccess *request) override;
    virtual bool is_empty(uint32_t cid) override;
    virtual MemoryAccess *top(uint32_t cid) override;
    virtual void pop(uint32_t cid) override;
    virtual uint32_t get_channel_id(MemoryAccess *request) override;
    virtual void print_stat() override;
    void set_dram_cycles(uint64_t cycles);
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
    cycle_type get_cycle() { return _cycles; }
    uint64_t get_core_command_counter(uint32_t core_id,
                                      const std::string& name) const;

    uint64_t MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col);
    uint64_t EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps, int num_readres);
    void update_stat(uint32_t cid);
    void log(Stage stage);

    std::unique_ptr<dramsim3::NewtonSim> _mem;
    std::vector<uint64_t> _total_processed_requests;
    std::vector<uint64_t> _processed_requests;
    uint64_t _mem_req_cnt = 0;
    int _burst_cycle;
    std::vector<std::vector<MemoryIOStat>> _stats;
    ProportionalWorkloadStat _estimated_workload;
    std::vector<uint64_t> _actual_request_counts;
    std::vector<uint64_t> _actual_read_request_counts;
    std::vector<uint64_t> _actual_write_request_counts;
    std::vector<uint64_t> _command_request_baseline;
    std::vector<uint64_t> _command_read_request_baseline;
    std::vector<uint64_t> _command_write_request_baseline;
    std::vector<std::unordered_map<std::string, uint64_t>>
        _command_counter_baseline;
    std::vector<uint64_t> _command_warmup_request_counts;
    std::vector<uint64_t> _command_warmup_read_request_counts;
    std::vector<uint64_t> _command_warmup_write_request_counts;
    std::vector<std::unordered_map<std::string, uint64_t>>
        _command_warmup_counter_counts;
    bool _command_has_warmup_sample = false;
    double _command_warmup_weight = 0.5;
    std::vector<std::unordered_map<std::string, uint64_t>>
        _estimated_command_counts;
    std::vector<std::vector<uint64_t>> _active_cycle_baseline;
    std::vector<std::vector<uint64_t>> _idle_cycle_baseline;
    std::vector<std::vector<uint64_t>> _sref_cycle_baseline;
    std::vector<std::vector<uint64_t>> _pim_active_cycle_baseline;
    std::vector<std::vector<uint64_t>> _pim_idle_cycle_baseline;
    std::vector<std::vector<uint64_t>> _estimated_pim_active_cycles;
    std::vector<std::vector<uint64_t>> _estimated_pim_idle_cycles;
    uint64_t _estimated_stage_pim_cycle_sum = 0;
    DecodePruningDramState _decode_pruning_state_baseline;
    std::string _decode_pruning_state_operation;
    std::unordered_map<std::string, DecodePruningDramState>
        _decode_pruning_state_accumulators;
    std::unordered_map<std::string, DecodePruningDramState>
        _decode_pruning_state_templates;
    uint64_t _stat_interval;

    void receive_predicting_config(size_t unstable_length, size_t sample_length, double inst_ratio);

    // stats
    uint64_t _stage_cycles;
    uint64_t _total_done_requests;
    double get_avg_bw_util() override;
    uint64_t get_avg_pim_cycle() override;
    void reset_pim_cycle() override;

    // One Transaction is added from DRAM in each transaction
    std::vector<bool> _push_valid;
    std::vector<bool> _pop_valid;

    std::vector<cycle_type> _last_push_cycle;
    std::vector<cycle_type> _last_pop_cycle;
};

#endif
