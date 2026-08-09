#pragma once

#include <list>
#include <array>
#include <memory>
#include <vector>

#include "../DRAM/Dram.h"
#include "../common_function.hpp"
#include "Sram.h"

class MyCore {
public:
    static constexpr size_t kTimingOpCount = 10;
    struct TimingBreakdown {
        cycle_type cycle = 0;
        cycle_type compute = 0;
        cycle_type memory = 0;
        cycle_type idle = 0;
        cycle_type load = 0;
        cycle_type store = 0;
        std::array<cycle_type, kTimingOpCount> op_compute{};
        std::array<cycle_type, kTimingOpCount> op_stall{};
    };
    MyCore(uint32_t id, const SysConfig& config);
    virtual bool running();
    virtual bool can_issue(Tile &next_tile);
    virtual bool can_issue_pim();
    virtual void issue(Tile &in_tile);

    virtual void issue_pim(Tile &in_tile);

    virtual void log();

    virtual Ptr<Tile> pop_finished_tile();

    virtual void cycle();

    virtual bool has_memory_request(uint32_t index) {
        return _memory_request_queues[index].size() > 0;
    }
    virtual void pop_memory_request(uint32_t index) {
        assert(has_memory_request(index));
        _memory_request_queues[index].pop();
    }
    virtual MemoryAccess *top_memory_request(uint32_t index) {
        return _memory_request_queues[index].front();
    }

    virtual void push_memory_request(MemoryAccess *request);
    virtual void push_memory_response(MemoryAccess *response);
    virtual void print_stats();
    virtual void apply_estimated_workload(const ProportionalWorkloadStat& workload);
    virtual void begin_proportional_timing_sampling();
    virtual void end_proportional_timing_sampling();
    virtual TimingBreakdown apply_estimated_timing(
        cycle_type skipped_work_cycles, cycle_type global_wait_cycles);
    virtual void apply_decode_pruning_timing(const TimingBreakdown& timing);
    virtual cycle_type get_compute_cycles() { return _stat_compute_cycle; }
    virtual void set_core_cycle(cycle_type cycle) { _core_cycle = cycle; }

    virtual bool can_issue_compute(Instruction &inst);
    virtual bool pim_can_issue_compute(Instruction &inst);
    virtual cycle_type get_inst_compute_cycles(Instruction &inst);

    const uint32_t _id;
    const SysConfig& _config;

    cycle_type _core_cycle;
    uint64_t _compute_end_cycle;
    cycle_type _stat_compute_cycle;
    cycle_type _stat_idle_cycle;
    cycle_type _stat_memory_cycle;
    cycle_type _accum_request_rr_cycle;
    cycle_type _max_request_rr_cycle{};
    cycle_type _min_request_rr_cycle{};
    cycle_type _memory_stall_cycle;
    cycle_type _compute_memory_stall_cycle;
    cycle_type _vector_memory_stall_cycle;
    cycle_type _layernorm_stall_cycle;
    cycle_type _rmsnorm_stall_cycle;
    cycle_type _rope_stall_cycle;
    cycle_type _softmax_stall_cycle;
    cycle_type _add_stall_cycle;
    cycle_type _mul_stall_cycle;
    cycle_type _gelu_stall_cycle;
    cycle_type _silu_stall_cycle;
    cycle_type _gemv_stall_cycle;

    cycle_type _load_memory_cycle;
    cycle_type _store_memory_cycle;

    /* Vector Unit Params */
    cycle_type _stat_vec_compute_cycle;
    cycle_type _stat_vec_memory_cycle;  // Does not acctuall count yet
    cycle_type _stat_vec_idle_cycle;    // Does not acctuall count yet

    cycle_type _stat_gemm_cycle;
    cycle_type _stat_layernorm_cycle;
    cycle_type _stat_rmsnorm_cycle;
    cycle_type _stat_rope_cycle;
    cycle_type _stat_add_cycle;
    cycle_type _stat_mul_cycle;
    cycle_type _stat_gelu_cycle;
    cycle_type _stat_silu_cycle;
    cycle_type _stat_softmax_cycle;
    cycle_type _stat_gemv_cycle;

    // PIM Instruction Count
    uint64_t _pim_pheader_count;
    uint64_t _pim_gwrite_count;
    uint64_t _pim_comp_count;
    uint64_t _pim_readers_count;

    // Memory Instruction Count
    uint64_t _write_count;
    uint64_t _read_count;

    // execution instruction count
    uint64_t _gemm_count;
    uint64_t _gemv_count;
    uint64_t _layernorm_count;
    uint64_t _rmsnorm_count;
    uint64_t _rope_count;
    uint64_t _softmax_count;
    uint64_t _add_count;
    uint64_t _mul_count;
    uint64_t _gelu_count;
    uint64_t _silu_count;
    uint64_t _im2col_count;
    uint64_t _dummy_count;

    ProportionalWorkloadStat _estimated_workload;
    TimingBreakdown _proportional_timing_baseline;
    TimingBreakdown _proportional_timing_sample_end;
    bool _proportional_timing_sample_complete = false;
    TimingBreakdown _estimated_timing;

    int _running_layer;

    std::deque<std::shared_ptr<Tile>> _tiles;
    std::deque<std::shared_ptr<Tile>> _pim_tiles;
    std::queue<std::shared_ptr<Tile>> _finished_tiles;

    std::queue<Instruction> _compute_pipeline;
    std::queue<Instruction> _vector_pipeline;
    std::vector<std::queue<Instruction>> _vector_pipelines;

    std::queue<Instruction> _ld_inst_queue;
    std::queue<Instruction> _st_inst_queue;
    std::queue<Instruction> _ex_inst_queue;

    // Execution Queue for PIM
    std::queue<Instruction> _pim_inst_queue;

    std::queue<Instruction> _ld_inst_queue_for_pim;
    std::queue<Instruction> _st_inst_queue_for_pim;
    std::queue<Instruction> _ex_inst_queue_for_pim;

    // make it to vector
    std::vector<std::queue<MemoryAccess *>> _memory_request_queues;
    std::queue<MemoryAccess *> _memory_response_queue;

    std::queue<MemoryAccess *> _pim_request_queue;
    std::queue<MemoryAccess *> _pim_response_queue;

    uint32_t _waiting_write_reqs;
    uint32_t _waiting_pim_reqs;

    int _current_spad;
    int _current_acc_spad;
    Sram _spad;
    Sram _acc_spad;

    Sram _pim_spad;
    Sram _pim_acc_spad;

    uint32_t _stat_systolic_inst_issue_count = 0;
    uint32_t _stat_systolic_preload_issue_count = 0;
    cycle_type get_vector_compute_cycles(Instruction& inst);
    cycle_type calculate_add_tree_iterations(uint32_t vector_size);
    cycle_type calculate_vector_op_iterations(uint32_t vector_size);

    std::vector<NPUStat> _stat;
    void issue_ex_inst(Instruction inst);
    void pim_issue_ex_inst(Instruction inst);
    Instruction get_first_ready_ex_inst();

    // NPU SA, VU cycle
    void systolic_cycle();
    void vector_unit_cycle();

    // Queue for SA block, PIM block
    void ld_queue_cycle();
    void st_queue_cycle();
    void ex_queue_cycle();

    void pim_queue_cycle();

    // Queue for SA block, PIM block
    void pim_ld_queue_cycle();
    void pim_st_queue_cycle();
    void pim_ex_queue_cycle();

    // Update stats
    void update_stats();
    TimingBreakdown timing_snapshot() const;
};
