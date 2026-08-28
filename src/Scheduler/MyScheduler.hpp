#pragma once
#include <chrono>
#include <array>
#include "../common_function.hpp"
#include "../Program/StageProgram.h"

class PIM;
class EventDrivenDram;
class MyCore;
class MyInterconnect;
class Client;
class DramDataContainer;

class MyScheduler {
public:
    MyScheduler(const SysConfig& config, const cycle_type *core_cycle,
                DramDataContainer* data_container = nullptr);
    ~MyScheduler() = default;
    void launch(Ptr<Model> model);
    bool can_accept_request() const;
    void add_request(std::shared_ptr<InferRequest> request);
    bool empty();
    bool running();
    bool has_completed_request();

    std::shared_ptr<InferRequest> pop_completed_request();
    bool has_stage_changed() const;
    void reset_has_stage_changed_status();
    Stage get_prev_stage() const;
    uint32_t count_active_operations() const;

    void cycle();
    void init_batches();
    void cleanup_batch(std::vector<Ptr<InferRequest>> batch_request);

    void make_program();
    void refresh_status();
    void finish_program();
    void refresh_stage();

    void issue_tile_per_core();

    bool finish_tile(uint32_t core_id, Tile& tile);
    Tile& top_tile(uint32_t core_id);
    void get_tile(uint32_t core_id);

    bool is_executable_tile_empty();
    uint32_t get_exist_tile_count();
    void print_stat();
    void print_op_stat();
    Ops getOpType(const std::string& opStr); // input op name return op type

    void set_scheduler_cycles(uint64_t cycles);

    typedef struct {
        uint32_t id;
        uint32_t request_id;
        std::string name;
        // xxx necessary?
        // bool launched;
        cycle_type start_cycle;
        uint32_t total_tiles;
        uint32_t remain_tiles;
        uint32_t launched_tiles;
        
        // PIM Bandwidth Stats
        cycle_type pim_start_cycle;
        cycle_type pim_end_cycle;
        uint64_t pim_inst_count;
        cycle_type logical_pim_end_cycle;
        uint64_t estimated_pim_inst_count;
        std::chrono::steady_clock::time_point host_start_time;
        double deferred_compile_time_sec;
    } RunningOperationStat;

    const cycle_type *_core_cycle;

    Ptr<Model> _model;
    std::unique_ptr<StageProgram> _model_program; // Inference program
    std::deque<Tile> _executable_tile_queue; // to store tile for compute
    std::map<uint32_t, std::deque<Tile>> _core_executable_tile_queue;  // Tile queue for each core

    const SysConfig& _config;

    std::unordered_map<uint32_t, RunningOperationStat> _finished_operation_stats;
    std::unordered_map<uint32_t, RunningOperationStat> _active_operation_stats;

    cycle_type _cycles;

    std::deque<std::shared_ptr<InferRequest>> _request_queue;
    std::queue<std::shared_ptr<InferRequest>> _completed_request_queue;

    uint32_t _max_batch_size;
    uint32_t _max_active_reqs;

    cycle_type _last_request_cycle;

    std::vector<Ptr<InferRequest>> _breq;

    uint32_t _active_reqs;

    Stage _stage;
    Stage _init_stage;          // default A, if you want to start from other stage, set it

    bool _test_single_op;
    std::string _test_single_op_name;
    Ops _test_single_op_type;

    bool _test_multi_layer;
    std::string _test_multi_layer_name;

    bool _acceletare_ctrl;

    bool _has_stage_changed;
    Stage _prev_stage;          // The original stage
    bool _output_token_iteration_enable = false;

    cycle_type program_start_cycle;

    // memory spec
    uint32_t _dram_channels;
    uint32_t _dram_banks_per_ch;

    // uint32_t _gwrite_latency;
    // uint32_t _gemv_latency;
    int _core_rr_id;

    // tile map, reduced to accelerate
    std::unordered_map<std::string, Tile*> _tile_map;
    uint32_t _get_tile_count;

    // added logic for Tile based prediction
    // predicting used cycle
    bool _sim_accelerate = false;
    void compute_estimated_cycle();
    void finish_last_mm_tile(uint32_t core_id, Tile& tile);
    bool update_stats_last_tile(uint32_t core_id, Tile& tile);
    std::size_t _unstable_length = 0;
    std::size_t _sample_length = 0;
    std::size_t _tile_position = 0;
    uint32_t _estimated_all_cycle = 0;
    uint32_t _actual_all_cycle = 0;
    uint32_t _unstable_cycle = 0;
    uint32_t _sample_cycle = 0;
    uint32_t _mean_cycle = 0;
    std::vector<uint32_t> _get_tile_cycle;
    std::vector<uint32_t> _finish_tile_cycle;
    std::double_t _inst_ratio;

    // GEMM Loop_wise acceleration variables
    std::unordered_map<uint32_t, std::vector<uint32_t>> _core_kloop_cycles; // 每个core的K_Loop完成周期
    std::unordered_map<uint32_t, uint32_t> _core_unstable_cycles; // 每个core的不稳定周期
    std::unordered_map<uint32_t, uint32_t> _core_sample_cycles; // 每个core的采样周期
    std::unordered_map<uint32_t, uint32_t> _core_remaining_cycles; // 每个core的剩余周期
    std::unordered_map<uint32_t, uint32_t> _core_k_inner_count; // 每个core的K_Loop计数
    std::unordered_map<uint32_t, bool> _core_stable; // 每个core是否达到稳定状态
    std::unordered_map<uint32_t, uint32_t> _core_estimated_cycles; // 每个core的估计总周期
    uint32_t _k_loop_size = 0; // K_Loop的大小

    // GEMM_Attn Loop_wise acceleration variables
    std::unordered_map<uint32_t, std::vector<uint32_t>> _core_head_cycles; // 每个core的head完成周期
    std::unordered_map<uint32_t, uint32_t> _core_head_unstable_cycles; // 每个core的第一个head周期
    std::unordered_map<uint32_t, uint32_t> _core_head_sample_cycles; // 每个core的第二个head周期
    std::unordered_map<uint32_t, uint32_t> _core_head_tile_count; // 每个core完成tile数量
    std::unordered_map<uint32_t, bool> _core_head_first_done; // 每个core是否完成了第一个head
    std::unordered_map<uint32_t, bool> _core_head_second_done; // 每个core是否完成了第二个head
    std::unordered_map<uint32_t, uint32_t> _core_head_remaining; // 每个core剩余head数量
    std::unordered_map<uint32_t, uint32_t> _core_head_estimated_cycles; // 每个core完成所有head的估计总周期
    uint32_t _total_heads = 0; // head数量
    uint8_t _latest_core = 0;
    uint8_t _front_heads = 0;

    // Proportional sampling variables. GEMM samples K-loop tiles, while
    // GEMM_Att warms up two complete heads and samples the next two heads as
    // one alternating execution group before predicting all remaining heads.
    double _proportional_sample_ratio = 0.25;
    uint32_t _proportional_sample_tiles = 0;
    uint32_t _proportional_common_kloops = 0;
    uint32_t _proportional_skipped_tiles = 0;
    bool _proportional_plan_ready = false;
    bool _proportional_prediction_applied = false;
    bool _proportional_tail_is_active = false;
    bool _proportional_attention_mode = false;
    bool _proportional_softmax_mode = false;
    uint32_t _proportional_common_head_rounds = 0;
    uint32_t _proportional_common_softmax_rounds = 0;
    std::vector<uint32_t> _proportional_active_cores;
    std::unordered_map<uint32_t, uint32_t> _proportional_prefix_tiles;
    std::unordered_map<uint32_t, uint32_t> _proportional_warmup_head_tiles;
    std::unordered_map<uint32_t, std::vector<cycle_type>> _proportional_finish_cycles;
    std::unordered_map<uint32_t, std::deque<Tile>> _proportional_tail_queues;
    std::unordered_map<uint32_t, ProportionalWorkloadStat>
        _proportional_estimated_workload;
    bool _proportional_workload_applied = false;
    bool _proportional_command_sample_started = false;
    bool _proportional_timing_applied = false;
    double _deferred_compile_time_sec = 0.0;
    std::unordered_map<uint32_t, cycle_type>
        _proportional_core_estimated_cycles;

    uint32_t _operation_id;

    std::deque<Tile> _total_tiles;
    std::shared_ptr<Operation> _current_op;
    std::vector<int> _inst_num;
    std::vector<int> _inst_move_num;
    std::vector<int> _inst_comp_num;
    uint32_t _tile_num_last;

    struct PredictingConfig {           //variables passed to dram simulator
        std::size_t _unstable_length;
        std::size_t _sample_length;
        double_t _inst_ratio;           //acceleration ratio
        bool _enable_sim_accelerate;    //only enables MatMul acceleration
        std::uint32_t _num_insts;       //total number of instructions in a tile
        std::uint32_t _tile_index;      //current tile position passed to DRAMSim3
    };
    PredictingConfig get_predicting_config();
    void init_predicting_config();
    void reset_proportional_sampling_state();
    void prepare_proportional_sampling();
    void prepare_proportional_attention_sampling();
    void prepare_proportional_softmax_sampling();
    uint32_t get_estimated_all_cycle();
    bool are_all_cores_stable();
    bool are_all_cores_head_second_done(); // 检查是否所有core都完成了第二个head
    bool proportional_ready_to_predict() const;
    bool proportional_tail_phase() const;
    bool proportional_operation_complete(uint32_t operation_id) const;
    bool release_proportional_tail_tiles();
    const std::unordered_map<uint32_t, ProportionalWorkloadStat>&
        proportional_estimated_workload() const;
    bool proportional_workload_applied() const;
    void mark_proportional_workload_applied();
    void apply_proportional_core_timing();
    bool decode_pruning_prediction_pending() const;
    cycle_type decode_pruning_prediction_cycles() const;
    void apply_decode_pruning_prediction();
    void apply_decode_pruning_dram_state(cycle_type skipped_dram_cycles);
    void complete_decode_pruning_prediction(cycle_type end_cycle);
    double deferred_compile_time_sec() const {
        return _deferred_compile_time_sec;
    }

    ProportionalWorkloadStat summarize_proportional_tile(const Tile& tile) const;
    void record_proportional_skipped_tile(uint32_t core_id, const Tile& tile);
    void record_proportional_skipped_tile(
        uint32_t core_id, const Tile& tile,
        const Tile& compiled_representative);
    void materialize_tile(Tile& tile);
    void materialize_tile_range(std::deque<Tile>& queue, uint32_t begin,
                                uint32_t end);
    void finalize_proportional_workload_plan();
    void print_current_operation_workload_stat() const;
    void begin_attention_round_command_trace();
    void record_attention_round_command_trace(uint32_t core_id,
                                              const Tile& tile);

    bool _attention_round_trace_active = false;
    bool _attention_round_trace_file_initialized = false;
    std::string _attention_round_trace_operation;
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, uint32_t>>
        _attention_round_remaining_tiles;
    std::unordered_map<uint32_t, uint32_t> _attention_round_index;
    std::unordered_map<uint32_t, uint64_t> _attention_round_last_act;
    std::unordered_map<uint32_t, uint64_t> _attention_round_last_pre;


    std::vector<std::pair<std::string, uint32_t>> _stage_stats;
    std::vector<std::pair<std::string, uint32_t>> _op_stats;

    // Global PIM Stats
    uint64_t _total_pim_inst_count = 0;
    cycle_type _total_pim_duration = 0;

    struct DecodeCoreTiming {
        cycle_type cycle = 0;
        cycle_type compute = 0;
        cycle_type memory = 0;
        cycle_type idle = 0;
        cycle_type load = 0;
        cycle_type store = 0;
        std::array<cycle_type, 10> op_compute{};
        std::array<cycle_type, 10> op_stall{};
    };
    struct DecodePruningTemplate {
        cycle_type operation_cycles = 0;
        std::vector<DecodeCoreTiming> core_timing;
        std::vector<ProportionalWorkloadStat> core_workload;
        std::vector<uint64_t> channel_write_commands;
    };
    struct DecodePruningAccumulator {
        uint32_t samples = 0;
        uint64_t operation_cycles = 0;
        std::vector<DecodeCoreTiming> core_timing;
        std::vector<ProportionalWorkloadStat> core_workload;
        std::vector<uint64_t> channel_write_commands;
    };
    std::unordered_map<std::string, DecodePruningTemplate>
        _decode_pruning_templates;
    std::unordered_map<std::string, DecodePruningAccumulator>
        _decode_pruning_accumulators;
    std::unordered_map<uint32_t, std::vector<DecodeCoreTiming>>
        _decode_pruning_start_timing;
    std::unordered_map<
        uint32_t,
        std::unordered_map<uint32_t, ProportionalWorkloadStat>>
        _decode_pruning_sample_workload;
    bool _decode_pruning_pending = false;
    cycle_type _decode_pruning_pending_cycles = 0;
    uint32_t _decode_pruning_pending_operation = 0;
    std::unordered_map<uint32_t, ProportionalWorkloadStat>
        _decode_pruning_pending_workload;

    bool is_decode_pruning_target(const std::string& name) const;
    DecodeCoreTiming decode_core_timing_snapshot(uint32_t core_id) const;
    void prepare_decode_pruning_prediction();
    void capture_decode_pruning_template(uint32_t operation_id);

    // 以下组件均用于完成Scheduler对于其余组件进行控制与更新操作
    void bind_system(Client* client, PIM* dram, EventDrivenDram* event_driven_dram,
                     MyInterconnect* icnt,
                     const std::vector<std::unique_ptr<MyCore>>& cores);
    void sync_accelerated_cycles(cycle_type core_cycle, cycle_type dram_delta, cycle_type icnt_delta);
    private:
    Stage iterative_decode_stage() const;
    void complete_request(const Ptr<InferRequest>& request);
    void advance_iterative_inference(Stage completed_stage);

    Client* _client = nullptr;
    PIM* _dram = nullptr;
    EventDrivenDram* _event_driven_dram = nullptr;
    DramDataContainer* _data_container = nullptr;
    MyInterconnect* _icnt = nullptr;
    std::vector<MyCore*> _cores;


};
