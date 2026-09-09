#include "simulator.hpp"

#include "DRAM/Dram.h"
#include "DRAM/EventDrivenDram.h"
#include "clock_math.hpp"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <limits>
#include <string>
#include <boost/core/demangle.hpp>
#include <utility>

#include "Scheduler/MyScheduler.hpp"

#ifndef ENABLE_DRAM_ALIGNMENT_TRACE
#define ENABLE_DRAM_ALIGNMENT_TRACE 0
#endif

// #define TEST_EVENT_DRIVEN_

namespace {

std::unique_ptr<IDramBackend> make_dram_backend(
    DramMode mode, const SysConfig& config,
    DramDataContainer* data_container) {
    if (mode == DramMode::CYCLE_ACCURATE) {
        return std::make_unique<PIM>(config, data_container);
    }
    return std::make_unique<EventDrivenDram>(config, data_container);
}

}  // namespace

#if ENABLE_DRAM_ALIGNMENT_TRACE
static void log_dram_completion_trace(DramMode mode, uint32_t channel, const MemoryAccess* access) {
    if (!Config::system_config.record_dram_completion_trace) {
        return;
    }
    if (access == nullptr) {
        return;
    }

    static std::ofstream cycle_file;
    static std::ofstream event_file;
    static bool cycle_file_open = false;
    static bool event_file_open = false;
    static uint64_t cycle_sequence = 0;
    static uint64_t event_sequence = 0;

    const bool is_event_mode = mode == DramMode::EVENT_DRIVEN;
    auto& sequence = is_event_mode ? event_sequence : cycle_sequence;
    constexpr uint64_t kCompletionTraceLimit = 1000;
    if (sequence >= kCompletionTraceLimit) {
        return;
    }
    auto& file = is_event_mode ? event_file : cycle_file;
    auto& file_open = is_event_mode ? event_file_open : cycle_file_open;
    if (!file_open) {
        const std::string file_name = is_event_mode ? "/dram_completion_event.csv" : "/dram_completion_cycle.csv";
        const std::string path = Config::system_config.log_dir + file_name;
        file.open(path, std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            throw std::runtime_error(
                "Cannot open DRAM completion trace file: " + path);
        }
        file_open = true;
        file << "completion_sequence,channel,id,core_id,mem_id,buffer_id,address,type,rank,bankgroup,bank,row,col,enter_cycle,finish_cycle\n";
        if (!file.good()) {
            throw std::runtime_error(
                "Failed to write DRAM completion trace header: " + path);
        }
    }

    const uint32_t rank = MyAddressAllocator::get_rank_index(access->dram_address);
    const uint32_t bankgroup = MyAddressAllocator::get_bankgroup_index(access->dram_address);
    const uint32_t bank = MyAddressAllocator::get_bank_index(access->dram_address);
    const uint32_t row = MyAddressAllocator::get_row_index(access->dram_address);
    const uint32_t col = MyAddressAllocator::get_col_index(access->dram_address);

    file << sequence++ << ',' << channel << ',' << access->id << ','
         << access->core_id << ',' << access->mem_id << ',' << access->buffer_id << ','
         << access->dram_address << ','
         << memAccessTypeString(access->req_type) << ','
         << rank << ',' << bankgroup << ',' << bank << ',' << row << ',' << col << ','
         << access->dram_enter_cycle << ','
         << access->dram_finish_cycle << '\n';
    if (!file.good()) {
        throw std::runtime_error("Failed to write DRAM completion trace");
    }
}
#endif

#ifdef TEST_EVENT_DRIVEN_
static std::ofstream g_dram_compare_file;
static bool g_dram_compare_open = false;

static void log_dram_compare_trace(const char* source, uint32_t channel, const MemoryAccess* access) {
    if (!g_dram_compare_open) {
        const std::string path =
            Config::system_config.log_dir + "/dram_compare_trace.csv";
        g_dram_compare_file.open(path, std::ios::out | std::ios::trunc);
        if (!g_dram_compare_file.is_open()) {
            throw std::runtime_error(
                "Cannot open DRAM comparison trace file: " + path);
        }
        g_dram_compare_open = true;
        g_dram_compare_file << "source,channel,id,address,type,enter_cycle,finish_cycle\n";
    }
    g_dram_compare_file << source << ',' << channel << ',' << access->id << ',' << access->dram_address << ','
                        << memAccessTypeString(access->req_type) << ',' << access->dram_enter_cycle << ','
                        << access->dram_finish_cycle << '\n';
    if (!g_dram_compare_file.good()) {
        throw std::runtime_error("Failed to write DRAM comparison trace");
    }
}
#endif

Simulator::Simulator(const SysConfig& config) :_config(config), _core_cycles(0){

    phsim::clock::validate_frequency(_config.core_freq);
    phsim::clock::validate_frequency(_config.icnt_freq);
    phsim::clock::validate_frequency(_config.dram_freq);

    _n_cores = _config.num_cores;
    _n_memories = _config.dram_channels;
    memory_offset = _n_cores * _n_memories;

    _cores.resize(_config.num_cores);

    // Construct only the selected DRAM implementation. Simulator and Scheduler
    // use the shared backend contract from this point onward.
    _dram_mode = config.dram_trace_simulation_mode ? DramMode::EVENT_DRIVEN : DramMode::CYCLE_ACCURATE;
    ValidateDramBackendCapabilities(_dram_mode, config.mem_config.enable_self_refresh);
    _data_container = std::make_unique<DramDataContainer>(config);
    _dram_backend = make_dram_backend(
        _dram_mode, config, _data_container.get());
    #ifdef TEST_EVENT_DRIVEN_
    if (_dram_mode == DramMode::CYCLE_ACCURATE) {
        _comparison_event_driven_dram = std::make_unique<EventDrivenDram>(
            config, _data_container.get());
    }
    #endif
    assert(_dram_backend != nullptr);

    _dram_cycle_count = 0;
    _icnt_cycle_count = 0;

    // Test both the dram model  -------  can be removed
    std::vector<addr_type> test_addrs = {0x0, 0x40, 0x80, 0xC0, 0x100, 0x92b76};
    for (addr_type addr : test_addrs) {
        // MyAddress Allocator result
        uint32_t ch_allocator = MyAddressAllocator::get_channel_index(addr);
        // Selected backend result.
        MemoryAccess temp;
        temp.dram_address = addr;
        const uint32_t ch_backend = _dram_backend->get_channel_id(&temp);
        assert(ch_allocator == ch_backend);
        if (_comparison_event_driven_dram) {
            const uint32_t ch_event =
                _comparison_event_driven_dram->get_channel_id(&temp);
            assert(ch_backend == ch_event);
        }
    }

    // init_diag_file();
    std::string mode_str = (_dram_mode == DramMode::CYCLE_ACCURATE) ? "CYCLE_ACCURATE" : "EVENT_DRIVEN";
    spdlog::info("current dram is running in the {} mode ", mode_str);

    _icnt = std::make_unique<MyInterconnect>(_config);  // Create interconnect object

    // Create Core Object based on the
    for (int core_index = 0; core_index < _n_cores; core_index++) {
        _cores[core_index] = std::make_unique<MyCore>(core_index, _config);
    }

    _client = std::make_unique<Client>(_config);

    _scheduler = std::make_unique<MyScheduler>(_config, &_core_cycles,
                                               _data_container.get());
    _scheduler->bind_system(_client.get(), _dram_backend.get(),
                            _icnt.get(), _cores);

    spdlog::info("Simulation structure initialized");

    if (!_dram_backend || !_icnt || !_scheduler || !_client) {
        spdlog::error("Failed to initialize critical simulator components");
        throw std::runtime_error("Critical component initialization failed");
    }

    for (int i = 0; i < _n_cores; i++) {
        if (!_cores[i]) {
            spdlog::error("Failed to initialize core {}", i);
            throw std::runtime_error(
                "Core initialization failed for core " +
                std::to_string(i));
        }
    }
}

Simulator::~Simulator() {
    // Scheduler and transport components only borrow MemoryAccess pointers.
    // Destroy all borrowers before the Core-owned request pools.
    _scheduler.reset();
    _comparison_event_driven_dram.reset();
    _dram_backend.reset();
    _icnt.reset();
    _cores.clear();
    _client.reset();
    _stage_stats.clear();
}


void Simulator::run(std::string model_name) {
    spdlog::info("======Start Simulation=====");

    if (_dram_mode == DramMode::CYCLE_ACCURATE) {
        spdlog::info("====== Start Simulation (Mode: CYCLE_ACCURATE DRAM) =====");
    } else {
        spdlog::info("====== Start Simulation (Mode: EVENT_DRIVEN DRAM) =======");
    }

    _scheduler->launch(_model);
    spdlog::info("assign model {}", model_name);

    auto start_time = std::chrono::high_resolution_clock::now(); // record the simulation start time

    cycle(); // The entire simulation process of current tile

    auto end_time = std::chrono::high_resolution_clock::now(); // record the simulation end time
    std::chrono::duration<double> simulation_time = end_time - start_time;
    /*
    spdlog::info("               Simulation Time Efficiency Report                ");
    spdlog::info("DRAM Mode Used    : {}", _dram_mode == DramMode::CYCLE_ACCURATE ? "CYCLE_ACCURATE (NewtonSim)" : "EVENT_DRIVEN (Window)");
    spdlog::info("Total Core Cycles : {}", _core_cycles);
    spdlog::info("Real Time (Sec)   : {:.6f} seconds", simulation_time.count());
    if (simulation_time.count() > 0) {
        spdlog::info("Simulation Speed  : {:.0f} Core Cycles / second", _core_cycles / simulation_time.count());
    }
    spdlog::info("================================================================");
    */
}

void Simulator::update_req_stat(MemoryAccessType t, uint64_t &read_cnt, uint64_t &write_cnt, uint64_t &gwrite_cnt, uint64_t &other_cnt) {
    switch (t) {
        case MemoryAccessType::READ:
            read_cnt++;
            break;
        case MemoryAccessType::WRITE:
            write_cnt++;
            break;
        case MemoryAccessType::GWRITE:
            gwrite_cnt++;
            break;
        default:
            other_cnt++;
            break;
    }
}

void Simulator::launch_model(Ptr<Model> model) {_model = std::move(model);}

cycle_type Simulator::advance_accelerated_time(
    cycle_type target_core_cycle) {
    if (target_core_cycle < _core_cycles) {
        throw std::logic_error(
            "Accelerated time cannot move backwards from core cycle " +
            std::to_string(_core_cycles) + " to " +
            std::to_string(target_core_cycle));
    }
    if (target_core_cycle == _core_cycles) {
        return 0;
    }

    // _dram_cycle_count and _icnt_cycle_count identify the next edges that
    // have not yet executed. Move them to the first edge at or after the
    // target Core timestamp. Absolute conversion preserves fractional phase
    // across any number of acceleration jumps.
    const cycle_type target_dram_cycle = phsim::clock::scale_cycles_ceil(
        target_core_cycle, _config.core_freq, _config.dram_freq);
    const cycle_type target_icnt_cycle = phsim::clock::scale_cycles_ceil(
        target_core_cycle, _config.core_freq, _config.icnt_freq);
    if (target_dram_cycle < _dram_cycle_count ||
        target_icnt_cycle < _icnt_cycle_count) {
        throw std::logic_error(
            "Accelerated time would move a dependent clock backwards");
    }
    const cycle_type delta_dram_cycles =
        target_dram_cycle - _dram_cycle_count;
    const cycle_type delta_icnt_cycles =
        target_icnt_cycle - _icnt_cycle_count;

    _core_cycles = target_core_cycle;

    for (auto& core : _cores) {
        core->set_core_cycle(_core_cycles);
    }
    _client->set_client_cycle(_core_cycles);
    _scheduler->set_scheduler_cycles(_core_cycles);

    _dram_backend->synchronize_cycles(delta_dram_cycles);
    _dram_cycle_count = target_dram_cycle;
    _icnt->set_icnt_cycles(delta_icnt_cycles);
    _icnt_cycle_count = target_icnt_cycle;
    assert(_icnt->get_icnt_cycle() == _icnt_cycle_count);
    return delta_dram_cycles;
}

void Simulator::cycle() {
    double g_core_time_sec = 0.0;
    double g_sched_client_time_sec = 0.0;
    double g_program_compile_time_sec = 0.0;
    double g_deferred_compile_time_sec = 0.0;
    uint64_t g_core_count = 0;
    uint64_t g_sched_client_count = 0;

    double g_dram_time_sec = 0.0;
    uint64_t g_dram_time_count = 0;
    double g_icnt_time_sec = 0.0;
    uint64_t g_icnt_time_count = 0;

    OpStat op_stat;
    ModelStat model_stat;
    uint32_t tile_count;
    while (running()) {
        int model_id = 0;
        set_cycle_mask();

        if (_cycle_mask & CORE_MASK) { // Core Cycle

            // scheduler + client
            const bool program_missing_before_cycle = (_scheduler->_model_program == nullptr);
            const double deferred_compile_before_scheduler =
                _scheduler->deferred_compile_time_sec();
            auto t_sc_begin = std::chrono::high_resolution_clock::now(); // Record the start time of the scheduler + client
            while (_client->has_request() &&
                   _scheduler->can_accept_request()) {
                std::shared_ptr<InferRequest> infer_request = _client->pop_request();
                _scheduler->add_request(infer_request);
            }
            _client->cycle();

            while (_scheduler->has_completed_request()) {
                std::shared_ptr<InferRequest> response = _scheduler->pop_completed_request();
                _client->receive_response(response);
            }

            if (_scheduler->has_stage_changed()) {  // default false, if one stage program finished, set true
                _scheduler->reset_has_stage_changed_status(); // disable stage change
                // _icnt->log(_scheduler->get_prev_stage());
                update_stage_stat();
            }
            _scheduler->cycle();

            if (_scheduler->decode_pruning_prediction_pending()) {
                const cycle_type delta_core_cycles =
                    _scheduler->decode_pruning_prediction_cycles();
                if (delta_core_cycles >
                    std::numeric_limits<cycle_type>::max() - _core_cycles) {
                    throw std::overflow_error(
                        "Decode Pruning clock advance exceeds uint64_t range");
                }
                _scheduler->apply_decode_pruning_prediction();
                const cycle_type delta_dram_cycles =
                    advance_accelerated_time(_core_cycles +
                                             delta_core_cycles);
                _scheduler->apply_decode_pruning_dram_state(
                    delta_dram_cycles);
                spdlog::info(
                    "Applied Decode Pruning prediction: +{} core cycles",
                    delta_core_cycles);
                _scheduler->complete_decode_pruning_prediction(
                    _core_cycles);
            }

            if (_scheduler->_sim_accelerate) {
                _predicting_config = _scheduler->get_predicting_config();
            }
            auto t_sc_end = std::chrono::high_resolution_clock::now();  // Record the end time of the scheduler + client
            const double sched_client_elapsed_sec =
                std::chrono::duration<double>(t_sc_end - t_sc_begin).count();
            const double deferred_compile_scheduler_sec =
                _scheduler->deferred_compile_time_sec() -
                deferred_compile_before_scheduler;
            g_deferred_compile_time_sec +=
                deferred_compile_scheduler_sec;
            const bool program_created_this_cycle =
                program_missing_before_cycle && (_scheduler->_model_program != nullptr);
            if (program_created_this_cycle) {
                g_program_compile_time_sec += sched_client_elapsed_sec;
            } else {
                g_sched_client_time_sec += std::max(
                    0.0, sched_client_elapsed_sec -
                             deferred_compile_scheduler_sec);
            }
            g_sched_client_count++;  // record the cycle times


            // Core
            const double deferred_compile_before_core =
                _scheduler->deferred_compile_time_sec();
            auto t_core_begin = std::chrono::high_resolution_clock::now();
            for (int core_id = 0; core_id < _n_cores; core_id++) {
                auto finished_tile = _cores[core_id]->pop_finished_tile();   // First, get the finish tile of each Core
                if (finished_tile == nullptr) { }
                else if (finished_tile->status == Tile::Status::FINISH) {
                    if(!_scheduler->_sim_accelerate) {  // NO Acceleration
                        _scheduler->finish_tile(core_id, *finished_tile);  // if the tile finished, returned to the scheduler
                    }
                    else if (_scheduler->_config.accelerate_method == "naive") {
                        // The naive tile pruning operation, based on proportion
                        if (_scheduler->_tile_position < _scheduler->_sample_length - 1) {
                            _scheduler->finish_tile(core_id, *finished_tile);
                        }
                        // Check if acceleration prediction is complete and apply it
                        else if (_scheduler->_tile_position == _scheduler->_sample_length - 1) {
                            _scheduler->finish_last_mm_tile(core_id, *finished_tile);
                            _scheduler->compute_estimated_cycle();
                            const cycle_type estimated_cycle =
                                _scheduler->get_estimated_all_cycle();
                            if (estimated_cycle > _core_cycles) {
                                spdlog::info("Applying naive acceleration: jumping from cycle {} to estimated cycle {}", _core_cycles, estimated_cycle);
                                advance_accelerated_time(estimated_cycle);
                                _scheduler->update_stats_last_tile(core_id, *finished_tile);
                                _scheduler->refresh_status();
                            } else {
                                spdlog::info(
                                    "Naive prediction {} is not ahead of current cycle {}; retiring the sampled operation without a jump",
                                    estimated_cycle, _core_cycles);
                                _scheduler->update_stats_last_tile(core_id, *finished_tile);
                                _scheduler->refresh_status();
                            }
                        }
                    }
                    else if (_scheduler->_config.accelerate_method == "Loop_wise") {
                        _scheduler->finish_tile(core_id, *finished_tile);
                        // Check all cores have reached a stable state
                        bool all_cores_stable = false;
                        if (_scheduler->_current_op->get_optype() == "GEMM_Att") {
                            all_cores_stable = _scheduler->are_all_cores_head_second_done();
                        } else { // GEMM
                            all_cores_stable = _scheduler->are_all_cores_stable();
                        }

                        // If all cores have reached a stable state, apply prediction
                        if (all_cores_stable) {
                            /*
                             *  目前实现了对于延时的采样、统计计算、与更新操作
                             *  当前仍然缺少对于指令执行次数的采样、统计与更新操作，需要后续完成
                             *  计算cycle的信息基于调度器内部tile执行与采样得到,
                             *  同理，在Scheduler中需要构建相应的数据容器，统计instruction的delta信息,
                             *  记录core、DRAM、Interconncet中的一些counter，统计相应的delta
                             */
                            _scheduler->compute_estimated_cycle();
                            // _scheduler->_active_operation_stats[_scheduler->_operation_id].remain_tiles = 0;
                            const cycle_type estimated_cycle =
                                _scheduler->get_estimated_all_cycle();
                            bool core_clear = true;
                            for (int clear = 0; clear < _n_cores; clear++) {
                                core_clear = core_clear & _cores[clear]->_tiles.empty();
                            }
                            if (estimated_cycle > _core_cycles && core_clear){
                                std::string red = "\033[1;31m";
                                std::string reset = "\033[0m";
                                spdlog::info("{}Applying Loop_wise acceleration: jumping from cycle {} to estimated cycle {} {}", red, _core_cycles, estimated_cycle, reset);
                                _scheduler->_active_operation_stats[_scheduler->_operation_id].remain_tiles = 0;
                                advance_accelerated_time(estimated_cycle);
                                _scheduler->update_stats_last_tile(core_id, *finished_tile);
                                _scheduler->refresh_status();
                            }
                            else if (core_clear) {
                                // Sampling can finish after the predicted completion cycle,
                                // especially with more cores and only a few K-loops per core.
                                // In that case no forward jump is required, but the predicted
                                // remainder still has to be retired or the operation stalls.
                                spdlog::info("Loop_wise prediction {} is not ahead of current cycle {}; retiring remaining predicted tiles without a jump",
                                             estimated_cycle, _core_cycles);
                                _scheduler->_active_operation_stats[_scheduler->_operation_id].remain_tiles = 0;
                                _scheduler->update_stats_last_tile(core_id, *finished_tile);
                                _scheduler->refresh_status();
                            }
                            else {
                                _scheduler->update_stats_last_tile(core_id, *finished_tile);
                                _scheduler->refresh_status();
                            }
                        }
                        else {
                            _scheduler->update_stats_last_tile(core_id, *finished_tile);
                            _scheduler->refresh_status();
                        }
                    }
                    else if (_scheduler->_config.accelerate_method == "Proportional") {
                        _scheduler->finish_tile(core_id, *finished_tile);

                        bool cores_clear = true;
                        for (int clear = 0; clear < _n_cores; ++clear) {
                            cores_clear = cores_clear && _cores[clear]->_tiles.empty();
                        }

                        if (_scheduler->proportional_ready_to_predict() && cores_clear) {
                            if (!_scheduler->proportional_workload_applied()) {
                                _scheduler->mark_proportional_workload_applied();

                                ProportionalWorkloadStat workload;
                                for (const auto& [workload_core, core_workload] :
                                     _scheduler->proportional_estimated_workload()) {
                                    (void)workload_core;
                                    workload += core_workload;
                                }

                                _icnt->apply_estimated_workload(workload);

                                _dram_backend->apply_estimated_workload(
                                    workload);
                                spdlog::info(
                                    "Applied proportional ICNT/DRAM request compensation: reads {}, writes {}, PIM P_HEADER/GWRITE/COMP/READRES {}/{}/{}/{}, read bytes {}, write bytes {}",
                                    workload.memory_reads,
                                    workload.memory_writes,
                                    workload.pim_pheader,
                                    workload.pim_gwrite,
                                    workload.pim_comp,
                                    workload.pim_readres,
                                    workload.memory_read_bytes,
                                    workload.memory_write_bytes);
                            }
                            _scheduler->compute_estimated_cycle();
                            _scheduler->apply_proportional_core_timing();
                            const cycle_type estimated_cycle = _scheduler->get_estimated_all_cycle();
                            cycle_type skipped_dram_cycles = 0;
                            if (estimated_cycle > _core_cycles) {
                                spdlog::info(
                                    "Applying Proportional acceleration: jumping from core cycle {} to {}",
                                    _core_cycles, estimated_cycle);
                                skipped_dram_cycles =
                                    advance_accelerated_time(estimated_cycle);
                            }
                            else {
                                spdlog::info(
                                    "Proportional prediction {} is not ahead of current core cycle {}; no clock jump required",
                                    estimated_cycle, _core_cycles);
                            }

                            const bool has_tail = _scheduler->release_proportional_tail_tiles();
                            if (skipped_dram_cycles > 0) {
                                _dram_backend->apply_estimated_time(
                                    skipped_dram_cycles, has_tail);
                            }
                            if (!has_tail) {
                                assert(_scheduler->proportional_operation_complete(finished_tile->operation_id));
                                _scheduler->update_stats_last_tile(core_id, *finished_tile);
                                _scheduler->refresh_status();
                            }
                        }
                        else if (_scheduler->proportional_tail_phase() &&
                                 _scheduler->proportional_operation_complete(finished_tile->operation_id)) {
                            _scheduler->update_stats_last_tile(core_id, *finished_tile);
                            _scheduler->refresh_status();
                        }
                    }
                }
                // Issue new tile to core
                if (_scheduler->empty()){ continue; }
                else {
                    Tile &tile = _scheduler->top_tile(core_id);  // check current tile in scheduler, if available issue
                    if ((tile.status != Tile::Status::EMPTY) && _cores[core_id]->can_issue(tile)) {
                        // if has tile and core is available, issue the tile from scheduler to core
                        if (tile.status == Tile::Status::INITIALIZED) {
                            _cores[core_id]->issue(tile); // issue tile to core
                            _scheduler->get_tile(core_id); // delete tile from scheduler
                        }
                    }
                }
                _cores[core_id]->cycle();
            }
            _core_cycles++;
            auto t_core_end = std::chrono::high_resolution_clock::now(); // End time of all the core
            const double core_elapsed_sec =
                std::chrono::duration<double>(t_core_end - t_core_begin)
                    .count();
            const double deferred_compile_core_sec =
                _scheduler->deferred_compile_time_sec() -
                deferred_compile_before_core;
            g_deferred_compile_time_sec += deferred_compile_core_sec;
            g_core_time_sec += std::max(
                0.0, core_elapsed_sec - deferred_compile_core_sec);
            g_core_count++;
        }

        // Advance the selected DRAM backend. CA consumes one NewtonSim tick;
        // ED intentionally performs no work until a timestamped request or
        // pending event is observed.
        if (_cycle_mask & DRAM_MASK) {
            auto t_dram_begin = std::chrono::high_resolution_clock::now();
            _dram_backend->advance_cycle(
                _dram_cycle_count, _predicting_config._unstable_length,
                _predicting_config._sample_length,
                _predicting_config._inst_ratio);
            _dram_cycle_count++;
            _icnt->reset_dram_interface_valid();
            auto t_dram_end = std::chrono::high_resolution_clock::now();
            g_dram_time_sec +=
                std::chrono::duration<double>(t_dram_end - t_dram_begin)
                    .count();
            g_dram_time_count++;
        }

        if (_cycle_mask & ICNT_MASK) {
            auto t_icnt_begin = std::chrono::high_resolution_clock::now();

            // Core <-> interconnect transport is backend-independent.
            for (int core_id = 0; core_id < _n_cores; core_id++) {
                for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
                    const uint32_t core_node =
                        core_id * _n_memories + mem_id;
                    if (_cores[core_id]->has_memory_request(mem_id)) {
                        MemoryAccess* front =
                            _cores[core_id]->top_memory_request(mem_id);
                        front->core_id = core_id;
                        front->mem_id = mem_id;
                        if (!_icnt->is_full(core_node, front)) {
                            // Map exactly once, when the interconnect accepts
                            // the request. Re-mapping on a back-pressured cycle
                            // would corrupt the logical-to-physical relation.
                            front->logical_dram_address = front->dram_address;
                            front->dram_address =
                                TwoLevelPageMapper::map_logical_address(
                                    front->logical_dram_address);
                            assert(MyAddressAllocator::get_channel_index(
                                       front->dram_address) == mem_id);
                            _icnt->push(core_node, get_dest_node(front), front);
                            _cores[core_id]->pop_memory_request(mem_id);
                        }
                    }
                    if (!_icnt->is_empty(core_node)) {
                        MemoryAccess* response = _icnt->top(core_node);
                        _icnt->pop(core_node);
                        _cores[core_id]->push_memory_response(response);
                    }
                }
            }

            // Interconnect <-> DRAM transport is shared. The backend hook
            // stamps ED arrival cycles and is a no-op for CA.
            for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
                const uint32_t memory_node = memory_offset + mem_id;
                bool request_accepted = false;
                if (!_icnt->is_empty(memory_node)) {
                    MemoryAccess* memory_req = _icnt->top(memory_node);
                    _dram_backend->prepare_request(memory_req,
                                                   _dram_cycle_count);
                    if (!_dram_backend->is_full(mem_id, memory_req) &&
                        _icnt->dram_push_valid(mem_id)) {
                        _dram_backend->push(mem_id, memory_req);
#ifdef TEST_EVENT_DRIVEN_
                        if (_comparison_event_driven_dram) {
                            auto deep_copy_req = memory_req->clone();
                            auto* event_req = deep_copy_req.get();
                            _event_driven_compare_copies.push_back(
                                std::move(deep_copy_req));
                            _comparison_event_driven_dram->prepare_request(
                                event_req, _dram_cycle_count);
                            _comparison_event_driven_dram->push(mem_id,
                                                                event_req);
                        }
#endif
                        _icnt->pop(memory_node);
                        _icnt->consume_dram_push(mem_id);
                        request_accepted = true;
                    }
                }
                if (!request_accepted) {
                    _dram_backend->schedule_pending_work(mem_id,
                                                         _dram_cycle_count);
#ifdef TEST_EVENT_DRIVEN_
                    if (_comparison_event_driven_dram) {
                        _comparison_event_driven_dram->schedule_pending_work(
                            mem_id, _dram_cycle_count);
                    }
#endif
                }

#ifdef TEST_EVENT_DRIVEN_
                if (_comparison_event_driven_dram &&
                    !_comparison_event_driven_dram->is_empty(mem_id)) {
                    auto* event_resp =
                        _comparison_event_driven_dram->top(mem_id);
                    log_dram_compare_trace("event", mem_id, event_resp);
                    _comparison_event_driven_dram->pop(mem_id);
                }
#endif

                if (!_dram_backend->is_empty(mem_id) &&
                    !_icnt->is_full(memory_node,
                                    _dram_backend->top(mem_id)) &&
                    _icnt->dram_pop_valid(mem_id)) {
                    MemoryAccess* response = _dram_backend->top(mem_id);
#ifdef TEST_EVENT_DRIVEN_
                    if (_comparison_event_driven_dram) {
                        log_dram_compare_trace("golden", mem_id, response);
                    }
#endif
#if ENABLE_DRAM_ALIGNMENT_TRACE
                    log_dram_completion_trace(_dram_mode, mem_id, response);
#endif
                    _icnt->push(memory_node, get_dest_node(response), response);
                    _dram_backend->pop(mem_id);
                    _icnt->consume_dram_pop(mem_id);
                }
            }
            _icnt->cycle();
            ++_icnt_cycle_count;
            assert(_icnt->get_icnt_cycle() == _icnt_cycle_count);
            auto t_icnt_end = std::chrono::high_resolution_clock::now();
            g_icnt_time_sec +=
                std::chrono::duration<double>(t_icnt_end - t_icnt_begin)
                    .count();
            g_icnt_time_count++;
        }
    }
    spdlog::info(">>>>>> Simulation Finished <<<<<<");

    size_t outstanding_memory_accesses = 0;
    for (const auto& core : _cores) {
        outstanding_memory_accesses += core->outstanding_memory_accesses();
    }
    spdlog::info("Outstanding MemoryAccess requests at simulation end: {}",
                 outstanding_memory_accesses);
    if (outstanding_memory_accesses != 0) {
        throw std::logic_error(
            "Simulation finished with " +
            std::to_string(outstanding_memory_accesses) +
            " outstanding MemoryAccess requests");
    }

    const double g_dram_icnt_time_sec = g_dram_time_sec + g_icnt_time_sec;
    const uint64_t g_dram_icnt_count = g_dram_time_count + g_icnt_time_count;
    const double total_component_time_sec = g_sched_client_time_sec + g_core_time_sec + g_dram_icnt_time_sec;
    const double total_compile_time_sec =
        g_program_compile_time_sec + g_deferred_compile_time_sec;
    const double total_component_time_including_compile_sec =
        total_component_time_sec + total_compile_time_sec;

    spdlog::info("================================================================");
    spdlog::info("Component Real Time Breakdown");
    spdlog::info("program compile time    : {:.6f} s", g_program_compile_time_sec);
    spdlog::info("deferred compile time   : {:.6f} s", g_deferred_compile_time_sec);
    spdlog::info("total compile time      : {:.6f} s", total_compile_time_sec);
    spdlog::info("client+scheduler time  : {:.6f} s ({:.2f}%)", g_sched_client_time_sec,
                 total_component_time_sec > 0.0 ? g_sched_client_time_sec / total_component_time_sec * 100.0 : 0.0);
    spdlog::info("core time              : {:.6f} s ({:.2f}%)", g_core_time_sec,
                 total_component_time_sec > 0.0 ? g_core_time_sec / total_component_time_sec * 100.0 : 0.0);
    spdlog::info("dram+icnt time         : {:.6f} s ({:.2f}%)", g_dram_icnt_time_sec,
                 total_component_time_sec > 0.0 ? g_dram_icnt_time_sec / total_component_time_sec * 100.0 : 0.0);
    spdlog::info("component timed total  : {:.6f} s", total_component_time_sec);
    spdlog::info("runtime excl compile   : {:.6f} s", total_component_time_sec);
    spdlog::info("component total incl compile : {:.6f} s", total_component_time_including_compile_sec);
    spdlog::info("client+scheduler count : {}", g_sched_client_count);
    spdlog::info("core count             : {}", g_core_count);
    spdlog::info("dram cycle count       : {}", g_dram_time_count);
    spdlog::info("icnt+dram op count     : {}", g_icnt_time_count);
    spdlog::info("dram+icnt count        : {}", g_dram_icnt_count);
    spdlog::info("================================================================");

    spdlog::info(">>>>>> Core Stats <<<<<<");
    /* Print simulation stats */
    for (int core_id = 0; core_id < _n_cores; core_id++) {
        _cores[core_id]->print_stats();
        _cores[core_id]->log();
    }
    spdlog::info(">>>>>> ICNT Stats <<<<<<");
    _icnt->print_stats();
    // _icnt->log();
    _dram_backend->print_stat();
    spdlog::info(">>>>>> Scheduler Stats <<<<<<");
    _scheduler->print_stat();
    _scheduler->print_op_stat();
    _client->write_request_stats();
    log_data_container_stat();
    log_virtual_memory_stat();
    log_stage_stat();
}


bool Simulator::running() {      // return ture if there is any instance is running
    bool running = false;

    for (auto &core : _cores) {
        running = running || core->running();
    }
    running = running || _icnt->running();
    running = running || _dram_backend->running();
    running = running || _scheduler->running();
    running = running || _client->running();
    return running;
}


void Simulator::set_cycle_mask() {
    _cycle_mask = phsim::clock::select_next_mask({{{
        _core_cycles, _config.core_freq, CORE_MASK}, {
        _dram_cycle_count, _config.dram_freq, DRAM_MASK}, {
        _icnt_cycle_count, _config.icnt_freq, ICNT_MASK}}});
}


uint32_t Simulator::get_dest_node(MemoryAccess *access) {
    // memory_offset = core size * dram_channels, for core, the reset dram_channels port is for the dram
    if (access->request) {
        return memory_offset + _dram_backend->get_channel_id(access);  // core to memory
    }
    else {
        return access->core_id * _config.dram_channels + access->mem_id; // memory to core
    }
}


void Simulator::update_stage_stat() {
    Stage done_stage = _scheduler->get_prev_stage();
    const double memory_bw_util = _dram_backend->get_avg_bw_util();
    const uint64_t pim_cycles = _dram_backend->get_avg_pim_cycle();

    _stage_stats.push_back(StageStat{.stage = done_stage,
                                     .done_cycle = _core_cycles,
                                     .pim_cycles = pim_cycles,
                                     .npu_cycles = 0,
                                     .mem_bw_util = memory_bw_util});
}

void Simulator::log_stage_stat() {
    if (_stage_stats.empty()) {
        return; // ensure the valid data
    }

    std::string fname = Config::system_config.log_dir + "/_summary.tsv";
    std::ofstream ofile(fname);
    if (!ofile.is_open()) {
        throw std::runtime_error("Cannot open stage statistics file: " + fname);
    }

    std::string header = "";
    header += "Stage\t";
    header += "total_cycles\t";
    header += "pim_cycles\t";
    header += "mem_bw_util\t";
    ofile << header + "\n";

    cycle_type prev_cycle = 0;

    for (int i = 0; i < _stage_stats.size(); i++) {
        StageStat stage_stat = _stage_stats[i];
        std::string stage_row = "";

        const cycle_type total_cycle = stage_stat.done_cycle - prev_cycle;
        prev_cycle = stage_stat.done_cycle;
        stage_row += stageToString(stage_stat.stage) + "\t";
        stage_row += std::to_string(total_cycle) + "\t";
        stage_row += std::to_string(stage_stat.pim_cycles) + "\t";
        stage_row += std::to_string(stage_stat.mem_bw_util) + "\t";

        ofile << stage_row + "\n";
    }
    ofile.close();
}

void Simulator::log_data_container_stat() const {
    if (!_data_container) {
        return;
    }

    nlohmann::json output;
    output["enabled"] = _data_container->enabled();
    output["burst_length"] = _data_container->burst_length();
    output["dq_bytes"] = _data_container->dq_bytes();
    output["burst_bytes"] = _data_container->burst_bytes();
    output["stored_column_count"] =
        _data_container->stored_column_count();
    output["peak_stored_column_count"] =
        _data_container->peak_stored_column_count();
    output["dram_payload_bytes"] =
        _data_container->dram_payload_bytes();
    output["pim_payload_bytes"] =
        _data_container->pim_payload_bytes();
    output["resident_payload_bytes"] =
        _data_container->resident_payload_bytes();
    output["peak_resident_payload_bytes"] =
        _data_container->peak_resident_payload_bytes();
    output["nonzero_payload_bytes"] =
        _data_container->nonzero_payload_bytes();
    output["content_fingerprint"] =
        std::to_string(_data_container->content_fingerprint());
    output["response_counts"] = {
        {"read", _data_container->read_response_count()},
        {"write", _data_container->write_response_count()},
        {"pheader", _data_container->pheader_response_count()},
        {"gwrite", _data_container->gwrite_response_count()},
        {"comp", _data_container->comp_response_count()},
        {"readres", _data_container->readres_response_count()}};
    output["payload_limit_bytes"] =
        _data_container->max_resident_payload_bytes();
    output["payload_limit_enabled"] =
        _data_container->max_resident_payload_bytes() != 0;
    output["channels"] = nlohmann::json::array();

    for (uint32_t channel = 0;
         channel < _data_container->channel_count(); ++channel) {
        output["channels"].push_back({
            {"channel", channel},
            {"pim_input_payload_bytes",
             _data_container->pim_input_payload_bytes(channel)},
            {"peak_pim_input_payload_bytes",
             _data_container->peak_pim_input_payload_bytes(channel)},
            {"pim_output_payload_bytes",
             _data_container->pim_output_payload_bytes(channel)},
            {"peak_pim_output_payload_bytes",
             _data_container->peak_pim_output_payload_bytes(channel)}});
    }

    const std::string file_name =
        Config::system_config.log_dir + "/data_container_stats.json";
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open DataContainer statistics file: " + file_name);
    }
    file << output.dump(2) << '\n';
    if (!file.good()) {
        throw std::runtime_error(
            "Failed to write DataContainer statistics file: " + file_name);
    }

    spdlog::info(
        "DataContainer: enabled={}, resident={} B, peak={} B, "
        "stored_columns={}, peak_stored_columns={}",
        _data_container->enabled(),
        _data_container->resident_payload_bytes(),
        _data_container->peak_resident_payload_bytes(),
        _data_container->stored_column_count(),
        _data_container->peak_stored_column_count());
}

void Simulator::log_virtual_memory_stat() const {
    nlohmann::json output;
    output["enabled"] = TwoLevelPageMapper::is_enabled();
    output["mapped_page_count"] =
        TwoLevelPageMapper::mapped_page_count();
    output["mapping_call_count"] =
        TwoLevelPageMapper::mapping_call_count();
    output["changed_mapping_count"] =
        TwoLevelPageMapper::changed_mapping_count();
    output["mapping_pair_xor"] =
        std::to_string(TwoLevelPageMapper::mapping_pair_xor());
    output["mapping_pair_sum"] =
        std::to_string(TwoLevelPageMapper::mapping_pair_sum());
    output["page_table_fingerprint"] =
        std::to_string(TwoLevelPageMapper::page_table_fingerprint());
    output["mapping_calls_by_channel"] =
        TwoLevelPageMapper::mapping_calls_by_channel();

    const std::string file_name =
        Config::system_config.log_dir + "/virtual_memory_stats.json";
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open virtual-memory statistics file: " + file_name);
    }
    file << output.dump(2) << '\n';
    if (!file.good()) {
        throw std::runtime_error(
            "Failed to write virtual-memory statistics file: " + file_name);
    }

    spdlog::info(
        "Virtual memory: enabled={}, mapped_pages={}, mapping_calls={}, "
        "changed_mappings={}",
        TwoLevelPageMapper::is_enabled(),
        TwoLevelPageMapper::mapped_page_count(),
        TwoLevelPageMapper::mapping_call_count(),
        TwoLevelPageMapper::changed_mapping_count());
}
