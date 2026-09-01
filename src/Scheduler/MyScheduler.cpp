#include "MyScheduler.hpp"
#include "../DRAM/Dram.h"
#include "../DRAM/EventDrivenDram.h"
#include "../Core/MyCore.hpp"
#include "../Interconnect/MyInterconnect.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "../Client/Client.h"

namespace {

std::string format_request_ids(
    const std::vector<Ptr<InferRequest>>& requests) {
    std::ostringstream output;
    for (size_t index = 0; index < requests.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << requests[index]->id;
    }
    return output.str();
}

}  // namespace

MyScheduler::MyScheduler(const SysConfig& config, const cycle_type *core_cycle,
                         DramDataContainer* data_container)
    : _config(config), _core_cycle(core_cycle), _cycles(0),
      _data_container(data_container) {
    _max_batch_size = config.max_batch_size;   // 256;   // config.max_batch_size;
    _max_active_reqs = config.max_active_reqs;
    _continuous_batching = config.batch_scheduler == "continuous";

    _init_stage = Stage::Prefill;
    _model_program = nullptr;
    _has_stage_changed = false;
    _stage = _init_stage;
    program_start_cycle = 0;
    _last_request_cycle = 0;

    _test_single_op = config.test_single_op;
    _test_single_op_name = config.test_single_op_name;
    _test_single_op_type = getOpType(_test_single_op_name);

    _test_multi_layer = config.test_multi_layer;
    _test_multi_layer_name = config.test_multi_layer_name;

    _output_token_iteration_enable =
        config.output_token_iteration_enable && !_test_single_op &&
        !_test_multi_layer;
    if (config.output_token_iteration_enable &&
        (_test_single_op || _test_multi_layer)) {
        spdlog::warn(
            "output_token_iteration_enable is ignored in single-op and "
            "multi-layer test modes");
    }
    if (_output_token_iteration_enable) {
        spdlog::info(
            "Output-token iteration enabled; Decode backend: {}",
            config.allocation_scheme == "NPU" ? "NPU/SA" : "PIM");
    }

    _acceletare_ctrl = config.accelerate_ctrl;

    for (int i=0; i<config.num_cores;i++) {
        _core_executable_tile_queue[i] = std::deque<Tile>();  // initial Tile queue for each core
    }

    _core_rr_id = 0;
    _active_reqs = 0;
    spdlog::info(
        "Batch scheduler: {} (max_batch_size={}, max_active_reqs={})",
        config.batch_scheduler, _max_batch_size, _max_active_reqs);
}

void MyScheduler::launch(Ptr<Model> model) {  // Register the model on memory
    _model = model;
    spdlog::info("MODEL {} Launched in Scheduler", model->get_name());
}

bool MyScheduler::can_accept_request() const {
    return _request_queue.size() < _max_active_reqs;
}

void MyScheduler::add_request(std::shared_ptr<InferRequest> request) {
    if (!can_accept_request()) {
        throw std::runtime_error(fmt::format(
            "Scheduler cannot admit request {}: max_active_reqs={} has been "
            "reached",
            request->id, _max_active_reqs));
    }
    _request_queue.push_back(request);
    if (_continuous_batching) {
        if (request->is_initiated) {
            throw std::logic_error(
                "A newly admitted continuous-batching request is already "
                "initialized");
        }
        _waiting_prefill_queue.push_back(request);
        ++_active_reqs;
    }
    if (_stage == Stage::Finish && _model_program == nullptr &&
        _breq.empty()) {
        _stage = _init_stage;
    }
    spdlog::info(
        "Scheduler admitted request {}: {}/{} resident requests",
        request->id, _request_queue.size(), _max_active_reqs);
    _last_request_cycle = _cycles;
}

bool MyScheduler::has_completed_request() { return !_completed_request_queue.empty(); }

bool MyScheduler::empty() { return _model_program == nullptr; }

bool MyScheduler::running() { return !_request_queue.empty() || !_completed_request_queue.empty(); }

std::shared_ptr<InferRequest> MyScheduler::pop_completed_request() {
    // spdlog::info("Scheduler::pop_completed_request()");
    auto completed_req = _completed_request_queue.front();
    _completed_request_queue.pop();
    return completed_req;
}

bool MyScheduler::has_stage_changed() const { return _has_stage_changed;}
void MyScheduler::reset_has_stage_changed_status() { _has_stage_changed = false; }
Stage MyScheduler::get_prev_stage() const { return _prev_stage; }
uint32_t MyScheduler::count_active_operations() const { return _active_operation_stats.size(); }

void MyScheduler::init_batches() {
    // Initial a batch for inference, based on the maximum batch_size and present request in the queue, generate a batch for inference
    uint32_t batch_size = 0;

    // add the active requests from request queue to _breq for batch inference
    const uint32_t batch_capacity =
        std::min(_max_batch_size, _max_active_reqs);
    for (auto request : _request_queue) {
        if (batch_size >= batch_capacity) {  // if _max_batch_size = 1 for single batch inference
            break;
        }
        if (!request->is_initiated) {
            _breq.push_back(request);
            _active_reqs++;
            batch_size++;
        }
    }
    //TODO:: Multi Batch Inference
}


bool MyScheduler::form_continuous_batch() {
    if (!_continuous_batching) {
        return false;
    }
    if (_model_program != nullptr || !_breq.empty()) {
        throw std::logic_error(
            "Cannot form a continuous batch while another batch is active");
    }

    const size_t batch_capacity =
        std::min(_max_batch_size, _max_active_reqs);
    if (batch_capacity == 0) {
        throw std::logic_error("Continuous batch capacity is zero");
    }

    bool prefill_wait_elapsed = false;
    if (!_waiting_prefill_queue.empty()) {
        const cycle_type arrival =
            _waiting_prefill_queue.front()->arrival_cycle;
        prefill_wait_elapsed =
            _cycles > arrival && _cycles - arrival > 64;
    }

    // Prefill and Decode are intentionally kept in separate hardware
    // programs. Once a resident prompt has waited through the existing
    // batching window, schedule it before another Decode round. Otherwise a
    // full Decode-ready queue could keep an admitted prompt waiting forever.
    const bool choose_prefill = prefill_wait_elapsed;

    if (choose_prefill) {
        const size_t available_slots = batch_capacity;
        while (!_waiting_prefill_queue.empty() &&
               _breq.size() < available_slots) {
            const auto& request = _waiting_prefill_queue.front();
            _breq.push_back(request);
            _waiting_prefill_queue.pop_front();
        }
        if (_breq.empty()) {
            throw std::logic_error(
                "Continuous Prefill selection produced an empty batch");
        }
        _stage = Stage::Prefill;
    } else if (!_ready_decode_queue.empty()) {
        while (!_ready_decode_queue.empty() &&
               _breq.size() < batch_capacity) {
            _breq.push_back(_ready_decode_queue.front());
            _ready_decode_queue.pop_front();
        }
        _stage = iterative_decode_stage();
    } else {
        return false;
    }

    _current_continuous_batch_id = _next_continuous_batch_id++;
    spdlog::info(
        "Continuous batch {} formed: stage={}, requests=[{}], batch_size={}",
        _current_continuous_batch_id, stageToString(_stage),
        format_request_ids(_breq), _breq.size());
    return true;
}


void MyScheduler::complete_continuous_batch(Stage completed_stage) {
    if (!_continuous_batching || _breq.empty()) {
        throw std::logic_error("No continuous batch is available to complete");
    }

    const std::string completed_ids = format_request_ids(_breq);
    if (completed_stage == Stage::Prefill) {
        for (const auto& request : _breq) {
            if (request->is_initiated) {
                throw std::logic_error(fmt::format(
                    "Request {} entered Prefill more than once", request->id));
            }
            request->is_initiated = true;
            request->prefill_completed_cycle = _cycles;
            request->prefill_completed = true;
            if (request->generated >= request->output_size) {
                complete_request(request);
            } else {
                _ready_decode_queue.push_back(request);
            }
        }
    } else if (completed_stage == Stage::Decode ||
               completed_stage == Stage::NPU_Decode) {
        for (const auto& request : _breq) {
            if (!request->is_initiated ||
                request->generated >= request->output_size) {
                throw std::logic_error(fmt::format(
                    "Request {} entered an invalid continuous Decode "
                    "iteration ({}/{})",
                    request->id, request->generated, request->output_size));
            }
            request->generated++;
            if (!request->first_token_generated) {
                request->first_token_cycle = _cycles;
                request->first_token_generated = true;
            }
            spdlog::info(
                "Scheduler:: Request {} generated token {}/{}",
                request->id, request->generated, request->output_size);
            if (request->generated >= request->output_size) {
                complete_request(request);
            } else {
                _ready_decode_queue.push_back(request);
            }
        }
    } else {
        throw std::logic_error(
            "Continuous batching only supports Prefill and Decode stages");
    }

    spdlog::info(
        "Continuous batch {} completed: stage={}, requests=[{}]",
        _current_continuous_batch_id, stageToString(completed_stage),
        completed_ids);
    _breq.clear();

    if (_request_queue.empty()) {
        _stage = Stage::Finish;
    } else if (!_ready_decode_queue.empty()) {
        _stage = iterative_decode_stage();
    } else {
        _stage = Stage::Prefill;
    }
}


void MyScheduler::cleanup_batch(std::vector<Ptr<InferRequest>> batch_request) {
    for (auto request : batch_request) {
        // iteration done -> update request stat in batch
        request->is_initiated = true;
        request->generated++;

        assert(_stage == Stage::Finish);
        complete_request(request);
    }
}

Stage MyScheduler::iterative_decode_stage() const {
    return _config.allocation_scheme == "NPU" ? Stage::NPU_Decode
                                               : Stage::Decode;
}

void MyScheduler::complete_request(const Ptr<InferRequest>& request) {
    request->is_initiated = true;
    _model->_stored_KVCache.erase(request.get());
    _completed_request_queue.push(request);

    if (!_output_token_iteration_enable) {
        spdlog::info(
            "Scheduler:: The inference process of request {} is done",
            request->id);
    }

    for (auto itr = _request_queue.begin(); itr != _request_queue.end();) {
        if ((*itr)->id == request->id) {
            itr = _request_queue.erase(itr);
            if (_active_reqs > 0) {
                _active_reqs--;
            }
            if (!_output_token_iteration_enable) {
                spdlog::info(
                    "Scheduler::Free the KV cache of the done request {} ",
                    request->id);
            }
        } else {
            ++itr;
        }
    }
    if (_output_token_iteration_enable) {
        spdlog::info(
            "Scheduler:: Request {} completed after generating {}/{} tokens; "
            "KV cache released",
            request->id, request->generated, request->output_size);
    }
}

void MyScheduler::advance_iterative_inference(Stage completed_stage) {
    if (completed_stage != Stage::Prefill &&
        completed_stage != Stage::Decode &&
        completed_stage != Stage::NPU_Decode) {
        _stage = Stage::Finish;
        return;
    }

    if (completed_stage == Stage::Prefill) {
        for (const auto& request : _breq) {
            request->is_initiated = true;
        }
    } else {
        for (const auto& request : _breq) {
            if (request->generated >= request->output_size) {
                throw std::runtime_error(fmt::format(
                    "Request {} entered an extra Decode iteration ({}/{})",
                    request->id, request->generated, request->output_size));
            }
            request->generated++;
            spdlog::info(
                "Scheduler:: Request {} generated token {}/{}",
                request->id, request->generated, request->output_size);
        }
    }

    std::vector<Ptr<InferRequest>> pending_requests;
    pending_requests.reserve(_breq.size());
    for (const auto& request : _breq) {
        if (request->generated >= request->output_size) {
            complete_request(request);
        } else {
            pending_requests.push_back(request);
        }
    }
    _breq = std::move(pending_requests);

    if (!_breq.empty()) {
        _stage = iterative_decode_stage();
    } else if (!_request_queue.empty()) {
        // More requests may be waiting behind max_batch_size.
        _stage = _init_stage;
    } else {
        _stage = Stage::Finish;
    }
}


void MyScheduler::cycle() {
    bool step_next_stage = _model_program == nullptr; // no model program

    if (step_next_stage && _continuous_batching && _breq.empty()) {
        form_continuous_batch();
    }
    else if (step_next_stage && _stage == _init_stage &&
             !_request_queue.empty() &&
             _cycles - _last_request_cycle > 64) {   // Init one inference batch
        init_batches();
    }

    // TODO:: Multi-batch inference
    bool program_none = _model_program == nullptr;
    bool exist_request = !_breq.empty();
    if (program_none && exist_request) {   // has request but no program
        if (_stage == Stage::Finish) { // Current request finished
            if (_continuous_batching) {
                throw std::logic_error(
                    "Continuous batch cannot be active in Finish stage");
            }
            cleanup_batch(_breq);
            _breq.clear();
            // The Client may still hold requests that were back-pressured by
            // max_active_reqs. Return to the initial stage so the next
            // admitted batch can start instead of leaving the scheduler
            // permanently parked at Finish.
            _stage = _init_stage;
        }
        else {
            std::string red = "\033[1;31m";
            std::string reset = "\033[0m";
            spdlog::info("{}----------Make Inference Program----------{}", red, reset);
            make_program();  // make new program
        }
    }
    _cycles++;
}


void MyScheduler::make_program() {
    Config::system_config.decode_pruning_compile_context = false;
    if (_test_single_op) {
        auto batch_for_single_op = std::make_shared<BatchedRequest>(_breq);
        spdlog::info("Create a New Program to test the single operation: {}, (batch.size: {})",_test_single_op_name, batch_for_single_op->_batch_size);
        _stage = Stage::Single_test;
        _model_program = std::make_unique<StageProgram>(_test_single_op_type, batch_for_single_op, _stage, _data_container);
        spdlog::info("*************************************************************");
        spdlog::info("Initialize Single-Layer Test Program for Inference");
        spdlog::info("*************************************************************");
    }
    else if (_test_multi_layer){
        auto batch_for_multi_layer = std::make_shared<BatchedRequest>(_breq);
        spdlog::info("Create a New Program to test the multi_layer operation: {}, (batch.size: {})", _test_multi_layer, batch_for_multi_layer->_batch_size);
        _stage = Stage::Multi_test;
        Config::system_config.decode_pruning_compile_context =
            _config.decode_pruning_enabled &&
            (_test_multi_layer_name == "decode" ||
             _test_multi_layer_name == "npu_decode");
        _model_program = std::make_unique<StageProgram>(_model, _test_multi_layer_name, batch_for_multi_layer, _stage, _data_container);
        Config::system_config.decode_pruning_compile_context = false;
        spdlog::info("*************************************************************");
        spdlog::info("Initialize Multi-Layer Test Program for Inference");
        spdlog::info("*************************************************************");
    }
    else if (_stage == Stage::Prefill) {
        spdlog::info("*************************************************************");
        spdlog::info("Initialize Prefill Stage of Model Inference");
        spdlog::info("*************************************************************");
        auto batch_for_prefill = std::make_shared<BatchedRequest>(_breq);
        spdlog::info("New Program for SA (batch.size: {})", batch_for_prefill->_reqs.size());
        _model_program = std::make_unique<StageProgram>(_model, batch_for_prefill, StagePlatform::SA, _stage, _data_container);
    }
    else if (_stage == Stage::Decode) {
        spdlog::info("*************************************************************");
        spdlog::info("Initialize Decode Stage of Model Inference");
        spdlog::info("*************************************************************");
        auto batch_for_decode = std::make_shared<BatchedRequest>(_breq);
        spdlog::info("New Program for PIM  (batch.size: {})", batch_for_decode->_reqs.size());
        Config::system_config.decode_pruning_compile_context =
            _config.decode_pruning_enabled;
        _model_program = std::make_unique<StageProgram>(_model, batch_for_decode, StagePlatform::PIM, _stage, _data_container);
        Config::system_config.decode_pruning_compile_context = false;
    }
    else if (_stage == Stage::NPU_Decode) {
        spdlog::info("*************************************************************");
        spdlog::info("Initialize Decode Stage of Model Inference");
        spdlog::info("*************************************************************");
        auto batch_for_decode = std::make_shared<BatchedRequest>(_breq);
        spdlog::info("New Program for SA (batch.size: {})", batch_for_decode->_reqs.size());
        Config::system_config.decode_pruning_compile_context =
            _config.decode_pruning_enabled;
        _model_program = std::make_unique<StageProgram>(_model, batch_for_decode, StagePlatform::SA, _stage, _data_container);
        Config::system_config.decode_pruning_compile_context = false;
    }
    program_start_cycle = _cycles;
    refresh_status();
}


// 最新的refresh status
void MyScheduler::refresh_status() {
    if (_model_program != nullptr) {
        if (_model_program->check_finish()) {  // empty op map
            finish_program();
        }
        else if (is_executable_tile_empty() && _active_operation_stats[_operation_id].remain_tiles == 0) {
            // if all tile queues are empty, current op is finished
            // auto op = _model_program->get_executable_operations().front();
            for (auto op : _model_program->get_executable_operations()) {
                if (_active_operation_stats.count(op->get_id())) {
                    continue;
                }
                spdlog::info("Start the execution of operation {}", op->get_name());

                if (count_active_operations()) {
                    if (_active_operation_stats.find(op->get_id()) != _active_operation_stats.end()) {
                        return;
                    }
                }
                assert(op->get_tiles().size());
                _executable_tile_queue = op->get_tiles();
                // `op` is the operation selected by this loop. The executable list may
                // contain multiple ready siblings, so using front() can attach the
                // acceleration state to a different operation.
                _current_op = op;
                const auto current_op_type = _current_op->get_optype();
                // Keep the legacy Loop_wise QKT/SV estimator disabled. Proportional has
                // a separate head-aware sampler below.
                const auto current_op_name = _current_op->get_name();
                const bool skip_attention_gemm_acceleration =
                    _config.accelerate_method != "Proportional" &&
                    (current_op_name.find(".QKGEMM") != std::string::npos ||
                     current_op_name.find(".SVGEMM") != std::string::npos);
                const bool proportional_supported =
                    _config.accelerate_method != "Proportional" ||
                    current_op_type == "GEMM" || current_op_type == "GEMM_Att" ||
                    current_op_type == "Softmax" ||
                    current_op_type == "PIMGEMV";
                const bool acceleration_candidate =
                    current_op_type == "GEMM" || current_op_type == "GEMM_Att" ||
                    (_config.accelerate_method == "Proportional" &&
                     (current_op_type == "Softmax" ||
                      current_op_type == "PIMGEMV"));
                _sim_accelerate = acceleration_candidate &&
                                  _acceletare_ctrl && !skip_attention_gemm_acceleration &&
                                  proportional_supported;
                if (_sim_accelerate && _config.accelerate_method == "Proportional") {
                    reset_proportional_sampling_state();
                }
                if (_sim_accelerate &&
                    (_current_op->get_optype() == "GEMM" ||
                     (_config.accelerate_method == "Proportional" &&
                      _current_op->get_optype() == "PIMGEMV"))) {
                    // Reset acceleration variables for a new operation
                    _tile_position = 0;
                    _estimated_all_cycle = 0;
                    _unstable_cycle = 0;
                    _sample_cycle = 0;
                    _mean_cycle = 0;
                    _finish_tile_cycle.clear();
                    _inst_num.clear();
                    _inst_move_num.clear();
                    _inst_comp_num.clear();

                    if (_config.accelerate_method == "naive") { //naive predicting without K_Loop control
                        init_predicting_config();
                        PredictingConfig config = get_predicting_config();
                        uint32_t sample_length = config._sample_length;
                        if (sample_length > 0 && _executable_tile_queue.size() > sample_length) {
                            auto it = _executable_tile_queue.begin();
                            std::advance(it, sample_length);
                            _executable_tile_queue.erase(it, _executable_tile_queue.end());
                        }
                    }
                    if (_config.accelerate_method == "Loop_wise") { // Loop_wise predicting with intervals of K_Loop results
                        init_predicting_config();
                    }
                    if (_config.accelerate_method == "Proportional") {
                        init_predicting_config();
                    }
                }
                if (_sim_accelerate && _current_op->get_optype() == "GEMM_Att" &&
                    _config.accelerate_method == "Loop_wise") {
                    // Reset acceleration variables for a new operation
                    _tile_position = 0;
                    _estimated_all_cycle = 0;
                    _unstable_cycle = 0;
                    _sample_cycle = 0;
                    _mean_cycle = 0;
                    _finish_tile_cycle.clear();
                    _inst_num.clear();
                    _inst_move_num.clear();
                    _inst_comp_num.clear();

                    // Initialize GEMM_Att acceleration variables
                    _total_heads = _config.model_n_head;
                    uint32_t heads_per_core = _total_heads / _config.num_cores;

                    // 为每个core初始化相关变量
                    for (int i = 0; i < _config.num_cores; i++) {
                        _core_head_cycles[i].clear();
                        _core_head_unstable_cycles[i] = 0;
                        _core_head_sample_cycles[i] = 0;
                        _core_head_tile_count[i] = 0;
                        _core_head_first_done[i] = false;
                        _core_head_second_done[i] = false;
                        _front_heads = 2;
                        _core_head_remaining[i] = heads_per_core - _front_heads; // 减去3个head
                        _core_head_estimated_cycles[i] = 0;
                    }
                }
                _active_operation_stats[op->get_id()] = RunningOperationStat{
                    .id = op->get_id(),
                    .name = op->get_name(),
                    // xxx necessary?
                    // .launched = true,
                    .start_cycle = *_core_cycle,
                    .total_tiles = static_cast<uint32_t>(_executable_tile_queue.size()),
                    .remain_tiles = static_cast<uint32_t>(_executable_tile_queue.size()),
                    .launched_tiles = 0,
                    .pim_start_cycle = 0,
                    .pim_end_cycle = 0,
                    .pim_inst_count = 0,
                    .logical_pim_end_cycle = 0,
                    .estimated_pim_inst_count = 0,
                    .host_start_time = std::chrono::steady_clock::now(),
                    .deferred_compile_time_sec = 0.0,
                };
                const bool decode_target =
                    _config.decode_pruning_enabled &&
                    is_decode_pruning_target(current_op_name);
                const std::string decode_template_key =
                    decode_target
                        ? decode_pruning_template_key(_current_op)
                        : std::string{};
                const bool decode_has_template =
                    _decode_pruning_templates.find(decode_template_key) !=
                    _decode_pruning_templates.end();
                if (decode_target) {
                    _decode_pruning_operation_keys[op->get_id()] =
                        decode_template_key;
                }
                if (decode_target && !decode_has_template) {
                    auto& baseline = _decode_pruning_start_timing[op->get_id()];
                    baseline.clear();
                    for (uint32_t core_id = 0; core_id < _cores.size();
                         ++core_id) {
                        baseline.push_back(
                            decode_core_timing_snapshot(core_id));
                    }
                    if (_config.dram_trace_simulation_mode) {
                        assert(_event_driven_dram != nullptr);
                        _event_driven_dram
                            ->begin_decode_pruning_state_sample(
                                decode_template_key);
                    } else {
                        assert(_dram != nullptr);
                        _dram->begin_decode_pruning_state_sample(
                            decode_template_key);
                    }
                }
                issue_tile_per_core();  // issue current tile to each core
                if (decode_target && decode_has_template) {
                    prepare_decode_pruning_prediction();
                }
                return;
            }
        }
    }
}


void MyScheduler::finish_program() {
    spdlog::info("Current Program {} start at core cycle {}, finished at core cycle: {}", _model_program->_name,program_start_cycle , _cycles);
    _model_program = nullptr;
    refresh_stage();
}


void MyScheduler::refresh_stage() {
    bool stage_done = _model_program == nullptr;
    if (stage_done) {
        std::string red = "\033[1;31m";
        std::string reset = "\033[0m";
        std::string stage_name = stageToString(_stage);
        spdlog::info("{}------- Stage {} Done -------{}", red, stage_name, reset);

        // Update stat
        _stage_stats.emplace_back(stage_name, _cycles);
        _prev_stage = _stage;

        if (_prev_stage == Stage::Single_test or _prev_stage == Stage::Multi_test) {
            _stage = Stage::Finish;
        }
        else if (_output_token_iteration_enable) {
            if (_continuous_batching) {
                complete_continuous_batch(_prev_stage);
            } else {
                advance_iterative_inference(_prev_stage);
            }
        }
        else {
            int stageValue = static_cast<int>(_stage);  // Update to the next stage
            stageValue++;
            _stage = static_cast<Stage>(stageValue);
        }
        _has_stage_changed = true;  // if change stage, _has_stage_changed is set to ture
    }
}


void MyScheduler::issue_tile_per_core() {
    // process the current
    while(!_executable_tile_queue.empty()) {
        Tile& tile = _executable_tile_queue.front();
        /* Barrier! */
        if (tile.status == Tile::Status::BAR)
            break;

        if (tile.core_id == -1) { // -1 is global id
            tile.core_id = _core_rr_id % _config.num_cores;
            _core_rr_id++; // increase with round robin
        } else {
            assert(tile.core_id < _config.num_cores);
            tile.core_id = (tile.core_id) % _config.num_cores;  // allocate core id
        }

        if (tile.pim_tile == true) { tile.core_id = 0; }  // The PIM Tile can only be executed by Core 0
        _core_executable_tile_queue[tile.core_id].push_back(std::move(tile));
        _executable_tile_queue.pop_front();
    }

    begin_attention_round_command_trace();

    if (_sim_accelerate && _config.accelerate_method == "Proportional") {
        prepare_proportional_sampling();
    }
}


bool MyScheduler::finish_tile(uint32_t core_id, Tile& tile) {  // Record the finished information
    bool result = false;
    spdlog::debug("Core {} Finish Tile {} at {}", core_id, tile.operation_id,  *_core_cycle);
    assert(_active_operation_stats.find(tile.operation_id) != _active_operation_stats.end());
    assert(_finished_operation_stats.find(tile.operation_id) == _finished_operation_stats.end());
    assert(_active_operation_stats[tile.operation_id].remain_tiles > 0);
    _active_operation_stats[tile.operation_id].remain_tiles--;

    // Update PIM stats for this tile
    if (tile.pim_inst_count > 0) {
        // The first PIM tile start time, pheader
        _active_operation_stats[tile.operation_id].pim_inst_count += tile.pim_inst_count;
        if (_active_operation_stats[tile.operation_id].pim_start_cycle == 0 || 
            tile.pim_start_cycle < _active_operation_stats[tile.operation_id].pim_start_cycle) {
            _active_operation_stats[tile.operation_id].pim_start_cycle = tile.pim_start_cycle;
        }
        // The last PIM tile end time
        if (_active_operation_stats[tile.operation_id].pim_end_cycle == 0 ||
            tile.pim_finish_cycle > _active_operation_stats[tile.operation_id].pim_end_cycle) {
            _active_operation_stats[tile.operation_id].pim_end_cycle = tile.pim_finish_cycle;
        }
    }

    _model_program->finish_operation_tile(tile);
    record_attention_round_command_trace(core_id, tile);
    if (_config.decode_pruning_enabled &&
        _decode_pruning_start_timing.count(tile.operation_id)) {
        _decode_pruning_sample_workload[tile.operation_id][core_id] +=
            summarize_proportional_tile(tile);
    }

    if (_sim_accelerate) {
        if (_config.accelerate_method == "naive") {
            _tile_position ++;
            if (_tile_position == _unstable_length) {
                _unstable_cycle = *_core_cycle;
            }
            else if ((_tile_position > _unstable_length) && (_tile_position <= _sample_length - 1)) {
                _finish_tile_cycle.push_back(*_core_cycle);
            }
        }
        else if (_config.accelerate_method == "Proportional") {
            if (_proportional_plan_ready && !_proportional_tail_is_active) {
                auto& finish_cycles = _proportional_finish_cycles[core_id];
                finish_cycles.push_back(*_core_cycle);
                if (_proportional_softmax_mode ||
                    _proportional_attention_mode) {
                    const uint32_t warmup_tiles = _proportional_softmax_mode
                        ? _config.softmax_warmup_rounds
                        : _proportional_warmup_head_tiles.at(core_id);
                    if (finish_cycles.size() == warmup_tiles) {
                        _cores.at(core_id)
                            ->begin_proportional_timing_sampling();
                    }
                }
                if (!_proportional_command_sample_started) {
                    bool warmup_complete = true;
                    for (uint32_t active_core : _proportional_active_cores) {
                        const uint32_t warmup_tiles = _proportional_softmax_mode
                            ? _config.softmax_warmup_rounds
                            : (_proportional_attention_mode
                                   ? _proportional_warmup_head_tiles.at(active_core)
                                   : _k_loop_size);
                        const auto& active_finish =
                            _proportional_finish_cycles[active_core];
                        warmup_complete = warmup_complete &&
                            active_finish.size() >= warmup_tiles;
                    }
                    if (warmup_complete) {
                        if (_proportional_attention_mode) {
                            if (_config.dram_trace_simulation_mode) {
                                assert(_event_driven_dram != nullptr);
                                _event_driven_dram->mark_proportional_command_warmup_complete(
                                    _config.attention_command_warmup_weight);
                            } else {
                                assert(_dram != nullptr);
                                _dram->mark_proportional_command_warmup_complete(
                                    _config.attention_command_warmup_weight);
                            }
                        } else if (_proportional_softmax_mode) {
                            if (_config.dram_trace_simulation_mode) {
                                assert(_event_driven_dram != nullptr);
                                _event_driven_dram->begin_proportional_command_sampling();
                            } else {
                                assert(_dram != nullptr);
                                _dram->begin_proportional_command_sampling();
                            }
                        } else {
                            // For ordinary GEMMs, retain the first complete
                            // K-loop as command warmup.  READ/WRITE estimation
                            // uses the complete retained prefix, while row
                            // locality is sampled only after this cold-start
                            // window.
                            if (_config.dram_trace_simulation_mode) {
                                assert(_event_driven_dram != nullptr);
                                _event_driven_dram->mark_proportional_command_warmup_complete(
                                    0.0);
                            } else {
                                assert(_dram != nullptr);
                                _dram->mark_proportional_command_warmup_complete(
                                    0.0);
                            }
                        }
                        _proportional_command_sample_started = true;
                        spdlog::info(
                            "Started proportional DRAM command sample window for {} after warmup",
                            _current_op->get_name());
                    }
                }
                const auto prefix_it = _proportional_prefix_tiles.find(core_id);
                if (prefix_it != _proportional_prefix_tiles.end() &&
                    finish_cycles.size() == prefix_it->second) {
                    _cores.at(core_id)->end_proportional_timing_sampling();
                    spdlog::info(
                        "Core {} completed {} proportional sampling tiles for operation {}",
                        core_id, prefix_it->second, _current_op->get_name());
                }
            }
            _operation_id = tile.operation_id;
            return result;
        }
        else if (_config.accelerate_method == "Loop_wise") {
            if (_current_op->get_optype() == "GEMM_Att") {
                // GEMM_Att的处理逻辑
                _core_head_tile_count[core_id]++;

                // 检查是否完成了一个head的所有tile
                _total_tiles = _current_op->get_tiles();
                uint32_t tiles_per_head = _total_tiles.size() / _total_heads;
                if (_core_head_tile_count[core_id] % tiles_per_head == 0 && (_core_head_tile_count[core_id] / tiles_per_head) >= (_front_heads - 1)) {
                    // 记录当前head的完成周期
                    _core_head_cycles[core_id].push_back(*_core_cycle);

                    // 检查是否完成了第2个head
                    if (!_core_head_first_done[core_id]) {
                        _core_head_first_done[core_id] = true;
                        _core_head_unstable_cycles[core_id] = *_core_cycle; // 第2个head的完成周期
                    }
                    // 检查是否完成了第3个head
                    else if (!_core_head_second_done[core_id]) {
                        _core_head_second_done[core_id] = true;
                        _core_head_sample_cycles[core_id] = *_core_cycle - _core_head_unstable_cycles[core_id]; // 第3个head的完成周期减去第2个head的完成周期

                        // 计算剩余head的预测时间
                        const cycle_type remaining_cycles =
                            _core_head_sample_cycles[core_id] *
                            _core_head_remaining[core_id];

                        // 计算总预测时间
                        _core_head_estimated_cycles[core_id] = *_core_cycle + remaining_cycles;

                        // 清空当前core的可执行tile队列
                        _core_executable_tile_queue[core_id].clear();
                    }
                }
            }
            else {  // GEMM Operation
                if (_core_k_inner_count[core_id] == 0) _core_kloop_cycles[core_id].push_back(*_core_cycle); //first tile finish cycle of per core, only to generate first delta cycle
                _core_k_inner_count[core_id]++;

                // 检查是否完成了一个K_Loop
                assert(_k_loop_size > 0);
                if ((_core_k_inner_count[core_id]) % _k_loop_size == 0) {
                    // 记录当前K_Loop的完成周期
                    _core_kloop_cycles[core_id].push_back(*_core_cycle);

                    // 检查是否达到稳定状态
                    if (_core_kloop_cycles[core_id].size() >= 3 && !_core_stable[core_id]) {
                        // 计算相邻3个K_Loop的完成时间差
                        const cycle_type prev_prev_cycle = _core_kloop_cycles[core_id][_core_kloop_cycles[core_id].size() - 3];
                        const cycle_type prev_cycle = _core_kloop_cycles[core_id][_core_kloop_cycles[core_id].size() - 2];
                        const cycle_type current_cycle = _core_kloop_cycles[core_id].back();
                        const cycle_type delta_cycle = current_cycle - prev_cycle;
                        const cycle_type prev_delta_cycle = prev_cycle - prev_prev_cycle;
                        double_t cycle_diff = (double)delta_cycle / prev_delta_cycle;

                        if (cycle_diff > 0.98 && cycle_diff < 1.02) {   // 需要在这里添加
                            _core_stable[core_id] = true;
                            _core_executable_tile_queue[core_id].clear();
                            // _active_operation_stats[tile.operation_id].remain_tiles = 0;

                            // 计算不稳定周期和采样周期
                            _core_unstable_cycles[core_id] = _core_kloop_cycles[core_id][1]; // 第一个K_Loop的完成周期

                            // 计算采样周期：从第二个K_Loop到稳定状态的K_Loop之间的周期差
                            _core_sample_cycles[core_id] = current_cycle - _core_kloop_cycles[core_id][1];

                            // 计算平均每个K_Loop的周期
                            uint32_t sample_kloop_count = _core_kloop_cycles[core_id].size() - 2;
                            const cycle_type avg_kloop_cycle =
                                _core_sample_cycles[core_id] /
                                sample_kloop_count;

                            // 计算剩余的K_Loop数量
                            uint32_t total_kloops = _total_tiles.size() / (_k_loop_size * _config.num_cores);
                            uint32_t remaining_kloops = total_kloops - _core_kloop_cycles[core_id].size() + 1;

                            // 计算剩余周期
                            _core_remaining_cycles[core_id] = avg_kloop_cycle * remaining_kloops;
                            _core_estimated_cycles[core_id] = _core_unstable_cycles[core_id] + _core_sample_cycles[core_id] + _core_remaining_cycles[core_id];
                            // std::string fname = "output_mobile_accelerated/" + _current_op->get_name() + "_loopwise_predicting_result.txt";
                            // std::ofstream outfile(fname, std::ios_base::app);
                            // if (outfile.is_open()) {
                            //     outfile << "Core " << core_id << " unstable cycle:" << _core_unstable_cycles[core_id] << "\n";
                            //     outfile << "Core " << core_id << " sample cycle:" << _core_sample_cycles[core_id] << "\n";
                            //     outfile << "Core " << core_id << " remaining cycle:" << _core_remaining_cycles[core_id] << "\n";
                            //     outfile << "Core " << core_id << " estimated all cycle:" << _core_estimated_cycles[core_id] << "\n";
                            //     outfile.close();
                            // }
                        }
                    }
                }
                // A core may exhaust its assigned tiles before meeting the stability threshold.
                // Its actual completion cycle is then the safest estimate for that core.
                if (!_core_stable[core_id] && _core_executable_tile_queue[core_id].empty()) {
                    _core_stable[core_id] = true;
                    _core_estimated_cycles[core_id] = *_core_cycle;
                    spdlog::info("Core {} exhausted its Loop_wise sample tiles before stabilization; using actual cycle {}",
                                 core_id, *_core_cycle);
                }
            }
            _operation_id = tile.operation_id;
            return result;
        }
    }

    _operation_id = tile.operation_id;

    if (_active_operation_stats[tile.operation_id].remain_tiles == 0) {  // no remain tile, current op finished,
        _op_stats.emplace_back(_active_operation_stats[tile.operation_id].name, _cycles);
        result = true;
        spdlog::info("Layer {} finish at {}", _active_operation_stats[tile.operation_id].name, *_core_cycle);
        spdlog::info("Total compute time {}", *_core_cycle - _active_operation_stats[tile.operation_id].start_cycle);
        const double host_total_time_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            _active_operation_stats[tile.operation_id].host_start_time)
                                               .count();
        const double deferred_compile_time_sec =
            _active_operation_stats[tile.operation_id]
                .deferred_compile_time_sec;
        const double host_simulation_time_sec =
            std::max(0.0, host_total_time_sec - deferred_compile_time_sec);
        spdlog::info(
            "Operation {} Deferred Compile Time: {:.9f} s",
            _active_operation_stats[tile.operation_id].name,
            deferred_compile_time_sec);
        spdlog::info(
            "Operation {} Host Simulation Time: {:.9f} s",
            _active_operation_stats[tile.operation_id].name,
            host_simulation_time_sec);
        spdlog::info(
            "Operation {} Host Total Time incl Deferred Compile: {:.9f} s",
            _active_operation_stats[tile.operation_id].name,
            host_total_time_sec);
        print_current_operation_workload_stat();
        // Calculate and Print PIM stats for this operation
        if (_active_operation_stats[tile.operation_id].pim_inst_count > 0) {
            const auto& pim_stat = _active_operation_stats[tile.operation_id];
            const uint64_t logical_count =
                pim_stat.pim_inst_count + pim_stat.estimated_pim_inst_count;
            const cycle_type logical_end = std::max(
                pim_stat.pim_end_cycle, pim_stat.logical_pim_end_cycle);
            cycle_type duration =
                logical_end - pim_stat.pim_start_cycle;
            if (duration > 0) {
                double pim_bw = static_cast<double>(logical_count) / duration;
                spdlog::info("Operation {} PIM Bandwidth: {:.4f} inst/cycle (Logical count: {}, measured {}, estimated {}, duration {})",
                    _active_operation_stats[tile.operation_id].name, pim_bw, 
                    logical_count, pim_stat.pim_inst_count,
                    pim_stat.estimated_pim_inst_count, duration);
                // Aggregate to global stats
                _total_pim_inst_count += logical_count;
                _total_pim_duration += duration;
                // spdlog::info("PIM Utilization of current tile is {:.4f}", pim_bw);
                // spdlog::info()
            }
        }
        capture_decode_pruning_template(tile.operation_id);
        _model_program->finish_operation(tile.operation_id);
        _finished_operation_stats[tile.operation_id] = _active_operation_stats[tile.operation_id];
        _active_operation_stats.erase(tile.operation_id);
    }
    refresh_status();
    return result;
}

void MyScheduler::compute_estimated_cycle(){
    if (_config.accelerate_method == "naive") {
        if (_tile_position == _sample_length) {
            _sample_cycle = _finish_tile_cycle.back() - _finish_tile_cycle.front();
            _mean_cycle = _sample_cycle / _finish_tile_cycle.size();
            _estimated_all_cycle =  _unstable_cycle + _sample_cycle +
                                        _mean_cycle * (_total_tiles.size() - _sample_length - _tile_num_last + 2 * _config.num_cores) +
                                        (_tile_num_last - 2 * _config.num_cores) * _mean_cycle * _inst_ratio;
            //unstable_cycle(absolute) + sample_cycle(relative delta) + middle_predicting_cycle(relative delta) + last_outerloop_cycle(relative delta)
            // std::ofstream outfile(fname);
            // if (!outfile.is_open()) {
            //     assert(0);
            // }
            // //outfile << "acutal all cycle:" << _actual_all_cycle << "\n";
            // outfile << "unstable cycle:" << _unstable_cycle << "\n";
            // outfile << "sample cycle:" << _sample_cycle << "\n";
            // outfile << "mean cycle:" << _mean_cycle << "\n";
            // outfile << "estimated all cycle:" << _estimated_all_cycle << "\n";
            // outfile.close();
        }
    } else if (_config.accelerate_method == "Loop_wise") {
        // Loop_wise方法的预测计算
        // 对于Loop_wise方法，预测计算已经在finish_tile()函数中完成
        // 这里我们只需要找出所有core中的最大估计周期作为整体的估计周期
        if (_current_op->get_optype() == "GEMM_Att") {
            // GEMM_Att的预测计算
            _estimated_all_cycle = 0;
            for (int core_id = 0; core_id < _config.num_cores; core_id++) {
                if (_core_head_estimated_cycles[core_id] > _estimated_all_cycle) {
                    _estimated_all_cycle = _core_head_estimated_cycles[core_id];
                    _latest_core = core_id;
                }
            }
        }
        else {
            // GEMM的预测计算
            _estimated_all_cycle = 0;
            for (int core_id = 0; core_id < _config.num_cores; core_id++) {
                if (_core_estimated_cycles[core_id] > _estimated_all_cycle) {
                    _estimated_all_cycle = _core_estimated_cycles[core_id];
                    _latest_core = core_id;
                }
            }
        }
    } else if (_config.accelerate_method == "Proportional") {
        _estimated_all_cycle = 0;
        _proportional_core_estimated_cycles.clear();
        const auto stat_it = _active_operation_stats.find(_operation_id);
        if (!_proportional_plan_ready || stat_it == _active_operation_stats.end()) {
            return;
        }

        if (_proportional_attention_mode) {
            for (uint32_t core_id : _proportional_active_cores) {
                const auto& finish = _proportional_finish_cycles.at(core_id);
                const uint32_t warmup_head_tiles =
                    _proportional_warmup_head_tiles.at(core_id);
                const uint32_t prefix_tiles = _proportional_prefix_tiles.at(core_id);
                assert(warmup_head_tiles > 0 && prefix_tiles > warmup_head_tiles);
                assert(finish.size() >= prefix_tiles);

                const cycle_type warmup_finish = finish[warmup_head_tiles - 1];
                const cycle_type sample_finish = finish[prefix_tiles - 1];
                const cycle_type sampled_head_pair_cycles =
                    sample_finish - warmup_finish;
                assert(_proportional_common_head_rounds >= 6);
                const uint32_t remaining_heads =
                    _proportional_common_head_rounds - 4;
                const cycle_type estimated_remaining =
                    (sampled_head_pair_cycles * remaining_heads + 1) / 2;
                const cycle_type core_estimated_cycle =
                    sample_finish + estimated_remaining;
                _proportional_core_estimated_cycles[core_id] =
                    core_estimated_cycle;

                _estimated_all_cycle = std::max<cycle_type>(_estimated_all_cycle, core_estimated_cycle);
                spdlog::info(
                    "Proportional attention estimate core {}: warmup head tiles {}, sampled head tiles {}, predicted prefix+middle head rounds {}, sampled pair cycles {}, remaining predicted heads {}, finish cycle {}",
                    core_id, warmup_head_tiles,
                    prefix_tiles - warmup_head_tiles,
                    _proportional_common_head_rounds,
                    sampled_head_pair_cycles, remaining_heads,
                    core_estimated_cycle);
            }
            return;
        }

        if (_proportional_softmax_mode) {
            const uint32_t warmup_rounds = _config.softmax_warmup_rounds;
            const uint32_t sampled_rounds = _config.softmax_sample_rounds;
            const uint32_t prefix_rounds = warmup_rounds + sampled_rounds;
            for (uint32_t core_id : _proportional_active_cores) {
                const auto& finish = _proportional_finish_cycles.at(core_id);
                assert(finish.size() >= prefix_rounds);

                const cycle_type sample_window =
                    finish[prefix_rounds - 1] - finish[warmup_rounds - 1];
                const cycle_type remaining_rounds =
                    _proportional_common_softmax_rounds - prefix_rounds;
                const cycle_type estimated_remaining =
                    (sample_window * remaining_rounds + sampled_rounds / 2) / sampled_rounds;
                const cycle_type core_estimated_cycle =
                    finish[prefix_rounds - 1] + estimated_remaining;
                _proportional_core_estimated_cycles[core_id] =
                    core_estimated_cycle;

                _estimated_all_cycle = std::max<cycle_type>(
                    _estimated_all_cycle, core_estimated_cycle);
                spdlog::info(
                    "Proportional Softmax estimate core {}: warmup rounds {}, sampled rounds {}, common rounds {}, sample window cycles {}, finish cycle {}",
                    core_id, warmup_rounds, sampled_rounds,
                    _proportional_common_softmax_rounds, sample_window,
                    core_estimated_cycle);
            }
            return;
        }

        const cycle_type start_cycle = stat_it->second.start_cycle;
        const uint32_t kloop_tiles = _k_loop_size;
        const uint32_t sample_tiles = _proportional_sample_tiles;

        for (uint32_t core_id : _proportional_active_cores) {
            const auto& finish = _proportional_finish_cycles.at(core_id);
            assert(finish.size() >= static_cast<size_t>(kloop_tiles + sample_tiles));

            const cycle_type b_sample = finish[sample_tiles - 1];
            const cycle_type b_group1 = finish[kloop_tiles - 1];
            const cycle_type b_group2_sample = finish[kloop_tiles + sample_tiles - 1];

            const cycle_type group1_cycles = b_group1 - start_cycle;
            const cycle_type group1_remaining = b_group1 - b_sample;
            const cycle_type group2_sample = b_group2_sample - b_group1;
            const cycle_type cross_group_sample = b_group2_sample - b_sample;
            const cycle_type estimated_duration =
                group1_cycles + group2_sample + group1_remaining +
                static_cast<cycle_type>(_proportional_common_kloops - 2) * cross_group_sample;
            const cycle_type core_estimated_cycle = start_cycle + estimated_duration;
            _proportional_core_estimated_cycles[core_id] =
                core_estimated_cycle;

            _estimated_all_cycle = std::max<cycle_type>(_estimated_all_cycle, core_estimated_cycle);
            spdlog::info(
                "Proportional estimate core {}: K-loop tiles {}, sampled {}, common loops {}, finish cycle {}",
                core_id, kloop_tiles, sample_tiles, _proportional_common_kloops,
                core_estimated_cycle);
        }
    }
}

void MyScheduler::begin_attention_round_command_trace() {
    _attention_round_trace_active = false;
    if (_current_op == nullptr || _current_op->get_optype() != "GEMM_Att" ||
        _sim_accelerate || _config.dram_trace_simulation_mode ||
        _dram == nullptr) {
        return;
    }

    _attention_round_remaining_tiles.clear();
    _attention_round_index.clear();
    _attention_round_last_act.clear();
    _attention_round_last_pre.clear();
    _attention_round_trace_operation = _current_op->get_name();

    for (uint32_t core_id = 0; core_id < _config.num_cores; ++core_id) {
        const auto queue_it = _core_executable_tile_queue.find(core_id);
        if (queue_it == _core_executable_tile_queue.end() ||
            queue_it->second.empty()) {
            continue;
        }
        for (const auto& tile : queue_it->second) {
            const uint64_t head_key =
                (static_cast<uint64_t>(tile.batch) << 32) | tile.head_index;
            _attention_round_remaining_tiles[core_id][head_key]++;
        }
        _attention_round_index[core_id] = 0;
        _attention_round_last_act[core_id] =
            _dram->get_core_command_counter(core_id, "num_act_cmds");
        _attention_round_last_pre[core_id] =
            _dram->get_core_command_counter(core_id, "num_pre_cmds");
    }

    if (_attention_round_remaining_tiles.empty()) {
        return;
    }
    if (!_attention_round_trace_file_initialized) {
        std::ofstream out(Config::system_config.log_dir +
                          "/attention_core_round_commands.tsv",
                          std::ios::out | std::ios::trunc);
        out << "operation\tcore\tround\tbatch\thead\tcore_cycle"
               "\tcumulative_act\tcumulative_pre\tdelta_act\tdelta_pre\n";
        _attention_round_trace_file_initialized = true;
    }
    _attention_round_trace_active = true;
}

void MyScheduler::record_attention_round_command_trace(
    uint32_t core_id, const Tile& tile) {
    if (!_attention_round_trace_active) {
        return;
    }
    const uint64_t head_key =
        (static_cast<uint64_t>(tile.batch) << 32) | tile.head_index;
    auto core_it = _attention_round_remaining_tiles.find(core_id);
    if (core_it == _attention_round_remaining_tiles.end()) {
        return;
    }
    auto head_it = core_it->second.find(head_key);
    if (head_it == core_it->second.end() || head_it->second == 0) {
        return;
    }
    if (--head_it->second != 0) {
        return;
    }

    const uint64_t cumulative_act =
        _dram->get_core_command_counter(core_id, "num_act_cmds");
    const uint64_t cumulative_pre =
        _dram->get_core_command_counter(core_id, "num_pre_cmds");
    const uint64_t delta_act = cumulative_act - _attention_round_last_act[core_id];
    const uint64_t delta_pre = cumulative_pre - _attention_round_last_pre[core_id];
    const uint32_t round = ++_attention_round_index[core_id];

    std::ofstream out(Config::system_config.log_dir +
                      "/attention_core_round_commands.tsv",
                      std::ios::out | std::ios::app);
    out << _attention_round_trace_operation << '\t' << core_id << '\t'
        << round << '\t' << tile.batch << '\t' << tile.head_index << '\t'
        << *_core_cycle << '\t' << cumulative_act << '\t' << cumulative_pre
        << '\t' << delta_act << '\t' << delta_pre << '\n';

    _attention_round_last_act[core_id] = cumulative_act;
    _attention_round_last_pre[core_id] = cumulative_pre;
}

void MyScheduler::finish_last_mm_tile(uint32_t core_id, Tile& tile) {
    spdlog::debug("Core {} Finish Tile {} at {}", core_id, tile.operation_id,  *_core_cycle);
    assert(_active_operation_stats.find(tile.operation_id) != _active_operation_stats.end());
    assert(_finished_operation_stats.find(tile.operation_id) == _finished_operation_stats.end());
    assert(_active_operation_stats[tile.operation_id].remain_tiles > 0);
    _active_operation_stats[tile.operation_id].remain_tiles--;

    // spdlog::info("Finish tile stage_platform:{}", stagePlatformToString(tile.stage_platform));
    _model_program->finish_operation_tile(tile);

    _tile_position++;
    _finish_tile_cycle.push_back(*_core_cycle);
}

bool MyScheduler::update_stats_last_tile(uint32_t core_id, Tile& tile) {
    bool result = false;
    if (_active_operation_stats[tile.operation_id].remain_tiles == 0) {  // 完成判断，若当前operation没有新的tile，表明执行完毕
        _op_stats.emplace_back(_active_operation_stats[tile.operation_id].name, _cycles);
        result = true;
        spdlog::info("Layer {} finish at {}", _active_operation_stats[tile.operation_id].name, *_core_cycle);
        spdlog::info("Total compute time {}", *_core_cycle - _active_operation_stats[tile.operation_id].start_cycle);
        const double host_total_time_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            _active_operation_stats[tile.operation_id].host_start_time)
                                               .count();
        const double deferred_compile_time_sec =
            _active_operation_stats[tile.operation_id]
                .deferred_compile_time_sec;
        const double host_simulation_time_sec =
            std::max(0.0, host_total_time_sec - deferred_compile_time_sec);
        spdlog::info(
            "Operation {} Deferred Compile Time: {:.9f} s",
            _active_operation_stats[tile.operation_id].name,
            deferred_compile_time_sec);
        spdlog::info(
            "Operation {} Host Simulation Time: {:.9f} s",
            _active_operation_stats[tile.operation_id].name,
            host_simulation_time_sec);
        spdlog::info(
            "Operation {} Host Total Time incl Deferred Compile: {:.9f} s",
            _active_operation_stats[tile.operation_id].name,
            host_total_time_sec);
        print_current_operation_workload_stat();
        // Calculate and Print PIM stats for this operation
        if (_active_operation_stats[tile.operation_id].pim_inst_count > 0) {
            const auto& pim_stat = _active_operation_stats[tile.operation_id];
            const uint64_t logical_count =
                pim_stat.pim_inst_count + pim_stat.estimated_pim_inst_count;
            const cycle_type logical_end = std::max(
                pim_stat.pim_end_cycle, pim_stat.logical_pim_end_cycle);
            cycle_type duration = logical_end - pim_stat.pim_start_cycle;
            if (duration > 0) {
                double pim_bw = static_cast<double>(logical_count) / duration;
                spdlog::info("Operation {} PIM Bandwidth: {:.4f} inst/cycle (Logical count: {}, measured {}, estimated {}, duration {})",
                    _active_operation_stats[tile.operation_id].name, pim_bw,
                    logical_count, pim_stat.pim_inst_count,
                    pim_stat.estimated_pim_inst_count, duration);
                // Aggregate to global stats
                _total_pim_inst_count += logical_count;
                _total_pim_duration += duration;
                // spdlog::info("PIM Utilization of current tile is {:.4f}", pim_bw);
                // spdlog::info()
            }
        }
        _model_program->finish_operation(tile.operation_id);
        _finished_operation_stats[tile.operation_id] = _active_operation_stats[tile.operation_id];
        _active_operation_stats.erase(tile.operation_id);
    }
    return result;  // 如果是最后一个tile，表明当前OP的计算结果已经完全生成
}


MyScheduler::PredictingConfig MyScheduler::get_predicting_config() {
    return{
        this->_unstable_length,
        this->_sample_length,
        this->_inst_ratio
    };
}

void MyScheduler::init_predicting_config() {
    _total_tiles = _current_op->get_tiles();
    auto op_outer_loop = _current_op->get_outer_loop();
    auto op_inner_loop = _current_op->get_inner_loop();
    auto op_outer_loop_M = op_outer_loop[0];
    auto op_outer_loop_K = op_outer_loop[1];
    auto op_inner_loop_K = op_inner_loop[1];

    _tile_num_last = (double)_total_tiles.size() / op_outer_loop_M;  //tile core_cycles will be lower in the end, when Lin isn't 128*n
    if (_config.accelerate_method == "naive") {
        for (int i = 0; i < _total_tiles.size(); i++) {
            auto inst = _total_tiles[i].instructions;
            _inst_num.push_back(inst.size());
            uint16_t inst_single_tile_move = 0;
            for (int j = 0; j < inst.size(); j++) {
                if (inst[j].opcode == Opcode::MOVIN || inst[j].opcode == Opcode::MOVOUT) {
                    inst_single_tile_move++;
                }
            }
            _inst_move_num.push_back(inst_single_tile_move);
            auto inst_single_tile_comp = inst.size() - inst_single_tile_move;
            _inst_comp_num.push_back(inst_single_tile_comp);
        }
        _inst_ratio = (double)_inst_comp_num.back() / _inst_comp_num.front();
        _unstable_length = _config.num_cores * 8; //core_num * unstable_tiles_per_core (8 is just from experience)
        _sample_length = _unstable_length + round(_total_tiles.size() * 0.2);
    }
    if (_config.accelerate_method == "Loop_wise") {
        // 初始化Loop_wise加速相关变量
        _k_loop_size = op_outer_loop_K; // 设置K_Loop的大小

        // 为每个core初始化相关变量
        for (int i = 0; i < _config.num_cores; i++) {
            _core_kloop_cycles[i].clear();
            _core_unstable_cycles[i] = 0;
            _core_sample_cycles[i] = 0;
            _core_remaining_cycles[i] = 0;
            _core_k_inner_count[i] = 0;
            _core_stable[i] = false;
            _core_estimated_cycles[i] = 0;
        }
    }
    if (_config.accelerate_method == "Proportional") {
        _k_loop_size = op_outer_loop_K;
        _proportional_sample_ratio = _config.accelerate_sample_ratio;
        reset_proportional_sampling_state();
    }
}

void MyScheduler::reset_proportional_sampling_state() {
    _proportional_sample_ratio = _config.accelerate_sample_ratio;
    _proportional_sample_tiles = 0;
    _proportional_common_kloops = 0;
    _proportional_common_head_rounds = 0;
    _proportional_common_softmax_rounds = 0;
    _proportional_skipped_tiles = 0;
    _proportional_plan_ready = false;
    _proportional_prediction_applied = false;
    _proportional_tail_is_active = false;
    _proportional_attention_mode = false;
    _proportional_softmax_mode = false;
    _proportional_active_cores.clear();
    _proportional_prefix_tiles.clear();
    _proportional_warmup_head_tiles.clear();
    _proportional_finish_cycles.clear();
    _proportional_tail_queues.clear();
    _proportional_estimated_workload.clear();
    _proportional_vm_replay_tiles.clear();
    _proportional_workload_applied = false;
    _proportional_command_sample_started = false;
    _proportional_timing_applied = false;
    _proportional_core_estimated_cycles.clear();
}

ProportionalWorkloadStat MyScheduler::summarize_proportional_tile(
    const Tile& tile) const {
    ProportionalWorkloadStat stat;
    stat.tiles = 1;
    stat.channel_memory_reads.resize(_config.dram_channels, 0);
    stat.channel_memory_writes.resize(_config.dram_channels, 0);
    stat.channel_pim_pheader.resize(_config.dram_channels, 0);
    stat.channel_pim_gwrite.resize(_config.dram_channels, 0);
    stat.channel_pim_comp.resize(_config.dram_channels, 0);
    stat.channel_pim_readres.resize(_config.dram_channels, 0);

    const auto vector_iterations = [this](uint32_t size) -> uint64_t {
        return (static_cast<uint64_t>(size) + _config.vector_core_width - 1) /
               _config.vector_core_width;
    };
    const auto normal_accesses = [this](const Instruction& inst) -> uint64_t {
        if (inst.skip || inst.src_addrs.empty()) {
            return 0;
        }
        return static_cast<uint64_t>(inst.src_addrs.size()) *
               (inst.per_ch_inst ? 1U : _config.dram_channels);
    };
    const auto pim_accesses = [](const Instruction& inst) -> uint64_t {
        if (inst.skip || inst.src_addrs.empty()) {
            return 0;
        }
        if (inst.opcode == Opcode::PIM_COMP_HASH) {
            return static_cast<uint64_t>(inst.src_addrs.size()) *
                   MyAddressAllocator::ranks * MyAddressAllocator::bankgroups *
                   MyAddressAllocator::banks * MyAddressAllocator::dram_channels;
        }
        return static_cast<uint64_t>(inst.src_addrs.size()) *
               MyAddressAllocator::dram_channels;
    };
    const auto add_normal_channels = [this](
        const Instruction& inst, std::vector<uint64_t>& channels) {
        if (inst.skip || inst.src_addrs.empty()) {
            return;
        }
        if (inst.per_ch_inst) {
            for (const auto address : inst.src_addrs) {
                const uint32_t channel =
                    MyAddressAllocator::get_channel_index(address);
                assert(channel < channels.size());
                channels[channel]++;
            }
        } else {
            for (uint32_t channel = 0; channel < _config.dram_channels;
                 ++channel) {
                channels[channel] += inst.src_addrs.size();
            }
        }
    };
    const auto add_pim_channels = [](const Instruction& inst,
                                     std::vector<uint64_t>& channels) {
        if (inst.skip || inst.src_addrs.empty()) {
            return;
        }
        uint64_t requests_per_channel = inst.src_addrs.size();
        if (inst.opcode == Opcode::PIM_COMP_HASH) {
            requests_per_channel *= MyAddressAllocator::ranks *
                                    MyAddressAllocator::bankgroups *
                                    MyAddressAllocator::banks;
        }
        for (auto& requests : channels) {
            requests += requests_per_channel;
        }
    };

    for (const auto& inst : tile.instructions) {
        switch (inst.opcode) {
            case Opcode::MOVIN: {
                const uint64_t count = normal_accesses(inst);
                add_normal_channels(inst, stat.channel_memory_reads);
                stat.memory_reads += count;
                stat.memory_read_bytes +=
                    count * MyAddressAllocator::dram_burst_size;
                break;
            }
            case Opcode::MOVOUT:
            case Opcode::MOVOUT_POOL: {
                const uint64_t count = normal_accesses(inst);
                add_normal_channels(inst, stat.channel_memory_writes);
                stat.memory_writes += count;
                stat.memory_write_bytes +=
                    count * MyAddressAllocator::dram_burst_size;
                break;
            }
            case Opcode::GEMM:
            case Opcode::GEMM_PRELOAD:
                stat.gemm++;
                stat.num_calculation +=
                    static_cast<uint64_t>(_config.core_height) *
                    _config.core_width * inst.size;
                break;
            case Opcode::GEMV: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.gemv += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::LAYERNORM: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.layernorm += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::RMSNORM: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.rmsnorm += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::ROPE: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.rope += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::SOFTMAX: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.softmax += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::ADD: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.add += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::MUL: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.mul += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::GELU: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.gelu += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::SILU: {
                const uint64_t iterations = vector_iterations(inst.size);
                stat.silu += iterations;
                stat.num_calculation += iterations * _config.vector_core_width;
                break;
            }
            case Opcode::IM2COL:
                stat.im2col++;
                break;
            case Opcode::DATA_CONVERT:
            case Opcode::DUMMY:
                stat.dummy++;
                break;
            case Opcode::PIM_HEADER:
                add_pim_channels(inst, stat.channel_pim_pheader);
                stat.pim_pheader += pim_accesses(inst);
                break;
            case Opcode::PIM_GWRITE:
                add_pim_channels(inst, stat.channel_pim_gwrite);
                stat.pim_gwrite += pim_accesses(inst);
                break;
            case Opcode::PIM_COMP:
            case Opcode::PIM_COMP_HASH:
                add_pim_channels(inst, stat.channel_pim_comp);
                stat.pim_comp += pim_accesses(inst);
                break;
            case Opcode::PIM_READRES:
                add_pim_channels(inst, stat.channel_pim_readres);
                stat.pim_readres += pim_accesses(inst);
                break;
            case Opcode::PIM_COMPS_READRES:
                add_pim_channels(inst, stat.channel_pim_comp);
                add_pim_channels(inst, stat.channel_pim_readres);
                stat.pim_comp += pim_accesses(inst);
                stat.pim_readres += pim_accesses(inst);
                break;
            case Opcode::BAR:
            case Opcode::COMP:
            case Opcode::GEMM_WRITE:
            case Opcode::SIZE:
                break;
        }
    }
    return stat;
}

void MyScheduler::record_proportional_skipped_tile(uint32_t core_id,
                                                    Tile& tile) {
    assert(!tile.deferred_compile);
    _proportional_estimated_workload[core_id] +=
        summarize_proportional_tile(tile);
    if (_config.virtual_mem_hash_enable) {
        _proportional_vm_replay_tiles[core_id].push_back(std::move(tile));
    }
}

void MyScheduler::record_proportional_skipped_tile(
    uint32_t core_id, Tile& tile,
    const Tile& compiled_representative) {
    if (!tile.deferred_compile) {
        record_proportional_skipped_tile(core_id, tile);
        return;
    }
    assert(!compiled_representative.deferred_compile);
    _proportional_estimated_workload[core_id] +=
        summarize_proportional_tile(compiled_representative);
    if (_config.virtual_mem_hash_enable) {
        _proportional_vm_replay_tiles[core_id].push_back(std::move(tile));
    }
}

uint64_t MyScheduler::replay_virtual_memory_tile(const Tile& tile) const {
    assert(_config.virtual_mem_hash_enable);
    uint64_t mapping_calls = 0;
    const auto map_address = [&mapping_calls](addr_type logical_address) {
        const uint32_t logical_channel =
            MyAddressAllocator::get_channel_index(logical_address);
        const addr_type physical_address =
            TwoLevelPageMapper::map_logical_address(logical_address);
        assert(MyAddressAllocator::get_channel_index(physical_address) ==
               logical_channel);
        ++mapping_calls;
    };
    const auto replay_normal = [&map_address](const Instruction& inst) {
        if (inst.skip || inst.src_addrs.empty()) {
            return;
        }
        if (inst.per_ch_inst) {
            for (const addr_type address : inst.src_addrs) {
                map_address(address);
            }
            return;
        }
        for (const addr_type address : inst.src_addrs) {
            for (uint32_t channel = 0;
                 channel < MyAddressAllocator::dram_channels; ++channel) {
                map_address(MyAddressAllocator::add_channel_index(
                    address, channel));
            }
        }
    };
    const auto replay_pim = [&map_address](const Instruction& inst) {
        if (inst.skip || inst.src_addrs.empty()) {
            return;
        }
        for (const addr_type address : inst.src_addrs) {
            if (inst.opcode == Opcode::PIM_COMP_HASH) {
                const uint32_t row =
                    MyAddressAllocator::get_row_index(address);
                const uint32_t column =
                    MyAddressAllocator::get_col_index(address);
                for (uint32_t rank = 0;
                     rank < MyAddressAllocator::ranks; ++rank) {
                    for (uint32_t bank = 0;
                         bank < MyAddressAllocator::banks; ++bank) {
                        for (uint32_t bankgroup = 0;
                             bankgroup < MyAddressAllocator::bankgroups;
                             ++bankgroup) {
                            for (uint32_t channel = 0;
                                 channel < MyAddressAllocator::dram_channels;
                                 ++channel) {
                                map_address(
                                    MyAddressAllocator::make_address_by_index(
                                        rank, bankgroup, bank, row, column,
                                        channel));
                            }
                        }
                    }
                }
            } else {
                for (uint32_t channel = 0;
                     channel < MyAddressAllocator::dram_channels; ++channel) {
                    map_address(MyAddressAllocator::add_channel_index(
                        address, channel));
                }
            }
        }
    };

    for (const auto& instruction : tile.instructions) {
        switch (instruction.opcode) {
            case Opcode::MOVIN:
            case Opcode::MOVOUT:
            case Opcode::MOVOUT_POOL:
                replay_normal(instruction);
                break;
            case Opcode::PIM_HEADER:
            case Opcode::PIM_GWRITE:
            case Opcode::PIM_COMP:
            case Opcode::PIM_COMP_HASH:
            case Opcode::PIM_READRES:
            case Opcode::PIM_COMPS_READRES:
                replay_pim(instruction);
                break;
            default:
                break;
        }
    }
    return mapping_calls;
}

uint64_t MyScheduler::replay_virtual_memory_tiles(
    std::unordered_map<uint32_t, std::deque<Tile>>& queues) {
    if (!_config.virtual_mem_hash_enable) {
        assert(queues.empty());
        return 0;
    }

    uint64_t mapping_calls = 0;
    uint64_t replayed_tiles = 0;
    bool replayed_one = true;
    while (replayed_one) {
        replayed_one = false;
        for (uint32_t core_id = 0; core_id < _config.num_cores; ++core_id) {
            auto queue_it = queues.find(core_id);
            if (queue_it == queues.end() || queue_it->second.empty()) {
                continue;
            }
            Tile tile = std::move(queue_it->second.front());
            queue_it->second.pop_front();
            materialize_tile(tile);
            mapping_calls += replay_virtual_memory_tile(tile);
            ++replayed_tiles;
            replayed_one = true;
        }
    }
    queues.clear();
    spdlog::info(
        "Replayed {} virtual-memory mappings from {} pruned tiles",
        mapping_calls, replayed_tiles);
    return mapping_calls;
}

void MyScheduler::materialize_tile(Tile& tile) {
    if (!tile.deferred_compile) {
        return;
    }
    const auto compile_begin = std::chrono::steady_clock::now();
    tile.materialize();
    const double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - compile_begin)
                                   .count();
    _deferred_compile_time_sec += elapsed_sec;
    if (_current_op != nullptr) {
        const auto stat_it =
            _active_operation_stats.find(_current_op->get_id());
        if (stat_it != _active_operation_stats.end()) {
            stat_it->second.deferred_compile_time_sec += elapsed_sec;
        }
    }
}

void MyScheduler::materialize_tile_range(std::deque<Tile>& queue,
                                         uint32_t begin,
                                         uint32_t end) {
    assert(begin <= end && end <= queue.size());
    for (uint32_t tile_index = begin; tile_index < end; ++tile_index) {
        materialize_tile(queue[tile_index]);
    }
}

bool MyScheduler::is_decode_pruning_target(const std::string& name) const {
    const bool decode_stage =
        _stage == Stage::Decode || _stage == Stage::NPU_Decode ||
        (_stage == Stage::Multi_test &&
         (_test_multi_layer_name == "decode" ||
          _test_multi_layer_name == "npu_decode"));
    const bool ffn_operation = name.find(".ffn.") != std::string::npos;
    const bool qkv_generation =
        name.find(".attn.QGen") != std::string::npos ||
        name.find(".attn.KGen") != std::string::npos ||
        name.find(".attn.VGen") != std::string::npos ||
        name.find(".attn.proj") != std::string::npos;
    // K/V-cache growth can change the logical addresses of attention
    // generation operations between Decode iterations even when the tile
    // geometry is unchanged.  A sampled tile therefore cannot reproduce
    // their exact virtual-memory side effects.  FFN operands remain stable
    // for a fixed request batch, so those operations can use mapping-only
    // replay while attention generation continues through the exact path.
    if (_config.virtual_mem_hash_enable) {
        return decode_stage && ffn_operation;
    }
    return decode_stage && (ffn_operation || qkv_generation);
}

std::string MyScheduler::decode_pruning_template_key(
    const Ptr<Operation>& operation) const {
    assert(operation != nullptr);

    // A name-only key can incorrectly reuse one sample when continuous
    // batching changes request membership or when an operation with the same
    // name is emitted with different tile geometry.  Request progress is
    // deliberately excluded so repeated token iterations for the same batch
    // can still reuse a compatible template.
    std::ostringstream key;
    key << operation->get_name()
        << "|stage=" << static_cast<uint32_t>(_stage)
        << "|requests=";
    for (size_t index = 0; index < _breq.size(); ++index) {
        if (index != 0) {
            key << ',';
        }
        const auto& request = _breq[index];
        key << request->id << ':' << request->input_size << ':'
            << request->output_size << ':' << request->is_initiated;
    }

    key << "|tiles=" << _executable_tile_queue.size() << "|inner=";
    for (const uint32_t extent : operation->get_inner_loop()) {
        key << extent << ',';
    }
    key << "|outer=";
    for (const uint32_t extent : operation->get_outer_loop()) {
        key << extent << ',';
    }
    return key.str();
}

MyScheduler::DecodeCoreTiming
MyScheduler::decode_core_timing_snapshot(uint32_t core_id) const {
    const auto source = _cores.at(core_id)->timing_snapshot();
    DecodeCoreTiming result;
    result.cycle = source.cycle;
    result.compute = source.compute;
    result.memory = source.memory;
    result.idle = source.idle;
    result.load = source.load;
    result.store = source.store;
    result.op_compute = source.op_compute;
    result.op_stall = source.op_stall;
    return result;
}

void MyScheduler::capture_decode_pruning_template(uint32_t operation_id) {
    if (!_config.decode_pruning_enabled) return;
    const auto active = _active_operation_stats.find(operation_id);
    const auto baseline = _decode_pruning_start_timing.find(operation_id);
    const auto key_it = _decode_pruning_operation_keys.find(operation_id);
    if (active == _active_operation_stats.end() ||
        baseline == _decode_pruning_start_timing.end() ||
        key_it == _decode_pruning_operation_keys.end() ||
        !is_decode_pruning_target(active->second.name) ||
        _decode_pruning_templates.count(key_it->second)) {
        return;
    }
    const std::string template_key = key_it->second;
    const bool npu_decode_stage =
        _stage == Stage::NPU_Decode ||
        (_stage == Stage::Multi_test &&
         _test_multi_layer_name == "npu_decode");
    const bool npu_attention_projection =
        npu_decode_stage &&
        (active->second.name.find(".attn.QGen") != std::string::npos ||
         active->second.name.find(".attn.KGen") != std::string::npos ||
         active->second.name.find(".attn.VGen") != std::string::npos ||
         active->second.name.find(".attn.proj") != std::string::npos);
    std::vector<uint64_t> sampled_write_requests;
    std::vector<uint64_t> sampled_write_commands;
    if (_config.dram_trace_simulation_mode) {
        assert(_event_driven_dram != nullptr);
        if (npu_attention_projection) {
            sampled_write_requests =
                _event_driven_dram
                    ->decode_pruning_sampled_write_requests(
                        template_key);
            sampled_write_commands =
                _event_driven_dram
                    ->decode_pruning_sampled_write_commands(
                        template_key);
        }
        _event_driven_dram->finish_decode_pruning_state_sample(
            template_key,
            _config.decode_pruning_sample_iterations);
    } else {
        assert(_dram != nullptr);
        if (npu_attention_projection) {
            sampled_write_requests =
                _dram->decode_pruning_sampled_write_requests(
                    template_key);
            sampled_write_commands =
                _dram->decode_pruning_sampled_write_commands(
                    template_key);
        }
        _dram->finish_decode_pruning_state_sample(
            template_key,
            _config.decode_pruning_sample_iterations);
    }

    DecodePruningTemplate sample;
    sample.operation_cycles =
        *_core_cycle - active->second.start_cycle;
    sample.core_timing.resize(_cores.size());
    sample.core_workload.resize(_cores.size());
    const auto subtract = [](cycle_type current,
                             cycle_type start) -> cycle_type {
        return current >= start ? current - start : 0;
    };
    for (uint32_t core_id = 0; core_id < _cores.size(); ++core_id) {
        const auto current = decode_core_timing_snapshot(core_id);
        const auto& start = baseline->second.at(core_id);
        auto& delta = sample.core_timing[core_id];
        delta.cycle = subtract(current.cycle, start.cycle);
        delta.compute = subtract(current.compute, start.compute);
        delta.memory = subtract(current.memory, start.memory);
        delta.idle = subtract(current.idle, start.idle);
        delta.load = subtract(current.load, start.load);
        delta.store = subtract(current.store, start.store);
        for (size_t op = 0; op < delta.op_compute.size(); ++op) {
            delta.op_compute[op] =
                subtract(current.op_compute[op], start.op_compute[op]);
            delta.op_stall[op] =
                subtract(current.op_stall[op], start.op_stall[op]);
        }
        const auto workload_it =
            _decode_pruning_sample_workload.find(operation_id);
        if (workload_it != _decode_pruning_sample_workload.end()) {
            const auto core_workload = workload_it->second.find(core_id);
            if (core_workload != workload_it->second.end()) {
                sample.core_workload[core_id] = core_workload->second;
            }
        }
    }
    if (npu_attention_projection) {
        std::vector<uint64_t> workload_write_requests(
            _config.dram_channels, 0);
        for (const auto& workload : sample.core_workload) {
            for (uint32_t channel = 0;
                 channel < _config.dram_channels; ++channel) {
                if (channel < workload.channel_memory_writes.size()) {
                    workload_write_requests[channel] +=
                        workload.channel_memory_writes[channel];
                }
            }
        }
        uint64_t sampled_writes = 0;
        for (const uint64_t requests : sampled_write_requests) {
            sampled_writes += requests;
        }
        // DRAMsim3's num_write_requests counter is not maintained on every
        // NPU Store path. In that case the completed sample can report zero
        // even though the sampled tiles contain output Stores. Preserve the
        // per-channel Store workload captured from those executed tiles.
        if (sampled_writes == 0) {
            sampled_write_requests = workload_write_requests;
            for (const uint64_t requests : sampled_write_requests) {
                sampled_writes += requests;
            }
        }
        for (auto& workload : sample.core_workload) {
            workload.memory_writes = 0;
            workload.memory_write_bytes = 0;
            workload.channel_memory_writes.assign(
                _config.dram_channels, 0);
        }
        assert(!sample.core_workload.empty());
        auto& write_workload = sample.core_workload.front();
        write_workload.memory_writes = sampled_writes;
        write_workload.memory_write_bytes =
            sampled_writes * MyAddressAllocator::dram_burst_size;
        write_workload.channel_memory_writes =
            sampled_write_requests;
        sample.channel_write_commands =
            sampled_write_commands;
        spdlog::info(
            "Decode Pruning captured {} actual NPU output Store "
            "requests for {}",
            sampled_writes, active->second.name);
    }
    _decode_pruning_start_timing.erase(baseline);
    _decode_pruning_sample_workload.erase(operation_id);

    auto& accumulator =
        _decode_pruning_accumulators[template_key];
    if (accumulator.core_timing.empty()) {
        accumulator.core_timing.resize(_cores.size());
        accumulator.core_workload.resize(_cores.size());
        if (!sample.channel_write_commands.empty()) {
            accumulator.channel_write_commands.assign(
                _config.dram_channels, 0);
        }
    }
    ++accumulator.samples;
    accumulator.operation_cycles += sample.operation_cycles;
    const auto add_timing = [](DecodeCoreTiming& sum,
                               const DecodeCoreTiming& value) {
        sum.cycle += value.cycle;
        sum.compute += value.compute;
        sum.memory += value.memory;
        sum.idle += value.idle;
        sum.load += value.load;
        sum.store += value.store;
        for (size_t op = 0; op < sum.op_compute.size(); ++op) {
            sum.op_compute[op] += value.op_compute[op];
            sum.op_stall[op] += value.op_stall[op];
        }
    };
    for (uint32_t core_id = 0; core_id < _cores.size(); ++core_id) {
        add_timing(accumulator.core_timing[core_id],
                   sample.core_timing[core_id]);
        // Decode FFN tensor shapes are independent of KV-cache length.
        accumulator.core_workload[core_id] =
            sample.core_workload[core_id];
    }
    for (uint32_t channel = 0;
         channel < sample.channel_write_commands.size(); ++channel) {
        accumulator.channel_write_commands[channel] +=
            sample.channel_write_commands[channel];
    }
    spdlog::info(
        "Decode Pruning sample {}/{} captured for {}: {} cycles",
        accumulator.samples, _config.decode_pruning_sample_iterations,
        active->second.name, sample.operation_cycles);
    _decode_pruning_operation_keys.erase(operation_id);

    if (accumulator.samples <
        _config.decode_pruning_sample_iterations) {
        return;
    }

    const auto average = [&accumulator](cycle_type sum) -> cycle_type {
        return static_cast<cycle_type>(std::llround(
            static_cast<long double>(sum) / accumulator.samples));
    };
    DecodePruningTemplate result;
    result.operation_cycles = static_cast<cycle_type>(std::llround(
        static_cast<long double>(accumulator.operation_cycles) /
        accumulator.samples));
    result.core_timing.resize(_cores.size());
    result.core_workload = accumulator.core_workload;
    result.channel_write_commands.resize(
        accumulator.channel_write_commands.size(), 0);
    for (uint32_t channel = 0;
         channel < accumulator.channel_write_commands.size(); ++channel) {
        result.channel_write_commands[channel] =
            average(accumulator.channel_write_commands[channel]);
    }
    for (uint32_t core_id = 0; core_id < _cores.size(); ++core_id) {
        const auto& sum = accumulator.core_timing[core_id];
        auto& value = result.core_timing[core_id];
        value.cycle = average(sum.cycle);
        value.compute = average(sum.compute);
        value.memory = average(sum.memory);
        value.idle = average(sum.idle);
        value.load = average(sum.load);
        value.store = average(sum.store);
        for (size_t op = 0; op < value.op_compute.size(); ++op) {
            value.op_compute[op] = average(sum.op_compute[op]);
            value.op_stall[op] = average(sum.op_stall[op]);
        }
    }
    _decode_pruning_templates.emplace(template_key, result);
    spdlog::info(
        "Decode Pruning template finalized for {} from {} samples: "
        "{} average cycles",
        active->second.name, accumulator.samples,
        result.operation_cycles);
}

void MyScheduler::prepare_decode_pruning_prediction() {
    assert(_current_op != nullptr);
    assert(!_decode_pruning_pending);
    const auto key_it =
        _decode_pruning_operation_keys.find(_current_op->get_id());
    assert(key_it != _decode_pruning_operation_keys.end());
    const auto template_it =
        _decode_pruning_templates.find(key_it->second);
    assert(template_it != _decode_pruning_templates.end());

    _decode_pruning_pending_workload.clear();
    _decode_pruning_pending_vm_tiles.clear();
    uint64_t deferred_tiles_skipped = 0;
    for (uint32_t core_id = 0; core_id < _cores.size(); ++core_id) {
        auto& queue = _core_executable_tile_queue[core_id];
        deferred_tiles_skipped += static_cast<uint64_t>(std::count_if(
            queue.begin(), queue.end(),
            [](const Tile& tile) { return tile.deferred_compile; }));
        _decode_pruning_pending_workload[core_id] =
            template_it->second.core_workload.at(core_id);
        if (_config.virtual_mem_hash_enable) {
            auto& replay_queue =
                _decode_pruning_pending_vm_tiles[core_id];
            while (!queue.empty()) {
                replay_queue.push_back(std::move(queue.front()));
                queue.pop_front();
            }
        } else {
            queue.clear();
        }
    }
    if (_config.virtual_mem_hash_enable) {
        spdlog::info(
            "Decode compile-time pruning removed {} deferred tiles from "
            "execution for {}; their address mappings will be replayed",
            deferred_tiles_skipped, _current_op->get_name());
    } else {
        spdlog::info(
            "Decode compile-time pruning discarded {} deferred tiles for "
            "{} without materializing instructions",
            deferred_tiles_skipped, _current_op->get_name());
    }

    _decode_pruning_pending = true;
    _decode_pruning_pending_cycles =
        template_it->second.operation_cycles;
    _decode_pruning_pending_operation = _current_op->get_id();
    _decode_pruning_pending_key = key_it->second;
    _active_operation_stats.at(_decode_pruning_pending_operation)
        .remain_tiles = 0;

    if (_config.dram_trace_simulation_mode) {
        assert(_event_driven_dram != nullptr);
        _event_driven_dram->begin_proportional_command_sampling();
    } else {
        assert(_dram != nullptr);
        _dram->begin_proportional_command_sampling();
    }
    spdlog::info(
        "Decode Pruning skips all {} tiles of {} and predicts {} cycles",
        _active_operation_stats.at(_decode_pruning_pending_operation)
            .total_tiles,
        _current_op->get_name(), _decode_pruning_pending_cycles);
}

bool MyScheduler::decode_pruning_prediction_pending() const {
    return _decode_pruning_pending;
}

cycle_type MyScheduler::decode_pruning_prediction_cycles() const {
    assert(_decode_pruning_pending);
    return _decode_pruning_pending_cycles;
}

void MyScheduler::apply_decode_pruning_prediction() {
    assert(_decode_pruning_pending);
    replay_virtual_memory_tiles(_decode_pruning_pending_vm_tiles);
    const auto& prediction =
        _decode_pruning_templates.at(_decode_pruning_pending_key);
    ProportionalWorkloadStat total;
    for (const auto& [core_id, workload] :
         _decode_pruning_pending_workload) {
        _cores.at(core_id)->apply_estimated_workload(workload);
        total += workload;
    }
    for (uint32_t core_id = 0; core_id < prediction.core_timing.size();
         ++core_id) {
        const auto& source = prediction.core_timing[core_id];
        MyCore::TimingBreakdown timing;
        timing.cycle = source.cycle;
        timing.compute = source.compute;
        timing.memory = source.memory;
        timing.idle = source.idle;
        timing.load = source.load;
        timing.store = source.store;
        timing.op_compute = source.op_compute;
        timing.op_stall = source.op_stall;
        _cores.at(core_id)->apply_decode_pruning_timing(timing);
    }
    _icnt->apply_estimated_workload(total);
    const auto* write_command_override =
        prediction.channel_write_commands.empty()
            ? nullptr : &prediction.channel_write_commands;
    if (_config.dram_trace_simulation_mode) {
        _event_driven_dram->apply_estimated_workload(
            total, write_command_override);
    } else {
        _dram->apply_estimated_workload(
            total, write_command_override);
    }
    if (_config.dram_channels > 0) {
        _active_operation_stats.at(_decode_pruning_pending_operation)
            .estimated_pim_inst_count +=
            total.pim_comp / _config.dram_channels;
    }

    Tile estimated_tile;
    estimated_tile.operation_id = _decode_pruning_pending_operation;
    estimated_tile.stat = TileStat(
        _active_operation_stats.at(_decode_pruning_pending_operation)
            .start_cycle);
    estimated_tile.stat.end_cycle =
        estimated_tile.stat.start_cycle + _decode_pruning_pending_cycles;
    estimated_tile.stat.compute_cycles = 0;
    for (const auto& timing : prediction.core_timing) {
        for (cycle_type op_cycle : timing.op_compute) {
            estimated_tile.stat.compute_cycles += op_cycle;
        }
    }
    estimated_tile.stat.memory_reads = total.memory_read_bytes;
    estimated_tile.stat.memory_writes = total.memory_write_bytes;
    estimated_tile.stat.num_calculation = total.num_calculation;
    estimated_tile.stat.logical_tiles = total.tiles;
    estimated_tile.stat.estimated = true;
    _current_op->reduce_tile(estimated_tile);
}

void MyScheduler::apply_decode_pruning_dram_state(
    cycle_type skipped_dram_cycles) {
    assert(_decode_pruning_pending);
    if (_config.dram_trace_simulation_mode) {
        assert(_event_driven_dram != nullptr);
        _event_driven_dram->apply_decode_pruning_state(
            _decode_pruning_pending_key, skipped_dram_cycles);
    } else {
        assert(_dram != nullptr);
        _dram->apply_decode_pruning_state(
            _decode_pruning_pending_key, skipped_dram_cycles);
    }
}

void MyScheduler::complete_decode_pruning_prediction(
    cycle_type end_cycle) {
    assert(_decode_pruning_pending);
    auto& operation =
        _active_operation_stats.at(_decode_pruning_pending_operation);
    if (operation.estimated_pim_inst_count > 0) {
        operation.pim_start_cycle = operation.start_cycle;
        operation.logical_pim_end_cycle = end_cycle;
    }
    Tile completed;
    completed.operation_id = _decode_pruning_pending_operation;
    update_stats_last_tile(0, completed);
    _decode_pruning_pending = false;
    _decode_pruning_pending_cycles = 0;
    _decode_pruning_operation_keys.erase(
        _decode_pruning_pending_operation);
    _decode_pruning_pending_operation = 0;
    _decode_pruning_pending_key.clear();
    _decode_pruning_pending_workload.clear();
    _decode_pruning_pending_vm_tiles.clear();
    refresh_status();
}

void MyScheduler::finalize_proportional_workload_plan() {
    if (_config.dram_trace_simulation_mode) {
        assert(_event_driven_dram != nullptr);
        _event_driven_dram->begin_proportional_command_sampling();
    } else {
        assert(_dram != nullptr);
        _dram->begin_proportional_command_sampling();
    }
    for (uint32_t core_id : _proportional_active_cores) {
        _cores.at(core_id)->begin_proportional_timing_sampling();
    }
    ProportionalWorkloadStat total;
    for (const auto& [core_id, workload] : _proportional_estimated_workload) {
        (void)core_id;
        total += workload;
    }
    assert(total.tiles == _proportional_skipped_tiles);

    Tile estimated_tile;
    estimated_tile.stat = TileStat(std::numeric_limits<uint64_t>::max());
    estimated_tile.stat.end_cycle = 0;
    estimated_tile.stat.logical_tiles = total.tiles;
    estimated_tile.stat.estimated = true;
    estimated_tile.stat.memory_reads = total.memory_read_bytes;
    estimated_tile.stat.memory_writes = total.memory_write_bytes;
    estimated_tile.stat.num_calculation = total.num_calculation;
    _current_op->reduce_tile(estimated_tile);

    spdlog::info(
        "Proportional workload compensation planned for {}: logical tiles {}, read requests {}, write requests {}, read bytes {}, write bytes {}, calculations {}",
        _current_op->get_name(), total.tiles, total.memory_reads,
        total.memory_writes, total.memory_read_bytes, total.memory_write_bytes,
        total.num_calculation);
}

void MyScheduler::print_current_operation_workload_stat() const {
    const OperationStat stat = _current_op->get_stat();
    spdlog::info(
        "Operation workload {}: Logical tiles {}, Measured tiles {}, Estimated tiles {}, Memory reads {}, Memory writes {}, Estimated memory reads {}, Estimated memory writes {}, Calculations {}, Estimated calculations {}",
        stat.op_name, stat.measured_tiles + stat.estimated_tiles,
        stat.measured_tiles, stat.estimated_tiles, stat.memory_reads,
        stat.memory_writes, stat.estimated_memory_reads,
        stat.estimated_memory_writes, stat.num_calculation,
        stat.estimated_num_calculation);
}

void MyScheduler::prepare_proportional_sampling() {
    if (_proportional_plan_ready || !_sim_accelerate ||
        _config.accelerate_method != "Proportional") {
        return;
    }

    if (_current_op->get_optype() == "GEMM_Att") {
        prepare_proportional_attention_sampling();
        return;
    }
    if (_current_op->get_optype() == "Softmax") {
        prepare_proportional_softmax_sampling();
        return;
    }

    if (_k_loop_size == 0 || !std::isfinite(_proportional_sample_ratio) ||
        _proportional_sample_ratio <= 0.0 || _proportional_sample_ratio > 1.0) {
        spdlog::warn(
            "Disable Proportional acceleration for {}: invalid K-loop size {} or sample ratio {}",
            _current_op->get_name(), _k_loop_size, _proportional_sample_ratio);
        _sim_accelerate = false;
        return;
    }

    uint32_t common_kloops = std::numeric_limits<uint32_t>::max();
    for (uint32_t core_id = 0; core_id < _config.num_cores; ++core_id) {
        const uint32_t tile_count = static_cast<uint32_t>(_core_executable_tile_queue[core_id].size());
        if (tile_count == 0) {
            continue;
        }
        _proportional_active_cores.push_back(core_id);
        common_kloops = std::min(common_kloops, tile_count / _k_loop_size);
    }

    _proportional_sample_tiles = std::clamp<uint32_t>(
        static_cast<uint32_t>(std::floor(
            _k_loop_size * _proportional_sample_ratio + 0.5)),
        1, _k_loop_size);

    if (_proportional_active_cores.empty() || common_kloops < 2) {
        spdlog::info(
            "Disable Proportional acceleration for {}: fewer than two complete K-loops per active core",
            _current_op->get_name());
        _sim_accelerate = false;
        _proportional_active_cores.clear();
        return;
    }

    const uint32_t prefix_tiles = _k_loop_size + _proportional_sample_tiles;
    const uint32_t common_tiles = common_kloops * _k_loop_size;
    const uint32_t skipped_per_core = common_tiles - prefix_tiles;
    if (skipped_per_core == 0) {
        spdlog::info(
            "Disable Proportional acceleration for {}: the selected ratio leaves no complete middle tiles to skip",
            _current_op->get_name());
        _sim_accelerate = false;
        _proportional_active_cores.clear();
        return;
    }

    uint32_t actual_tiles = 0;
    for (uint32_t core_id : _proportional_active_cores) {
        auto& queue = _core_executable_tile_queue[core_id];
        const uint32_t original_count = static_cast<uint32_t>(queue.size());
        const uint32_t tail_count = original_count - common_tiles;

        materialize_tile_range(queue, 0, prefix_tiles);
        materialize_tile_range(queue, common_tiles, original_count);
        for (uint32_t tile_index = prefix_tiles; tile_index < common_tiles;
             ++tile_index) {
            const uint32_t representative_index =
                tile_index % _k_loop_size;
            record_proportional_skipped_tile(
                core_id, queue[tile_index], queue[representative_index]);
        }

        auto tail_begin = queue.begin() + common_tiles;
        for (auto it = tail_begin; it != queue.end(); ++it) {
            _proportional_tail_queues[core_id].push_back(std::move(*it));
        }
        queue.erase(tail_begin, queue.end());
        queue.erase(queue.begin() + prefix_tiles, queue.end());

        _proportional_prefix_tiles[core_id] = prefix_tiles;
        _proportional_finish_cycles[core_id].clear();
        actual_tiles += prefix_tiles + tail_count;
        _proportional_skipped_tiles += skipped_per_core;
    }

    const auto stat_it = _active_operation_stats.find(_current_op->get_id());
    assert(stat_it != _active_operation_stats.end());
    stat_it->second.remain_tiles = actual_tiles;

    _proportional_common_kloops = common_kloops;
    finalize_proportional_workload_plan();
    _proportional_plan_ready = true;
    spdlog::info(
        "Proportional sampling {}: ratio {:.3f}, K-loop tiles {}, sampled tiles/core {}, common loops {}, skipped tiles {}, retained tail tiles {}",
        _current_op->get_name(), _proportional_sample_ratio, _k_loop_size,
        _proportional_sample_tiles, _proportional_common_kloops,
        _proportional_skipped_tiles,
        static_cast<uint32_t>(_total_tiles.size()) - _proportional_skipped_tiles -
            prefix_tiles * static_cast<uint32_t>(_proportional_active_cores.size()));
}

void MyScheduler::prepare_proportional_attention_sampling() {
    struct CoreHeadPlan {
        uint32_t core_id;
        std::vector<uint32_t> head_ends;
    };

    std::vector<CoreHeadPlan> plans;
    uint32_t common_head_rounds = std::numeric_limits<uint32_t>::max();
    bool has_unassigned_core = false;

    for (uint32_t core_id = 0; core_id < _config.num_cores; ++core_id) {
        const auto& queue = _core_executable_tile_queue[core_id];
        if (queue.empty()) {
            has_unassigned_core = true;
            break;
        }

        CoreHeadPlan plan{core_id, {}};
        uint32_t previous_batch = queue.front().batch;
        uint32_t previous_head = queue.front().head_index;
        for (uint32_t tile_index = 1; tile_index < queue.size(); ++tile_index) {
            const auto& tile = queue[tile_index];
            if (tile.batch != previous_batch || tile.head_index != previous_head) {
                plan.head_ends.push_back(tile_index);
                previous_batch = tile.batch;
                previous_head = tile.head_index;
            }
        }
        plan.head_ends.push_back(static_cast<uint32_t>(queue.size()));

        common_head_rounds = std::min<uint32_t>(
            common_head_rounds, static_cast<uint32_t>(plan.head_ends.size()));
        plans.push_back(std::move(plan));
    }

    // Two heads are retained as warmup. The following two heads form a complete
    // alternating sample pair. All remaining common heads are predicted.
    if (has_unassigned_core || plans.empty() || common_head_rounds < 6) {
        spdlog::info(
            "Disable Proportional attention acceleration for {}: every core needs at least six complete head rounds for 2-head warmup, 2-head sampling, and prediction",
            _current_op->get_name());
        _sim_accelerate = false;
        return;
    }

    const uint32_t predicted_head_rounds = common_head_rounds;
    assert(predicted_head_rounds >= 6);

    uint32_t actual_tiles = 0;
    uint32_t retained_tail_tiles = 0;
    for (const auto& plan : plans) {
        auto& queue = _core_executable_tile_queue[plan.core_id];
        const uint32_t warmup_end = plan.head_ends[1];
        const uint32_t prefix_end = plan.head_ends[3];
        const uint32_t common_end =
            plan.head_ends[predicted_head_rounds - 1];
        const uint32_t original_count = static_cast<uint32_t>(queue.size());
        const uint32_t tail_count = original_count - common_end;
        const uint32_t skipped_count = common_end - prefix_end;

        materialize_tile_range(queue, 0, prefix_end);
        materialize_tile_range(queue, common_end, original_count);
        for (uint32_t tile_index = prefix_end; tile_index < common_end;
             ++tile_index) {
            const auto head_it = std::lower_bound(
                plan.head_ends.begin(), plan.head_ends.end(),
                tile_index + 1);
            assert(head_it != plan.head_ends.end());
            const uint32_t head_round = static_cast<uint32_t>(
                std::distance(plan.head_ends.begin(), head_it));
            const uint32_t head_start =
                head_round == 0 ? 0 : plan.head_ends[head_round - 1];
            const uint32_t offset = tile_index - head_start;
            const uint32_t representative_round =
                2 + ((head_round - 2) % 2);
            const uint32_t representative_start =
                plan.head_ends[representative_round - 1];
            const uint32_t representative_end =
                plan.head_ends[representative_round];
            const uint32_t representative_index = std::min<uint32_t>(
                representative_start + offset,
                representative_end - 1);
            record_proportional_skipped_tile(
                plan.core_id, queue[tile_index],
                queue[representative_index]);
        }

        for (auto it = queue.begin() + common_end; it != queue.end(); ++it) {
            _proportional_tail_queues[plan.core_id].push_back(std::move(*it));
        }
        queue.erase(queue.begin() + common_end, queue.end());
        queue.erase(queue.begin() + prefix_end, queue.end());

        _proportional_active_cores.push_back(plan.core_id);
        _proportional_warmup_head_tiles[plan.core_id] = warmup_end;
        _proportional_prefix_tiles[plan.core_id] = prefix_end;
        _proportional_finish_cycles[plan.core_id].clear();
        _proportional_skipped_tiles += skipped_count;
        retained_tail_tiles += tail_count;
        actual_tiles += prefix_end + tail_count;
    }

    const auto stat_it = _active_operation_stats.find(_current_op->get_id());
    assert(stat_it != _active_operation_stats.end());
    stat_it->second.remain_tiles = actual_tiles;

    _proportional_attention_mode = true;
    _proportional_common_head_rounds = predicted_head_rounds;
    finalize_proportional_workload_plan();
    _proportional_plan_ready = true;
    spdlog::info(
        "Proportional attention sampling {}: warmup heads/core 2, sampled heads/core 2, common head rounds {}, skipped tiles {}, retained irregular tail tiles {}",
        _current_op->get_name(), _proportional_common_head_rounds,
        _proportional_skipped_tiles, retained_tail_tiles);
}

void MyScheduler::prepare_proportional_softmax_sampling() {
    const uint32_t warmup_rounds = _config.softmax_warmup_rounds;
    const uint32_t sampled_rounds = _config.softmax_sample_rounds;
    const uint32_t prefix_rounds = warmup_rounds + sampled_rounds;

    if (warmup_rounds == 0 || sampled_rounds == 0) {
        spdlog::warn(
            "Disable Proportional Softmax acceleration for {}: warmup and sample rounds must both be nonzero",
            _current_op->get_name());
        _sim_accelerate = false;
        return;
    }

    uint32_t common_rounds = std::numeric_limits<uint32_t>::max();
    bool has_unassigned_core = false;
    for (uint32_t core_id = 0; core_id < _config.num_cores; ++core_id) {
        const uint32_t tile_count = static_cast<uint32_t>(
            _core_executable_tile_queue[core_id].size());
        if (tile_count == 0) {
            has_unassigned_core = true;
            break;
        }
        _proportional_active_cores.push_back(core_id);
        common_rounds = std::min(common_rounds, tile_count);
    }

    if (has_unassigned_core || common_rounds <= prefix_rounds) {
        spdlog::info(
            "Disable Proportional Softmax acceleration for {}: every core needs more than {} complete outer-tile rounds",
            _current_op->get_name(), prefix_rounds);
        _sim_accelerate = false;
        _proportional_active_cores.clear();
        return;
    }

    uint32_t actual_tiles = 0;
    uint32_t retained_tail_tiles = 0;
    for (uint32_t core_id : _proportional_active_cores) {
        auto& queue = _core_executable_tile_queue[core_id];
        const uint32_t original_count = static_cast<uint32_t>(queue.size());
        const uint32_t tail_count = original_count - common_rounds;
        const uint32_t skipped_count = common_rounds - prefix_rounds;

        materialize_tile_range(queue, 0, prefix_rounds);
        materialize_tile_range(queue, common_rounds, original_count);
        for (uint32_t tile_index = prefix_rounds; tile_index < common_rounds;
             ++tile_index) {
            const uint32_t representative_index =
                warmup_rounds +
                ((tile_index - warmup_rounds) % sampled_rounds);
            record_proportional_skipped_tile(
                core_id, queue[tile_index], queue[representative_index]);
        }

        for (auto it = queue.begin() + common_rounds; it != queue.end(); ++it) {
            _proportional_tail_queues[core_id].push_back(std::move(*it));
        }
        queue.erase(queue.begin() + common_rounds, queue.end());
        queue.erase(queue.begin() + prefix_rounds, queue.end());

        _proportional_prefix_tiles[core_id] = prefix_rounds;
        _proportional_finish_cycles[core_id].clear();
        _proportional_skipped_tiles += skipped_count;
        retained_tail_tiles += tail_count;
        actual_tiles += prefix_rounds + tail_count;
    }

    const auto stat_it = _active_operation_stats.find(_current_op->get_id());
    assert(stat_it != _active_operation_stats.end());
    stat_it->second.remain_tiles = actual_tiles;

    _proportional_softmax_mode = true;
    _proportional_common_softmax_rounds = common_rounds;
    finalize_proportional_workload_plan();
    _proportional_plan_ready = true;
    spdlog::info(
        "Proportional Softmax sampling {}: warmup rounds/core {}, sampled rounds/core {}, common rounds {}, skipped tiles {}, retained partial-round tail tiles {}",
        _current_op->get_name(), warmup_rounds, sampled_rounds,
        _proportional_common_softmax_rounds, _proportional_skipped_tiles,
        retained_tail_tiles);
}

bool MyScheduler::proportional_ready_to_predict() const {
    if (!_proportional_plan_ready || _proportional_prediction_applied ||
        _proportional_tail_is_active) {
        return false;
    }
    for (uint32_t core_id : _proportional_active_cores) {
        const auto prefix_it = _proportional_prefix_tiles.find(core_id);
        const auto finish_it = _proportional_finish_cycles.find(core_id);
        if (prefix_it == _proportional_prefix_tiles.end() ||
            finish_it == _proportional_finish_cycles.end() ||
            finish_it->second.size() < prefix_it->second) {
            return false;
        }
    }
    return true;
}

bool MyScheduler::proportional_tail_phase() const {
    return _proportional_tail_is_active;
}

bool MyScheduler::proportional_operation_complete(uint32_t operation_id) const {
    const auto stat_it = _active_operation_stats.find(operation_id);
    return stat_it != _active_operation_stats.end() && stat_it->second.remain_tiles == 0;
}

bool MyScheduler::release_proportional_tail_tiles() {
    assert(_proportional_plan_ready);
    assert(!_proportional_prediction_applied);

    bool has_tail = false;
    for (uint32_t core_id : _proportional_active_cores) {
        auto& tail = _proportional_tail_queues[core_id];
        auto& queue = _core_executable_tile_queue[core_id];
        while (!tail.empty()) {
            queue.push_back(std::move(tail.front()));
            tail.pop_front();
            has_tail = true;
        }
    }
    _proportional_prediction_applied = true;
    _proportional_tail_is_active = has_tail;
    return has_tail;
}

const std::unordered_map<uint32_t, ProportionalWorkloadStat>&
MyScheduler::proportional_estimated_workload() const {
    return _proportional_estimated_workload;
}

bool MyScheduler::proportional_workload_applied() const {
    return _proportional_workload_applied;
}

void MyScheduler::mark_proportional_workload_applied() {
    assert(!_proportional_workload_applied);
    replay_virtual_memory_tiles(_proportional_vm_replay_tiles);
    for (const auto& [core_id, workload] : _proportional_estimated_workload) {
        assert(core_id < _cores.size());
        _cores[core_id]->apply_estimated_workload(workload);
        if (_config.dram_channels > 0) {
            _active_operation_stats.at(_current_op->get_id())
                .estimated_pim_inst_count +=
                workload.pim_comp / _config.dram_channels;
        }
    }
    _proportional_workload_applied = true;
}

void MyScheduler::apply_proportional_core_timing() {
    if (_proportional_timing_applied) return;
    assert(_proportional_plan_ready);
    assert(!_proportional_core_estimated_cycles.empty());

    uint64_t estimated_operation_compute_cycles = 0;
    for (uint32_t core_id : _proportional_active_cores) {
        const cycle_type current_cycle = _cores.at(core_id)->_core_cycle;
        const cycle_type core_finish =
            _proportional_core_estimated_cycles.at(core_id);
        const cycle_type skipped_work_cycles = core_finish > current_cycle
            ? core_finish - current_cycle : 0;
        const cycle_type wait_start =
            std::max<cycle_type>(current_cycle, core_finish);
        const cycle_type global_wait_cycles =
            _estimated_all_cycle > wait_start
                ? _estimated_all_cycle - wait_start : 0;
        const auto estimated = _cores.at(core_id)->apply_estimated_timing(
            skipped_work_cycles, global_wait_cycles);
        for (cycle_type op_cycles : estimated.op_compute) {
            estimated_operation_compute_cycles += op_cycles;
        }
    }

    if (estimated_operation_compute_cycles > 0) {
        Tile timing_tile;
        const OperationStat operation_stat = _current_op->get_stat();
        timing_tile.stat = TileStat(operation_stat.start_cycle);
        timing_tile.stat.logical_tiles = 0;
        timing_tile.stat.estimated = true;
        timing_tile.stat.compute_cycles = estimated_operation_compute_cycles;
        _current_op->reduce_tile(timing_tile);
    }
    auto& operation = _active_operation_stats.at(_current_op->get_id());
    if (operation.estimated_pim_inst_count > 0) {
        operation.logical_pim_end_cycle = std::max<cycle_type>(
            operation.logical_pim_end_cycle, _estimated_all_cycle);
    }
    _proportional_timing_applied = true;
}

cycle_type MyScheduler::get_estimated_all_cycle() {
    return _estimated_all_cycle;
}

bool MyScheduler::are_all_cores_stable() {
    for (int core_id = 0; core_id < _config.num_cores; core_id++) {
        if (!_core_stable[core_id]) {
            return false;
        }
    }
    return true;
}

bool MyScheduler::are_all_cores_head_second_done() {
    for (int core_id = 0; core_id < _config.num_cores; core_id++) {
        if (!_core_head_second_done[core_id]) {
            return false;
        }
    }
    return true;
}



Tile& MyScheduler::top_tile(uint32_t core_id) {
    // return the first executable tile
    static Tile empty_tile = Tile{.status = Tile::Status::EMPTY};
    if (_executable_tile_queue.empty() && _core_executable_tile_queue[core_id].empty()) {  // both queues are empty
        return empty_tile;
    } else if (!_core_executable_tile_queue[core_id].empty()) {
        Tile& tile = _core_executable_tile_queue[core_id].front();  // get tile from the of each core
        materialize_tile(tile);
        return tile;
    }
    else {
        Tile& tile = _executable_tile_queue.front();  // get tile from the unique queue
        if (tile.status == Tile::Status::BAR) {
            return empty_tile;
        } else {
            tile.stage_platform = StagePlatform::SA;
            materialize_tile(tile);
            return tile;
        }
    }
}


void MyScheduler::get_tile(uint32_t core_id) {
    if (_executable_tile_queue.empty() && _core_executable_tile_queue[core_id].empty()) {
        return;
    }
    else {
        Tile& tile = _core_executable_tile_queue[core_id].front();

        if (tile.pim_tile == true) {assert(core_id == 0);}  // Double check the PIM Tile can only be executed by Core 0
        if (tile.status == Tile::Status::BAR) {
            RunningOperationStat stat = _finished_operation_stats[tile.operation_id];
            if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
                /* POP only if all lauched tiles are finished */
                _executable_tile_queue.pop_front();
                _finished_operation_stats[tile.operation_id].launched_tiles++;
                _finished_operation_stats[tile.operation_id].remain_tiles--;
            }
            return;
        }
        else {
            const uint32_t operation_id = tile.operation_id;
            auto& operation_stat = _active_operation_stats[operation_id];
            operation_stat.launched_tiles++;
            const std::string operation_name = operation_stat.name;
            _core_executable_tile_queue[core_id].pop_front();
            spdlog::info("At cycle {} Core {} get Tile, pop a tile from the Scheduler executable tile queue, {} exist for operation {}",
                *_core_cycle, core_id, get_exist_tile_count(), operation_name);
        }
    }
}

bool MyScheduler::is_executable_tile_empty() {
    if (!_executable_tile_queue.empty()) {
        return false;
    }
    for (uint32_t core_id = 0; core_id < _config.num_cores; core_id++) {
        if (!_core_executable_tile_queue[core_id].empty()) {
            return false;
        }
    }
    return true;
}

uint32_t MyScheduler::get_exist_tile_count() {
    uint32_t exist_tile_count = 0;
    if (!_executable_tile_queue.empty()) {
        exist_tile_count += _executable_tile_queue.size();
    }
    for (uint32_t core_id = 0; core_id < _config.num_cores; core_id++) {
        exist_tile_count += _core_executable_tile_queue[core_id].size();
    }
    return exist_tile_count;
}


void MyScheduler::print_stat() {
    cycle_type prev_cycles = 0;
    // Print Global PIM Bandwidth Utilization
    if (_total_pim_duration > 0) {
        double total_pim_bw = (double)_total_pim_inst_count * (_config.mem_config.burst_cycle / 2) / _total_pim_duration;
        spdlog::info("Total Inference PIM Bandwidth Utilization: {:.2f}% (Total Count: {}, Total Duration: {})",
            total_pim_bw * 100, _total_pim_inst_count, _total_pim_duration);
    }

    for (auto stage_stat : _stage_stats) {
        auto stage_name = stage_stat.first;
        auto stage_cycles = stage_stat.second;
        auto exec_cycles = stage_cycles - prev_cycles;
        spdlog::info("Stage {} : {} cycles", stage_name, exec_cycles);
        prev_cycles = stage_cycles;
    }
}


void MyScheduler::print_op_stat() {
    cycle_type prev_cycles = 0;
    for (auto op_stat : _op_stats) {
        auto op_name = op_stat.first;
        auto op_cycles = op_stat.second;
        auto exec_cycles = op_cycles - prev_cycles;
        spdlog::info("Op {} : {} cycles", op_name, exec_cycles);
        prev_cycles = op_cycles;
    }
}


Ops MyScheduler::getOpType(const std::string& opStr) {
    static const std::unordered_map<std::string, Ops> opMap = {
        {"rmsnorm", Ops::RMSNorm},
        {"layernorm", Ops::LayerNorm},
        {"gemm", Ops::GEMM},
        {"gemm_att", Ops::GEMM_Att},
        {"gemv", Ops::GEMV},
        {"gemv_att", Ops::GEMV_Att},
        {"split", Ops::Split},
        {"softmax", Ops::Softmax},
        {"add", Ops::Add},
        {"mul", Ops::Mul},
        {"gelu", Ops::Gelu},
        {"silu", Ops::SiLU},
        {"reshape", Ops::Reshape},
        {"transpose", Ops::Transpose},
        {"concat", Ops::Concat},
        {"data_convert", Ops::DataConvert},
        {"dataconvert", Ops::DataConvert},
        {"gemv_softmax", Ops::GEMV_Softmax},
        {"gemv_add", Ops::GEMV_Add},
        {"pim_gemv", Ops::PIM_GEMV}, //PIM Operation
        {"pim_gemv_qkt", Ops::PIM_GEMV_QKT},
        {"pim_gemv_sv", Ops::PIM_GEMV_SV},
        {"pim_gemv_att", Ops::PIM_GEMV_Att},
    };

    // Convert input string to lowercase
    std::string lowerOpStr = opStr;
    std::transform(lowerOpStr.begin(), lowerOpStr.end(), lowerOpStr.begin(), ::tolower);

    // find and return the corresponding Operation
    auto it = opMap.find(lowerOpStr);
    if (it != opMap.end()) {
        return it->second;
    }
    // not find
    throw std::invalid_argument("Unknown operation string: " + opStr);
}


void MyScheduler::set_scheduler_cycles(uint64_t cycles) {
    _cycles = cycles;
}


// Bind Implementation
void MyScheduler::bind_system(Client* client, PIM* dram,
                              EventDrivenDram* event_driven_dram,
                              MyInterconnect* icnt,
                              const std::vector<std::unique_ptr<MyCore>>& cores) {
    _client = client;
    _dram = dram;
    _event_driven_dram = event_driven_dram;
    _icnt = icnt;
    _cores.clear();
    _cores.reserve(cores.size());
    for (const auto& core : cores) {
        _cores.push_back(core.get());
    }
}

void MyScheduler::sync_accelerated_cycles(cycle_type core_cycle, cycle_type dram_delta, cycle_type icnt_delta) {
    assert(_dram != nullptr);
    assert(_icnt != nullptr);

    for (auto* core : _cores) {
        core->set_core_cycle(core_cycle);
    }

    _client->set_client_cycle(core_cycle);
    set_scheduler_cycles(core_cycle);
    _dram->set_dram_cycles(dram_delta);
    _icnt->set_icnt_cycles(icnt_delta);
}
