#include "Dram.h"

// >>> gsheo

PIM::PIM(const SysConfig& config, DramDataContainer* data_container)
    : Dram(config),
      _mem(std::make_unique<dramsim3::NewtonSim>(config.memory_config_path_, config.log_dir)),
      _data_container(data_container) {
    _total_processed_requests.resize(config.dram_channels);
    _processed_requests.resize(config.dram_channels);

    for (int ch = 0; ch < config.dram_channels; ch++) {
        _total_processed_requests[ch] = 0;
        _processed_requests[ch] = 0;
    }

    _stat_interval = 1000;
    _stats.resize(config.dram_channels);
    for (size_t i = 0; i < config.dram_channels; ++i) {
        _stats[i].push_back(MemoryIOStat(0, i, _stat_interval));
    }

    _cycles = 0;
    //_burst_cycle = _mem->GetBurstLength() / 2;  // double data rate
    _burst_cycle = _mem->GetBurstCycle();
    _total_done_requests = 0;
    _stage_cycles = 0;

    _push_valid.resize(MyAddressAllocator::dram_channels, true);
    _pop_valid.resize(MyAddressAllocator::dram_channels, true);

    _last_push_cycle.resize(MyAddressAllocator::dram_channels, 0);
    _last_pop_cycle.resize(MyAddressAllocator::dram_channels, 0);
    _actual_request_counts.resize(MyAddressAllocator::dram_channels, 0);
    _actual_read_request_counts.resize(MyAddressAllocator::dram_channels, 0);
    _actual_write_request_counts.resize(MyAddressAllocator::dram_channels, 0);
    _command_request_baseline.resize(MyAddressAllocator::dram_channels, 0);
    _command_read_request_baseline.resize(MyAddressAllocator::dram_channels, 0);
    _command_write_request_baseline.resize(MyAddressAllocator::dram_channels, 0);
    _command_counter_baseline.resize(MyAddressAllocator::dram_channels);
    _command_warmup_request_counts.resize(MyAddressAllocator::dram_channels, 0);
    _command_warmup_read_request_counts.resize(
        MyAddressAllocator::dram_channels, 0);
    _command_warmup_write_request_counts.resize(
        MyAddressAllocator::dram_channels, 0);
    _command_warmup_counter_counts.resize(MyAddressAllocator::dram_channels);
    _estimated_command_counts.resize(MyAddressAllocator::dram_channels);
    _active_cycle_baseline.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));
    _idle_cycle_baseline.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));
    _sref_cycle_baseline.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));
    _pim_active_cycle_baseline.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));
    _pim_idle_cycle_baseline.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));
    _estimated_pim_active_cycles.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));
    _estimated_pim_idle_cycles.assign(MyAddressAllocator::dram_channels,
        std::vector<uint64_t>(config.mem_config.ranks, 0));

    // >>> Address mapping test
    /*
    int ch = 1;
    int ra = 0;
    int bg = 2;
    int ba = 1;
    int row = 231;
    int col = 10;
    uint64_t dram_addr = AddressConfig::make_address(ch, ra, bg, ba, row, col);
    uint64_t newtonsim_addr = MakeAddress(ch, ra, bg, ba, row, col);
    // MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col)
    uint64_t my_dram_addr = MyAddressAllocator::make_address_with_channel(bg, ba, ra, row, col, ch);
    uint64_t address_make_by_index = MyAddressAllocator::make_address_by_index_with_shift(ra, bg, ba, row, col, ch);
    // MyAddressAllocator::make_address_with_channel(uint32_t inner_row_loop_index, uint32_t middle_row_loop_index, uint32_t outer_row_loop_index, uint32_t row, uint32_t col, uint32_t channel)
    // assert(address_make_by_index == newtonsim_addr);
    // <<< Address mapping test
    spdlog::info("Newton init");
    */
}

bool PIM::running() { return false; }

void PIM::cycle() {
    _mem->ClockTick();
    _cycles++;
    _stage_cycles++;
    int interval = 100000;
    if (_cycles % interval == 0) {
        spdlog::info("-------------DRAM BW Check--------------");
        for (int ch = 0; ch < _config.dram_channels; ch++) {
            float util = ((float)_processed_requests[ch] * _burst_cycle) / interval * 100;
            spdlog::info("DRAM CH[{}]: BW Util {:.2f}%", ch, util);
            _total_processed_requests[ch] += _processed_requests[ch];
            _processed_requests[ch] = 0;
        }
    }

    // update stats
    if (_cycles % _stat_interval == 0) {
        double duration = (double)_stat_interval / (_config.dram_freq * 1000000.0);
        int banks_per_channel = _config.mem_config.banks * _config.mem_config.ranks;
        double total_buffer_size_kb = (_config.mem_config.input_buffer_size + _config.mem_config.output_buffer_size * PIM_Parameters::PU_num_per_channel)/ 1024.0;
        double static_energy = total_buffer_size_kb * _config.mem_config.pim_buffer_static_power_per_kb * duration;
        static_energy += banks_per_channel * _config.mem_config.pim_static_power_per_pu * duration;

        for (auto ch = 0; ch < _config.dram_channels; ++ch) {
            auto stat = MemoryIOStat(_cycles, ch, _stat_interval);
            stat.pim_energy += static_energy;
            _stats[ch].push_back(stat);
        }
    }

    for (auto ch = 0; ch < _config.dram_channels; ++ch) {
        _push_valid[ch] = true;
        _pop_valid[ch] = true;
    }

}

uint64_t PIM::MakeAddress(int channel, int rank, int bankgroup, int bank, int row, int col) {
    return _mem->MakeAddress(channel, rank, bankgroup, bank, row, col);
}

uint64_t PIM::EncodePIMHeader(int channel, int row, bool for_gwrite, int num_comps,
                              int num_readres) {
    return _mem->EncodePIMHeader(channel, row, for_gwrite, num_comps, num_readres);
}

bool PIM::is_full(uint32_t cid, MemoryAccess *request) {
    request->dram_address = MyAddressAllocator::add_channel_index(request->dram_address, cid);
    bool full = !_mem->WillAcceptTransaction(request->dram_address, int(request->req_type));
    // if (cid == 0) {
    // if (full)
    //     spdlog::info("MEMORY channel {} is full!! {}", cid,
    //     memAccessTypeString(request->req_type));
    // else
    //     spdlog::info("MEMORY channel {} can receive mem_req {}", cid,
    //                  memAccessTypeString(request->req_type));
    // }
    if (full) {
        // spdlog::info("Current DRAM is full, current write_buffer_");
        // , return queue size is {}", _mem->response_queues_[cid].
    }

    return full;
}

void PIM::push(uint32_t cid, MemoryAccess *request) {
    // std::string acc_type_str = memAccessTypeString(request->req_type);
    // if (cid == 0) spdlog::info("{} cid:{}", acc_type_str, cid)

    // uint32_t mem_ch = get_channel_id(request);
    // assert(mem_ch == cid);
    /*
    auto data_aligned = MyAddressAllocator::dram_burst_size;

    const addr_type atomic_bytes = _mem->GetBurstLength() * _mem->GetBusBits() / 8;
    // const addr_type atomic_bytes =
    //     _mem->GetBurstLength() * _mem->GetBusBits() / 8;
    const addr_type target_addr = request->dram_address;
    // align address
    const addr_type start_addr = target_addr - (target_addr % atomic_bytes);

    assert(start_addr == target_addr);
    assert(request->size == atomic_bytes);
    */

    const addr_type target_addr = request->dram_address;
    auto channel_index = MyAddressAllocator::get_channel_index(target_addr);
    assert(cid == channel_index);

    // assert(_push_valid[channel_index] == true);
    _push_valid[channel_index] = false;

    assert(_last_push_cycle[channel_index] < _cycles);
    _last_push_cycle[channel_index] = _cycles;

    // spdlog::info("Channel {} Pushed at dram cycle {}", cid, _cycles);
    request->dram_enter_cycle = _cycles;
    // MyAddressAllocator::check_addrs({request->dram_address});
    int count = 0;
    request->request = false;
    _mem_req_cnt++;
    _actual_request_counts[cid]++;
    if (request->req_type == MemoryAccessType::READ) {
        _actual_read_request_counts[cid]++;
    } else if (request->req_type == MemoryAccessType::WRITE) {
        _actual_write_request_counts[cid]++;
    }
    _mem->AddTransaction(target_addr, int(request->req_type), request,
                         request->core_id);

    // 后期用于生成不同的读写操作信息
    // spdlog:: info("(CycleAccurate DRAM) A {} Memory Transaction {} is pushed in to channel {} at dram cycles {}, current pending read Transaction is {} ",
    //    memAccessTypeString(request->req_type),channel_index, request->dram_address, request->dram_enter_cycle, _mem->pending_read_q_.size());
}

bool PIM::is_empty(uint32_t cid) {
    // spdlog::info("pim is_empty(" + std::to_string(cid) + "):" + std::to_string(_mem->IsEmpty(cid)));
    return _mem->IsEmpty(cid);
}

MemoryAccess *PIM::top(uint32_t cid) {
    assert(!is_empty(cid));
    // This is the operation for data container
    auto* memory_response = (MemoryAccess *)_mem->Top(cid);
    if (_data_container != nullptr) {
        if (!memory_response->request && !memory_response->data_ready) {
            const addr_type addr = memory_response->dram_address;
            if (memory_response->req_type == MemoryAccessType::READ) {
                memory_response->data = _data_container->read_burst(addr);
            }
            else if ((memory_response->req_type == MemoryAccessType::WRITE) && !memory_response->data.empty()) {
                _data_container->write_burst(addr, memory_response->data);
            }
            memory_response->data_ready = true;
        }
    }

    return memory_response;
}
void PIM::update_stat(uint32_t cid) {
    // READ, WRITE, GWRITE, COMP, READRES, P_HEADER, COMPS_READRES, SIZE
    MemoryAccess *memory_response = top(cid);

    bool response = false;
    switch (memory_response->req_type) {
        case MemoryAccessType::READ:
            _stats[cid].back().memory_reads += memory_response->size;
            response = true;
            break;
        case MemoryAccessType::WRITE:
            _stats[cid].back().memory_writes += memory_response->size;
            response = true;
            break;
        case MemoryAccessType::READRES:
        case MemoryAccessType::COMPS_READRES:
            _stats[cid].back().pim_reads += memory_response->size;
            _stats[cid].back().pim_energy += _config.mem_config.pim_buffer_dynamic_power_per_bit * memory_response->size * 8;
            response = true;
            break;
        case MemoryAccessType::GWRITE:
            _stats[cid].back().pim_writes += memory_response->size;
            _stats[cid].back().pim_energy += _config.mem_config.pim_buffer_dynamic_power_per_bit * memory_response->size * 8;
            response = true;
            break;
        case MemoryAccessType::COMP:
            _stats[cid].back().pim_comps += memory_response->size;
            _stats[cid].back().pim_energy += _config.mem_config.pim_compute_power_per_mac * memory_response->size;
            response = true;
            break;
        case MemoryAccessType::COMP_HASH:
            _stats[cid].back().pim_comps += memory_response->size;
            _stats[cid].back().pim_energy += _config.mem_config.pim_compute_power_per_mac * memory_response->size;
            response = true;
            break;
            // default:
            //     ast(0);
    }

    if (response) {
        _processed_requests[cid]++;
        _total_done_requests++;
    }
}

void PIM::pop(uint32_t cid) {
    // make sure update stat before mem-pop
    update_stat(cid);
    assert(!is_empty(cid) && _pop_valid[cid]==true);
    _pop_valid[cid] = false;
    _mem->Pop(cid);
    // spdlog::info("Channel {} Popped at dram cycle {}", cid, _cycles);
}

uint32_t PIM::get_channel_id(MemoryAccess *access) {
    // spdlog::info("pim get_channel_id()");
    return _mem->GetChannel(access->dram_address);
}

void PIM::print_stat() {
    // spdlog::info("pim print_stat()");
    uint64_t total_reqs = 0;
    for (int ch = 0; ch < _config.dram_channels; ch++) {
        const uint64_t channel_requests =
            _total_processed_requests[ch] + _processed_requests[ch];
        float util = ((float)channel_requests * _burst_cycle) / _cycles * 100;
        spdlog::info("DRAM CH[{}]: AVG BW Util {:.2f}%", ch, util);
        total_reqs += channel_requests;
    }
    float util = ((float)total_reqs * _burst_cycle / _config.dram_channels) / _cycles * 100;
    spdlog::info("DRAM: AVG BW Util {:.2f}%", util);
    spdlog::info("DRAM total cycles: {}", _cycles);
    spdlog::info("DRAM total processed memory requests: {}", _mem_req_cnt);
    spdlog::info(
        "DRAM estimated logical requests read/write/pim {}/{}/{}, bytes {}/{}",
        _estimated_workload.memory_reads, _estimated_workload.memory_writes,
        _estimated_workload.pim_pheader + _estimated_workload.pim_gwrite +
            _estimated_workload.pim_comp + _estimated_workload.pim_readres,
        _estimated_workload.memory_read_bytes,
        _estimated_workload.memory_write_bytes);

    std::ofstream command_out(
        Config::system_config.log_dir + "/proportional_command_compensation.json",
        std::ofstream::out);
    command_out << "{";
    static const std::array<const char*, 21> command_counters = {
        "num_read_cmds", "num_write_cmds", "num_act_cmds", "num_pre_cmds",
        "num_write_buf_hits", "num_read_row_hits", "num_write_row_hits",
        "num_ondemand_pres",
        "num_pheader_cmds", "num_gwrite_cmds", "num_comp_cmds",
        "num_readres_cmds", "num_pim_cmds", "num_pim_activate_cmds",
        "num_pim_precharge_cmds", "pim_cycles", "num_cycles", "num_ref_cmds",
        "num_refb_cmds", "num_reads_done", "num_writes_done"};
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        if (channel != 0) command_out << ",";
        command_out << "\"" << channel << "\":{";
        bool first = true;
        for (const char* counter : command_counters) {
            if (!first) command_out << ",";
            first = false;
            const uint64_t logical = _mem->GetCounter(channel, counter);
            const uint64_t estimated = _estimated_command_counts[channel][counter];
            command_out << "\"" << counter << "\":" << logical - estimated
                        << ",\"estimated_" << counter << "\":" << estimated
                        << ",\"logical_" << counter << "\":" << logical;
        }
        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            const uint64_t logical_active = _mem->GetVecCounter(
                channel, "pim_rank_active_cycles", rank);
            const uint64_t logical_idle = _mem->GetVecCounter(
                channel, "pim_all_bank_idle_cycles", rank);
            const uint64_t estimated_active =
                _estimated_pim_active_cycles[channel][rank];
            const uint64_t estimated_idle =
                _estimated_pim_idle_cycles[channel][rank];
            command_out
                << ",\"pim_rank_active_cycles[" << rank << "]\":"
                << logical_active - estimated_active
                << ",\"estimated_pim_rank_active_cycles[" << rank << "]\":"
                << estimated_active
                << ",\"logical_pim_rank_active_cycles[" << rank << "]\":"
                << logical_active
                << ",\"pim_all_bank_idle_cycles[" << rank << "]\":"
                << logical_idle - estimated_idle
                << ",\"estimated_pim_all_bank_idle_cycles[" << rank << "]\":"
                << estimated_idle
                << ",\"logical_pim_all_bank_idle_cycles[" << rank << "]\":"
                << logical_idle;
        }
        command_out << "}";
    }
    command_out << "}";
    command_out.close();

    _mem->PrintStats();

    // Calculate and print Area Overhead
    int num_channels = _config.mem_config.channels;
    int num_ranks = _config.mem_config.ranks;
    int num_banks = _config.mem_config.banks;

    int pu_num = _config.mem_config.PU_num;
    int total_pus = num_channels * pu_num;

    double pim_pu_area = _config.mem_config.pim_pu_area;
    double pim_ctrl_area = _config.mem_config.pim_controller_area_overhead;
    double buffer_area_per_kb = _config.mem_config.pim_buffer_area_per_kb;

    // Buffer size in KB per bank
    double total_buffer_size_kb = (num_channels * _config.mem_config.input_buffer_size +
        num_channels * _config.mem_config.output_buffer_size * pu_num) / 1024.0;

    double total_pu_area = total_pus * pim_pu_area;
    double total_ctrl_area = total_pus * pim_ctrl_area;
    double total_buffer_area = total_buffer_size_kb * buffer_area_per_kb;

    // Hybrid Bonding Area Calculation
    // Total Bandwidth = PU_per_channel * DQ * (2.0 / tCK_ns) × channel
    double bus_width_bytes = _config.mem_config.bus_width / 8.0;
    double tCK_ns = _config.mem_config.tCK;
    double bandwidth_per_channel_gbs = bus_width_bytes * (2.0 / tCK_ns);
    double total_bandwidth_gbs = bandwidth_per_channel_gbs * num_channels;

    double total_hybrid_bonding_area = total_bandwidth_gbs * _config.mem_config.hybrid_bonding_bw_area_ratio;
    double total_area = total_pu_area + total_ctrl_area + total_buffer_area + total_hybrid_bonding_area;

    spdlog::info("PIM Area Overhead Analysis:");
    spdlog::info("PIM Structure: {} channel, {} pus per channel, with {}KB global input buffer and {}KB output buffer",
        num_channels, pu_num, _config.mem_config.input_buffer_size/1024.0, _config.mem_config.output_buffer_size/1024.0);
    spdlog::info("Total PIM Units: {}", total_pus);
    spdlog::info("Total PU Area: {:.4f} mm2", total_pu_area);
    spdlog::info("Total Controller Area: {:.4f} mm2", total_ctrl_area);
    spdlog::info("Total Buffer Area: {:.4f} mm2", total_buffer_area);
    spdlog::info("Total Hybrid Bonding Area: {:.4f} mm2", total_hybrid_bonding_area);
    spdlog::info("Total Area Overhead: {:.4f} mm2", total_area);
}

void PIM::log(Stage stage) {
    std::string fname = Config::system_config.log_dir + "/mem_io_" + stageToString(stage) + "_ch_";
    for (size_t i = 0; i < _stats.size(); ++i) {
        Logger::log(_stats[i], fname + std::to_string(i));
        auto last_stat = _stats[i].back();
        _stats[i].clear();
        _stats[i].push_back(last_stat);
    }
}

double PIM::get_avg_bw_util() {
    const double avg_bw_util =
        _stage_cycles == 0 || _config.dram_channels == 0
            ? 0.0
            : ((double)_total_done_requests * _burst_cycle /
               _config.dram_channels) /
                  _stage_cycles * 100;

    // reset
    _total_done_requests = 0;
    _stage_cycles = 0;
    return avg_bw_util;
}

uint64_t PIM::get_avg_pim_cycle() {
    const uint64_t avg_pim_cycle =
        _mem->GetAvgPIMCycles() +
        (_config.dram_channels == 0
             ? 0
             : _estimated_stage_pim_cycle_sum / _config.dram_channels);
    reset_pim_cycle();
    return avg_pim_cycle;
}

void PIM::reset_pim_cycle() {
    _mem->ResetPIMCycle();
    _estimated_stage_pim_cycle_sum = 0;
}

void PIM::receive_predicting_config(size_t unstable_length, size_t sample_length, double inst_ratio) {
    _mem->receive_predicting_config(unstable_length, sample_length, inst_ratio);
}

void PIM::set_dram_cycles(uint64_t cycles) {
    _cycles += cycles;
    // The caller supplies a skipped interval, not an absolute timestamp.
    // Keep the stage-local denominator consistent with cycle(), especially
    // after get_avg_bw_util() resets it at a Prefill/Decode boundary.
    _stage_cycles += cycles;
}

uint64_t PIM::get_core_command_counter(
    uint32_t core_id, const std::string& name) const {
    return _mem->GetCoreCommandCounter(core_id, name);
}

void PIM::begin_proportional_command_sampling() {
    static const std::array<const char*, 19> counters = {
        "num_read_cmds", "num_write_cmds", "num_act_cmds", "num_pre_cmds",
        "num_write_buf_hits", "num_read_row_hits", "num_write_row_hits",
        "num_ondemand_pres",
        "num_pheader_cmds", "num_gwrite_cmds", "num_comp_cmds",
        "num_readres_cmds", "num_pim_cmds", "num_pim_activate_cmds",
        "num_pim_precharge_cmds", "pim_cycles", "num_cycles", "num_ref_cmds",
        "num_refb_cmds"};
    _command_has_warmup_sample = false;
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        _command_warmup_request_counts[channel] = 0;
        _command_warmup_read_request_counts[channel] = 0;
        _command_warmup_write_request_counts[channel] = 0;
        _command_warmup_counter_counts[channel].clear();
        _command_request_baseline[channel] = _actual_request_counts[channel];
        _command_read_request_baseline[channel] =
            _actual_read_request_counts[channel];
        _command_write_request_baseline[channel] =
            _actual_write_request_counts[channel];
        auto& baseline = _command_counter_baseline[channel];
        baseline.clear();
        for (const char* counter : counters) {
            baseline[counter] = _mem->GetCounter(channel, counter);
        }
        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            _active_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(channel, "rank_active_cycles", rank);
            _idle_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(channel, "all_bank_idle_cycles", rank);
            _sref_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(channel, "sref_cycles", rank);
            _pim_active_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(
                    channel, "pim_rank_active_cycles", rank);
            _pim_idle_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(
                    channel, "pim_all_bank_idle_cycles", rank);
        }
    }
}

void PIM::begin_decode_pruning_state_sample(
    const std::string& operation) {
    static const std::array<const char*, 14> counters = {
        "num_act_cmds", "num_pre_cmds", "num_write_buf_hits",
        "num_read_row_hits", "num_write_row_hits", "num_ondemand_pres",
        "num_pim_activate_cmds", "num_pim_precharge_cmds",
        "pim_cycles", "num_ref_cmds", "num_refb_cmds", "num_cycles",
        "num_write_requests", "num_write_cmds"};
    _decode_pruning_state_operation = operation;
    auto& baseline = _decode_pruning_state_baseline;
    baseline = DecodePruningDramState{};
    baseline.counters.resize(_config.dram_channels);
    baseline.rank_active_cycles.resize(_config.dram_channels);
    baseline.sref_cycles.resize(_config.dram_channels);
    baseline.pim_rank_active_cycles.resize(_config.dram_channels);
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        for (const char* counter : counters) {
            baseline.counters[channel][counter] =
                _mem->GetCounter(channel, counter);
        }
        baseline.rank_active_cycles[channel].resize(
            _config.mem_config.ranks, 0);
        baseline.sref_cycles[channel].resize(
            _config.mem_config.ranks, 0);
        baseline.pim_rank_active_cycles[channel].resize(
            _config.mem_config.ranks, 0);
        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            baseline.rank_active_cycles[channel][rank] =
                _mem->GetVecCounter(
                    channel, "rank_active_cycles", rank);
            baseline.sref_cycles[channel][rank] =
                _mem->GetVecCounter(channel, "sref_cycles", rank);
            baseline.pim_rank_active_cycles[channel][rank] =
                _mem->GetVecCounter(
                    channel, "pim_rank_active_cycles", rank);
        }
    }
}

void PIM::finish_decode_pruning_state_sample(
    const std::string& operation, uint32_t required_samples) {
    assert(operation == _decode_pruning_state_operation);
    assert(required_samples > 0);
    auto& accumulator =
        _decode_pruning_state_accumulators[operation];
    if (accumulator.counters.empty()) {
        accumulator.counters.resize(_config.dram_channels);
        accumulator.rank_active_cycles.resize(_config.dram_channels);
        accumulator.sref_cycles.resize(_config.dram_channels);
        accumulator.pim_rank_active_cycles.resize(_config.dram_channels);
        for (uint32_t channel = 0; channel < _config.dram_channels;
             ++channel) {
            accumulator.rank_active_cycles[channel].resize(
                _config.mem_config.ranks, 0);
            accumulator.sref_cycles[channel].resize(
                _config.mem_config.ranks, 0);
            accumulator.pim_rank_active_cycles[channel].resize(
                _config.mem_config.ranks, 0);
        }
    }
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        for (const auto& [counter, baseline] :
             _decode_pruning_state_baseline.counters[channel]) {
            const uint64_t current =
                _mem->GetCounter(channel, counter);
            accumulator.counters[channel][counter] +=
                current >= baseline ? current - baseline : 0;
        }
        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            const auto add_delta = [](uint64_t& sum, uint64_t current,
                                      uint64_t baseline) {
                sum += current >= baseline ? current - baseline : 0;
            };
            add_delta(
                accumulator.rank_active_cycles[channel][rank],
                _mem->GetVecCounter(
                    channel, "rank_active_cycles", rank),
                _decode_pruning_state_baseline
                    .rank_active_cycles[channel][rank]);
            add_delta(
                accumulator.sref_cycles[channel][rank],
                _mem->GetVecCounter(channel, "sref_cycles", rank),
                _decode_pruning_state_baseline.sref_cycles[channel][rank]);
            add_delta(
                accumulator.pim_rank_active_cycles[channel][rank],
                _mem->GetVecCounter(
                    channel, "pim_rank_active_cycles", rank),
                _decode_pruning_state_baseline
                    .pim_rank_active_cycles[channel][rank]);
        }
    }
    ++accumulator.samples;
    if (accumulator.samples < required_samples) return;

    DecodePruningDramState result = accumulator;
    const auto average = [&accumulator](uint64_t value) {
        return static_cast<uint64_t>(std::llround(
            static_cast<long double>(value) / accumulator.samples));
    };
    result.samples = accumulator.samples;
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        for (auto& [counter, value] : result.counters[channel]) {
            value = average(value);
        }
        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            result.rank_active_cycles[channel][rank] =
                average(result.rank_active_cycles[channel][rank]);
            result.sref_cycles[channel][rank] =
                average(result.sref_cycles[channel][rank]);
            result.pim_rank_active_cycles[channel][rank] =
                average(result.pim_rank_active_cycles[channel][rank]);
        }
    }
    _decode_pruning_state_templates[operation] = std::move(result);
}

std::vector<uint64_t> PIM::decode_pruning_sampled_write_requests(
    const std::string& operation) const {
    std::vector<uint64_t> writes(_config.dram_channels, 0);
    const auto template_it =
        _decode_pruning_state_templates.find(operation);
    if (template_it != _decode_pruning_state_templates.end()) {
        for (uint32_t channel = 0;
             channel < _config.dram_channels; ++channel) {
            writes[channel] =
                template_it->second.counters[channel].at(
                    "num_write_requests");
        }
        return writes;
    }
    assert(operation == _decode_pruning_state_operation);
    for (uint32_t channel = 0;
         channel < _config.dram_channels; ++channel) {
        const uint64_t current =
            _mem->GetCounter(channel, "num_write_requests");
        const uint64_t baseline =
            _decode_pruning_state_baseline.counters[channel].at(
                "num_write_requests");
        writes[channel] =
            current >= baseline ? current - baseline : 0;
    }
    return writes;
}

std::vector<uint64_t> PIM::decode_pruning_sampled_write_commands(
    const std::string& operation) const {
    std::vector<uint64_t> commands(_config.dram_channels, 0);
    const auto template_it =
        _decode_pruning_state_templates.find(operation);
    if (template_it != _decode_pruning_state_templates.end()) {
        for (uint32_t channel = 0;
             channel < _config.dram_channels; ++channel) {
            commands[channel] =
                template_it->second.counters[channel].at(
                    "num_write_cmds");
        }
        return commands;
    }
    assert(operation == _decode_pruning_state_operation);
    for (uint32_t channel = 0;
         channel < _config.dram_channels; ++channel) {
        const uint64_t current =
            _mem->GetCounter(channel, "num_write_cmds");
        const uint64_t baseline =
            _decode_pruning_state_baseline.counters[channel].at(
                "num_write_cmds");
        commands[channel] =
            current >= baseline ? current - baseline : 0;
    }
    return commands;
}

void PIM::apply_decode_pruning_state(
    const std::string& operation, cycle_type skipped_dram_cycles) {
    const auto template_it =
        _decode_pruning_state_templates.find(operation);
    assert(template_it != _decode_pruning_state_templates.end());
    const auto& state = template_it->second;
    const auto scale = [skipped_dram_cycles](
                           uint64_t value,
                           uint64_t sampled_cycles) -> uint64_t {
        if (value == 0 || sampled_cycles == 0 ||
            skipped_dram_cycles == 0) {
            return 0;
        }
        return static_cast<uint64_t>(std::llround(
            static_cast<long double>(value) * skipped_dram_cycles /
            sampled_cycles));
    };
    static const std::array<const char*, 10> replay_counters = {
        "num_act_cmds", "num_pre_cmds", "num_write_buf_hits",
        "num_read_row_hits", "num_write_row_hits", "num_ondemand_pres",
        "num_pim_activate_cmds", "num_pim_precharge_cmds",
        "num_ref_cmds", "num_refb_cmds"};
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        const auto& counters = state.counters[channel];
        const uint64_t sampled_cycles = counters.at("num_cycles");
        for (const char* counter : replay_counters) {
            const uint64_t value = counters.at(counter);
            _mem->AddCounter(channel, counter, value);
            _estimated_command_counts[channel][counter] += value;
        }
        const uint64_t estimated_pim_cycles =
            scale(counters.at("pim_cycles"), sampled_cycles);
        _mem->AddCounter(channel, "pim_cycles", estimated_pim_cycles);
        _mem->AddCounter(channel, "num_cycles", skipped_dram_cycles);
        _estimated_command_counts[channel]["pim_cycles"] +=
            estimated_pim_cycles;
        _estimated_command_counts[channel]["num_cycles"] +=
            skipped_dram_cycles;
        _estimated_stage_pim_cycle_sum += estimated_pim_cycles;

        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            uint64_t active = scale(
                state.rank_active_cycles[channel][rank], sampled_cycles);
            uint64_t sref = scale(
                state.sref_cycles[channel][rank], sampled_cycles);
            active = std::min<uint64_t>(active, skipped_dram_cycles);
            sref = std::min<uint64_t>(
                sref, skipped_dram_cycles - active);
            const uint64_t idle =
                skipped_dram_cycles - active - sref;
            _mem->AddVecCounter(
                channel, "rank_active_cycles", rank, active);
            _mem->AddVecCounter(
                channel, "all_bank_idle_cycles", rank, idle);
            _mem->AddVecCounter(channel, "sref_cycles", rank, sref);

            uint64_t pim_active = scale(
                state.pim_rank_active_cycles[channel][rank],
                sampled_cycles);
            pim_active = std::min<uint64_t>(
                pim_active, skipped_dram_cycles);
            const uint64_t pim_idle =
                skipped_dram_cycles - pim_active;
            _mem->AddVecCounter(
                channel, "pim_rank_active_cycles", rank, pim_active);
            _mem->AddVecCounter(
                channel, "pim_all_bank_idle_cycles", rank, pim_idle);
            _estimated_pim_active_cycles[channel][rank] += pim_active;
            _estimated_pim_idle_cycles[channel][rank] += pim_idle;
        }
        spdlog::info(
            "Decode Pruning DRAM CH[{}] state replay for {}: "
            "ACT/PRE {}/{}, PIM cycles {}, elapsed {}",
            channel, operation, counters.at("num_act_cmds"),
            counters.at("num_pre_cmds"), estimated_pim_cycles,
            skipped_dram_cycles);
    }
}

void PIM::mark_proportional_command_warmup_complete(double warmup_weight) {
    static const std::array<const char*, 19> counters = {
        "num_read_cmds", "num_write_cmds", "num_act_cmds", "num_pre_cmds",
        "num_write_buf_hits", "num_read_row_hits", "num_write_row_hits",
        "num_ondemand_pres",
        "num_pheader_cmds", "num_gwrite_cmds", "num_comp_cmds",
        "num_readres_cmds", "num_pim_cmds", "num_pim_activate_cmds",
        "num_pim_precharge_cmds", "pim_cycles", "num_cycles", "num_ref_cmds",
        "num_refb_cmds"};
    assert(warmup_weight >= 0.0 && warmup_weight <= 1.0);
    _command_warmup_weight = warmup_weight;
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        _command_warmup_request_counts[channel] =
            _actual_request_counts[channel] - _command_request_baseline[channel];
        _command_warmup_read_request_counts[channel] =
            _actual_read_request_counts[channel] -
            _command_read_request_baseline[channel];
        _command_warmup_write_request_counts[channel] =
            _actual_write_request_counts[channel] -
            _command_write_request_baseline[channel];
        auto& warmup = _command_warmup_counter_counts[channel];
        for (const char* counter : counters) {
            const uint64_t current = _mem->GetCounter(channel, counter);
            warmup[counter] = current - _command_counter_baseline[channel][counter];
            _command_counter_baseline[channel][counter] = current;
        }
        _command_request_baseline[channel] = _actual_request_counts[channel];
        _command_read_request_baseline[channel] =
            _actual_read_request_counts[channel];
        _command_write_request_baseline[channel] =
            _actual_write_request_counts[channel];
        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            _active_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(channel, "rank_active_cycles", rank);
            _idle_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(channel, "all_bank_idle_cycles", rank);
            _sref_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(channel, "sref_cycles", rank);
            _pim_active_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(
                    channel, "pim_rank_active_cycles", rank);
            _pim_idle_cycle_baseline[channel][rank] =
                _mem->GetVecCounter(
                    channel, "pim_all_bank_idle_cycles", rank);
        }
    }
    _command_has_warmup_sample = true;
    spdlog::info(
        "Marked proportional DRAM command warmup window with warmup/sample weights {:.3f}/{:.3f}",
        _command_warmup_weight, 1.0 - _command_warmup_weight);
}

void PIM::apply_estimated_time(cycle_type skipped_dram_cycles) {
    if (skipped_dram_cycles == 0) return;
    const auto scale = [skipped_dram_cycles](uint64_t measured,
                                             uint64_t sampled) -> uint64_t {
        if (measured == 0 || sampled == 0) return 0;
        return static_cast<uint64_t>(std::llround(
            static_cast<long double>(measured) * skipped_dram_cycles /
            sampled));
    };
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        const uint64_t sampled_cycles =
            _mem->GetCounter(channel, "num_cycles") -
            _command_counter_baseline[channel]["num_cycles"];
        const uint64_t estimated_ref = scale(
            _mem->GetCounter(channel, "num_ref_cmds") -
                _command_counter_baseline[channel]["num_ref_cmds"],
            sampled_cycles);
        const uint64_t estimated_refb = scale(
            _mem->GetCounter(channel, "num_refb_cmds") -
                _command_counter_baseline[channel]["num_refb_cmds"],
            sampled_cycles);
        const uint64_t estimated_pim_cycles = scale(
            _mem->GetCounter(channel, "pim_cycles") -
                _command_counter_baseline[channel]["pim_cycles"],
            sampled_cycles);
        _mem->AddCounter(channel, "num_cycles", skipped_dram_cycles);
        _mem->AddCounter(channel, "num_ref_cmds", estimated_ref);
        _mem->AddCounter(channel, "num_refb_cmds", estimated_refb);
        _mem->AddCounter(channel, "pim_cycles", estimated_pim_cycles);
        _estimated_command_counts[channel]["num_cycles"] += skipped_dram_cycles;
        _estimated_command_counts[channel]["num_ref_cmds"] += estimated_ref;
        _estimated_command_counts[channel]["num_refb_cmds"] += estimated_refb;
        _estimated_command_counts[channel]["pim_cycles"] +=
            estimated_pim_cycles;
        _estimated_stage_pim_cycle_sum += estimated_pim_cycles;

        for (int rank = 0; rank < _config.mem_config.ranks; ++rank) {
            const uint64_t sampled_active =
                _mem->GetVecCounter(channel, "rank_active_cycles", rank) -
                _active_cycle_baseline[channel][rank];
            const uint64_t sampled_sref =
                _mem->GetVecCounter(channel, "sref_cycles", rank) -
                _sref_cycle_baseline[channel][rank];
            uint64_t estimated_active = scale(sampled_active, sampled_cycles);
            uint64_t estimated_sref = scale(sampled_sref, sampled_cycles);
            estimated_active = std::min<uint64_t>(estimated_active,
                                                  skipped_dram_cycles);
            estimated_sref = std::min<uint64_t>(
                estimated_sref, skipped_dram_cycles - estimated_active);
            const uint64_t estimated_idle =
                skipped_dram_cycles - estimated_active - estimated_sref;
            _mem->AddVecCounter(channel, "rank_active_cycles", rank,
                                estimated_active);
            _mem->AddVecCounter(channel, "all_bank_idle_cycles", rank,
                                estimated_idle);
            _mem->AddVecCounter(channel, "sref_cycles", rank,
                                estimated_sref);

            const uint64_t sampled_pim_active =
                _mem->GetVecCounter(
                    channel, "pim_rank_active_cycles", rank) -
                _pim_active_cycle_baseline[channel][rank];
            uint64_t estimated_pim_active =
                scale(sampled_pim_active, sampled_cycles);
            estimated_pim_active = std::min<uint64_t>(
                estimated_pim_active, skipped_dram_cycles);
            const uint64_t estimated_pim_idle =
                skipped_dram_cycles - estimated_pim_active;
            _mem->AddVecCounter(channel, "pim_rank_active_cycles", rank,
                                estimated_pim_active);
            _mem->AddVecCounter(
                channel, "pim_all_bank_idle_cycles", rank,
                estimated_pim_idle);
            _estimated_pim_active_cycles[channel][rank] +=
                estimated_pim_active;
            _estimated_pim_idle_cycles[channel][rank] +=
                estimated_pim_idle;
        }
        spdlog::info(
            "DRAM CH[{}] time compensation: sampled cycles {}, skipped cycles {}, estimated REF/REFB {}/{}, PIM cycles {}",
            channel, sampled_cycles, skipped_dram_cycles, estimated_ref,
            estimated_refb, estimated_pim_cycles);
    }
}

void PIM::apply_estimated_workload(
    const ProportionalWorkloadStat& workload,
    const std::vector<uint64_t>* write_command_override) {
    _estimated_workload += workload;
    uint64_t total_requests = 0;
    for (uint32_t channel = 0; channel < _config.dram_channels; ++channel) {
        const auto channel_value = [channel](const std::vector<uint64_t>& values) {
            return channel < values.size() ? values[channel] : 0ULL;
        };
        const uint64_t reads = channel_value(workload.channel_memory_reads);
        const uint64_t writes = channel_value(workload.channel_memory_writes);
        const uint64_t pheaders = channel_value(workload.channel_pim_pheader);
        const uint64_t gwrites = channel_value(workload.channel_pim_gwrite);
        const uint64_t comps = channel_value(workload.channel_pim_comp);
        const uint64_t readres = channel_value(workload.channel_pim_readres);
        const uint64_t channel_requests =
            reads + writes + pheaders + gwrites + comps + readres;

        const uint64_t measured_requests =
            _actual_request_counts[channel] -
            _command_request_baseline[channel];
        const uint64_t measured_read_requests =
            _actual_read_request_counts[channel] -
            _command_read_request_baseline[channel];
        const uint64_t measured_write_requests =
            _actual_write_request_counts[channel] -
            _command_write_request_baseline[channel];
        const auto measured_counter = [&](const char* counter) -> uint64_t {
            return _mem->GetCounter(channel, counter) -
                   _command_counter_baseline[channel][counter];
        };
        const auto estimate = [&](const char* counter,
                                  uint64_t skipped_requests,
                                  uint64_t sampled_requests) -> uint64_t {
            if (skipped_requests == 0 || sampled_requests == 0) return 0;
            const uint64_t measured = measured_counter(counter);
            return static_cast<uint64_t>(std::llround(
                static_cast<long double>(measured) * skipped_requests /
                sampled_requests));
        };
        const auto weighted_estimate = [&](const char* counter,
                                           uint64_t skipped_requests,
                                           uint64_t warmup_requests,
                                           uint64_t sample_requests) -> uint64_t {
            if (!_command_has_warmup_sample) {
                return estimate(counter, skipped_requests, sample_requests);
            }
            const uint64_t warmup_commands =
                _command_warmup_counter_counts[channel][counter];
            const uint64_t sample_commands = measured_counter(counter);
            long double weighted_rate = 0.0L;
            long double available_weight = 0.0L;
            if (warmup_requests > 0 && _command_warmup_weight > 0.0) {
                weighted_rate += static_cast<long double>(_command_warmup_weight) *
                    warmup_commands / warmup_requests;
                available_weight += _command_warmup_weight;
            }
            const double sample_weight = 1.0 - _command_warmup_weight;
            if (sample_requests > 0 && sample_weight > 0.0) {
                weighted_rate += static_cast<long double>(sample_weight) *
                    sample_commands / sample_requests;
                available_weight += sample_weight;
            }
            if (skipped_requests == 0 || available_weight == 0.0L) return 0;
            return static_cast<uint64_t>(std::llround(
                weighted_rate * skipped_requests / available_weight));
        };
        const auto combined_estimate = [&](const char* counter,
                                           uint64_t skipped_requests,
                                           uint64_t warmup_requests,
                                           uint64_t sample_requests) -> uint64_t {
            const uint64_t combined_requests =
                warmup_requests + sample_requests;
            if (skipped_requests == 0 || combined_requests == 0) return 0;
            const uint64_t combined_commands =
                _command_warmup_counter_counts[channel][counter] +
                measured_counter(counter);
            return static_cast<uint64_t>(std::llround(
                static_cast<long double>(combined_commands) *
                skipped_requests / combined_requests));
        };
        const uint64_t sampled_reads =
            _mem->GetCounter(channel, "num_read_cmds") -
            _command_counter_baseline[channel]["num_read_cmds"];
        const uint64_t sampled_writes =
            _mem->GetCounter(channel, "num_write_cmds") -
            _command_counter_baseline[channel]["num_write_cmds"];
        const uint64_t sampled_pheaders =
            _mem->GetCounter(channel, "num_pheader_cmds") -
            _command_counter_baseline[channel]["num_pheader_cmds"];
        const uint64_t sampled_gwrites =
            _mem->GetCounter(channel, "num_gwrite_cmds") -
            _command_counter_baseline[channel]["num_gwrite_cmds"];
        const uint64_t sampled_comps =
            _mem->GetCounter(channel, "num_comp_cmds") -
            _command_counter_baseline[channel]["num_comp_cmds"];
        const uint64_t sampled_readres =
            _mem->GetCounter(channel, "num_readres_cmds") -
            _command_counter_baseline[channel]["num_readres_cmds"];
        uint64_t estimated_read_cmds = _command_has_warmup_sample
            ? combined_estimate(
                  "num_read_cmds", reads,
                  _command_warmup_read_request_counts[channel],
                  measured_read_requests)
            : estimate("num_read_cmds", reads, sampled_reads);
        uint64_t estimated_write_cmds = _command_has_warmup_sample
            ? combined_estimate(
                  "num_write_cmds", writes,
                  _command_warmup_write_request_counts[channel],
                  measured_write_requests)
            : estimate("num_write_cmds", writes, sampled_writes);
        if (write_command_override != nullptr &&
            channel < write_command_override->size()) {
            estimated_write_cmds =
                write_command_override->at(channel);
        }
        // Decode Pruning skips a whole operation, so there may be no
        // within-operation command sample. One logical MOVIN/MOVOUT or PIM
        // request still maps to one corresponding command; use that exact
        // identity for the primary command counters and leave row-state
        // counters to the conservative sampled estimator below.
        if (reads > 0 && sampled_reads == 0) {
            estimated_read_cmds = reads;
        }
        if (write_command_override == nullptr &&
            writes > 0 && sampled_writes == 0) {
            estimated_write_cmds = writes;
        }
        const uint64_t estimated_write_buf_hits =
            _command_has_warmup_sample
                ? combined_estimate(
                      "num_write_buf_hits", writes,
                      _command_warmup_write_request_counts[channel],
                      measured_write_requests)
                : weighted_estimate(
                      "num_write_buf_hits", writes,
                      _command_warmup_write_request_counts[channel],
                      measured_write_requests);
        const uint64_t estimated_write_row_hits =
            _command_has_warmup_sample
                ? combined_estimate(
                      "num_write_row_hits", writes,
                      _command_warmup_write_request_counts[channel],
                      measured_write_requests)
                : weighted_estimate(
                      "num_write_row_hits", writes,
                      _command_warmup_write_request_counts[channel],
                      measured_write_requests);
        const std::array<std::pair<const char*, uint64_t>, 17> additions = {{
            {"num_read_cmds", estimated_read_cmds},
            {"num_write_cmds", estimated_write_cmds},
            {"num_act_cmds", weighted_estimate(
                "num_act_cmds", channel_requests,
                _command_warmup_request_counts[channel], measured_requests)},
            {"num_pre_cmds", weighted_estimate(
                "num_pre_cmds", channel_requests,
                _command_warmup_request_counts[channel], measured_requests)},
            {"num_write_buf_hits", estimated_write_buf_hits},
            {"num_read_row_hits", weighted_estimate(
                "num_read_row_hits", reads,
                _command_warmup_read_request_counts[channel],
                measured_read_requests)},
            {"num_write_row_hits", estimated_write_row_hits},
            {"num_ondemand_pres", weighted_estimate(
                "num_ondemand_pres", channel_requests,
                _command_warmup_request_counts[channel], measured_requests)},
            {"num_pheader_cmds", sampled_pheaders == 0 ? pheaders :
                estimate("num_pheader_cmds", pheaders, sampled_pheaders)},
            {"num_gwrite_cmds", sampled_gwrites == 0 ? gwrites :
                estimate("num_gwrite_cmds", gwrites, sampled_gwrites)},
            {"num_comp_cmds", sampled_comps == 0 ? comps :
                estimate("num_comp_cmds", comps, sampled_comps)},
            {"num_readres_cmds", sampled_readres == 0 ? readres :
                estimate("num_readres_cmds", readres, sampled_readres)},
            {"num_pim_cmds",
             (sampled_pheaders + sampled_gwrites + sampled_comps +
              sampled_readres) == 0
                 ? pheaders + gwrites + comps + readres
                 : estimate(
                       "num_pim_cmds",
                       pheaders + gwrites + comps + readres,
                       sampled_pheaders + sampled_gwrites + sampled_comps +
                           sampled_readres)},
            {"num_pim_activate_cmds", estimate("num_pim_activate_cmds", channel_requests, measured_requests)},
            {"num_pim_precharge_cmds", estimate("num_pim_precharge_cmds", channel_requests, measured_requests)},
            // Done counters represent completed logical requests. Every
            // pruned MOVIN/MOVOUT would contribute exactly one completion.
            {"num_reads_done", reads},
            {"num_writes_done", writes}}};
        for (const auto& [counter, value] : additions) {
            _mem->AddCounter(channel, counter, value);
            _estimated_command_counts[channel][counter] += value;
        }
        spdlog::info(
            "DRAM CH[{}] command compensation: warmup/sample requests {}/{}, weights {:.3f}/{:.3f}, skipped requests {}, estimated read/write/act/pre {}/{}/{}/{}",
            channel, _command_warmup_request_counts[channel], measured_requests,
            _command_has_warmup_sample ? _command_warmup_weight : 0.0,
            _command_has_warmup_sample ? 1.0 - _command_warmup_weight : 1.0,
            channel_requests,
            additions[0].second, additions[1].second,
            additions[2].second, additions[3].second);

        _stats[channel].back().memory_reads +=
            reads * MyAddressAllocator::dram_burst_size;
        _stats[channel].back().memory_writes +=
            writes * MyAddressAllocator::dram_burst_size;
        _stats[channel].back().pim_reads +=
            readres * MyAddressAllocator::dram_burst_size;
        _stats[channel].back().pim_writes +=
            gwrites * MyAddressAllocator::dram_burst_size;
        _stats[channel].back().pim_comps +=
            comps * MyAddressAllocator::dram_burst_size;

        _processed_requests[channel] += channel_requests;
        total_requests += channel_requests;
    }
    _mem_req_cnt += total_requests;
    _total_done_requests += total_requests;
}

// <<< gsheo
