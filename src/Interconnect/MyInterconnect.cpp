
#include "MyInterconnect.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>

#include "../../ext/booksim/include/booksim2/Interconnect.hpp"

/*
    bool running();
    void cycle();
    void push(uint32_t src, uint32_t dest, MemoryAccess *request);
    bool is_full(uint32_t src, MemoryAccess *request);
    bool is_empty(uint32_t nid);
    MemoryAccess *top(uint32_t nid);
    void pop(uint32_t nid);
    void print_stats();
 */


MyInterconnect::MyInterconnect(const SysConfig& config) : _config(config) {
    spdlog::info("Initialize My Interconnect");

    // period information (us) =  1 / MHZ
    _icnt_period = 1.0 / static_cast<double>(_config.icnt_freq);
    _dram_period = 1.0 / static_cast<double>(_config.dram_freq);
    _icnt_freq = _config.icnt_freq;
    _dram_freq = _config.dram_freq;
    _dram_time = 0.0;
    _icnt_time = 0.0;

    _latency = config.icnt_latency;
    _n_nodes = config.num_cores * config.dram_channels + config.dram_channels;

    _n_cores = config.num_cores;
    _n_memories = config.dram_channels;

    memory_offset = config.num_cores * config.dram_channels;

    _in_buffers.resize(_n_nodes);
    _out_buffers.resize(_n_nodes);
    _busy_node.resize(_n_nodes);

    for(int node = 0; node < _n_nodes; node++) {
        _busy_node[node] = false;
    }

    // TODO: make it configurable
    _mem_cycle_interval = 500;
    _stats.resize(config.dram_channels);
    _measured_traffic.resize(config.dram_channels);
    _estimated_traffic.resize(config.dram_channels);
    for (size_t i = 0; i < config.dram_channels; ++i) {
        _stats[i].emplace_back(0, i, _mem_cycle_interval);
    }

    _dram_push_valid.assign(config.dram_channels, true);
    _dram_pop_valid.assign(config.dram_channels, true);

    // 设置所同步的 DRAM tick
    _current_dram_tick_sync_by_icnt = 0;
    _next_dram_tick_sync_by_icnt = 1;
}

bool MyInterconnect::running() {
    return false;
}

void MyInterconnect::cycle() {
    // Send request from Core to DRAM, and get DRAM response back to Core
    for (int node = 0; node < _n_nodes; node++) {
        int src_node = (_rr_start + node) % _n_nodes;
        if (!_in_buffers[src_node].empty() && _in_buffers[src_node].front().finish_cycle <= _cycles) {
            uint32_t dest = _in_buffers[src_node].front().dest;
            if (!_busy_node[dest]) {
                _out_buffers[dest].push(_in_buffers[src_node].front().access);
                _busy_node[dest] = true;
                // spdlog::info("Interconnect transport a memory access from source: {} to dest: {} at ICNT cycle {}",src_node, dest, _cycles);
                _in_buffers[src_node].pop();
            }
        }
    }

    // Record the work states of interconnect
    for (auto ch_idx = 0; ch_idx < _config.dram_channels; ++ch_idx) {
        if (_stats[ch_idx].back().start_cycle + _mem_cycle_interval < get_core_cycle()) {
            auto stat = MemoryIOStat((get_core_cycle() / _mem_cycle_interval) * _mem_cycle_interval, ch_idx, _mem_cycle_interval);
            _stats[ch_idx].push_back(stat);
        }
    }

    // set the busy node flag for each port
    for(int node = 0; node < _n_nodes; node++) {
        _busy_node[node] = false;
    }

    _rr_start = (_rr_start + 1) % _n_nodes;
    _cycles++;
}


void MyInterconnect::push(uint32_t src, uint32_t dest, MemoryAccess *request) {
    // -- initialize entity
    MyInterconnect::Entity entity;
    if (_in_buffers[src].empty())
        entity.finish_cycle = _cycles + _latency;
    else
        entity.finish_cycle = _in_buffers[src].back().finish_cycle + 1;

    entity.add_cycle = _cycles;
    entity.src = src;
    entity.dest = dest;
    entity.access = request;

    // -- push to _in_buffer
    _in_buffers[src].push(entity);

    // spdlog::info("At time {}, push a memory request to interconnect from source: {} to dest: {} ",entity.add_cycle,entity.src, entity.dest);
}

bool MyInterconnect::is_full(uint32_t nid, MemoryAccess* request) {
    // TODO: unlimit buffer size
    return false;
}

bool MyInterconnect::is_empty(uint32_t nid) {
    return _out_buffers[nid].empty();
}

MemoryAccess* MyInterconnect::top(uint32_t nid) {
    assert(!is_empty(nid));
    return _out_buffers[nid].front();
}


MemoryAccess* MyInterconnect::top(uint32_t nid, cycle_type update_dram_enter_cycle) {
    assert(!is_empty(nid));
    _out_buffers[nid].front()->dram_enter_cycle = update_dram_enter_cycle;
    return _out_buffers[nid].front();
}

void MyInterconnect::pop(uint32_t nid) {
    auto mem_access = _out_buffers[nid].front();
    // spdlog::trace("PUSH {}", _cycles);
    if (nid < memory_offset) {
        update_stat(*mem_access, nid % _config.dram_channels);
    }
    _out_buffers[nid].pop();
}


cycle_type MyInterconnect::get_core_cycle() {
    return (cycle_type)((double)_cycles * (double)Config::system_config.core_freq /
                        (double)Config::system_config.icnt_freq);
}


cycle_type MyInterconnect::get_dram_cycle() {
    double dram_cycle = std::ceil((double)_cycles * (double)Config::system_config.dram_freq / (double)Config::system_config.icnt_freq);
    spdlog::info("get dram cycle is {}", dram_cycle);
    return (cycle_type)(dram_cycle);

}

cycle_type MyInterconnect::get_icnt_cycle() {
    return _cycles;
}


void MyInterconnect::print_stats() {
    std::ofstream json_out(
        Config::system_config.log_dir + "/icnt_traffic.json",
        std::ofstream::out);
    json_out << "{";
    for (uint32_t channel = 0; channel < _n_memories; ++channel) {
        const auto& measured = _measured_traffic[channel];
        const auto& estimated = _estimated_traffic[channel];
        spdlog::info(
            "ICNT channel {} logical read/write traffic = {}/{} bytes (measured {}/{}, estimated {}/{})",
            channel,
            measured.memory_read_bytes + estimated.memory_read_bytes,
            measured.memory_write_bytes + estimated.memory_write_bytes,
            measured.memory_read_bytes, measured.memory_write_bytes,
            estimated.memory_read_bytes, estimated.memory_write_bytes);
        spdlog::info(
            "ICNT channel {} logical PIM requests P_HEADER/GWRITE/COMP/READRES = {}/{}/{}/{} (measured {}/{}/{}/{}, estimated {}/{}/{}/{})",
            channel,
            measured.pim_pheader_requests + estimated.pim_pheader_requests,
            measured.pim_gwrite_requests + estimated.pim_gwrite_requests,
            measured.pim_comp_requests + estimated.pim_comp_requests,
            measured.pim_readres_requests + estimated.pim_readres_requests,
            measured.pim_pheader_requests, measured.pim_gwrite_requests,
            measured.pim_comp_requests, measured.pim_readres_requests,
            estimated.pim_pheader_requests, estimated.pim_gwrite_requests,
            estimated.pim_comp_requests, estimated.pim_readres_requests);

        if (channel != 0) {
            json_out << ",";
        }
        json_out << "\"" << channel << "\":{"
                 << "\"channel\":" << channel
                 << ",\"measured_read_bytes\":"
                 << measured.memory_read_bytes
                 << ",\"estimated_read_bytes\":"
                 << estimated.memory_read_bytes
                 << ",\"logical_read_bytes\":"
                 << measured.memory_read_bytes + estimated.memory_read_bytes
                 << ",\"measured_write_bytes\":"
                 << measured.memory_write_bytes
                 << ",\"estimated_write_bytes\":"
                 << estimated.memory_write_bytes
                 << ",\"logical_write_bytes\":"
                 << measured.memory_write_bytes + estimated.memory_write_bytes
                 << ",\"measured_pheader_requests\":"
                 << measured.pim_pheader_requests
                 << ",\"estimated_pheader_requests\":"
                 << estimated.pim_pheader_requests
                 << ",\"logical_pheader_requests\":"
                 << measured.pim_pheader_requests +
                        estimated.pim_pheader_requests
                 << ",\"measured_gwrite_requests\":"
                 << measured.pim_gwrite_requests
                 << ",\"estimated_gwrite_requests\":"
                 << estimated.pim_gwrite_requests
                 << ",\"logical_gwrite_requests\":"
                 << measured.pim_gwrite_requests +
                        estimated.pim_gwrite_requests
                 << ",\"measured_comp_requests\":"
                 << measured.pim_comp_requests
                 << ",\"estimated_comp_requests\":"
                 << estimated.pim_comp_requests
                 << ",\"logical_comp_requests\":"
                 << measured.pim_comp_requests +
                        estimated.pim_comp_requests
                 << ",\"measured_readres_requests\":"
                 << measured.pim_readres_requests
                 << ",\"estimated_readres_requests\":"
                 << estimated.pim_readres_requests
                 << ",\"logical_readres_requests\":"
                 << measured.pim_readres_requests +
                        estimated.pim_readres_requests
                 << "}";
    }
    json_out << "}";
}


void MyInterconnect::update_stat(MemoryAccess memory_access, uint64_t ch_idx) {
    assert(ch_idx < _stats.size());
    auto& measured = _measured_traffic[ch_idx];
    auto& interval = _stats[ch_idx].back();
    // READ, WRITE, GWRITE, COMP, READRES, P_HEADER, COMPS_READRES, SIZE
    switch (memory_access.req_type) {
        case MemoryAccessType::READ:
            interval.memory_reads += memory_access.size;
            measured.memory_read_bytes += memory_access.size;
            break;
        case MemoryAccessType::WRITE:
            interval.memory_writes += memory_access.size;
            measured.memory_write_bytes += memory_access.size;
            break;
        case MemoryAccessType::P_HEADER:
            measured.pim_pheader_requests++;
            break;
        case MemoryAccessType::READRES:
            interval.pim_reads++;
            measured.pim_readres_requests++;
            break;
        case MemoryAccessType::GWRITE:
            interval.pim_writes++;
            measured.pim_gwrite_requests++;
            break;
        case MemoryAccessType::COMP:
        case MemoryAccessType::COMP_HASH:
            interval.pim_comps++;
            measured.pim_comp_requests++;
            break;
        case MemoryAccessType::COMPS_READRES:
            interval.pim_comps++;
            interval.pim_reads++;
            measured.pim_comp_requests++;
            measured.pim_readres_requests++;
            break;
        case MemoryAccessType::SIZE:
            break;
    }
}

void MyInterconnect::apply_estimated_workload(
    const ProportionalWorkloadStat& workload) {
    const auto channel_value = [](const std::vector<uint64_t>& values,
                                  uint32_t channel) {
        return channel < values.size() ? values[channel] : 0ULL;
    };
    for (uint32_t channel = 0; channel < _n_memories; ++channel) {
        auto& estimated = _estimated_traffic[channel];
        auto& interval = _stats[channel].back();
        const uint64_t read_bytes =
            channel_value(workload.channel_memory_reads, channel) *
            MyAddressAllocator::dram_burst_size;
        const uint64_t write_bytes =
            channel_value(workload.channel_memory_writes, channel) *
            MyAddressAllocator::dram_burst_size;
        const uint64_t pheaders =
            channel_value(workload.channel_pim_pheader, channel);
        const uint64_t gwrites =
            channel_value(workload.channel_pim_gwrite, channel);
        const uint64_t comps =
            channel_value(workload.channel_pim_comp, channel);
        const uint64_t readres =
            channel_value(workload.channel_pim_readres, channel);

        interval.memory_reads += read_bytes;
        interval.memory_writes += write_bytes;
        interval.pim_writes += gwrites;
        interval.pim_comps += comps;
        interval.pim_reads += readres;

        estimated.memory_read_bytes += read_bytes;
        estimated.memory_write_bytes += write_bytes;
        estimated.pim_pheader_requests += pheaders;
        estimated.pim_gwrite_requests += gwrites;
        estimated.pim_comp_requests += comps;
        estimated.pim_readres_requests += readres;
    }
}


void MyInterconnect::set_icnt_cycles(uint64_t cycles) {
    _cycles = _cycles + cycles;
}


void MyInterconnect::reset_dram_interface_valid()
{
    std::fill(_dram_push_valid.begin(), _dram_push_valid.end(), true);
    std::fill(_dram_pop_valid.begin(), _dram_pop_valid.end(), true);
}

bool MyInterconnect::dram_push_valid(int mem_id)
{
    return _dram_push_valid[mem_id];
}

bool MyInterconnect::dram_pop_valid(int mem_id)
{
    return _dram_pop_valid[mem_id];
}

void MyInterconnect::consume_dram_push(int mem_id)
{
    _dram_push_valid[mem_id] = false;
}

void MyInterconnect::consume_dram_pop(int mem_id)
{
    _dram_pop_valid[mem_id] = false;
}

uint64_t MyInterconnect::get_dram_tick(uint64_t global_cycle) {
    return global_cycle * _config.dram_freq / _config.icnt_freq;     // 根据你的时钟调度方式替换这里
}
