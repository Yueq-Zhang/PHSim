
#include "MyInterconnect.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

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
    spdlog::info("Initialize My Interconnect ({})",
                 config.icnt_type == IcntType::BOOKSIM2 ? "booksim2"
                                                        : "simple");

    // period information (us) =  1 / MHZ
    _icnt_period = 1.0 / static_cast<double>(_config.icnt_freq);
    _dram_period = 1.0 / static_cast<double>(_config.dram_freq);
    _icnt_freq = _config.icnt_freq;
    _dram_freq = _config.dram_freq;
    _dram_time = 0.0;
    _icnt_time = 0.0;

    _latency = config.icnt_latency;
    _cycles = 0;
    _rr_start = 0;
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

    if (config.icnt_type == IcntType::SIMPLE) {
        spdlog::info(
            "Simple interconnect queue capacities: input={} packets/node, "
            "output={} packets/node (0 means unbounded)",
            config.icnt_input_buffer_size,
            config.icnt_output_buffer_size);
    }

    if (config.icnt_type == IcntType::BOOKSIM2) {
        if (config.icnt_config_path.empty()) {
            throw std::invalid_argument(
                "BookSim interconnect requires icnt_config_path");
        }
        if (!std::filesystem::is_regular_file(config.icnt_config_path)) {
            throw std::invalid_argument(
                "BookSim configuration file does not exist: " +
                config.icnt_config_path);
        }
        _booksim = std::make_unique<booksim2::Interconnect>(
            config.icnt_config_path, static_cast<int>(_n_nodes));
        if (_booksim->get_network_node_count() !=
            static_cast<int>(_n_nodes)) {
            throw std::invalid_argument(
                "BookSim topology node count " +
                std::to_string(_booksim->get_network_node_count()) +
                " does not match PHSim interconnect node count " +
                std::to_string(_n_nodes));
        }
        spdlog::info(
            "BookSim active: config='{}', nodes={}, flit_size={} B",
            config.icnt_config_path, _n_nodes, _booksim->get_flit_size());
    }

    // 设置所同步的 DRAM tick
    _current_dram_tick_sync_by_icnt = 0;
    _next_dram_tick_sync_by_icnt = 1;
}

bool MyInterconnect::running() {
    return _booksim &&
           _booksim_injected_packets != _booksim_ejected_packets;
}

void MyInterconnect::cycle() {
    if (_booksim) {
        _booksim->run();
    } else {
        // Send request from Core to DRAM, and get DRAM response back to Core
        for (int node = 0; node < _n_nodes; node++) {
            int src_node = (_rr_start + node) % _n_nodes;
            if (!_in_buffers[src_node].empty() &&
                _in_buffers[src_node].front().finish_cycle <= _cycles) {
                uint32_t dest = _in_buffers[src_node].front().dest;
                const bool output_has_space =
                    _config.icnt_output_buffer_size == 0 ||
                    _out_buffers[dest].size() <
                        _config.icnt_output_buffer_size;
                if (!_busy_node[dest] && output_has_space) {
                    _out_buffers[dest].push(
                        _in_buffers[src_node].front().access);
                    _busy_node[dest] = true;
                    _in_buffers[src_node].pop();
                }
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

    if (!_booksim) {
        // set the busy node flag for each port
        for(int node = 0; node < _n_nodes; node++) {
            _busy_node[node] = false;
        }
        _rr_start = (_rr_start + 1) % _n_nodes;
    }
    _cycles++;
}


void MyInterconnect::push(uint32_t src, uint32_t dest, MemoryAccess *request) {
    if (src >= _n_nodes || dest >= _n_nodes || request == nullptr) {
        throw std::out_of_range(
            "Interconnect push has an invalid node or null request");
    }
    if (is_full(src, request)) {
        throw std::overflow_error(
            "Interconnect input buffer capacity exceeded at node " +
            std::to_string(src));
    }
    if (_booksim) {
        const uint32_t packet_size = get_booksim_packet_size(request);
        if (packet_size > static_cast<uint32_t>(
                              std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "BookSim packet size exceeds the supported int range");
        }
        _booksim->push(request, 0, request->dram_address,
                       static_cast<int>(packet_size),
                       get_booksim_type(request), static_cast<int>(src),
                       static_cast<int>(dest));
        const bool inserted =
            _booksim_inflight_payload_bytes.emplace(request, packet_size)
                .second;
        if (!inserted) {
            throw std::logic_error(
                "The same MemoryAccess was injected into BookSim twice");
        }
        ++_booksim_injected_packets;
        _booksim_injected_payload_bytes += packet_size;
        return;
    }

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
    if (nid >= _n_nodes || request == nullptr) {
        throw std::out_of_range(
            "Interconnect fullness query has an invalid source or request");
    }
    if (_booksim) {
        return _booksim->is_full(
            nid, 0, get_booksim_packet_size(request));
    }
    return _config.icnt_input_buffer_size != 0 &&
           _in_buffers[nid].size() >= _config.icnt_input_buffer_size;
}

bool MyInterconnect::is_empty(uint32_t nid) {
    if (nid >= _n_nodes) {
        throw std::out_of_range("Interconnect node index out of range");
    }
    if (_booksim) {
        return _booksim->is_empty(nid, 0);
    }
    return _out_buffers[nid].empty();
}

MemoryAccess* MyInterconnect::top(uint32_t nid) {
    assert(!is_empty(nid));
    if (_booksim) {
        return const_cast<MemoryAccess*>(
            static_cast<const MemoryAccess*>(_booksim->top(nid, 0)));
    }
    return _out_buffers[nid].front();
}


MemoryAccess* MyInterconnect::top(uint32_t nid, cycle_type update_dram_enter_cycle) {
    assert(!is_empty(nid));
    MemoryAccess* access = top(nid);
    access->dram_enter_cycle = update_dram_enter_cycle;
    return access;
}

void MyInterconnect::pop(uint32_t nid) {
    if (is_empty(nid)) {
        throw std::underflow_error("Cannot pop an empty interconnect output");
    }
    auto mem_access = top(nid);
    // spdlog::trace("PUSH {}", _cycles);
    if (nid < memory_offset) {
        update_stat(*mem_access, nid % _config.dram_channels);
    }
    if (_booksim) {
        const auto size_it = _booksim_inflight_payload_bytes.find(mem_access);
        if (size_it == _booksim_inflight_payload_bytes.end()) {
            throw std::logic_error(
                "BookSim ejected a packet without injection metadata");
        }
        const uint32_t packet_size = size_it->second;
        _booksim->pop(nid, 0);
        _booksim_inflight_payload_bytes.erase(size_it);
        ++_booksim_ejected_packets;
        _booksim_ejected_payload_bytes += packet_size;
    } else {
        _out_buffers[nid].pop();
    }
}

booksim2::Interconnect::Type MyInterconnect::get_booksim_type(
    const MemoryAccess* access) const {
    if (access == nullptr) {
        throw std::invalid_argument(
            "Cannot classify a null BookSim packet");
    }
    const bool write_like =
        access->req_type == MemoryAccessType::WRITE ||
        access->req_type == MemoryAccessType::GWRITE;
    if (access->request) {
        return write_like ? booksim2::Interconnect::Type::WRITE
                          : booksim2::Interconnect::Type::READ;
    }
    return write_like ? booksim2::Interconnect::Type::WRITE_REPLY
                      : booksim2::Interconnect::Type::READ_REPLY;
}

uint32_t MyInterconnect::get_booksim_packet_size(
    const MemoryAccess* access) const {
    if (access == nullptr) {
        throw std::invalid_argument(
            "Cannot size a null BookSim packet");
    }

    const bool request_carries_data =
        access->request &&
        (access->req_type == MemoryAccessType::WRITE ||
         access->req_type == MemoryAccessType::GWRITE);
    const bool response_carries_data =
        !access->request &&
        (access->req_type == MemoryAccessType::READ ||
         access->req_type == MemoryAccessType::READRES ||
         access->req_type == MemoryAccessType::COMPS_READRES);

    uint64_t payload_bytes = _config.icnt_ctrl_size;
    if (request_carries_data || response_carries_data) {
        payload_bytes = std::max<uint64_t>(access->size,
                                           access->data.size());
        if (payload_bytes == 0) {
            payload_bytes = _config.icnt_ctrl_size;
        }
    }
    if (payload_bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error(
            "BookSim packet payload does not fit in uint32_t");
    }
    return static_cast<uint32_t>(payload_bytes);
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
    json_out.close();

    if (_booksim) {
        nlohmann::json booksim_stats = {
            {"backend", "booksim2"},
            {"config_path", _config.icnt_config_path},
            {"nodes", _n_nodes},
            {"flit_size_bytes", _booksim->get_flit_size()},
            {"phsim_icnt_cycles", _cycles},
            {"booksim_cycles", _booksim->get_cycle()},
            {"injected_packets", _booksim_injected_packets},
            {"ejected_packets", _booksim_ejected_packets},
            {"injected_payload_bytes",
             _booksim_injected_payload_bytes},
            {"ejected_payload_bytes",
             _booksim_ejected_payload_bytes}};
        const std::string path =
            Config::system_config.log_dir + "/booksim2_stats.json";
        std::ofstream booksim_out(path, std::ofstream::out);
        if (!booksim_out.is_open()) {
            throw std::runtime_error(
                "Cannot open BookSim statistics file: " + path);
        }
        booksim_out << booksim_stats.dump(2) << '\n';
        if (!booksim_out.good()) {
            throw std::runtime_error(
                "Failed to write BookSim statistics file: " + path);
        }
        spdlog::info(
            "BookSim packets injected/ejected={}/{}, payload bytes={}/{}, "
            "BookSim cycles={}",
            _booksim_injected_packets, _booksim_ejected_packets,
            _booksim_injected_payload_bytes,
            _booksim_ejected_payload_bytes, _booksim->get_cycle());
    }
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
