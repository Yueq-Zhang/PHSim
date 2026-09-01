
#include <list>
#include <unordered_map>

#include "../common_function.hpp"
#include "../../ext/booksim/include/booksim2/Interconnect.hpp"

class MyInterconnect {
public:
    MyInterconnect(const SysConfig& config);
    virtual ~MyInterconnect() = default;
    bool running();
    void cycle();
    void push(uint32_t src, uint32_t dest, MemoryAccess *request);
    bool is_full(uint32_t src, MemoryAccess *request);
    bool is_empty(uint32_t nid);
    MemoryAccess *top(uint32_t nid);
    MemoryAccess* top(uint32_t nid, cycle_type update_dram_enter_cycle);
    void pop(uint32_t nid);
    void print_stats();
    cycle_type get_core_cycle();

    void update_stat(MemoryAccess memory_access, uint64_t ch_idx);
    void apply_estimated_workload(const ProportionalWorkloadStat& workload);

    void set_icnt_cycles(uint64_t cycles);

    void reset_dram_interface_valid();
    bool dram_push_valid(int mem_id);
    bool dram_pop_valid(int mem_id);
    void consume_dram_push(int mem_id);
    void consume_dram_pop(int mem_id);

    cycle_type get_icnt_cycle();
    uint64_t get_dram_tick(uint64_t global_cycle); // 基于当前的ICNT cycle 得到dram cycle
    cycle_type get_dram_cycle();

    bool using_booksim() const noexcept { return _booksim != nullptr; }


protected:
    const SysConfig& _config;
    // period information (us) =  1 / MHZ
    double _icnt_period;
    double _dram_period;
    uint32_t _icnt_freq;
    uint32_t _dram_freq;
    double _icnt_time;
    double _dram_time;

    uint32_t _n_nodes;
    uint32_t _n_cores;
    uint32_t _n_memories;

    uint32_t memory_offset;

    uint64_t _cycles;

    uint32_t _latency;
    double _bandwidth;
    uint32_t _rr_start;

    // Check The IOstate
    std::vector<std::vector<MemoryIOStat>> _stats;
    MemoryIOStat _stat;
    uint64_t _mem_cycle_interval;

    struct TrafficTotals {
        uint64_t memory_read_bytes = 0;
        uint64_t memory_write_bytes = 0;
        uint64_t pim_pheader_requests = 0;
        uint64_t pim_gwrite_requests = 0;
        uint64_t pim_comp_requests = 0;
        uint64_t pim_readres_requests = 0;
    };
    std::vector<TrafficTotals> _measured_traffic;
    std::vector<TrafficTotals> _estimated_traffic;

    struct Entity {
        cycle_type add_cycle;
        cycle_type finish_cycle;
        uint32_t src;
        uint32_t dest;
        MemoryAccess* access;
    };

    std::vector<std::queue<Entity>> _in_buffers;
    std::vector<std::queue<MemoryAccess*>> _out_buffers;
    std::vector<bool> _busy_node;


    // std::vector<u_int32_t> _busy_cycle;
    std::vector<bool>  _dram_push_valid;
    std::vector<bool>  _dram_pop_valid;

    uint64_t _current_dram_tick_sync_by_icnt;
    uint64_t _next_dram_tick_sync_by_icnt;

    std::unique_ptr<booksim2::Interconnect> _booksim;
    uint64_t _booksim_injected_packets = 0;
    uint64_t _booksim_ejected_packets = 0;
    uint64_t _booksim_injected_payload_bytes = 0;
    uint64_t _booksim_ejected_payload_bytes = 0;
    uint64_t _input_full_query_events = 0;
    uint64_t _booksim_input_full_query_events = 0;
    uint64_t _output_full_blocked_packet_cycles = 0;
    std::vector<uint64_t> _max_input_buffer_occupancy;
    std::vector<uint64_t> _max_output_buffer_occupancy;
    std::unordered_map<const MemoryAccess*, uint32_t>
        _booksim_inflight_payload_bytes;

    booksim2::Interconnect::Type get_booksim_type(
        const MemoryAccess* access) const;
    uint32_t get_booksim_packet_size(const MemoryAccess* access) const;
};
