#pragma once

#include "DRAM/DramDataContainer.h"
#include "DRAM/IDramBackend.h"
#include "common_function.hpp"

#include "Model/Model.h"
#include "Client/Client.h"

#include "Scheduler/MyScheduler.hpp"
#include "Core/MyCore.hpp"
#include "Interconnect/MyInterconnect.hpp"



#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3

class EventDrivenDram;

class Simulator {
public:
    Simulator(const SysConfig& config);
    ~Simulator();

    void launch_model(Ptr<Model> model);
    void run(std::string model_name);

    void cycle();
    bool running();
    void set_cycle_mask();
    uint32_t get_dest_node(MemoryAccess *access);
    void update_stage_stat();
    void log_stage_stat();
    void log_data_container_stat() const;
    void log_virtual_memory_stat() const;
    cycle_type advance_accelerated_time(cycle_type target_core_cycle);

    const SysConfig& _config;

    uint32_t _n_cores;
    uint32_t _n_memories;
    uint32_t memory_offset;

    std::vector<std::unique_ptr<MyCore>> _cores;
    std::unique_ptr<MyInterconnect> _icnt;

    void update_req_stat(MemoryAccessType t, uint64_t &read_cnt, uint64_t &write_cnt, uint64_t &gwrite_cnt, uint64_t &other_cnt);
    // ===== debug: req_type statistics =====
    uint64_t _stat_core2icnt_read   = 0;
    uint64_t _stat_core2icnt_write  = 0;
    uint64_t _stat_core2icnt_gwrite = 0;
    uint64_t _stat_core2icnt_other  = 0;

    uint64_t _stat_icnt2cycle_read   = 0;
    uint64_t _stat_icnt2cycle_write  = 0;
    uint64_t _stat_icnt2cycle_gwrite = 0;
    uint64_t _stat_icnt2cycle_other  = 0;

    uint64_t _stat_icnt2event_read   = 0;
    uint64_t _stat_icnt2event_write  = 0;
    uint64_t _stat_icnt2event_gwrite = 0;
    uint64_t _stat_icnt2event_other  = 0;

    uint64_t _stat_dramresp_read   = 0;
    uint64_t _stat_dramresp_write  = 0;
    uint64_t _stat_dramresp_gwrite = 0;
    uint64_t _stat_dramresp_other  = 0;


    DramMode _dram_mode;
    std::unique_ptr<DramDataContainer> _data_container;
    std::unique_ptr<IDramBackend> _dram_backend;
    // Populated only by the optional CA-vs-ED trace comparison build.
    std::unique_ptr<EventDrivenDram> _comparison_event_driven_dram;
    cycle_type _dram_cycle_count;
    cycle_type _icnt_cycle_count;

    // 对于两种DRAM执行Trace的记录与对比
    std::vector<std::unique_ptr<MemoryAccess>> _event_driven_compare_copies;
    uint32_t _trace_log_limit = 0;
    uint32_t _trace_print_count = 0;
    uint32_t _trace_print_start = 0;
    uint32_t _trace_print_end = 0;
    uint32_t _cycle_dram_completed = 0;
    uint32_t _event_dram_completed = 0;

    std::unique_ptr<MyScheduler> _scheduler;
    std::unique_ptr<Client> _client;

    addr_type _dram_ch_stride_size;
    uint64_t _core_cycles;

    uint32_t _cycle_mask;
    bool _single_run;
    Ptr<Model> _model;

    struct StageStat {
        Stage stage;
        uint64_t done_cycle;
        uint64_t pim_cycles;
        uint64_t npu_cycles;
        double mem_bw_util;
    };

    // For Channel Pruning
    uint32_t _cycle_finish_tile;
    uint32_t _cycle_get_tile;
    std::vector<uint32_t> _cycle_finish;
    std::vector<uint32_t> _cycle_get;
    std::unordered_map<std::string, Tile*> _core_tile_map;
    MyScheduler::PredictingConfig _predicting_config;

    std::vector<StageStat> _stage_stats;

    // For Event-Driven Simulation
    struct TraceRecord { // Record the 
        addr_type address;
        cycle_type cycle;  // 发送时的周期 或 完成时的DRAM周期
    };

    // 周期精确DRAM
    std::vector<TraceRecord> _cycle_dram_send;     // 发送的trace
    std::vector<TraceRecord> _cycle_dram_complete; // 完成的trace

    // 事件精确DRAM
    std::vector<TraceRecord> _event_dram_send;     // 发送的trace
    std::vector<TraceRecord> _event_dram_complete; // 完成的trace

    // 全局最大完成时间（不受trace_log_limit限制）
    cycle_type _newton_max_finish_cycle = 0;
    cycle_type _event_max_finish_cycle = 0;

    cycle_type _event_max_finish_cycle_internal = 0;

    cycle_type event_max() const { return _event_max_finish_cycle_internal; }

    void update_event_max(cycle_type v, const char* who) {
        if (v > _event_max_finish_cycle_internal) {
            spdlog::error("[EVENT_MAX] {} -> {} (caller: {})",
                          _event_max_finish_cycle_internal, v, who);
            _event_max_finish_cycle_internal = v;
        }
    }

    // ========== DRAM模式控制开关 ==========
    bool _use_event_dram = true;   // true=事件驱动, false=周期精确

};
