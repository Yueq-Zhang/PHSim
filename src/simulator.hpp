#pragma once

#include "DRAM/EventDrivenDram.h"

#include "DRAM/Dram.h"
#include "common_function.hpp"

#include "Model/Model.h"
#include "Interconnect/Interconnect.h"
#include "Client/Client.h"
#include "Core/Core.h"

#include "Scheduler/MyScheduler.hpp"
#include "Core/MyCore.hpp"
#include "Interconnect/MyInterconnect.hpp"



#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3

class Simulator {
public:
    Simulator(const SysConfig& config);
    ~Simulator();

    void launch_model(Ptr<Model> model);
    void run(std::string model_name);

    // addr_type get_addr_align() { return _dram->get_addr_align(); }
    void cycle();
    bool running();
    void set_cycle_mask();
    uint32_t get_dest_node(MemoryAccess *access);
    void update_stage_stat();
    void log_stage_stat();

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
    std::unique_ptr<PIM> _dram; // std::unique_ptr<MemorySystem> _dram;
    std::unique_ptr<EventDrivenDram> _event_driven_dram; // event driven dram structure
    cycle_type _dram_cycle_count;

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

    // period information (us)
    double _core_period;
    double _icnt_period;
    double _dram_period;
    //
    double _core_time;
    double _icnt_time;
    double _dram_time;

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

    // 当前这一轮事件发生的物理时刻（秒）
    // 由 set_cycle_mask() 中的 minimum_time 得到
    double _curr_event_time = 0.0;

    // ED模式下：每个memory channel下一次最早允许被dram采样新trace的物理时刻
    // 单位与 _core_time / _icnt_time / _dram_time 相同，都是绝对物理时间
    std::vector<double> _ed_next_push_time;

    // 将任意物理时刻 t 对齐到不早于 t 的第一个 dram 上升沿时刻”
    double align_to_dram_edge(double t) const;

    // 将已经对齐到dram边沿的物理时刻转换成 dram cycle 编号
    cycle_type time_to_dram_cycle(double t) const;

};
