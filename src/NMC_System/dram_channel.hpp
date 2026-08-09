#ifndef DRAM_CHANNEL_HPP
#define DRAM_CHANNEL_HPP

#include <iostream>
#include <limits>
#include <algorithm>
#include "dram_rank.hpp"

// [新增] 定义事务提交的状态枚举
enum class TransactionStatus {
    REJECTED = 0,        // 把 0 留给失败状态
    ISSUED = 1,              // 成功发射，拿到准确的 complete_cycle
    QUEUED_WRITE_DRAIN = 2,  // 遭遇写排空阻塞，被撤回
    QUEUED_TIMING = 3        // 遭遇时序冲突阻塞，被撤回
};

inline std::ofstream& get_predict_refresh_log() {
    static std::ofstream f("refresh_predict.txt",
                           std::ios::out | std::ios::trunc);
    static bool header_written = false;
    if (!header_written) {
        f << "rank_id,current_cycle,refresh_exec_at,"
             "unblock_at,next_refresh_cycle\n";
        f.flush();
        header_written = true;
    }
    return f;
}

extern std::ofstream callback_record_file;

class DRAMChannel {
public:
    DRAMChannel(MemConfig& MemConfig, int channel_id);
    // bool AddTransaction(const std::shared_ptr<Memory_Transaction>& trans);
    // [修改] 将返回值从 bool 改为 TransactionStatus
    // TransactionStatus AddTransaction(const std::shared_ptr<Memory_Transaction>& trans);
    // bool AddPIMTransaction(const std::shared_ptr<PIM_Transaction> trans);

    void ScheduleTransaction();
    // void IssueTransaction(std::shared_ptr<Memory_Transaction> trans);
    void WriteBufferDrain();
    void ArrangeWriteBufferDrain();
    // uint64_t WriteBufferDrainValidTime(std::shared_ptr<Memory_Transaction> trans);
    void OtherRankTimingUpdate(uint64_t processing_cycle, bool is_write, int rank_id);
    // Address AddressMapping(uint64_t hex_addr);

    cycle_type PredictEventCycle(std::shared_ptr<Event> event);  // 新创建函数用于完成预测
    void IssueActivateEvent(std::shared_ptr<Event> event);
    void IssuePrechargeEvent(std::shared_ptr<Event> event);
    void IssueExecuteEvent(std::shared_ptr<Event> event);
    void IssueRankRefresh(cycle_type refresh_cycle);
    cycle_type GetNextRankRefreshCycle() const;

    void GetPrechargeCycle(std::shared_ptr<Event> event);
    void GetActivateCycle(std::shared_ptr<Event> event);
    void GetExecuteCycle(std::shared_ptr<Event> event);

    void RefreshEarliestQueue();

    MemConfig& config_;

    // 创建容器用于存储event, 用于完成换行判断以及后续使用
    std::deque<std::shared_ptr<Event>> execute_queue;
    std::deque<std::shared_ptr<Event>> activate_queue;
    std::deque<std::shared_ptr<Event>> precharge_queue;
    std::deque<std::shared_ptr<Event>> return_queue;

    // 创建一个容器来管理每一个bank中 precharge 与 exec的管理问题
    std::unordered_map<uint32_t, std::deque<std::shared_ptr<Event>>>  bank_row_buffer_conflict_event_queue;

    int channel_id;
    cycle_type _dram_cycle      = 0;
    cycle_type _pre_dram_cycle  = 0;

    std::pair<int, cycle_type> earliest_queue = {0, std::numeric_limits<uint64_t>::max()}; // exec=0; activate=1; precharge=2; rank_refresh=3


    bool is_unified_queue_;

    int write_draining_;
    std::vector<uint64_t> write_address_buffer_;

    uint64_t last_transaction_input_clk_ = 0;
    std::vector<DRAMRank*> dram_ranks;

    uint64_t clk_;
    // 数据总线空闲时间（真实执行用）
    uint64_t data_bus_busy_until_ = 0;

    // 预测事务详细结构体，记录一次DRAM访问的预测信息（不修改状态）
    struct PredictedTxn {
        uint64_t start_cycle = 0;       // 读/写列命令最早能发出的时间（即开始传输数据的时间）
        uint64_t complete_cycle = 0;    // 预测完成时间（数据完全传输结束）
        uint64_t data_bus_start = 0;    // 数据开始占用总线的时间
        uint64_t data_bus_end = 0;      // 数据释放总线的时间

        int rank = -1;          // 预测的目标rank编号
        int bankgroup = -1;     // 预测的目标bank group编号
        int bank = -1;          // 预测的目标bank编号
        int row = -1;           // 预测的行地址

        bool is_write = false;  // 是否为写操作
        bool row_hit = false;   // 是否发生行命中（不需要额外的activate）
    };

    // ========== 精确预测接口（只读，不修改状态）==========
    // 对指定地址的访问进行详细预测，返回包含开始周期、完成周期等信息的结构体
    PredictedTxn PredictTransactionDetailed(
        addr_type addr,
        bool is_write,
        uint64_t now
    );

    // 仅预测访问的完成周期（简化接口）
    uint64_t PredictCompleteCycle(
        addr_type addr,
        bool is_write,
        uint64_t now
    );

    // ========== 辅助预测函数 ==========
    // 预测给定bank当前周期是否受刷新影响，若受影响则返回刷新完成的时间
    uint64_t PredictBankRefreshImpact(
        DRAMBank* bank,
        DRAMRank* rank,
        uint64_t current_cycle
    );

    // 预测给定rank当前周期是否处于刷新中，若是则返回刷新完成时间
    uint64_t PredictRankRefreshImpact(
        DRAMRank* rank,
        uint64_t current_cycle
    );

    // 预测激活命令最早可执行的时间（考虑rank级激活限制）
    uint64_t PredictActivateTime(
        const std::vector<std::pair<int, uint64_t>>& rank_activate_recorder,
        int target_bg,
        uint64_t base_cycle
    );

    // 仅预测读/写列命令最早能发出的时刻（用于快速比较）
    uint64_t PredictStartCycle(addr_type addr, bool is_write, uint64_t now);

    /*
    // [新增] 获取写排空结束时间的预估函数
    uint64_t GetWriteDrainFinishTime() {
        if (!write_draining_) return clk_;
        // 使用你 hpp 里的变量名：channel_transaction_write_buffer
        return clk_ + channel_transaction_write_buffer.size() * config_.write_delay;
    }

    // [新增] 暴露底层真实读队列大小
    size_t get_real_read_queue_size() const {
        return channel_transaction_read_queue.size();
    }

    // [新增] 暴露底层真实写缓冲大小
    size_t get_real_write_buffer_size() const {
        return channel_transaction_write_buffer.size();
    }
    */
};


inline DRAMChannel::DRAMChannel(MemConfig& MemConfig, int Channel_id) :config_(MemConfig), channel_id(Channel_id),is_unified_queue_(config_.unified_queue), write_draining_(0), clk_(0) {
    /*
    if (is_unified_queue_) {
        channel_transaction_unified_buffer.reserve(config_.trans_queue_size);
    }
    else {
        channel_transaction_read_queue.reserve(config_.trans_queue_size);
        channel_transaction_write_buffer.reserve(config_.trans_queue_size);
        write_address_buffer_.reserve(config_.trans_queue_size);
    }
    */


    for (auto rank_id = 0; rank_id < config_.ranks; rank_id++) {
        dram_ranks.emplace_back(new DRAMRank(config_, channel_id, rank_id));
    }
}

inline void DRAMChannel::GetPrechargeCycle(std::shared_ptr<Event> event) {
    event->precharge_cycle = std::max(event->precharge_cycle + 1, dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_activate) ;
    if (event->precharge_cycle < earliest_queue.second) {
        earliest_queue = {2, event->precharge_cycle};
    }
}

inline void DRAMChannel::IssuePrechargeEvent(std::shared_ptr<Event> event) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    // 发送Precharge, 将event从precharge queue中删除, 放置在activate_queue中; 如果Activate中之前为空，计算首个Activate的时间, 同时更新下一个Precharge的时间
    dram_ranks[event->rank_index]->IssuePrecharge(event);
    activate_queue.push_back(event);
    if (activate_queue.size() == 1) {
        GetActivateCycle(activate_queue.front());
    }
    precharge_queue.pop_front();
    if (!precharge_queue.empty()) {
        // TODO::处理顺序没有发送变化，后续可能存在优化空间(不需要逐个比对时间, 而是添加一定的经验内容)
        GetPrechargeCycle(precharge_queue.front()); // 计算下一个Event完成Precharge的时间
    }
    RefreshEarliestQueue();
}


inline void DRAMChannel::GetActivateCycle(std::shared_ptr<Event> event) {

    // 首先在Bankgroup与Bank两层级，查找对应Bank的最早刷新时间
    event->activate_cycle = std::max({event->activate_cycle,
        dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_activate,
        dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_activate});

    // 之后需要进行tFAW或者t32AW的的情况判定, 确定是否符合window
    if (dram_ranks[event->rank_index]->rank_activate_recoder_.size() >= 4) {
        size_t idx = dram_ranks[event->rank_index]->rank_activate_recoder_.size() - 4;
        event->activate_cycle = std::max(event->activate_cycle, dram_ranks[event->rank_index]->rank_activate_recoder_[idx].second + config_.tFAW);
    }

    if (config_.IsGDDR() && dram_ranks[event->rank_index]->rank_activate_recoder_.size() >= 32) {
        size_t idx = dram_ranks[event->rank_index]->rank_activate_recoder_.size() - 32;
        event->activate_cycle = std::max(event->activate_cycle, dram_ranks[event->rank_index]->rank_activate_recoder_[idx].second + config_.t32AW);
    }

    if (event->activate_cycle < earliest_queue.second) {
        earliest_queue = {1, event->activate_cycle};
    }
}

inline void DRAMChannel::IssueActivateEvent(std::shared_ptr<Event> event) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    // 发送Activate
    dram_ranks[event->rank_index]->IssueActivate(event);
    // 发送之后，当前Activate队列删除发送事件，发送到execute队列中等待执行, 对于首个指令, 则计算发送时间
    execute_queue.push_back(event);
    if (execute_queue.size() == 1) {
        GetExecuteCycle(execute_queue.front());
    }

    // 在发送activate之后，检查当前bank的Pending_precharge_event, 清空queue, 以及pending_precharge_event
    auto operation_bank = dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];

    if (!operation_bank->pending_precharge_order_queue.empty() && event->row_index == operation_bank->pending_precharge_order_queue.front()) {
        operation_bank->pending_precharge_order_queue.pop_front();
        auto it = operation_bank->pending_precharge_event.find(event->row_index);
        if (it != operation_bank->pending_precharge_event.end()) {
            for (auto same_row_event: it->second) {
                execute_queue.push_back(same_row_event);
                operation_bank->open_row_exec_event_count++;
            }
            operation_bank->pending_precharge_event.erase(it);
        }
    }
    // execute_queue.push_back() [event->row_index]

    // 将执行完成的Activate Event移除, 队列非空的话同时更新下一个Activate的时间
    activate_queue.pop_front();
    if (!activate_queue.empty()) {
        // TODO::处理顺序没有发送变化，后续可能存在优化空间(不需要逐个比对时间, 而是添加一定的经验内容)
        GetActivateCycle(activate_queue.front()); // 计算下一个Event进行Activate的时间
    }
    RefreshEarliestQueue();
}


inline void DRAMChannel::GetExecuteCycle(std::shared_ptr<Event> event) {
    bool is_write = event->req_type == MemoryAccessType::WRITE;
    if (is_write) {
        event->execute_cycle = std::max({event->execute_cycle + 1,
                                dram_ranks[event->rank_index]->to_do_write,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_write,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_write});
    }
    else {
        event->execute_cycle = std::max({event->execute_cycle + 1,
                                dram_ranks[event->rank_index]->to_do_read,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->to_do_read,
                                dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index]->to_do_read});
    }

    if (event->execute_cycle < earliest_queue.second) {
        earliest_queue = {0, event->execute_cycle};
    }
}


inline void DRAMChannel::IssueExecuteEvent(std::shared_ptr<Event> event) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    // 完成对于Execute的执行, 同时更新目前Bank中的信息
    bool is_write = event->req_type == MemoryAccessType::WRITE;
    if (is_write) {
        for (auto rank : dram_ranks) {
            if (rank->rank_id == event->rank_index) {
                rank->IssueWrite(event);  // 执行指令发送到对应的rank
            }
            else { // 更新其他rank
                rank->to_do_read = event->execute_cycle + EventDrivenParams::write_to_read_o;
                rank->to_do_write = event->execute_cycle + EventDrivenParams::write_to_write_o;
            }
        }
        event->complete_cycle = event->execute_cycle + config_.write_delay;
    }
    else {  // Read
        for (auto rank : dram_ranks) {
            if (rank->rank_id == event->rank_index) {
                rank->IssueRead(event);  // 执行指令发送到对应的rank
            }
            else { // 更新其他rank
                rank->to_do_read = event->execute_cycle + EventDrivenParams::read_to_read_o;
                rank->to_do_write = event->execute_cycle + EventDrivenParams::read_to_write_o;
            }
        }
        event->complete_cycle = event->execute_cycle + config_.read_delay;
    }
    // else PIM Operations

    // process the pending precharge event
    auto operation_bank = dram_ranks[event->rank_index]->dram_bankgroups[event->bankgroup_index]->dram_banks[event->bank_index];

    if (operation_bank->open_row_exec_event_count == 0 && !operation_bank->pending_precharge_order_queue.empty()) {
        // 找到待换行的event信息，进入precharge_queue中等待后续操作
        auto operation_row = operation_bank->pending_precharge_order_queue.front();
        auto it = operation_bank->pending_precharge_event.find(operation_row);
        if (it != operation_bank->pending_precharge_event.end() && !it->second.empty()) {
            auto pending_event = it->second.front();
            it->second.pop_front();
            if (it->second.empty()) {
                operation_bank->pending_precharge_event.erase(it);
            }
            pending_event->need_precharge = true;
            pending_event->precharge_cycle = event->execute_cycle + 1;
            operation_bank->pending_precharge = true;
            precharge_queue.push_back(pending_event);
            if (precharge_queue.size() == 1) {
                GetPrechargeCycle(precharge_queue.front());
            }
        }
    }

    // 执行完成后, 将event放置在return_queue,之后清除execute_queue中的event
    return_queue.push_back(event);
    spdlog::debug("(EventDriven DRAM) Event {} is Push back to return queue at cycle {}, current return queue size is {}",
        event->dram_address, event->execute_cycle, return_queue.size());

    execute_queue.pop_front();
    if (!execute_queue.empty()) {
        // TODO::处理顺序没有发送变化，后续可能存在优化空间(不需要逐个比对时间, 而是添加一定的经验内容), 或者可以比对一下，之后选择进行放置
        GetExecuteCycle(execute_queue.front()); // 计算下一个Event的执行时间
    }
    RefreshEarliestQueue();
}


inline cycle_type DRAMChannel::GetNextRankRefreshCycle() const {
    cycle_type next_cycle = std::numeric_limits<cycle_type>::max();
    for (auto rank : dram_ranks) {
        if (rank->rank_refresh) {
            next_cycle = std::min(next_cycle, static_cast<cycle_type>(rank->next_refresh_cycle));
        }
    }
    return next_cycle;
}

inline void DRAMChannel::IssueRankRefresh(cycle_type refresh_cycle) {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};
    for (auto rank : dram_ranks) {
        if (!rank->rank_refresh || rank->next_refresh_cycle > refresh_cycle) {
            continue;
        }

        rank->current_trans_refresh_count =
            (refresh_cycle - rank->rank_refresh_offset_) / rank->rank_refresh_interval_ + 1;
        const auto rank_refresh_time = rank->RankRefreshReady();
        spdlog::debug("(EventDriven DRAM) Rank Refresh at cycle {} for Channel {}, Rank {}, actual refresh cycle {}, next refresh {}",
            refresh_cycle, channel_id, rank->rank_id, rank_refresh_time, rank->next_refresh_cycle);
        rank->RankRefresh(rank_refresh_time);
    }
    RefreshEarliestQueue();
}

inline void DRAMChannel::RefreshEarliestQueue() {
    earliest_queue = {-1, std::numeric_limits<cycle_type>::max()};

    if (!execute_queue.empty()) {
        if (execute_queue.front()->execute_cycle <= earliest_queue.second) {
            earliest_queue = {0, execute_queue.front()->execute_cycle};
        }
    }

    if (!activate_queue.empty()) {
        if (activate_queue.front()->activate_cycle < earliest_queue.second) {
            earliest_queue = {1, activate_queue.front()->activate_cycle};
        }

    }

    if (!precharge_queue.empty()) {
        if (precharge_queue.front()->precharge_cycle < earliest_queue.second) {
            earliest_queue = {2, precharge_queue.front()->precharge_cycle};
        }
    }

    if (execute_queue.empty() && activate_queue.empty() && precharge_queue.empty()) {
        const auto next_rank_refresh = GetNextRankRefreshCycle();
        if (next_rank_refresh < earliest_queue.second) {
            earliest_queue = {3, next_rank_refresh};
        }
    }
}


///////////////////////////////  下面的大概率用不上了 ////////////////////////////////////////////////////////
/*
inline TransactionStatus DRAMChannel::AddTransaction(const std::shared_ptr<Memory_Transaction>& trans) {

    static std::ofstream g_dram_add_log;
    static bool g_dram_add_log_open = false;
    if (!g_dram_add_log_open) {
        g_dram_add_log.open("dram_add_log.txt", std::ios::out | std::ios::trunc);
        g_dram_add_log_open = true;
    }

    g_dram_add_log << "ADD"
                   << " ch=" << channel_id
                   << " addr=0x" << std::hex << trans->addr << std::dec
                   << " is_write=" << trans->is_write
                   << " gen_before=" << trans->generate_cycle
                   << " proc_before=" << trans->processing_cycle
                   << "\n";
    g_dram_add_log.flush();

    if (trans->generate_cycle <= last_transaction_input_clk_) {
        trans->generate_cycle = last_transaction_input_clk_ + 1;
    }
    trans->generate_cycle + 1; // 原代码保留
    trans->processing_cycle = trans->generate_cycle + 1;

    last_transaction_input_clk_ = trans->generate_cycle;

    // 1. 统一队列的情况
    if (is_unified_queue_ and channel_transaction_unified_buffer.size()<channel_transaction_unified_buffer.capacity()) {
        channel_transaction_unified_buffer.emplace_back(trans);
        trans->address = AddressMapping(trans->addr);

        ScheduleTransaction();
        return TransactionStatus::ISSUED;
    }
    else {
        // 2. 写请求的情况
        if (trans->is_write) {
            if (channel_transaction_write_buffer.size() < channel_transaction_write_buffer.capacity()) {
                trans->complete_cycle = trans->generate_cycle + 1;
                auto write_trans = std::make_shared<Memory_Transaction>(*trans);
                write_trans->address = AddressMapping(write_trans->addr);
                channel_transaction_write_buffer.emplace_back(write_trans);
                write_address_buffer_.emplace_back(write_trans->addr);

                g_dram_add_log << "  -> ENQUEUE_WRITE_BUFFER"
                               << " ch=" << channel_id
                               << " addr=0x" << std::hex << trans->addr << std::dec
                               << " finish=" << trans->complete_cycle
                               << " wb_size=" << channel_transaction_write_buffer.size()
                               << "\n";
                g_dram_add_log.flush();
            }
            ScheduleTransaction();
            return TransactionStatus::ISSUED;
        }
        // 3. 读请求的情况
        else {
            trans->is_issued = false;  // 初始化发射标记
            trans->address = AddressMapping(trans->addr);
            channel_transaction_read_queue.emplace_back(trans);

            g_dram_add_log << "  -> ENQUEUE_READ_QUEUE"
                           << " ch=" << channel_id
                           << " addr=0x" << std::hex << trans->addr << std::dec
                           << " rq_size=" << channel_transaction_read_queue.size()
                           << "\n";
            g_dram_add_log.flush();

            ScheduleTransaction();     // 尝试调用物理调度

            // 检查底层有没有发射，如果没有，将其撤回并报错阻塞
            if (trans->is_issued) {
                return TransactionStatus::ISSUED;
            } else {
                // 没有被发射，从队列尾部撤回！消灭幽灵完成！
                channel_transaction_read_queue.pop_back();

                if (write_draining_) {
                    return TransactionStatus::QUEUED_WRITE_DRAIN;
                } else {
                    return TransactionStatus::QUEUED_TIMING;
                }
            }
        }
    }
}


inline void DRAMChannel::ScheduleTransaction() {

    // Unified queue 模式：每次只发一条
    if (is_unified_queue_) {
        if (!channel_transaction_unified_buffer.empty()) {
            IssueTransaction(channel_transaction_unified_buffer.front());
            channel_transaction_unified_buffer.erase(channel_transaction_unified_buffer.begin());
        }
        return;
    }

    // 非 unified queue：先检查是否要进入写排空模式
    if (write_draining_ == 0) {
        if (channel_transaction_write_buffer.size() >= channel_transaction_write_buffer.capacity() ||
            channel_transaction_write_buffer.size() > 8) {
            write_draining_ = static_cast<int>(channel_transaction_write_buffer.size());
            }
    }

    // 若处于写排空模式：本轮只排 1 条写
    // 不要同时把读队列清掉
    if (write_draining_ > 0) {
        WriteBufferDrain();
        return;
    }

    // 非写排空模式：每轮最多只发 1 条读
    // 不要 while 清空整个读队列
    if (!channel_transaction_read_queue.empty()) {
        IssueTransaction(channel_transaction_read_queue.front());
        channel_transaction_read_queue.erase(channel_transaction_read_queue.begin());
        return;
    }

    // 没有可发请求则直接返回
    return;
}

inline Address DRAMChannel::AddressMapping(uint64_t hex_addr) {
    // 原逻辑：把地址再右移 shift_bits 后，按 MemConfig 位域拆 channel/rank/bg/ba/row/col。
    // 但当前项目里的 dram_address 已经是 MyAddressAllocator 编码后的“未 shift 地址”，
    // CA 侧也已经改成直接用 MyAddressAllocator::get_*() 解码，不能在 ED 里再额外右移一次。


    hex_addr >>= config_.shift_bits;
    int channel = (hex_addr >> config_.ch_pos) & config_.ch_mask;
    int rank = (hex_addr >> config_.ra_pos) & config_.ra_mask;
    int bg = (hex_addr >> config_.bg_pos) & config_.bg_mask;
    int ba = (hex_addr >> config_.ba_pos) & config_.ba_mask;
    int ro = (hex_addr >> config_.ro_pos) & config_.ro_mask;
    int co = (hex_addr >> config_.co_pos) & config_.co_mask;
    return Address(channel, rank, bg, ba, ro, co);

    int channel = MyAddressAllocator::get_channel_index(hex_addr);
    int rank = MyAddressAllocator::get_rank_index(hex_addr);
    int bg = MyAddressAllocator::get_bankgroup_index(hex_addr);
    int ba = MyAddressAllocator::get_bank_index(hex_addr);
    int ro = MyAddressAllocator::get_row_index(hex_addr);
    int co = MyAddressAllocator::get_col_index(hex_addr);
    return Address(channel, rank, bg, ba, ro, co);
}


inline void DRAMChannel::IssueTransaction(std::shared_ptr<Memory_Transaction> trans){
    assert(dram_ranks.size() == (size_t)config_.ranks);
    assert(trans->address.rank >= 0 && trans->address.rank < config_.ranks);

    trans->is_issued = true; // [核心] 明确标记该请求已获得真实物理时序

    if (clk_ < trans->generate_cycle) {
        clk_ = trans->generate_cycle;
    }
    // spdlog::info("At {} clk, A {} Transaction of address = {}, which generate at {} is issued ", clk_, trans->is_write ? "Write": "Read", trans->addr, trans->generate_cycle);

    // 按照层级逐级向下发送，一直到bank层级，同时在各层级中进行判断，获得当前Issue可以向下执行的最早的时间，在此过程中生成执行操作的延迟，在bank层级最终就是完成了相应的处理，并将其出队，完成了Trans的析构处理。
    if (dram_ranks[trans->address.rank]->in_self_refresh) {
        dram_ranks[trans->address.rank]->ExitSelfRefresh();
    }

    // 基于transaction作用的内容进行逐级传递，直到到达对应的Bank。
    auto processing_cycle = dram_ranks[trans->address.rank]->IssueTransactionToBankGroup(trans);

    // Serialize data-bus bursts to prevent overlap and match the timing model.
    uint64_t bus_start =
        (trans->complete_cycle >= (uint64_t)config_.burst_cycle)
        ? (trans->complete_cycle - (uint64_t)config_.burst_cycle)
        : 0;

    bus_start = std::max(bus_start, data_bus_busy_until_);

    uint64_t bus_finish = bus_start + (uint64_t)config_.burst_cycle;

    trans->complete_cycle = bus_finish;
    data_bus_busy_until_ = bus_finish;
    for (int i = 0; i < config_.ranks; i++) {
        if (i!=trans->address.rank) {
            OtherRankTimingUpdate(processing_cycle, trans->is_write, i);
        }
    }
    clk_++;  // 之后如果进行了precharge或者Activate操作，对于clk直接执行++

}

inline void DRAMChannel::WriteBufferDrain() {
    // 若没有处于写排空，或者写buffer为空，直接返回
    if (write_draining_ <= 0 || channel_transaction_write_buffer.empty()) {
        if (write_draining_ <= 0) {
            write_address_buffer_.clear();
        }
        return;
    }

    uint64_t trans_valid_time = 0;
    int issue_index = 0;
    uint64_t trans_valid_min_time = std::numeric_limits<uint64_t>::max();

    // 在当前 write buffer 中挑一条最早可发的写
    for (int i = 0; i < (int)channel_transaction_write_buffer.size(); i++) {
        trans_valid_time = WriteBufferDrainValidTime(channel_transaction_write_buffer[i]);
        if (trans_valid_time < trans_valid_min_time) {
            trans_valid_min_time = trans_valid_time;
            issue_index = i;
        }
    }

    // 标记这是写排空发出的事务
    channel_transaction_write_buffer[issue_index]->processing_cycle = clk_;
    channel_transaction_write_buffer[issue_index]->write_drain = true;

    // 真正发这 1 条写
    IssueTransaction(channel_transaction_write_buffer[issue_index]);

    // 从 write buffer 中删除
    channel_transaction_write_buffer.erase(channel_transaction_write_buffer.begin() + issue_index);

    // 写排空计数减一
    write_draining_ -= 1;

    // 若本轮排空结束，清理 write address buffer
    if (write_draining_ <= 0 || channel_transaction_write_buffer.empty()) {
        write_draining_ = 0;
        write_address_buffer_.clear();
    }

    return;
}



inline void DRAMChannel::OtherRankTimingUpdate(uint64_t processing_cycle, bool is_write, int rank_id) {
    // spdlog::info("Update the Read and Write Timing of Rank {}", rank_id);

    assert(rank_id >= 0 && rank_id < (int)dram_ranks.size());
    if (is_write) {
        dram_ranks[rank_id]->to_do_read = dram_ranks[rank_id]->to_do_read > processing_cycle + config_.other_rank_write_to_read_latency ? dram_ranks[rank_id]->to_do_read : processing_cycle + config_.other_rank_write_to_read_latency;
        dram_ranks[rank_id]->to_do_write = dram_ranks[rank_id]->to_do_write > processing_cycle + config_.other_rank_write_to_write_latency ? dram_ranks[rank_id]->to_do_write : processing_cycle + config_.other_rank_write_to_write_latency;
    }
    else {
        dram_ranks[rank_id]->to_do_read = dram_ranks[rank_id]->to_do_read > processing_cycle + config_.other_rank_read_to_read_latency ? dram_ranks[rank_id]->to_do_read : processing_cycle + config_.other_rank_read_to_read_latency;
        dram_ranks[rank_id]->to_do_write = dram_ranks[rank_id]->to_do_write > processing_cycle + config_.other_rank_read_to_write_latency ? dram_ranks[rank_id]->to_do_write : processing_cycle + config_.other_rank_read_to_write_latency;
    }
}


inline uint64_t DRAMChannel::WriteBufferDrainValidTime(std::shared_ptr<Memory_Transaction> trans) {
    uint64_t trans_valid_time = clk_;

    if (trans->address.row == dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->open_row) {  // Row buffer hit
        // to Write Rank
        trans_valid_time = trans_valid_time < dram_ranks[trans->address.rank]->to_do_write ? dram_ranks[trans->address.rank]->to_do_write : trans_valid_time;
        // to Write BankGroup
        trans_valid_time = trans_valid_time < dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->to_do_write ? \
            dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->to_do_write : trans_valid_time;
        // to Write Bank
        trans_valid_time = trans_valid_time < dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->to_do_write ? \
            dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->to_do_write : trans_valid_time;
    }
    else {  // Row buffer miss
        // Precharge
        trans_valid_time = trans_valid_time < dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->to_do_precharge ? \
            dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->to_do_precharge : trans_valid_time;
        // Activate
        trans_valid_time = trans_valid_time + config_.tRP < dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->to_do_activate ? \
            dram_ranks[trans->address.rank]->dram_bankgroups[trans->address.bankgroup]->dram_banks[trans->address.bank]->to_do_activate : trans_valid_time + config_.tRP;
        // trans->processing_cycle = trans->processing_cycle + config_.tRP < to_do_activate ? trans->processing_cycle + config_.tRP : to_do_activate;
        trans_valid_time = trans_valid_time + config_.activate_write;
    }
    return trans_valid_time;
}


inline void DRAMChannel::ArrangeWriteBufferDrain(){
    // 一种新的排列方法，用于实现速度提升 -- 之后再尝试一下


}



inline bool DRAMChannel::AddPIMTransaction(const std::shared_ptr<PIM_Transaction> trans) {
    if (trans->generate_cycle <= last_transaction_input_clk_) {
        trans->generate_cycle = last_transaction_input_clk_ + 1;
    }
    trans->generate_cycle + 1;   // The Transaction input latency = 1
    trans->processing_cycle = trans->generate_cycle + 1;

    return true;
}


// Bank Refresh 影响
inline uint64_t DRAMChannel::PredictBankRefreshImpact(
    DRAMBank* bank,
    DRAMRank* rank,
    uint64_t current_cycle) {

    // 计算当前是第几次 refresh
    unsigned int current_refresh_count =
        (current_cycle - bank->refresh_offset_) / bank->refresh_interval_;

    // 计算 refresh 的理想时间（idle time）
    uint64_t refresh_idle_time =
        (current_refresh_count - 1) * bank->refresh_interval_ + bank->refresh_offset_;

    int refresh_execute_block_time = 0;

    // 复刻 BankRefresh() 的逻辑
    if (bank->bank_state == BankState::OPEN &&
        current_refresh_count - bank->bank_refresh_count == 1) {

        // Bank 是 OPEN：需要等 precharge 完成
        refresh_execute_block_time += std::max(
            static_cast<int>(bank->to_do_precharge - refresh_idle_time), 0
        );
        refresh_execute_block_time += 1;  // precharge command
        refresh_execute_block_time += config_.tRP;

    } else if (bank->bank_state == BankState::CLOSED) {

        // Bank 是 CLOSED：需要等之前的 activate 完成
        refresh_execute_block_time += std::max(
            static_cast<int>(bank->to_do_activate - refresh_idle_time), 0
        );

    } else {
        refresh_execute_block_time = 0;
    }

    uint64_t latest_refresh_time = refresh_idle_time + refresh_execute_block_time;

    // 考虑 rank_activate_recorder 的 tRC 约束
    if (!rank->rank_activate_recoder_.empty()) {
        uint64_t last_activate = rank->rank_activate_recoder_.back().second;
        latest_refresh_time = std::max(latest_refresh_time, last_activate + config_.tRC);
    }

    // Refresh 之后，bank->to_do_activate 会被设置为：
    uint64_t virtual_to_do_activate = latest_refresh_time + config_.tRFCb;

    // 返回当前事务最早能开始的时间
    return std::max(current_cycle, virtual_to_do_activate);
}

//Rank Refresh 影响
inline uint64_t DRAMChannel::PredictRankRefreshImpact(
    DRAMRank* rank,
    uint64_t current_cycle) {

    // 计算需要第几次 refresh  (当前时间 - 刷新偏移量) / 刷新间隔 + 1
    int current_trans_refresh_count =
            (current_cycle - rank->rank_refresh_offset_) / rank->rank_refresh_interval_ + 1;
    // (刷新次数 - 1) * 间隔 + 偏移量
    uint64_t refresh_time =
            (current_trans_refresh_count - 1) * rank->rank_refresh_interval_ +
            rank->rank_refresh_offset_;

    //判断预测的这次刷新是否是相对于该Rank已记录的最后一次刷新 (rank->refresh_count) 的“下一次”刷新
    if (current_trans_refresh_count - rank->refresh_count == 1) {
        uint64_t to_refresh_time = 0;

        // 遍历所有 bank，找最大等待时间
        for (auto bg: rank->dram_bankgroups) {
            for (auto bank: bg->dram_banks) {
                if (bank->bank_state == BankState::OPEN) {
                    // OPEN 状态：需要等 precharge
                    if (refresh_time < bank->to_do_precharge) {
                        refresh_time = bank->to_do_precharge;
                    }
                    refresh_time += 1;
                    to_refresh_time = config_.tRP;
                } else {
                    // CLOSED 状态：需要等上一个事务的 precharge 延迟 Bank关闭后，必须经过至少 tRP时间，才能开始刷新操作
                    uint64_t wait = bank->latest_trans_time + config_.tRP;
                    if (refresh_time + to_refresh_time < wait) {
                        to_refresh_time = wait - refresh_time;
                    }
                }
            }
        }
        refresh_time += to_refresh_time;
    }

    // Refresh 之后，所有 bank 的 to_do_activate 会被设为：
    uint64_t virtual_to_do_activate = refresh_time + config_.tRFC;

    uint64_t result = std::max(current_cycle, virtual_to_do_activate);

    // 打印预测refresh
    auto& log = get_predict_refresh_log();
    log << rank->rank_id        << ","
        << current_cycle        << ","
        << refresh_time         << ","  // 预测refresh执行时间
        << virtual_to_do_activate << ","// 预测unblock时间
        << rank->next_refresh_cycle << "\n";
    log.flush();
    return result;
}

// Activate 时间 包含 FAW/32AW
inline uint64_t DRAMChannel::PredictActivateTime(const std::vector<std::pair<int, uint64_t> > &rank_activate_recorder, int target_bg, uint64_t base_cycle) {
    uint64_t act_cycle = base_cycle;

    // (1) tRRD_L / tRRD_S 约束
    if (!rank_activate_recorder.empty()) {
        auto &last = rank_activate_recorder.back();
        if (last.first == target_bg) {
            act_cycle = std::max(act_cycle, last.second + config_.tRRD_L);
        } else {
            act_cycle = std::max(act_cycle, last.second + config_.tRRD_S);
        }
    }

    // (2) FAW 约束（4-activate window）
    if (rank_activate_recorder.size() >= 4) {
        size_t idx = rank_activate_recorder.size() - 4;
        act_cycle = std::max(act_cycle,
            rank_activate_recorder[idx].second + config_.tFAW);
    }

    // (3) 32AW 约束（GDDR 特有）
    if (config_.IsGDDR() && rank_activate_recorder.size() >= 32) {
        act_cycle = std::max(act_cycle,
            rank_activate_recorder[0].second + config_.t32AW);
    }

    // 返回 activate 命令执行的时间
    return act_cycle;
}


//
inline DRAMChannel::PredictedTxn
DRAMChannel::PredictTransactionDetailed(
    addr_type addr,      // 要预测的 DRAM 地址
    bool is_write,       // 是否为写操作
    uint64_t now)        // 当前模拟周期（用于外部时间参考）
{
    // 创建预测结果结构体，并设置读写标志
    PredictedTxn pred;
    pred.is_write = is_write;

    // 1) 地址映射：将线性地址解析为 rank、bankgroup、bank 和行号
    Address mapped_addr = AddressMapping(addr);
    pred.rank = mapped_addr.rank;
    pred.bankgroup = mapped_addr.bankgroup;
    pred.bank = mapped_addr.bank;
    pred.row = mapped_addr.row;

    // 2) 当前预测起点 = max(外部传入的 now, channel 内部时钟 clk_)
    //    确保预测不会早于当前系统时间
    uint64_t proc_cycle = std::max(now, clk_);

    // 获取对应的 rank、bankgroup、bank 指针，便于后续访问状态
    auto* rank = dram_ranks[mapped_addr.rank];
    auto* bg   = rank->dram_bankgroups[mapped_addr.bankgroup];
    auto* bank = bg->dram_banks[mapped_addr.bank];

    // 3) Rank 级刷新影响：如果 rank 处于刷新窗口内，则将 proc_cycle 推进到刷新完成
    if (rank->rank_refresh &&
        proc_cycle >= rank->next_refresh_cycle) {
        proc_cycle = PredictRankRefreshImpact(rank, proc_cycle);
    }

    // 4) Bank 级刷新影响：如果 bank 处于刷新窗口内，推进到刷新完成
    if (bank->bank_refresh &&
        proc_cycle >= bank->next_bank_refresh_time) {
        proc_cycle = PredictBankRefreshImpact(bank, rank, proc_cycle);
    }

    // 5) Rank 级跨 BankGroup 时序约束：读写操作必须满足 rank 级的读写窗口限制
    if (is_write)
        proc_cycle = std::max(proc_cycle, rank->to_do_write);
    else
        proc_cycle = std::max(proc_cycle, rank->to_do_read);

    // 6) BankGroup 级时序约束：同一 bankgroup 内的读写操作需满足 bankgroup 级限制
    if (is_write)
        proc_cycle = std::max(proc_cycle, bg->to_do_write);
    else
        proc_cycle = std::max(proc_cycle, bg->to_do_read);

    // 7) Bank 级状态处理：根据当前 bank 状态（CLOSED / OPEN且行命中 / OPEN且行冲突）
    //    分别计算列命令（READ/WRITE）最早可发出的时间 col_issue
    bool row_hit = (bank->bank_state == BankState::OPEN &&
                    bank->open_row == mapped_addr.row);
    pred.row_hit = row_hit;

    uint64_t col_issue = 0;

    if (bank->bank_state == BankState::CLOSED) {
        // A：bank 关闭 -> 需要先激活（ACT），然后才能发读写命令
        // 考虑 bank 自身的激活就绪时间
        proc_cycle = std::max(proc_cycle, bank->to_do_activate);
        // 预测激活命令实际能发出的时间（考虑 rank 级激活并发限制）
        uint64_t act_cycle = PredictActivateTime(
            rank->rank_activate_recoder_,
            mapped_addr.bankgroup,
            proc_cycle);
        uint64_t act_done = act_cycle + 1;           // 激活完成周期
        // 列命令最早发出时间 = 激活完成 + 激活到读/写的延迟（tRCD 等）
        col_issue = act_done + (is_write
            ? (uint64_t)config_.activate_write   // 写操作对应的激活后延迟
            : (uint64_t)config_.activate_read);   // 读操作对应的激活后延迟

    } else if (row_hit) {
        // B：行命中 -> 无需激活，直接发读写命令
        // 需要考虑 bank 级的读写就绪时间
        col_issue = is_write
            ? std::max(proc_cycle, bank->to_do_write)
            : std::max(proc_cycle, bank->to_do_read);

    } else {
        // C：行冲突（bank 打开但行号不匹配）-> 需要先预充电（PRE），再激活（ACT），再读写
        // 先满足 bank 预充电就绪时间，然后加上 tRP（预充电时间）
        proc_cycle = std::max(proc_cycle, bank->to_do_precharge);
        proc_cycle += (uint64_t)config_.tRP;
        // 再满足激活就绪时间
        proc_cycle = std::max(proc_cycle, bank->to_do_activate);
        // 预测激活命令实际发出时间
        uint64_t act_cycle = PredictActivateTime(
            rank->rank_activate_recoder_,
            mapped_addr.bankgroup,
            proc_cycle);
        uint64_t act_done = act_cycle + 1;
        // 列命令发出时间 = 激活完成 + 激活到读/写的延迟
        col_issue = act_done + (is_write
            ? (uint64_t)config_.activate_write
            : (uint64_t)config_.activate_read);
    }

    // 记录列命令最早发出时间（即开始传输数据的周期）
    pred.start_cycle = col_issue;

    // 8) 预测完成时间（不考虑数据总线冲突）：开始周期 + 固定尾部延迟
    uint64_t tail = is_write
        ? (uint64_t)config_.write_delay   // 写操作从列命令发出到数据结束的延迟
        : (uint64_t)config_.read_delay;    // 读操作从列命令发出到数据结束的延迟
    uint64_t complete_no_bus = col_issue + tail;

    // 9) 数据总线占用预测（只读，不修改真实 data_bus_busy_until_）
    //    计算总线占用区间 [bus_start, bus_end)，实际完成周期取总线释放时间
    //    总线开始时间 = 完成周期（无总线冲突时） - burst_cycle，但至少不小于当前数据总线繁忙时刻
    uint64_t bus_start =
        (complete_no_bus >= (uint64_t)config_.burst_cycle)
        ? (complete_no_bus - (uint64_t)config_.burst_cycle)
        : 0;
    bus_start = std::max(bus_start, data_bus_busy_until_);
    uint64_t bus_end = bus_start + (uint64_t)config_.burst_cycle;

    pred.data_bus_start = bus_start;
    pred.data_bus_end = bus_end;
    pred.complete_cycle = bus_end;   // 最终完成周期为总线释放的时刻

    return pred;
}

// 预测请求的列命令（READ/WRITE）最早可发出的周期（即开始传输数据的周期）
inline uint64_t DRAMChannel::PredictStartCycle(
    addr_type addr,      // 要预测的 DRAM 地址
    bool is_write,       // 是否为写操作
    uint64_t now)        // 当前模拟周期（外部时间参考）
{
    // 调用详细预测函数获取完整预测结果，然后返回其中的 start_cycle 字段
    return PredictTransactionDetailed(addr, is_write, now).start_cycle;
}

// 预测请求的完成周期（考虑数据总线冲突后，数据完全传输结束的时刻）
inline uint64_t DRAMChannel::PredictCompleteCycle(
    addr_type addr,      // 要预测的 DRAM 地址
    bool is_write,       // 是否为写操作
    uint64_t now)        // 当前模拟周期（外部时间参考）
{
    // 调用详细预测函数获取完整预测结果，然后返回其中的 complete_cycle 字段
    return PredictTransactionDetailed(addr, is_write, now).complete_cycle;
}


inline cycle_type DRAMChannel::PredictEventCycle(std::shared_ptr<Event> event) {
    // 所有的rank index 信息都可以得到
    // 获取对应 rank, bankgroup, bank 指针，用于后续访问状态
    auto* rank = dram_ranks[event->rank_index];
    auto* bg = rank->dram_bankgroups[event->bankgroup_index];
    auto* bank = bg->dram_banks[event->bank_index];

    auto proc_cycle = std::max(clk_, event->add_cycle); // 处理时间首先取为当前DRAM时间与Event的添加时间最大值

    // 3) Rank 级刷新影响：如果 rank 处于刷新窗口内，则将 proc_cycle 推进到刷新完成
    if (rank->rank_refresh && proc_cycle >= rank->next_refresh_cycle) {
        proc_cycle = PredictRankRefreshImpact(rank, proc_cycle);
    }

    // 4) Bank 级刷新影响：如果 bank 处于刷新窗口内，推进到刷新完成
    if (bank->bank_refresh && proc_cycle >= bank->next_bank_refresh_time) {
        proc_cycle = PredictBankRefreshImpact(bank, rank, proc_cycle);
    }

    bool is_write = event->req_type == MemoryAccessType::WRITE;

    // 5) Rank 级跨 BankGroup 时序约束：读写操作必须满足 rank 级的读写窗口限制
    if (is_write)
        proc_cycle = std::max(proc_cycle, rank->to_do_write);
    else
        proc_cycle = std::max(proc_cycle, rank->to_do_read);

    // 6) BankGroup 级时序约束：同一 bankgroup 内的读写操作需满足 bankgroup 级限制
    if (is_write)
        proc_cycle = std::max(proc_cycle, bg->to_do_write);
    else
        proc_cycle = std::max(proc_cycle, bg->to_do_read);

    // 7) Bank级时序约束: 对于同一个Bank, 需要考虑对应的换行开销
    bool row_hit = (bank->bank_state == BankState::OPEN && bank->open_row == event->row_index);

    cycle_type col_operation_cycle = 0;

    if (bank->bank_state == BankState::CLOSED) {
        // A：bank 关闭 -> 需要先激活（ACT），然后才能发读写命令
        // 考虑 bank 自身的激活就绪时间
        proc_cycle = std::max(proc_cycle, bank->to_do_activate);
        // 预测激活命令实际能发出的时间(考虑 rank 级激活限制), 其中包括FAW与32AW
        uint64_t act_cycle = PredictActivateTime(rank->rank_activate_recoder_, event->bankgroup_index, proc_cycle);
        uint64_t act_done = act_cycle + 1;           // 激活完成周期

        // 列命令最早发出时间 = 激活完成 + 激活到读/写的延迟
        col_operation_cycle = act_done + (is_write
            ? (uint64_t)config_.activate_write   // 写操作对应的激活后延迟
            : (uint64_t)config_.activate_read);  // 读操作对应的激活后延迟

    } else if (row_hit) {
        // B：行命中 -> 无需激活，直接发读写命令, 考虑对应Bank的时间
        // 需要考虑 bank 级的读写就绪时间
        col_operation_cycle = is_write
            ? std::max(proc_cycle, bank->to_do_write)
            : std::max(proc_cycle, bank->to_do_read);
    } else {
        // C：行冲突（bank 打开但行号不匹配）-> 需要先预充电（PRE），再激活（ACT），再读写
        // 先满足 bank 预充电就绪时间，然后加上 tRP（预充电时间）
        proc_cycle = std::max(proc_cycle, bank->to_do_precharge);
        proc_cycle += (uint64_t)config_.tRP;
        // 再满足激活就绪时间
        proc_cycle = std::max(proc_cycle, bank->to_do_activate);
        // 预测激活命令实际发出时间
        uint64_t act_cycle = PredictActivateTime(rank->rank_activate_recoder_, event->bankgroup_index, proc_cycle);
        uint64_t act_done = act_cycle + 1;
        // 列命令发出时间 = 激活完成 + 激活到读/写的延迟
        col_operation_cycle = act_done + (is_write
            ? (uint64_t)config_.activate_write
            : (uint64_t)config_.activate_read);
    }
    return col_operation_cycle;
}

*/
#endif
