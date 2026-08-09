#ifndef DRAM_BANK_HPP
#define DRAM_BANK_HPP

#include "../common_function.hpp"

enum class BankState {OPEN, CLOSED, SREF, PD};
enum class RowBufPolicy {OPEN_PAGE, CLOSE_PAGE};

class DRAMBank {
public:
    DRAMBank(MemConfig& MemConfig, std::vector<std::pair<int, uint64_t>>* Rank_Activate_Recorder, int Rank_id, int BankGroup_id, int Bank_id);
    ~DRAMBank();
    // uint64_t ProcessTransaction(std::shared_ptr<Memory_Transaction> trans);
    int BankRefresh(unsigned int current_refresh_count);

    // void RowSwitch(std::shared_ptr<Memory_Transaction> trans);  // precharge + activate

    // --------------------------------------------------------
    void Activate(std::shared_ptr<Event> event);
    void Precharge(std::shared_ptr<Event> event);
    void Read(std::shared_ptr<Event> event);
    void Write(std::shared_ptr<Event> event);
    void PIM(std::shared_ptr<Event> event);

    void AddPendingPrechargeEvent(std::shared_ptr<Event> event);


    void PushPendingSwitchRow(std::shared_ptr<Event> event);

    MemConfig& config_;
    BankState bank_state;
    std::vector<std::pair<int, uint64_t>>* rank_activate_recorder;
    int rank_id, bankgroup_id, bank_id;

    // about row buffer hit
    RowBufPolicy row_buf_policy_;
    int open_row;  // Currently open row

    int open_row_exec_event_count;
    std::deque<int> pending_precharge_order_queue;  // 存储了当前待进行precharge的event的顺序
    std::unordered_map<int, std::deque<std::shared_ptr<Event>>> pending_precharge_event; // Bank中按照行号存储
    bool pending_precharge; // true when a precharge command is already queued for this bank
    bool pending_activate;  // 当为true时, 标志着当前bank的activate的指令已经发送到Pending队列中
    int row_hit_count;  // consecutive accesses to one row
    int row_miss_count;

    // about refresh
    uint64_t latest_refresh_time = 0;
    uint64_t next_bank_refresh_time;
    bool bank_refresh = false;
    int bank_refresh_count = 0;
    int refresh_interval_ = 0;
    int refresh_offset_ = 0;

    // 相关时间参数信息
    uint64_t latest_trans_time = 0;

    // 下一次相关操作允许执行的时间
    uint64_t to_do_precharge;
    uint64_t to_do_activate;    // 之后需要考虑，时间窗口内多次激活的问题
    uint64_t to_do_read;
    uint64_t to_do_write;

    std::multimap<uint32_t, std::shared_ptr<Event>> pending_activate_row;
};


#endif