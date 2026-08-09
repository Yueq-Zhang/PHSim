#ifndef __DRAM_SYSTEM_H
#define __DRAM_SYSTEM_H

#include <fstream>
#include <string>
#include <vector>

#include "common.h"
#include "configuration.h"
// #include "controller.h"
#include "dram_controller.h"
// #include "neupims_controller.h"
#include "newton_controller.h"
#include "timing.h"

namespace dramsim3 {

class BaseDRAMSystem {
  public:
    BaseDRAMSystem(Config &config, const std::string &output_dir,
                   std::function<void(uint64_t)> read_callback,
                   std::function<void(uint64_t)> write_callback);
    virtual ~BaseDRAMSystem() {}
    void RegisterCallbacks(std::function<void(uint64_t)> read_callback,
                           std::function<void(uint64_t)> write_callback);
    void PrintEpochStats();
    void PrintStats();
    void ResetStats();

    virtual bool WillAcceptTransaction(uint64_t hex_addr, TransactionType req_type) const = 0;
    virtual bool AddTransaction(uint64_t hex_addr, TransactionType req_type,
                                uint32_t core_id = 0) = 0;
    virtual void ClockTick() = 0;
    int GetChannel(uint64_t hex_addr) const;
    uint64_t GetCounter(uint32_t channel, const std::string& name) const;
    void AddCounter(uint32_t channel, const std::string& name, uint64_t value);
    uint64_t GetVecCounter(uint32_t channel, const std::string& name,
                           int pos) const;
    void AddVecCounter(uint32_t channel, const std::string& name, int pos,
                       uint64_t value);
    uint64_t GetCoreCommandCounter(uint32_t core_id,
                                   const std::string& name) const;

    std::function<void(uint64_t req_id)> read_callback_, write_callback_;
    std::function<void(uint64_t req_id)> pim_callback_;

    static int total_channels_;

    virtual uint64_t GetAvgPIMCycles() = 0;
    virtual void ResetPIMCycle() = 0;

  uint64_t clk_;

  protected:
    uint64_t id_;
    uint64_t last_req_clk_;
    Config &config_;
    Timing timing_;
    uint64_t parallel_cycles_;
    uint64_t serial_cycles_;

    std::vector<Controller *> ctrls_;

#ifdef ADDR_TRACE
    std::ofstream address_trace_;
#endif // ADDR_TRACE
};

// hmmm not sure this is the best naming...
class JedecDRAMSystem : public BaseDRAMSystem {
  public:
    JedecDRAMSystem(Config &config, const std::string &output_dir,
                    std::function<void(uint64_t)> read_callback,
                    std::function<void(uint64_t)> write_callback,
                    std::function<void(uint64_t)> pim_callback);
    ~JedecDRAMSystem();
    bool WillAcceptTransaction(uint64_t hex_addr, TransactionType req_type) const override;
    bool AddTransaction(uint64_t hex_addr, TransactionType req_type,
                        uint32_t core_id = 0) override;
    void ClockTick() override;
    uint64_t GetAvgPIMCycles() override;
    void ResetPIMCycle() override;
};

// Model a memorysystem with an infinite bandwidth and a fixed latency (possibly
// zero) To establish a baseline for what a 'good' memory standard can and
// cannot do for a given application
class IdealDRAMSystem : public BaseDRAMSystem {
  public:
    IdealDRAMSystem(Config &config, const std::string &output_dir,
                    std::function<void(uint64_t)> read_callback,
                    std::function<void(uint64_t)> write_callback);
    ~IdealDRAMSystem();
    bool WillAcceptTransaction(uint64_t hex_addr, TransactionType req_type) const override {
        return true;
    };
    bool AddTransaction(uint64_t hex_addr, TransactionType req_type,
                        uint32_t core_id = 0) override;
    void ClockTick() override;

  private:
    int latency_;
    std::vector<Transaction> infinite_buffer_q_;
};

} // namespace dramsim3
#endif // __DRAM_SYSTEM_H
