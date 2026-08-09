#ifndef __TIMING_H
#define __TIMING_H

#include "common.h"
#include "configuration.h"
#include <vector>

namespace dramsim3 {

class Timing {
  public:
    Timing(const Config &config);
    std::vector<std::vector<std::pair<CommandType, int>>> same_bank;
    std::vector<std::vector<std::pair<CommandType, int>>> other_banks_same_bankgroup;
    std::vector<std::vector<std::pair<CommandType, int>>> other_bankgroups_same_rank;
    std::vector<std::vector<std::pair<CommandType, int>>> other_ranks;
    std::vector<std::vector<std::pair<CommandType, int>>> same_rank;

    std::vector<std::vector<std::pair<CommandType, int>>> same_channel;

    int burst_cycle;
    int pim_precharge_latency; // Precharge time of PIM Banks of channel
    int pim_activate_latency;

    int pim_precharge_to_activate;
    int pim_activate_to_read;

    void update_pim_act_timing_for_dual_bank(uint32_t comps_per_pim_row);

};

} // namespace dramsim3
#endif
