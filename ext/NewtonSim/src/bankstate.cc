#include "bankstate.h"

#include <boost/mpl/assert.hpp>

namespace dramsim3 {

BankState::BankState(bool enable_dual_buffer)
    : enable_dual_buffer_(enable_dual_buffer),
      state_(State::CLOSED),
      pim_state_(State::CLOSED),
      pim_open_row_(-1),
      cmd_timing_(static_cast<int>(CommandType::SIZE)),
      open_row_(-1),
      row_hit_count_(0),
      pim_enter_count_(0),
      pim_lock_(false) {
    cmd_timing_[static_cast<int>(CommandType::READ)] = 0;
    cmd_timing_[static_cast<int>(CommandType::READ_PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::WRITE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::WRITE_PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::ACTIVATE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::REFRESH)] = 0;
    cmd_timing_[static_cast<int>(CommandType::SREF_ENTER)] = 0;
    cmd_timing_[static_cast<int>(CommandType::SREF_EXIT)] = 0;
    // PIM Operations
    cmd_timing_[static_cast<int>(CommandType::P_HEADER)] = 0;
    cmd_timing_[static_cast<int>(CommandType::GWRITE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::G_ACT)] = 0;
    cmd_timing_[static_cast<int>(CommandType::G_PRE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::COMP)] = 0;
    cmd_timing_[static_cast<int>(CommandType::COMP_HASH)] = 0;
    cmd_timing_[static_cast<int>(CommandType::READRES)] = 0;
    cmd_timing_[static_cast<int>(CommandType::PIM_PRECHARGE)] = 0;
    cmd_timing_[static_cast<int>(CommandType::PIM_ACTIVE)] = 0;
}

/* Return the cmd required before issuing the cmd in the current state */
Command BankState::GetReadyCommand(const Command &cmd, uint64_t clk) const {
    if (!enable_dual_buffer_) {
        return GetReadyCommandSingle(cmd, clk);
    }
    // 下面代码是针对两个Row Buffer的实现方案，分别进行处理
    // p_header for gwrite = normal buffer cmd
    // p_header for comp-readres = pim buffer cmd
    if (cmd.IsPIMBufferCommand()) {
        return GetReadyPIMCommand(cmd, clk);
    }
    else if (cmd.IsNormalBufferCommand()) {
        return GetReadyNormalCommand(cmd, clk);
    }
    else {
        throw std::runtime_error("(GetReadyCommand) Not Valid Command");
    }
}

void BankState::UpdateState(const Command &cmd) {
    if (!enable_dual_buffer_) {  // 单个row buffer的情况
        UpdateStateSingle(cmd);
    }
    else { // 具有两个row Buffer的情况
        if (cmd.IsPIMBufferCommand()) {
            UpdatePIMState(cmd);
        } else if (cmd.IsNormalBufferCommand())
            UpdateNormalState(cmd);
        else {
            PrintError("(UpdateState) Not Valid Command");
        }
    }
}

Command BankState::GetReadyPIMCommand(const Command &cmd, uint64_t clk) const {
    CommandType required_type = CommandType::SIZE;
    switch (pim_state_) {
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:
                    if (state_ == State::OPEN && cmd.Row() == open_row_) {
                        required_type = CommandType::PRECHARGE;
                    } else {
                        return Command(cmd);
                    }
                    break;
                case CommandType::G_ACT:
                    if (state_ == State::OPEN && cmd.Row() == open_row_) {
                        PrintError("(GetReadyPIMCommand) Already Normal Buffer open this row");
                    }
                    required_type = cmd.cmd_type;
                    break;
                case CommandType::COMP:
                    required_type = CommandType::G_ACT;
                    break;
                case CommandType::COMP_HASH:  // The Ready Instructions of COMP_HASH
                    required_type = CommandType::ACTIVATE;
                    break;
                case CommandType::GWRITE:
                case CommandType::READRES:
                case CommandType::PIM_PRECHARGE:  // no need to precharge
                case CommandType::G_PRE:
                default:
                    std::cerr << "(GetReadyPIMCommand) Channel:" << cmd.Channel()
                              << " Rank:" << cmd.Rank() << " Bankgroup:" << cmd.Bankgroup()
                              << " Bank:" << cmd.Bank() << std::endl;
                    std::cerr << "PIM State: CLOSED, but try " << cmd.CommandTypeString()
                              << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::COMP:
                    if (cmd.Row() == pim_open_row_) {
                        return Command(cmd);
                    }
                    else {
                        required_type = CommandType::PIM_PRECHARGE;
                    }
                    break;
                case CommandType::COMP_HASH: // The Ready Instructions of COMP_HASH
                    if (cmd.Row() == pim_open_row_) {
                        return Command(cmd);
                    }
                    else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::REFRESH:
                case CommandType::REFRESH_BANK:
                case CommandType::SREF_ENTER:
                    required_type = CommandType::PIM_PRECHARGE;
                    break;
                case CommandType::PIM_PRECHARGE:
                case CommandType::G_PRE:
                case CommandType::READRES:
                    required_type = cmd.cmd_type;
                    break;
                case CommandType::G_ACT:
                case CommandType::PIM_ACTIVE:
                    // 作用到当前的某些节点有问题，但是后续没问题了
                    break;
                default:
                    std::cerr << "(GetReadyPIMCommand) Unknown type!" << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::SREF:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:
                case CommandType::GWRITE:
                case CommandType::G_ACT:
                    required_type = CommandType::SREF_EXIT;
                    break;
                case CommandType::COMP:
                case CommandType::READRES:
                case CommandType::PIM_PRECHARGE:
                // <<< gsheo
                default:
                    std::cerr << "(GetReadyPIMCommand) Unknown type!" << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::PD:
        case State::SIZE:
            std::cerr << "In unknown state" << std::endl;
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
            break;
    }
    if (required_type != CommandType::SIZE) {
        if (clk >= cmd_timing_[static_cast<int>(required_type)]) {
            return Command(required_type, cmd.addr, cmd.hex_addr, cmd.is_last_comps, cmd.num_comps);
        }
        // Command fordebug = Command(required_type, cmd.addr, cmd.hex_addr);
        // std::cout << "(GetReadyCommand) cur:" << cmd.CommandTypeString()
        //           << ", required:" << fordebug.CommandTypeString();
        // std::cout << ", clk:" << clk << " , cmd_timing:"
        //           << cmd_timing_[static_cast<int>(required_type)] <<
        //           std::endl;
    }
    return Command();
}
Command BankState::GetReadyNormalCommand(const Command &cmd, uint64_t clk) const {
    CommandType required_type = CommandType::SIZE;
    switch (state_) {
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:  // p_header for gwrite
                    if (pim_state_ == State::OPEN && pim_open_row_ == cmd.Row()) {
                        required_type = CommandType::PIM_PRECHARGE;
                    } else {
                        return Command(cmd);
                    }
                    break;
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::GWRITE:
                    if (pim_state_ == State::OPEN && cmd.Row() == pim_open_row_) {
                        required_type = CommandType::PIM_PRECHARGE;
                        PrintColor(Color::RED,
                                   "(GetReadyPIMCommand) Already PIM Buffer open this row");
                        break;
                    }
                    required_type = CommandType::ACTIVATE;
                    break;
                case CommandType::REFRESH:
                    if (pim_state_ == State::CLOSED) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PIM_PRECHARGE;
                    }
                    break;
                default:
                    std::cerr << "(GetReadyNormalCommand) Unknown type!" << std::endl;
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::P_HEADER:
                    if (open_row_ == cmd.Row()) {
                        return Command(cmd);
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::READ:
                case CommandType::WRITE:
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::GWRITE:
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                        // PrintError("Why not handle?");
                    }
                    break;
                case CommandType::REFRESH:
                    required_type = CommandType::PRECHARGE;
                    break;
                default:
                    std::cerr << "(GetReadyNormalCommand) Unknown type!" << std::endl;
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::SREF:
        case State::PD:
        case State::SIZE:
            std::cerr << "In unknown state" << std::endl;
            AbruptExit(__FILE__, __LINE__);
            break;
    }

    if (required_type != CommandType::SIZE) {
        if (clk >= cmd_timing_[static_cast<int>(required_type)]) {
            return Command(required_type, cmd.addr, cmd.hex_addr);
        }
    }
    return Command();
}

void BankState::UpdateNormalState(const Command &cmd) {
    switch (state_) {
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    row_hit_count_++;
                    break;
                case CommandType::GWRITE:
                    break;
                case CommandType::PRECHARGE:
                    state_ = State::CLOSED;
                    open_row_ = -1;
                    row_hit_count_ = 0;
                    break;
                case CommandType::ACTIVATE:
                case CommandType::REFRESH:
                default:
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::REFRESH:
                    if (pim_state_ == State::CLOSED) {
                        break;
                    } else {
                        AbruptExit(__FILE__, __LINE__);
                    }
                case CommandType::REFRESH_BANK:
                    break;
                case CommandType::ACTIVATE:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    break;
                case CommandType::SREF_ENTER:
                    state_ = State::SREF;
                    break;
                case CommandType::GWRITE:
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::READ_PRECHARGE:
                case CommandType::WRITE_PRECHARGE:
                case CommandType::PRECHARGE:
                case CommandType::SREF_EXIT:
                default:
                    std::cout << cmd << std::endl;
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        default:
            AbruptExit(__FILE__, __LINE__);
    }
    return;
}

void BankState::UpdatePIMState(const Command &cmd) {
    // PrintDebug("UpdatePIMState");
    switch (pim_state_) {
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::COMP:
                case CommandType::READRES:
                case CommandType::GWRITE:
                case CommandType::G_ACT:
                case CommandType::PIM_ACTIVE:
                    break;
                case CommandType::PIM_PRECHARGE:
                case CommandType::G_PRE:
                    pim_state_ = State::CLOSED;
                    pim_open_row_ = -1;
                    break;
                default:
                    std::cout << "(UpdatePIMState) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::G_ACT:
                case CommandType::PIM_ACTIVE:
                    pim_state_ = State::OPEN;
                    pim_open_row_ = cmd.Row();
                    break;
                case CommandType::GWRITE:
                case CommandType::READRES:
                case CommandType::PIM_PRECHARGE:
                case CommandType::G_PRE:
                    break;
                case CommandType::SREF_ENTER:
                case CommandType::SREF_EXIT:
                case CommandType::REFRESH:  // handle in normal
                case CommandType::REFRESH_BANK:
                case CommandType::COMP:
                default:
                    std::cout << "(UpdatePIMState) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::SREF:
            AbruptExit(__FILE__, __LINE__);
            break;
        default:
            std::cout << "(UpdatePIMState) addr: " << HexString(cmd.hex_addr)
                      << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                      << " bank:" << cmd.Bank() << " ";
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
    }
    // PrintDebug("UpdatePIMState Done");
    return;
}

void BankState::UpdateTiming(CommandType cmd_type, uint64_t time) {
    // if (cmd_type == CommandType::COMPS_READRES)
    //     PrintImportant("(UpdateTiming) COMPS_READRES", "time:", time);
    cmd_timing_[static_cast<int>(cmd_type)] = std::max(cmd_timing_[static_cast<int>(cmd_type)], time);
    return;
}

/* Return the cmd required before issuing the cmd in the current state */
Command BankState::GetReadyCommandSingle(const Command &cmd, uint64_t clk) const {
    // In case of single buffer, no usage of PIM_PRECHARGE
    CommandType required_type = CommandType::SIZE;
    // 完成Bank执行状态判断
    switch (state_) {
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    required_type = CommandType::ACTIVATE;  // 读写指令，如果Bank是关闭状态，添加Activate指令，完成激活
                    break;
                case CommandType::P_HEADER:
                case CommandType::GWRITE:
                case CommandType::READRES:
                case CommandType::PIM_ACTIVE:
                case CommandType::G_ACT:
                    required_type = cmd.cmd_type;  // 这些PIM指令可以直接执行
                    break;
                case CommandType::COMP:
                    required_type = CommandType::PIM_ACTIVE;  // 当前的Page是关闭状态，要进行COMP操作，添加PIM_Activate，完成全部激活
                    break;
                case CommandType::COMP_HASH:
                    required_type = CommandType::ACTIVATE;
                    break;
                case CommandType::REFRESH:
                    required_type = cmd.cmd_type;  // Refresh指令可以立即执行
                    break;
                default:
                    PrintWarning("(GetReadyCommandSingle) Unknown type! addr: ",
                                 HexString(cmd.hex_addr), "channel:", cmd.Channel(),
                                 "rank:", cmd.Rank(), "bg:", cmd.Bankgroup(), "bank:", cmd.Bank());
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    if (pim_lock_) {  // 设定了在PIM计算过程不能完成读写操作，目前没有使用这个逻辑进行限制
                        PrintStateAndCommand(cmd);
                        std::cerr << "Try to getReadyCommand for READ/WRITE during PIM lock" << std::endl;
                        AbruptExit(__FILE__, __LINE__);
                    }
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;
                    }
                    break;
                case CommandType::P_HEADER:
                case CommandType::GWRITE:
                case CommandType::READRES:
                case CommandType::G_ACT:
                case CommandType::PIM_ACTIVE:
                case CommandType::PIM_PRECHARGE:
                case CommandType::G_PRE:
                    required_type = cmd.cmd_type;  // 三种PIM计算，直接返回对应的类型
                    break;
                case CommandType::COMP:
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PIM_PRECHARGE;  // 后续考虑替换为 PIM_PRECHARGE
                    }
                    break;
                    // >>> gsheo
                case CommandType::COMP_HASH:
                    if (cmd.Row() == open_row_) {
                        required_type = cmd.cmd_type;
                    } else {
                        required_type = CommandType::PRECHARGE;  // 后续考虑替换为 PIM_PRECHARGE
                    }
                    break;
                case CommandType::REFRESH:
                    required_type = CommandType::PRECHARGE;
                    break;
                // <<< gsheo
                default:
                    std::cerr << "(GetReadyCommandSingle) Unknown type!" << std::endl;
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
                    break;
            }
            break;
        case State::PD:
        case State::SIZE:
        default:
            std::cerr << "In unknown state" << std::endl;
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
            break;
    }

    // 完成Bank执行时间判断
    if (required_type != CommandType::SIZE) {
        if (clk >= cmd_timing_[static_cast<int>(required_type)]) {
            return Command(required_type, cmd.addr, cmd.hex_addr, cmd.is_last_comps, cmd.num_comps);
        }
    }
    return Command();
}

void BankState::UpdateStateSingle(const Command &cmd) {
    switch (state_) {
        case State::OPEN:
            switch (cmd.cmd_type) {
                case CommandType::READ:
                case CommandType::WRITE:
                    row_hit_count_++;
                    break;
                case CommandType::P_HEADER:
                case CommandType::GWRITE:
                case CommandType::READRES:
                    break;
                case CommandType::COMP:
                case CommandType::COMP_HASH:
                    if (cmd.Row() == open_row_) {
                        row_hit_count_++;
                    }
                    break;
                case CommandType::PIM_PRECHARGE:
                case CommandType::PRECHARGE:
                    state_ = State::CLOSED;
                    open_row_ = -1;
                    row_hit_count_ = 0;
                    break;
                case CommandType::PIM_ACTIVE:  // PrintError("G_ACT in OPEN state");
                case CommandType::G_ACT:
                    break;
                case CommandType::ACTIVATE:
                case CommandType::REFRESH:
                default:
                    std::cout << "(UpdateStateSingle) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        case State::CLOSED:
            switch (cmd.cmd_type) {
                case CommandType::REFRESH:
                    break;
                case CommandType::ACTIVATE:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    break;
                case CommandType::SREF_ENTER:
                    state_ = State::SREF;
                    break;
                // >>> gsheo
                case CommandType::G_ACT:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    pim_lock_ = true;
                    pim_enter_count_++;
                    break;
                case CommandType::PIM_ACTIVE:
                    state_ = State::OPEN;
                    open_row_ = cmd.Row();
                    break;
                case CommandType::P_HEADER:
                case CommandType::GWRITE:
                case CommandType::READRES:
                    break;
                case CommandType::PIM_PRECHARGE:
                case CommandType::G_PRE:
                    break;  // 对于调用Precharge同时当前Bank关闭的情况，可以直接跳过
                case CommandType::READ:
                case CommandType::WRITE:
                case CommandType::PRECHARGE:
                case CommandType::COMP:
                default:
                    std::cout << "(UpdateStateSingle) addr: " << HexString(cmd.hex_addr)
                              << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                              << " bank:" << cmd.Bank() << " ";
                    PrintStateAndCommand(cmd);
                    AbruptExit(__FILE__, __LINE__);
            }
            break;
        default:
            std::cout << "(UpdateStateSingle) addr: " << HexString(cmd.hex_addr)
                      << " rank:" << cmd.Rank() << " bg:" << cmd.Bankgroup()
                      << " bank:" << cmd.Bank() << " ";
            PrintStateAndCommand(cmd);
            AbruptExit(__FILE__, __LINE__);
    }
}

}  // namespace dramsim3
