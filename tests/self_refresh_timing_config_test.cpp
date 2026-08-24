#include "common_function.hpp"
#include "channel_state.h"
#include "configuration.h"
#include "timing.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect_true(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  PASS: " << name << '\n';
        return;
    }
    ++failures;
    std::cerr << "  FAIL: " << name << '\n';
}

template <typename Function>
void expect_throws(Function&& function, const std::string& name) {
    try {
        function();
        ++failures;
        std::cerr << "  FAIL: " << name << " did not throw\n";
    } catch (const std::exception&) {
        std::cout << "  PASS: " << name << '\n';
    }
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    output << text;
}

std::string replace_once(std::string text, const std::string& original,
                         const std::string& replacement) {
    const auto position = text.find(original);
    if (position == std::string::npos) {
        throw std::runtime_error("Missing fixture text: " + original);
    }
    text.replace(position, original.size(), replacement);
    return text;
}

int timing_delay(const std::vector<std::pair<dramsim3::CommandType, int>>& constraints,
                 dramsim3::CommandType target) {
    for (const auto& [command, delay] : constraints) {
        if (command == target) {
            return delay;
        }
    }
    return -1;
}

void check_parsers_and_timing(const std::filesystem::path& memory_config,
                              const std::filesystem::path& pim_config,
                              const std::filesystem::path& output_dir,
                              int expected_tckesr) {
    MemConfig mem_config(memory_config.string(), pim_config.string(),
                         output_dir.string());
    expect_true(mem_config.tCKESR == expected_tckesr,
                "MemConfig reads tCKESR=" + std::to_string(expected_tckesr));

    dramsim3::Config newton_config(memory_config.string(), output_dir.string());
    expect_true(newton_config.tCKESR == expected_tckesr,
                "NewtonSim Config reads tCKESR=" +
                    std::to_string(expected_tckesr));

    // Timing currently also derives PIM constraints from address-layout globals
    // that the full simulator initializes before constructing NewtonSim.
    MyAddressAllocator::ranks = newton_config.ranks;
    MyAddressAllocator::bankgroups = newton_config.bankgroups;
    MyAddressAllocator::banks = newton_config.banks_per_group;
    MyAddressAllocator::BL_num_per_row =
        newton_config.columns / newton_config.BL;
    MyAddressAllocator::dram_burst_size =
        newton_config.BL * newton_config.bus_width / 8;
    MyAddressAllocator::AddrGranularity_Hash_Bytes = 1024;

    dramsim3::Timing timing(newton_config);
    const auto sref_enter = static_cast<int>(dramsim3::CommandType::SREF_ENTER);
    const auto sref_exit = static_cast<int>(dramsim3::CommandType::SREF_EXIT);
    expect_true(
        timing_delay(timing.same_rank[sref_enter],
                     dramsim3::CommandType::SREF_EXIT) == expected_tckesr,
        "SREF_ENTER to SREF_EXIT uses tCKESR");
    expect_true(
        timing_delay(timing.same_rank[sref_exit],
                     dramsim3::CommandType::ACTIVATE) == newton_config.tXS,
        "SREF_EXIT to ACTIVATE uses tXS");

    dramsim3::ChannelState channel_state(0, newton_config, timing);
    const dramsim3::Address rank_address(-1, 0, -1, -1, -1, -1);
    const dramsim3::Command enter_command(
        dramsim3::CommandType::SREF_ENTER, rank_address, 0);
    constexpr uint64_t enter_cycle = 100;
    expect_true(
        channel_state.GetReadyCommand(enter_command, enter_cycle).cmd_type ==
            dramsim3::CommandType::SREF_ENTER,
        "SREF_ENTER is issuable for an idle rank");
    channel_state.UpdateTimingAndStates(enter_command, enter_cycle);
    expect_true(channel_state.IsRankSelfRefreshing(0),
                "SREF_ENTER moves the rank into self-refresh");

    const dramsim3::Command exit_command(
        dramsim3::CommandType::SREF_EXIT, rank_address, 0);
    expect_true(
        !channel_state
             .GetReadyCommand(exit_command,
                              enter_cycle + expected_tckesr - 1)
             .IsValid(),
        "SREF_EXIT is blocked before tCKESR");
    expect_true(
        channel_state
                .GetReadyCommand(exit_command,
                                 enter_cycle + expected_tckesr)
                .cmd_type == dramsim3::CommandType::SREF_EXIT,
        "SREF_EXIT becomes ready at tCKESR");
    const uint64_t exit_cycle = enter_cycle + expected_tckesr;
    channel_state.UpdateTimingAndStates(exit_command, exit_cycle);
    expect_true(!channel_state.IsRankSelfRefreshing(0),
                "SREF_EXIT returns the rank to the closed state");

    const dramsim3::Address bank_address(0, 0, 0, 0, 0, 0);
    const dramsim3::Command read_command(
        dramsim3::CommandType::READ, bank_address, 0);
    expect_true(
        !channel_state
             .GetReadyCommand(read_command,
                              exit_cycle + newton_config.tXS - 1)
             .IsValid(),
        "ACTIVATE is blocked before tXS after SREF_EXIT");
    expect_true(
        channel_state
                .GetReadyCommand(read_command,
                                 exit_cycle + newton_config.tXS)
                .cmd_type == dramsim3::CommandType::ACTIVATE,
        "ACTIVATE becomes ready at tXS after SREF_EXIT");
}

}  // namespace

int main() {
    const std::filesystem::path source_dir = PH_SIM_SOURCE_DIR;
    const auto fixture = source_dir / "tests/fixtures/smoke/memory_config.ini";
    const auto pim_config = source_dir / "tests/fixtures/smoke/pim_config.json";
    const auto temp_dir = std::filesystem::temp_directory_path() /
                          "phsim-self-refresh-timing-config-test";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    try {
        const std::string canonical_text = read_text(fixture);
        const std::string legacy_text = replace_once(
            canonical_text, "tCKESR = 7", "tCKSRE = 7");
        const std::string duplicate_text = replace_once(
            canonical_text, "tCKESR = 7", "tCKESR = 7\ntCKSRE = 7");
        const std::string conflict_text = replace_once(
            canonical_text, "tCKESR = 7", "tCKESR = 7\ntCKSRE = 9");
        const std::string default_text = replace_once(
            canonical_text, "tCKESR = 7", "");
        const std::string dual_buffer_text = replace_once(
            canonical_text, "pim_type = SINGLE", "pim_type = DUAL");

        const auto canonical_path = temp_dir / "canonical.ini";
        const auto legacy_path = temp_dir / "legacy.ini";
        const auto duplicate_path = temp_dir / "duplicate.ini";
        const auto conflict_path = temp_dir / "conflict.ini";
        const auto default_path = temp_dir / "default.ini";
        const auto dual_buffer_path = temp_dir / "dual-buffer.ini";
        write_text(canonical_path, canonical_text);
        write_text(legacy_path, legacy_text);
        write_text(duplicate_path, duplicate_text);
        write_text(conflict_path, conflict_text);
        write_text(default_path, default_text);
        write_text(dual_buffer_path, dual_buffer_text);

        check_parsers_and_timing(canonical_path, pim_config, temp_dir, 7);
        check_parsers_and_timing(legacy_path, pim_config, temp_dir, 7);
        check_parsers_and_timing(duplicate_path, pim_config, temp_dir, 7);
        check_parsers_and_timing(default_path, pim_config, temp_dir, 12);
        check_parsers_and_timing(dual_buffer_path, pim_config, temp_dir, 7);

        expect_throws(
            [&] {
                MemConfig config(conflict_path.string(), pim_config.string(),
                                 temp_dir.string());
            },
            "MemConfig rejects conflicting tCKESR/tCKSRE");
        expect_throws(
            [&] { dramsim3::Config config(conflict_path.string(), temp_dir.string()); },
            "NewtonSim Config rejects conflicting tCKESR/tCKSRE");

        ValidateDramBackendCapabilities(DramMode::CYCLE_ACCURATE, true);
        ValidateDramBackendCapabilities(DramMode::EVENT_DRIVEN, false);
        expect_throws(
            [] { ValidateDramBackendCapabilities(DramMode::EVENT_DRIVEN, true); },
            "EventDriven rejects enabled self-refresh");
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: unexpected exception: " << error.what() << '\n';
    }

    std::filesystem::remove_all(temp_dir);
    if (failures == 0) {
        std::cout << "RESULT PASS: self-refresh configuration and timing checks\n";
        return 0;
    }
    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
