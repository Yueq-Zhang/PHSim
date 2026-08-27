#include "common_function.hpp"
#include "configuration.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

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

void expect_valid(const std::filesystem::path& memory_config,
                  const std::filesystem::path& pim_config,
                  const std::filesystem::path& output_dir,
                  const std::string& name) {
    try {
        MemConfig outer(memory_config.string(), pim_config.string(),
                        output_dir.string());
        dramsim3::Config newton(memory_config.string(), output_dir.string());

        const auto expect_equal = [&](uint64_t outer_value,
                                      uint64_t newton_value,
                                      const std::string& field) {
            if (outer_value != newton_value) {
                throw std::runtime_error(
                    name + " mismatch for " + field + ": MemConfig=" +
                    std::to_string(outer_value) + ", NewtonSim=" +
                    std::to_string(newton_value));
            }
        };
        expect_equal(outer.channel_size, newton.channel_size, "channel_size");
        expect_equal(outer.channels, newton.channels, "channels");
        expect_equal(outer.ranks, newton.ranks, "ranks");
        expect_equal(outer.banks, newton.banks, "banks");
        expect_equal(outer.bankgroups, newton.bankgroups, "bankgroups");
        expect_equal(outer.banks_per_group, newton.banks_per_group,
                     "banks_per_group");
        expect_equal(outer.rows, newton.rows, "rows");
        expect_equal(outer.columns, newton.columns, "columns");
        expect_equal(outer.device_width, newton.device_width, "device_width");
        expect_equal(outer.bus_width, newton.bus_width, "bus_width");
        expect_equal(outer.devices_per_rank, newton.devices_per_rank,
                     "devices_per_rank");
        expect_equal(outer.BL, newton.BL, "BL");
        expect_equal(outer.request_size_bytes, newton.request_size_bytes,
                     "request_size_bytes");
        expect_equal(outer.shift_bits, newton.shift_bits, "shift_bits");
        expect_equal(outer.ch_pos, newton.ch_pos, "ch_pos");
        expect_equal(outer.ra_pos, newton.ra_pos, "ra_pos");
        expect_equal(outer.bg_pos, newton.bg_pos, "bg_pos");
        expect_equal(outer.ba_pos, newton.ba_pos, "ba_pos");
        expect_equal(outer.ro_pos, newton.ro_pos, "ro_pos");
        expect_equal(outer.co_pos, newton.co_pos, "co_pos");
        expect_equal(outer.ch_mask, newton.ch_mask, "ch_mask");
        expect_equal(outer.ra_mask, newton.ra_mask, "ra_mask");
        expect_equal(outer.bg_mask, newton.bg_mask, "bg_mask");
        expect_equal(outer.ba_mask, newton.ba_mask, "ba_mask");
        expect_equal(outer.ro_mask, newton.ro_mask, "ro_mask");
        expect_equal(outer.co_mask, newton.co_mask, "co_mask");
        std::cout << "  PASS: " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: " << name << ": " << error.what() << '\n';
    }
}

void expect_invalid_call(const std::function<void()>& construct,
                         const std::string& message_fragment,
                         const std::string& name) {
    try {
        construct();
        ++failures;
        std::cerr << "  FAIL: " << name << " did not throw\n";
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(message_fragment) !=
            std::string::npos) {
            std::cout << "  PASS: " << name << '\n';
        } else {
            ++failures;
            std::cerr << "  FAIL: " << name << " threw unexpected message: "
                      << error.what() << '\n';
        }
    }
}

void expect_invalid(const std::filesystem::path& memory_config,
                    const std::filesystem::path& pim_config,
                    const std::filesystem::path& output_dir,
                    const std::string& message_fragment,
                    const std::string& name) {
    expect_invalid_call(
        [&] {
            MemConfig config(memory_config.string(), pim_config.string(),
                             output_dir.string());
        },
        message_fragment, name + " in MemConfig");
    expect_invalid_call(
        [&] {
            dramsim3::Config config(memory_config.string(),
                                    output_dir.string());
        },
        message_fragment, name + " in NewtonSim Config");
}

}  // namespace

int main() {
    const std::filesystem::path source_dir = PH_SIM_SOURCE_DIR;
    const auto smoke_memory =
        source_dir / "tests/fixtures/smoke/memory_config.ini";
    const auto smoke_pim = source_dir / "tests/fixtures/smoke/pim_config.json";
    const auto temp_dir = std::filesystem::temp_directory_path() /
                          "phsim-dram-geometry-config-test";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    try {
        expect_valid(smoke_memory, smoke_pim, temp_dir,
                     "smoke configuration has a valid DRAM layout");

        const auto collapsed_memory =
            source_dir / "configs/memory_config/GDDR6_8Gb_x16.ini";
        const auto generic_pim =
            source_dir / "configs/pim_config/pim_config.json";
        expect_valid(collapsed_memory, generic_pim, temp_dir,
                     "disabled bankgroups use the shared collapsed layout");
        MemConfig collapsed(collapsed_memory.string(), generic_pim.string(),
                            temp_dir.string());
        if (collapsed.bankgroups != 1 || collapsed.banks_per_group != 16 ||
            collapsed.banks != 16) {
            ++failures;
            std::cerr << "  FAIL: bankgroup_enable=false should produce "
                         "1 group with 16 banks\n";
        } else {
            std::cout << "  PASS: bankgroup_enable=false collapses 4x4 "
                         "banks to 1x16\n";
        }

        const std::vector<std::string> case_names = {
            "Base", "Small", "Tiny", "Mobile", "Nano", "Server"};
        const std::vector<std::string> memory_names = {
            "Base_8xGDDR6.ini", "Small_4xGDDR6.ini", "Tiny_4xLPDDR5.ini",
            "Mobile_2xLPDDR5.ini", "Nano_2xLPDDR5.ini", "Server_8xHBM2.ini"};
        for (std::size_t index = 0; index < case_names.size(); ++index) {
            const auto case_dir = source_dir / "configs/Cases" / case_names[index];
            expect_valid(case_dir / memory_names[index],
                         case_dir / (case_names[index] + "_pim_config.json"),
                         temp_dir, case_names[index] + " DRAM geometry");
        }

        const std::string fixture_text = read_text(smoke_memory);
        const std::vector<std::pair<std::string, std::string>> variants = {
            {"channels.ini", replace_once(fixture_text, "channels = 2",
                                           "channels = 3")},
            {"bus-width.ini", replace_once(fixture_text, "bus_width = 16",
                                            "bus_width = 10")},
            {"device-width.ini", replace_once(fixture_text,
                                               "device_width = 16",
                                               "device_width = 0")},
            {"columns.ini", replace_once(fixture_text, "columns = 1024",
                                          "columns = 1000")},
            {"ranks.ini", replace_once(fixture_text, "channel_size = 1024",
                                        "channel_size = 3072")},
            {"mapping-duplicate.ini",
             replace_once(fixture_text,
                          "address_mapping = rorababgchco",
                          "address_mapping = rorabachchco")},
            {"mapping-unknown.ini",
             replace_once(fixture_text,
                          "address_mapping = rorababgchco",
                          "address_mapping = rorabaxxchco")},
            {"mapping-short.ini",
             replace_once(fixture_text,
                          "address_mapping = rorababgchco",
                          "address_mapping = rorababgch")}};
        for (const auto& [filename, contents] : variants) {
            write_text(temp_dir / filename, contents);
        }

        std::string overflow_text = replace_once(
            fixture_text, "rows = 32768", "rows = 1073741824");
        overflow_text = replace_once(overflow_text, "columns = 1024",
                                     "columns = 1073741824");
        overflow_text = replace_once(overflow_text, "device_width = 16",
                                     "device_width = 1073741824");
        overflow_text = replace_once(overflow_text, "bus_width = 16",
                                     "bus_width = 1073741824");
        write_text(temp_dir / "capacity-overflow.ini", overflow_text);

        expect_invalid(temp_dir / "channels.ini", smoke_pim, temp_dir,
                       "channels must be a positive power of two",
                       "reject non-power-of-two channels");
        expect_invalid(temp_dir / "bus-width.ini", smoke_pim, temp_dir,
                       "bus_width must be a positive multiple of 8 bits",
                       "reject a non-byte-aligned bus width");
        expect_invalid(temp_dir / "device-width.ini", smoke_pim, temp_dir,
                       "device_width must be greater than zero",
                       "reject zero device width before division");
        expect_invalid(temp_dir / "columns.ini", smoke_pim, temp_dir,
                       "columns must be a positive power of two",
                       "reject non-power-of-two columns");
        expect_invalid(temp_dir / "ranks.ini", smoke_pim, temp_dir,
                       "derived ranks must be a positive power of two",
                       "reject a non-power-of-two derived rank count");
        expect_invalid(temp_dir / "capacity-overflow.ini", smoke_pim,
                       temp_dir, "overflows uint64_t",
                       "reject capacity multiplication overflow");
        expect_invalid(temp_dir / "mapping-duplicate.ini", smoke_pim,
                       temp_dir, "duplicate address field 'ch'",
                       "reject duplicate address fields");
        expect_invalid(temp_dir / "mapping-unknown.ini", smoke_pim,
                       temp_dir, "unknown address field 'xx'",
                       "reject unknown address fields");
        expect_invalid(temp_dir / "mapping-short.ini", smoke_pim,
                       temp_dir, "exactly 6 two-character fields",
                       "reject incomplete address mappings");
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: unexpected exception: " << error.what() << '\n';
    }

    std::filesystem::remove_all(temp_dir);
    if (failures == 0) {
        std::cout << "RESULT PASS: DRAM geometry validation\n";
        return 0;
    }
    std::cerr << "RESULT FAIL: " << failures << " check(s) failed\n";
    return 1;
}
