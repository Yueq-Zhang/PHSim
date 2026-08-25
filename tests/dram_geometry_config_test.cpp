#include "common_function.hpp"

#include <filesystem>
#include <fstream>
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
        MemConfig config(memory_config.string(), pim_config.string(),
                         output_dir.string());
        std::cout << "  PASS: " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "  FAIL: " << name << ": " << error.what() << '\n';
    }
}

void expect_invalid(const std::filesystem::path& memory_config,
                    const std::filesystem::path& pim_config,
                    const std::filesystem::path& output_dir,
                    const std::string& message_fragment,
                    const std::string& name) {
    try {
        MemConfig config(memory_config.string(), pim_config.string(),
                         output_dir.string());
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
                                        "channel_size = 3072")}};
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
