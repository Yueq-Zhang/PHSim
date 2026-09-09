#include "Interconnect/MyInterconnect.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void configure_simple(SysConfig& config, uint32_t input_capacity,
                      uint32_t output_capacity) {
    config.icnt_type = IcntType::SIMPLE;
    config.icnt_freq = 1000;
    config.dram_freq = 1000;
    config.core_freq = 1000;
    config.icnt_latency = 1;
    config.icnt_ctrl_size = 8;
    config.icnt_input_buffer_size = input_capacity;
    config.icnt_output_buffer_size = output_capacity;
    config.num_cores = 1;
    config.dram_channels = 1;
}

std::vector<MemoryAccess> make_accesses(size_t count) {
    std::vector<MemoryAccess> accesses(count);
    for (size_t index = 0; index < count; ++index) {
        accesses[index].id = static_cast<uint32_t>(index);
        accesses[index].size = 16;
        accesses[index].req_type = MemoryAccessType::READ;
        accesses[index].request = true;
    }
    return accesses;
}

void test_simple_unbounded_default() {
    SysConfig& config = Config::system_config;
    configure_simple(config, 0, 0);
    MyInterconnect interconnect(config);
    auto accesses = make_accesses(64);
    require(!interconnect.running(),
            "an empty Simple interconnect must be idle");
    for (auto& access : accesses) {
        require(!interconnect.is_full(0, &access),
                "zero Simple input capacity must remain unbounded");
        interconnect.push(0, 1, &access);
    }
    require(!interconnect.is_full(0, &accesses.front()),
            "unbounded Simple input must not become full");
    require(interconnect.running(),
            "queued Simple input packets must keep the interconnect running");
}

void test_simple_finite_backpressure() {
    SysConfig& config = Config::system_config;
    configure_simple(config, 2, 1);
    MyInterconnect interconnect(config);
    auto accesses = make_accesses(3);
    require(!interconnect.running(),
            "an empty finite Simple interconnect must be idle");

    interconnect.push(0, 1, &accesses[0]);
    interconnect.push(0, 1, &accesses[1]);
    require(interconnect.running(),
            "queued finite Simple input packets must be reported as active");
    require(interconnect.is_full(0, &accesses[2]),
            "finite Simple input must report full at capacity");
    bool rejected = false;
    try {
        interconnect.push(0, 1, &accesses[2]);
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    require(rejected, "Simple push must reject input overflow");

    interconnect.cycle();
    interconnect.cycle();
    require(interconnect.running(),
            "a buffered Simple output packet must keep the interconnect active");
    require(interconnect.top(1) == &accesses[0],
            "Simple interconnect must preserve packet order");
    interconnect.push(0, 1, &accesses[2]);
    require(interconnect.is_full(0, &accesses[2]),
            "Simple input must refill while the destination is blocked");
    interconnect.cycle();
    require(interconnect.is_full(0, &accesses[2]),
            "full Simple output must backpressure the input queue");

    interconnect.pop(1);
    interconnect.cycle();
    require(interconnect.top(1) == &accesses[1],
            "Simple output must resume after one packet is consumed");
    interconnect.pop(1);
    interconnect.cycle();
    require(interconnect.top(1) == &accesses[2],
            "Simple backpressure must not drop packets");
    interconnect.pop(1);
    require(!interconnect.running(),
            "a fully drained Simple interconnect must return to idle");
}

void drain_booksim(booksim2::Interconnect& interconnect, int destination,
                   size_t expected_packets,
                   std::unordered_set<const void*>& received) {
    for (int cycle = 0;
         cycle < 10000 && received.size() < expected_packets; ++cycle) {
        interconnect.run();
        while (!interconnect.is_empty(destination, 0)) {
            received.insert(interconnect.top(destination, 0));
            interconnect.pop(destination, 0);
        }
    }
    require(received.size() == expected_packets,
            "BookSim drain timed out or lost packets");
}

void test_booksim_unbounded_default(const std::filesystem::path& root) {
    booksim2::Interconnect interconnect(
        (root / "configs/booksim2_configs/mesh_2x2_unbounded.icnt")
            .string(),
        4);
    std::vector<int> packets(32);
    for (auto& packet : packets) {
        require(!interconnect.is_full(0, 0, 16),
                "omitted BookSim input capacity must remain unbounded");
        interconnect.push(&packet, 0, 0, 16,
                          booksim2::Interconnect::Type::READ, 0, 3);
    }
    require(!interconnect.is_full(0, 0, 16),
            "unbounded BookSim input must not become full");
    std::unordered_set<const void*> received;
    drain_booksim(interconnect, 3, packets.size(), received);
}

void test_booksim_finite_backpressure(const std::filesystem::path& root) {
    booksim2::Interconnect interconnect(
        (root / "configs/booksim2_configs/mesh_2x2_backpressure.icnt")
            .string(),
        4);
    std::vector<int> packets(12);
    interconnect.push(&packets[0], 0, 0, 16,
                      booksim2::Interconnect::Type::READ, 0, 3);
    interconnect.push(&packets[1], 0, 0, 16,
                      booksim2::Interconnect::Type::READ, 0, 3);
    require(interconnect.is_full(0, 0, 16),
            "finite BookSim input must report full at capacity");
    bool rejected = false;
    try {
        interconnect.push(&packets[2], 0, 0, 16,
                          booksim2::Interconnect::Type::READ, 0, 3);
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    require(rejected, "BookSim push must reject input overflow");

    size_t injected = 2;
    for (int cycle = 0; cycle < 300; ++cycle) {
        interconnect.run();
        if (injected < packets.size() &&
            !interconnect.is_full(0, 0, 16)) {
            interconnect.push(&packets[injected], 0, 0, 16,
                              booksim2::Interconnect::Type::READ, 0, 3);
            ++injected;
        }
    }

    std::unordered_set<const void*> received;
    size_t initially_buffered = 0;
    while (!interconnect.is_empty(3, 0)) {
        received.insert(interconnect.top(3, 0));
        interconnect.pop(3, 0);
        ++initially_buffered;
    }
    require(initially_buffered > 0 && initially_buffered <= 4,
            "finite BookSim boundary must hold at most one packet per request VC");

    for (int cycle = 0;
         cycle < 10000 &&
         (injected < packets.size() || received.size() < packets.size());
         ++cycle) {
        interconnect.run();
        if (injected < packets.size() &&
            !interconnect.is_full(0, 0, 16)) {
            interconnect.push(&packets[injected], 0, 0, 16,
                              booksim2::Interconnect::Type::READ, 0, 3);
            ++injected;
        }
        while (!interconnect.is_empty(3, 0)) {
            received.insert(interconnect.top(3, 0));
            interconnect.pop(3, 0);
        }
    }
    require(injected == packets.size(),
            "BookSim input did not resume after backpressure cleared");
    require(received.size() == packets.size(),
            "BookSim finite buffers dropped or duplicated packets");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: interconnect_backpressure_test "
                     "simple|booksim-unbounded|booksim-finite\n";
        return 2;
    }
    const std::filesystem::path source_root = PH_SIM_SOURCE_DIR;
    const std::string mode = argv[1];
    if (mode == "simple") {
        test_simple_unbounded_default();
        test_simple_finite_backpressure();
    } else if (mode == "booksim-unbounded") {
        test_booksim_unbounded_default(source_root);
    } else if (mode == "booksim-finite") {
        test_booksim_finite_backpressure(source_root);
    } else {
        std::cerr << "unknown test mode: " << mode << '\n';
        return 2;
    }
    std::cout << "RESULT PASS: interconnect capacity and backpressure\n";
    return 0;
}
