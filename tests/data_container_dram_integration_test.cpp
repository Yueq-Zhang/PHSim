#include "DRAM/Dram.h"
#include "DRAM/DramDataContainer.h"
#include "DRAM/IDramBackend.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef PH_SIM_SOURCE_DIR
#define PH_SIM_SOURCE_DIR "."
#endif

namespace DRAMDataContainer {
std::unique_ptr<DramDataContainer> storage;

void init(const SysConfig& config) {
    storage = std::make_unique<DramDataContainer>(config);
}
void cleanup() { storage.reset(); }
void data_write(DramDataContainer::BurstData data, uint32_t ch, uint32_t ra,
                uint32_t bg, uint32_t ba, uint32_t row, uint32_t col) {
    storage->write_burst(std::move(data), ch, ra, bg, ba, row, col);
}
DramDataContainer::BurstData data_read(uint32_t ch, uint32_t ra, uint32_t bg,
                                       uint32_t ba, uint32_t row,
                                       uint32_t col) {
    return storage->read_burst(ch, ra, bg, ba, row, col);
}
std::vector<uint8_t> flatten_burst(
    const DramDataContainer::BurstData& data) {
    return storage->flatten_burst(data);
}
}  // namespace DRAMDataContainer

namespace {

using Burst = std::vector<std::vector<uint8_t>>;

constexpr uint32_t kMaxDramCyclesPerRequest = 10000;
int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "  FAIL: " << message << '\n';
    }
}

void begin_case(const char* name) {
    std::cout << "CASE " << name << '\n';
}

std::string source_path(const std::string& relative) {
    return std::string(PH_SIM_SOURCE_DIR) + "/" + relative;
}

void configure_system(SysConfig& config) {
    const std::string memory_config =
        source_path("configs/Cases/Nano/Nano_2xLPDDR5.ini");
    const std::string pim_config =
        source_path("configs/Cases/Nano/Nano_pim_config.json");

    config.memory_config_path_ = memory_config;
    config.log_dir = "/tmp";
    config.mem_config = MemConfig(memory_config, pim_config, "/tmp");
    config.dram_channels = config.mem_config.channels;
    config.dram_data_container_enable = true;
    config.dram_freq = 800;
    config.pim_input_buffer_size = config.mem_config.input_buffer_size;
    config.pim_output_buffer_size = config.mem_config.output_buffer_size;

    MyAddressAllocator::configure_address_decoder(config.mem_config);
    PIM_Parameters::init(config);
    DRAMDataContainer::cleanup();
    DRAMDataContainer::init(config);
}

Burst burst_from_flat(const std::vector<uint8_t>& flat,
                      const SysConfig& config) {
    const uint32_t burst_length = config.mem_config.BL;
    const uint32_t bytes_per_column = config.mem_config.bus_width / 8;
    if (flat.size() != burst_length * bytes_per_column) {
        throw std::invalid_argument("flat payload size does not match one DRAM burst");
    }

    Burst burst(burst_length, std::vector<uint8_t>(bytes_per_column, 0));
    for (uint32_t column = 0; column < burst_length; ++column) {
        for (uint32_t byte = 0; byte < bytes_per_column; ++byte) {
            burst[column][byte] = flat[column * bytes_per_column + byte];
        }
    }
    return burst;
}

std::vector<uint8_t> make_payload(uint8_t base, const SysConfig& config) {
    const uint32_t size = config.mem_config.BL * config.mem_config.bus_width / 8;
    std::vector<uint8_t> payload(size);
    for (uint32_t i = 0; i < size; ++i) {
        payload[i] = static_cast<uint8_t>(base + i);
    }
    return payload;
}

MemoryAccess* submit_and_wait(PIM& dram, uint32_t channel,
                              MemoryAccess& request) {
    dram.cycle();
    uint32_t waited = 0;
    while (dram.is_full(channel, &request)) {
        if (++waited > kMaxDramCyclesPerRequest) {
            throw std::runtime_error("DRAM request remained full beyond cycle limit");
        }
        dram.cycle();
    }

    dram.push(channel, &request);
    for (uint32_t cycle = 0; cycle < kMaxDramCyclesPerRequest; ++cycle) {
        dram.cycle();
        if (!dram.is_empty(channel)) {
            return dram.top(channel);
        }
    }
    throw std::runtime_error("DRAM response exceeded cycle limit");
}

MemoryAccess make_request(addr_type address, MemoryAccessType type,
                          const SysConfig& config,
                          std::vector<uint8_t> data = {}) {
    MemoryAccess request{};
    request.id = 1;
    request.logical_dram_address = address;
    request.dram_address = address;
    request.size = config.mem_config.BL * config.mem_config.bus_width / 8;
    request.req_type = type;
    request.request = true;
    request.core_id = 0;
    request.mem_id = MyAddressAllocator::get_channel_index(address);
    request.data = std::move(data);
    request.data_ready = false;
    return request;
}

addr_type make_active_path_address(uint32_t channel, uint32_t row) {
    return MyAddressAllocator::make_address_by_index(
        0, 0, 0, row, 0, channel);
}

void consume_response(PIM& dram, uint32_t channel,
                      MemoryAccess* response, MemoryAccess* expected) {
    expect(response == expected,
           "NewtonSim should return the original MemoryAccess object");
    expect(!response->request, "completed request should be marked as a response");
    expect(response->data_ready,
           "DataContainer hook should mark the completed response data-ready");
    expect(response->dram_finish_cycle >= response->dram_enter_cycle,
           "DRAM completion cycle should not precede its entry cycle");
    dram.pop(channel);
}

void test_seeded_read(PIM& dram, const SysConfig& config) {
    begin_case("cycle_accurate_seeded_read");
    const addr_type address = make_active_path_address(0, 1);
    const std::vector<uint8_t> seeded = make_payload(0x10, config);
    DRAMDataContainer::data_write(
        burst_from_flat(seeded, config), 0, 0, 0, 0, 1, 0);

    MemoryAccess read =
        make_request(address, MemoryAccessType::READ, config);
    MemoryAccess* response = submit_and_wait(dram, 0, read);
    expect(response->data == seeded,
           "CycleAccurate READ should return the seeded DataContainer bytes");
    consume_response(dram, 0, response, &read);
}

void test_write_then_read(PIM& dram, const SysConfig& config) {
    begin_case("cycle_accurate_write_then_read");
    const addr_type address = make_active_path_address(0, 2);
    const std::vector<uint8_t> written = make_payload(0x40, config);

    MemoryAccess write =
        make_request(address, MemoryAccessType::WRITE, config, written);
    MemoryAccess* write_response = submit_and_wait(dram, 0, write);
    consume_response(dram, 0, write_response, &write);

    const Burst stored = DRAMDataContainer::data_read(0, 0, 0, 0, 2, 0);
    expect(DRAMDataContainer::flatten_burst(stored) == written,
           "completed WRITE should update the addressed DataContainer burst");

    MemoryAccess read =
        make_request(address, MemoryAccessType::READ, config);
    MemoryAccess* read_response = submit_and_wait(dram, 0, read);
    expect(read_response->data == written,
           "a READ after WRITE should observe the new payload");
    consume_response(dram, 0, read_response, &read);
}

void test_channel_isolation(PIM& dram, const SysConfig& config) {
    begin_case("cycle_accurate_channel_isolation");
    const addr_type address_ch0 = make_active_path_address(0, 3);
    const addr_type address_ch1 = make_active_path_address(1, 3);
    const std::vector<uint8_t> payload_ch0 = make_payload(0x20, config);
    const std::vector<uint8_t> payload_ch1 = make_payload(0x80, config);

    DRAMDataContainer::data_write(
        burst_from_flat(payload_ch0, config), 0, 0, 0, 0, 3, 0);
    DRAMDataContainer::data_write(
        burst_from_flat(payload_ch1, config), 1, 0, 0, 0, 3, 0);

    MemoryAccess read_ch1 =
        make_request(address_ch1, MemoryAccessType::READ, config);
    MemoryAccess* response = submit_and_wait(dram, 1, read_ch1);
    expect(response->data == payload_ch1,
           "channel 1 READ should not return channel 0 data");
    consume_response(dram, 1, response, &read_ch1);

    const Burst ch0 = DRAMDataContainer::data_read(0, 0, 0, 0, 3, 0);
    expect(DRAMDataContainer::flatten_burst(ch0) == payload_ch0,
           "channel 1 access should leave channel 0 storage unchanged");
    expect(MyAddressAllocator::get_channel_index(address_ch0) == 0,
           "channel 0 generated address should decode as channel 0");
    expect(MyAddressAllocator::get_channel_index(address_ch1) == 1,
           "channel 1 generated address should decode as channel 1");
}

void test_empty_write_payload_is_ignored(PIM& dram, const SysConfig& config) {
    begin_case("empty_write_payload_is_ignored");
    const addr_type address = make_active_path_address(0, 4);
    const std::vector<uint8_t> original = make_payload(0x60, config);
    DRAMDataContainer::data_write(
        burst_from_flat(original, config), 0, 0, 0, 0, 4, 0);

    MemoryAccess write =
        make_request(address, MemoryAccessType::WRITE, config);
    MemoryAccess* response = submit_and_wait(dram, 0, write);
    consume_response(dram, 0, response, &write);

    const Burst stored = DRAMDataContainer::data_read(0, 0, 0, 0, 4, 0);
    expect(DRAMDataContainer::flatten_burst(stored) == original,
           "WRITE without data should leave the DataContainer unchanged");
}

}  // namespace

int main() {
    try {
        SysConfig config;
        configure_system(config);
        PIM dram(config, DRAMDataContainer::storage.get());

        test_seeded_read(dram, config);
        test_write_then_read(dram, config);
        test_channel_isolation(dram, config);
        test_empty_write_payload_is_ignored(dram, config);

        DRAMDataContainer::cleanup();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "UNCAUGHT NON-STANDARD EXCEPTION\n";
    }

    if (failures == 0) {
        std::cout << "RESULT PASS: 4 CycleAccurate DataContainer integration cases\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " assertion(s) failed\n";
    return 1;
}
static_assert(std::is_base_of<IDramBackend, PIM>::value,
              "PIM must implement the common DRAM backend data path");
