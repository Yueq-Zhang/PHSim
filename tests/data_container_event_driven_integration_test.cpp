#include "DRAM/EventDrivenDram.h"
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

constexpr uint32_t kMaxDramCycles = 10000;
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
    config.output_path_ = "/tmp";
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
        throw std::invalid_argument(
            "flat payload size does not match one DRAM burst");
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

addr_type make_vm_mapped_address(const SysConfig& config,
                                 addr_type logical_address) {
    const uint64_t total_bytes =
        static_cast<uint64_t>(config.mem_config.channels) *
        config.mem_config.channel_size * 1024 * 1024;
    TwoLevelDeterministicMapper mapper(
        total_bytes / TwoLevelDeterministicMapper::PAGE_SIZE_BYTES);
    mapper.configure(config.mem_config);
    return mapper.map(logical_address);
}

class EventDrivenHarness {
public:
    EventDrivenHarness(EventDrivenDram& dram, uint32_t channels)
        : dram_(dram), cycles_(channels, 1) {}

    void push(MemoryAccess& request) {
        const uint32_t channel = request.mem_id;
        request.dram_enter_cycle = cycles_.at(channel);
        uint32_t waited = 0;
        while (dram_.is_full(channel, &request)) {
            advance(channel);
            if (++waited > kMaxDramCycles) {
                throw std::runtime_error(
                    "EventDriven request remained full beyond cycle limit");
            }
            request.dram_enter_cycle = cycles_.at(channel);
        }
        dram_.push(channel, &request);
    }

    MemoryAccess* wait_one(uint32_t channel) {
        for (uint32_t waited = 0; waited < kMaxDramCycles; ++waited) {
            if (!dram_.is_empty(channel)) {
                return dram_.top(channel);
            }
            advance(channel);
        }
        throw std::runtime_error(
            "EventDriven response exceeded cycle limit");
    }

    void pop(uint32_t channel) {
        dram_.pop(channel);
    }

private:
    void advance(uint32_t channel) {
        ++cycles_.at(channel);
        dram_.schedule_pending_operation(channel, cycles_.at(channel));
    }

    EventDrivenDram& dram_;
    std::vector<cycle_type> cycles_;
};

void consume_response(EventDrivenHarness& harness, uint32_t channel,
                      MemoryAccess* response, MemoryAccess* expected) {
    expect(response == expected,
           "EventDriven DRAM should return the original MemoryAccess object");
    expect(!response->request,
           "completed EventDriven request should be marked as a response");
    expect(response->data_ready,
           "EventDriven DataContainer hook should mark the response data-ready");
    expect(response->dram_finish_cycle >= response->dram_enter_cycle,
           "EventDriven completion should not precede request entry");
    harness.pop(channel);
}

void test_seeded_read(EventDrivenHarness& harness, const SysConfig& config) {
    begin_case("event_driven_seeded_read");
    const addr_type address = make_active_path_address(0, 1);
    const std::vector<uint8_t> seeded = make_payload(0x10, config);
    DRAMDataContainer::data_write(
        burst_from_flat(seeded, config), 0, 0, 0, 0, 1, 0);

    MemoryAccess read = make_request(address, MemoryAccessType::READ, config);
    harness.push(read);
    MemoryAccess* response = harness.wait_one(0);
    expect(response->data == seeded,
           "EventDriven READ should return seeded DataContainer bytes");
    consume_response(harness, 0, response, &read);
}

void test_write_then_read(EventDrivenHarness& harness,
                          const SysConfig& config) {
    begin_case("event_driven_write_then_read");
    const addr_type address = make_active_path_address(0, 2);
    const std::vector<uint8_t> written = make_payload(0x40, config);

    MemoryAccess write =
        make_request(address, MemoryAccessType::WRITE, config, written);
    harness.push(write);
    MemoryAccess* write_response = harness.wait_one(0);
    consume_response(harness, 0, write_response, &write);

    const Burst stored = DRAMDataContainer::data_read(0, 0, 0, 0, 2, 0);
    expect(DRAMDataContainer::flatten_burst(stored) == written,
           "EventDriven logical WRITE completion should update DataContainer");

    MemoryAccess read = make_request(address, MemoryAccessType::READ, config);
    harness.push(read);
    MemoryAccess* read_response = harness.wait_one(0);
    expect(read_response->data == written,
           "EventDriven READ after WRITE should observe the new payload");
    consume_response(harness, 0, read_response, &read);
}

void test_merged_reads(EventDrivenHarness& harness, const SysConfig& config) {
    begin_case("event_driven_merged_reads");
    const addr_type address = make_active_path_address(0, 3);
    const std::vector<uint8_t> seeded = make_payload(0x70, config);
    DRAMDataContainer::data_write(
        burst_from_flat(seeded, config), 0, 0, 0, 0, 3, 0);

    MemoryAccess read_a = make_request(address, MemoryAccessType::READ, config);
    MemoryAccess read_b = make_request(address, MemoryAccessType::READ, config);
    harness.push(read_a);
    harness.push(read_b);

    MemoryAccess* response_a = harness.wait_one(0);
    expect(response_a == &read_a,
           "first merged READ should retain its original request identity");
    expect(response_a->data == seeded,
           "first merged READ should receive the full DataContainer payload");
    consume_response(harness, 0, response_a, &read_a);

    MemoryAccess* response_b = harness.wait_one(0);
    expect(response_b == &read_b,
           "second merged READ should retain its original request identity");
    expect(response_b->data == seeded,
           "second merged READ should receive an independent payload copy");
    consume_response(harness, 0, response_b, &read_b);
}

void test_write_buffer_forwarding(EventDrivenHarness& harness,
                                  const SysConfig& config) {
    begin_case("event_driven_write_buffer_forwarding");
    const addr_type address = make_active_path_address(0, 4);
    const std::vector<uint8_t> written = make_payload(0x90, config);

    MemoryAccess write =
        make_request(address, MemoryAccessType::WRITE, config, written);
    MemoryAccess read = make_request(address, MemoryAccessType::READ, config);
    harness.push(write);
    harness.push(read);

    MemoryAccess* write_response = harness.wait_one(0);
    consume_response(harness, 0, write_response, &write);
    MemoryAccess* read_response = harness.wait_one(0);
    expect(read_response->data == written,
           "forwarded READ should observe the preceding logical WRITE payload");
    consume_response(harness, 0, read_response, &read);
}

void test_merged_writes_last_writer_wins(EventDrivenHarness& harness,
                                         const SysConfig& config) {
    begin_case("event_driven_merged_writes_last_writer_wins");
    const addr_type address = make_active_path_address(0, 5);
    const std::vector<uint8_t> first = make_payload(0x20, config);
    const std::vector<uint8_t> second = make_payload(0xC0, config);

    MemoryAccess write_a =
        make_request(address, MemoryAccessType::WRITE, config, first);
    MemoryAccess write_b =
        make_request(address, MemoryAccessType::WRITE, config, second);
    harness.push(write_a);
    harness.push(write_b);

    consume_response(harness, 0, harness.wait_one(0), &write_a);
    consume_response(harness, 0, harness.wait_one(0), &write_b);

    const Burst stored = DRAMDataContainer::data_read(0, 0, 0, 0, 5, 0);
    expect(DRAMDataContainer::flatten_burst(stored) == second,
           "last logical WRITE in a merged group should win");
}

void test_empty_write_payload_is_ignored(EventDrivenHarness& harness,
                                         const SysConfig& config) {
    begin_case("event_driven_empty_write_payload_is_ignored");
    const addr_type address = make_active_path_address(0, 6);
    const std::vector<uint8_t> original = make_payload(0x50, config);
    DRAMDataContainer::data_write(
        burst_from_flat(original, config), 0, 0, 0, 0, 6, 0);

    MemoryAccess write = make_request(address, MemoryAccessType::WRITE, config);
    harness.push(write);
    consume_response(harness, 0, harness.wait_one(0), &write);

    const Burst stored = DRAMDataContainer::data_read(0, 0, 0, 0, 6, 0);
    expect(DRAMDataContainer::flatten_burst(stored) == original,
           "EventDriven WRITE without payload should leave data unchanged");
}

void test_channel_isolation(EventDrivenHarness& harness,
                            const SysConfig& config) {
    begin_case("event_driven_channel_isolation");
    const addr_type address_ch0 = make_active_path_address(0, 7);
    const addr_type address_ch1 = make_active_path_address(1, 7);
    const std::vector<uint8_t> payload_ch0 = make_payload(0x30, config);
    const std::vector<uint8_t> payload_ch1 = make_payload(0xA0, config);
    DRAMDataContainer::data_write(
        burst_from_flat(payload_ch0, config), 0, 0, 0, 0, 7, 0);
    DRAMDataContainer::data_write(
        burst_from_flat(payload_ch1, config), 1, 0, 0, 0, 7, 0);

    MemoryAccess read_ch1 =
        make_request(address_ch1, MemoryAccessType::READ, config);
    harness.push(read_ch1);
    MemoryAccess* response = harness.wait_one(1);
    expect(response->data == payload_ch1,
           "channel 1 READ should not return channel 0 data");
    consume_response(harness, 1, response, &read_ch1);

    expect(DRAMDataContainer::flatten_burst(
               DRAMDataContainer::data_read(0, 0, 0, 0, 7, 0)) == payload_ch0,
           "channel 1 access should leave channel 0 data unchanged");
    expect(MyAddressAllocator::get_channel_index(address_ch0) == 0,
           "generated channel 0 address should decode as channel 0");
    expect(MyAddressAllocator::get_channel_index(address_ch1) == 1,
           "generated channel 1 address should decode as channel 1");
}

void test_repeated_top_is_idempotent(EventDrivenHarness& harness,
                                     const SysConfig& config) {
    begin_case("event_driven_repeated_top_is_idempotent");
    const addr_type address = make_active_path_address(0, 8);
    const std::vector<uint8_t> original = make_payload(0x15, config);
    const std::vector<uint8_t> replacement = make_payload(0xD0, config);
    DRAMDataContainer::data_write(
        burst_from_flat(original, config), 0, 0, 0, 0, 8, 0);

    MemoryAccess read = make_request(address, MemoryAccessType::READ, config);
    harness.push(read);
    MemoryAccess* first_top = harness.wait_one(0);
    expect(first_top->data == original,
           "first top should snapshot the completed READ payload");

    DRAMDataContainer::data_write(
        burst_from_flat(replacement, config), 0, 0, 0, 0, 8, 0);
    MemoryAccess* second_top = harness.wait_one(0);
    expect(second_top == first_top,
           "repeated top before pop should return the same response object");
    expect(second_top->data == original,
           "data_ready should prevent repeated top from re-reading storage");
    consume_response(harness, 0, second_top, &read);
}

void test_short_write_zero_fills_tail(EventDrivenHarness& harness,
                                      const SysConfig& config) {
    begin_case("event_driven_short_write_zero_fills_tail");
    const addr_type address = make_active_path_address(0, 9);
    const std::vector<uint8_t> short_payload = {0xDE, 0xAD, 0xBE};
    const size_t full_size =
        config.mem_config.BL * config.mem_config.bus_width / 8;
    std::vector<uint8_t> expected(full_size, 0);
    expected[0] = 0xDE;
    expected[1] = 0xAD;
    expected[2] = 0xBE;

    MemoryAccess write = make_request(
        address, MemoryAccessType::WRITE, config, short_payload);
    harness.push(write);
    consume_response(harness, 0, harness.wait_one(0), &write);

    expect(DRAMDataContainer::flatten_burst(
               DRAMDataContainer::data_read(0, 0, 0, 0, 9, 0)) == expected,
           "short EventDriven WRITE should zero-fill the remaining burst bytes");
}

void test_pim_payload_sequence(EventDrivenHarness& harness,
                               const SysConfig& config) {
    begin_case("event_driven_pim_payload_sequence");
    const addr_type address = make_active_path_address(0, 10);
    const std::vector<uint8_t> input = make_payload(0x21, config);
    const std::vector<uint8_t> result_first = make_payload(0xB1, config);
    const std::vector<uint8_t> result_second = make_payload(0xD1, config);
    std::vector<uint8_t> result = result_first;
    result.insert(result.end(), result_second.begin(), result_second.end());

    MemoryAccess header =
        make_request(address, MemoryAccessType::P_HEADER, config);
    harness.push(header);
    consume_response(harness, 0, harness.wait_one(0), &header);

    MemoryAccess gwrite =
        make_request(address, MemoryAccessType::GWRITE, config, input);
    harness.push(gwrite);
    consume_response(harness, 0, harness.wait_one(0), &gwrite);
    expect(DRAMDataContainer::storage->pim_input_payload_bytes(0) ==
               input.size(),
           "EventDriven GWRITE should materialize one input burst");

    MemoryAccess comp =
        make_request(address, MemoryAccessType::COMP, config, result);
    harness.push(comp);
    consume_response(harness, 0, harness.wait_one(0), &comp);
    expect(DRAMDataContainer::storage->pim_output_payload_bytes(0) ==
               result.size(),
           "EventDriven COMP should publish the supplied functional result");

    MemoryAccess readres =
        make_request(address, MemoryAccessType::READRES, config);
    harness.push(readres);
    MemoryAccess* response = harness.wait_one(0);
    expect(response->data == result_first,
           "EventDriven READRES should return the first PIM output burst");
    consume_response(harness, 0, response, &readres);

    MemoryAccess second_readres =
        make_request(address, MemoryAccessType::READRES, config);
    harness.push(second_readres);
    response = harness.wait_one(0);
    expect(response->data == result_second,
           "EventDriven READRES should advance to the second output burst");
    consume_response(harness, 0, response, &second_readres);

    MemoryAccess reset =
        make_request(address, MemoryAccessType::P_HEADER, config);
    harness.push(reset);
    consume_response(harness, 0, harness.wait_one(0), &reset);
    expect(DRAMDataContainer::storage->pim_input_payload_bytes(0) == 0 &&
               DRAMDataContainer::storage->pim_output_payload_bytes(0) == 0,
           "a new EventDriven P_HEADER should clear prior PIM payloads");
    expect(DRAMDataContainer::storage->peak_pim_input_payload_bytes(0) ==
               input.size() &&
               DRAMDataContainer::storage->peak_pim_output_payload_bytes(0) ==
                   result.size(),
           "EventDriven PIM peaks should survive a P_HEADER phase reset");

    MemoryAccess fallback_gwrite =
        make_request(address, MemoryAccessType::GWRITE, config, input);
    harness.push(fallback_gwrite);
    consume_response(harness, 0, harness.wait_one(0), &fallback_gwrite);
    MemoryAccess fallback_comp =
        make_request(address, MemoryAccessType::COMP, config);
    harness.push(fallback_comp);
    consume_response(harness, 0, harness.wait_one(0), &fallback_comp);
    MemoryAccess fallback_readres =
        make_request(address, MemoryAccessType::READRES, config);
    harness.push(fallback_readres);
    response = harness.wait_one(0);
    expect(response->data == input,
           "COMP without a functional result should preserve opaque input bytes");
    consume_response(harness, 0, response, &fallback_readres);
}

void test_vm_mapped_pim_payload(EventDrivenHarness& harness,
                                const SysConfig& config) {
    begin_case("event_driven_vm_mapped_pim_payload");
    const uint64_t burst_bytes =
        static_cast<uint64_t>(config.mem_config.BL) *
        config.mem_config.bus_width / 8;
    const addr_type logical =
        TwoLevelDeterministicMapper::HASH_UNIT_BYTES / burst_bytes;
    const addr_type physical = make_vm_mapped_address(config, logical);
    const uint32_t channel =
        MyAddressAllocator::get_channel_index(physical);
    const std::vector<uint8_t> input = make_payload(0x32, config);
    const std::vector<uint8_t> result = make_payload(0xC2, config);

    expect(physical != logical,
           "VM diagnostic address should be changed by the hash mapper");

    MemoryAccess header =
        make_request(physical, MemoryAccessType::P_HEADER, config);
    header.logical_dram_address = logical;
    harness.push(header);
    consume_response(harness, channel, harness.wait_one(channel), &header);
    MemoryAccess gwrite =
        make_request(physical, MemoryAccessType::GWRITE, config, input);
    gwrite.logical_dram_address = logical;
    harness.push(gwrite);
    consume_response(harness, channel, harness.wait_one(channel), &gwrite);
    MemoryAccess comp =
        make_request(physical, MemoryAccessType::COMP_HASH, config, result);
    comp.logical_dram_address = logical;
    harness.push(comp);
    consume_response(harness, channel, harness.wait_one(channel), &comp);
    MemoryAccess readres =
        make_request(physical, MemoryAccessType::READRES, config);
    readres.logical_dram_address = logical;
    harness.push(readres);
    MemoryAccess* response = harness.wait_one(channel);
    expect(response->data == result,
           "EventDriven READRES should follow the VM-mapped channel state");
    expect(response->logical_dram_address == logical &&
               response->dram_address == physical,
           "DataContainer should preserve logical and mapped physical addresses");
    consume_response(harness, channel, response, &readres);
}

}  // namespace

int main() {
    try {
        SysConfig config;
        configure_system(config);
        {
            EventDrivenDram dram(config, DRAMDataContainer::storage.get());
            EventDrivenHarness harness(dram, config.dram_channels);

            test_seeded_read(harness, config);
            test_write_then_read(harness, config);
            test_merged_reads(harness, config);
            test_write_buffer_forwarding(harness, config);
            test_merged_writes_last_writer_wins(harness, config);
            test_empty_write_payload_is_ignored(harness, config);
            test_channel_isolation(harness, config);
            test_repeated_top_is_idempotent(harness, config);
            test_short_write_zero_fills_tail(harness, config);
        }

        // Logical WRITE responses can precede the final physical write-buffer
        // drain. Use a fresh backend so those intentionally outstanding
        // normal events cannot contaminate the PIM row-state checks.
        DRAMDataContainer::cleanup();
        DRAMDataContainer::init(config);
        {
            EventDrivenDram dram(config, DRAMDataContainer::storage.get());
            EventDrivenHarness harness(dram, config.dram_channels);
            test_pim_payload_sequence(harness, config);
            test_vm_mapped_pim_payload(harness, config);
        }

        DRAMDataContainer::cleanup();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "UNCAUGHT NON-STANDARD EXCEPTION\n";
    }

    if (failures == 0) {
        std::cout
            << "RESULT PASS: 11 EventDriven DataContainer integration cases\n";
        return 0;
    }

    std::cerr << "RESULT FAIL: " << failures << " assertion(s) failed\n";
    return 1;
}
static_assert(std::is_base_of<IDramBackend, EventDrivenDram>::value,
              "EventDrivenDram must implement the common DRAM backend data path");
