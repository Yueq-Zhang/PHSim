#include "clock_math.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr std::uint32_t kCoreMask = 1U << 1;
constexpr std::uint32_t kDramMask = 1U << 2;
constexpr std::uint32_t kIcntMask = 1U << 3;

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_exact_edge_selection() {
    using phsim::clock::Edge;
    using phsim::clock::select_next_mask;

    expect(select_next_mask({{{0, 1000, kCoreMask},
                              {0, 2000, kDramMask},
                              {0, 1500, kIcntMask}}}) ==
               (kCoreMask | kDramMask | kIcntMask),
           "Initial clock edges must be simultaneous");

    // 1/1000 and 2/2000 are the same physical time. ICNT is later.
    expect(select_next_mask({{{1, 1000, kCoreMask},
                              {2, 2000, kDramMask},
                              {2, 1500, kIcntMask}}}) ==
               (kCoreMask | kDramMask),
           "Coincident Core and DRAM edges must share a timestamp");
}

void test_cycles_beyond_double_integer_precision() {
    using phsim::clock::compare_edges;
    using phsim::clock::select_next_mask;

    constexpr std::uint64_t large = 9007199254740993ULL;  // 2^53 + 1
    expect(compare_edges(large, 1000, large + 1, 1000) < 0,
           "Adjacent cycles above 2^53 must remain ordered");
    expect(select_next_mask({{{large, 1000, kCoreMask},
                              {large + 1, 1000, kDramMask},
                              {large + 2, 1000, kIcntMask}}}) == kCoreMask,
           "Large adjacent timestamps must not collapse into a tie");

    // The three edges below are coincident at a timestamp beyond the point
    // where double can represent every integer cycle. Their 1000:800:400 MHz
    // relation must remain exact regardless of simulation length.
    constexpr std::uint64_t long_core_cycle = 9007199254740995ULL;
    constexpr std::uint64_t long_epoch = long_core_cycle / 5;
    expect(select_next_mask({{{long_core_cycle, 1000, kCoreMask},
                              {long_epoch * 4, 800, kDramMask},
                              {long_epoch * 2, 400, kIcntMask}}}) ==
               (kCoreMask | kDramMask | kIcntMask),
           "Long-run cross-domain coincidence must remain exact");
}

void test_exact_frequency_scaling() {
    using phsim::clock::scale_cycles_ceil;
    using phsim::clock::scale_cycles_floor;

    expect(scale_cycles_floor(5, 3, 2) == 3,
           "Floor clock scaling is incorrect");
    expect(scale_cycles_ceil(5, 3, 2) == 4,
           "Ceiling clock scaling is incorrect");
    expect(scale_cycles_floor(9007199254740993ULL, 1000, 2000) ==
               18014398509481986ULL,
           "Large exact clock scaling is incorrect");

    // Conversion is based on the absolute timestamp. Repeated one-cycle
    // truncation would incorrectly produce zero for this example.
    expect(scale_cycles_floor(5, 3, 2) >
               5 * scale_cycles_floor(1, 3, 2),
           "Absolute scaling must preserve fractional phase across jumps");
}

void test_invalid_and_overflow_inputs() {
    bool invalid_frequency = false;
    try {
        (void)phsim::clock::scale_cycles_floor(1, 0, 1);
    } catch (const std::invalid_argument&) {
        invalid_frequency = true;
    }
    expect(invalid_frequency, "Zero clock frequency must be rejected");

    bool overflow = false;
    try {
        (void)phsim::clock::scale_cycles_floor(
            std::numeric_limits<std::uint64_t>::max(), 1, 2);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    expect(overflow, "Clock scaling overflow must be rejected");
}

}  // namespace

int main() {
    test_exact_edge_selection();
    test_cycles_beyond_double_integer_precision();
    test_exact_frequency_scaling();
    test_invalid_and_overflow_inputs();
    return 0;
}
