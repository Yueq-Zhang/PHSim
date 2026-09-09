#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace phsim::clock {

using cycle_type = std::uint64_t;

inline void validate_frequency(std::uint32_t frequency) {
    if (frequency == 0) {
        throw std::invalid_argument("Clock frequency must be greater than zero");
    }
}

// Compare lhs_cycle / lhs_frequency with rhs_cycle / rhs_frequency without
// converting either timestamp to floating point and without overflowing a
// 64-bit cross product.
inline int compare_edges(cycle_type lhs_cycle, std::uint32_t lhs_frequency,
                         cycle_type rhs_cycle, std::uint32_t rhs_frequency) {
    validate_frequency(lhs_frequency);
    validate_frequency(rhs_frequency);

    const cycle_type lhs_whole = lhs_cycle / lhs_frequency;
    const cycle_type rhs_whole = rhs_cycle / rhs_frequency;
    if (lhs_whole < rhs_whole) return -1;
    if (lhs_whole > rhs_whole) return 1;

    const std::uint64_t lhs_remainder = lhs_cycle % lhs_frequency;
    const std::uint64_t rhs_remainder = rhs_cycle % rhs_frequency;
    const std::uint64_t lhs_fraction =
        lhs_remainder * static_cast<std::uint64_t>(rhs_frequency);
    const std::uint64_t rhs_fraction =
        rhs_remainder * static_cast<std::uint64_t>(lhs_frequency);
    if (lhs_fraction < rhs_fraction) return -1;
    if (lhs_fraction > rhs_fraction) return 1;
    return 0;
}

struct Edge {
    cycle_type cycle;
    std::uint32_t frequency;
    std::uint32_t mask;
};

inline std::uint32_t select_next_mask(const std::array<Edge, 3>& edges) {
    Edge earliest = edges.front();
    for (const Edge& edge : edges) {
        if (compare_edges(edge.cycle, edge.frequency,
                          earliest.cycle, earliest.frequency) < 0) {
            earliest = edge;
        }
    }

    std::uint32_t mask = 0;
    for (const Edge& edge : edges) {
        if (compare_edges(edge.cycle, edge.frequency,
                          earliest.cycle, earliest.frequency) == 0) {
            mask |= edge.mask;
        }
    }
    return mask;
}

inline cycle_type scale_cycles(cycle_type source_cycles,
                               std::uint32_t source_frequency,
                               std::uint32_t destination_frequency,
                               bool round_up) {
    validate_frequency(source_frequency);
    validate_frequency(destination_frequency);

    const cycle_type whole = source_cycles / source_frequency;
    const std::uint64_t remainder = source_cycles % source_frequency;
    const std::uint64_t fractional_numerator =
        remainder * static_cast<std::uint64_t>(destination_frequency);
    cycle_type fractional = fractional_numerator / source_frequency;
    if (round_up && fractional_numerator % source_frequency != 0) {
        ++fractional;
    }

    constexpr cycle_type max_cycle =
        std::numeric_limits<cycle_type>::max();
    if (whole > (max_cycle - fractional) / destination_frequency) {
        throw std::overflow_error("Scaled clock cycle exceeds uint64_t range");
    }
    return whole * destination_frequency + fractional;
}

inline cycle_type scale_cycles_floor(cycle_type source_cycles,
                                     std::uint32_t source_frequency,
                                     std::uint32_t destination_frequency) {
    return scale_cycles(source_cycles, source_frequency,
                        destination_frequency, false);
}

inline cycle_type scale_cycles_ceil(cycle_type source_cycles,
                                    std::uint32_t source_frequency,
                                    std::uint32_t destination_frequency) {
    return scale_cycles(source_cycles, source_frequency,
                        destination_frequency, true);
}

}  // namespace phsim::clock
