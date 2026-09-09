#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phsim {

inline uint64_t AvailablePingPongSramBytes(uint32_t spad_size_kib) {
    return static_cast<uint64_t>(spad_size_kib) * 1024ULL / 2ULL;
}

inline std::string FormatSramTileDimensions(
    const std::vector<uint32_t>& dimensions) {
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << dimensions[index];
    }
    output << ']';
    return output.str();
}

[[noreturn]] inline void ThrowSramTileCannotFit(
    const std::string& operation_name, uint64_t required_bytes,
    uint64_t available_bytes, const std::vector<uint32_t>& tile_dimensions,
    const std::string& detail = {}) {
    std::ostringstream message;
    message << operation_name
            << " minimum tile cannot fit in SRAM: requires "
            << required_bytes << " bytes, but one ping-pong SRAM half provides "
            << available_bytes << " bytes; tile="
            << FormatSramTileDimensions(tile_dimensions);
    if (!detail.empty()) {
        message << "; " << detail;
    }
    throw std::invalid_argument(message.str());
}

inline void ValidateSramTileFits(
    uint64_t required_bytes, uint64_t available_bytes,
    const std::string& operation_name,
    const std::vector<uint32_t>& tile_dimensions,
    const std::string& detail = {}) {
    if (required_bytes > available_bytes) {
        ThrowSramTileCannotFit(operation_name, required_bytes,
                               available_bytes, tile_dimensions, detail);
    }
}

inline bool CanDoubleSramTileCount(uint32_t tile_count) {
    return tile_count <= std::numeric_limits<uint32_t>::max() / 2U;
}

inline uint32_t CalculateSramTileDimension(uint32_t total_size,
                                           uint64_t tile_count) {
    if (tile_count == 0) {
        throw std::logic_error("SRAM tile count must be positive");
    }
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(total_size) + tile_count - 1ULL) /
        tile_count);
}

inline uint32_t CalculateAlignedSramTileDimension(
    uint32_t total_size, uint64_t tile_count, uint32_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument("SRAM tile alignment must be positive");
    }
    const uint64_t unaligned =
        CalculateSramTileDimension(total_size, tile_count);
    const uint64_t aligned =
        ((unaligned + alignment - 1ULL) / alignment) * alignment;
    if (aligned > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("aligned SRAM tile dimension exceeds uint32_t");
    }
    return static_cast<uint32_t>(aligned);
}

inline void DoubleSramTileCountOrThrow(
    uint32_t& tile_count, const std::string& operation_name,
    size_t dimension_index) {
    if (!CanDoubleSramTileCount(tile_count)) {
        throw std::overflow_error(
            operation_name + " tile count overflow on dimension " +
            std::to_string(dimension_index) +
            " while reducing SRAM usage");
    }
    tile_count *= 2U;
}

inline uint32_t HalveSramTileDimensionOrThrow(
    uint32_t current_size, const std::string& operation_name,
    uint64_t required_bytes, uint64_t available_bytes,
    const std::vector<uint32_t>& tile_dimensions,
    const std::string& detail = {}) {
    if (current_size <= 1U) {
        ThrowSramTileCannotFit(operation_name, required_bytes,
                               available_bytes, tile_dimensions, detail);
    }
    const uint32_t next_size = current_size / 2U + current_size % 2U;
    assert(next_size < current_size);
    return next_size;
}

inline void HalveSramTileDimensionAndDoubleCount(
    std::vector<uint32_t>& inner_loop, size_t inner_dimension,
    std::vector<uint32_t>& outer_loop, size_t outer_dimension,
    const std::string& operation_name, uint64_t required_bytes,
    uint64_t available_bytes, const std::string& detail = {}) {
    if (inner_dimension >= inner_loop.size() ||
        outer_dimension >= outer_loop.size()) {
        throw std::logic_error(operation_name +
                               " has an invalid SRAM tiling dimension");
    }

    const uint32_t next_size = HalveSramTileDimensionOrThrow(
        inner_loop[inner_dimension], operation_name, required_bytes,
        available_bytes, inner_loop, detail);
    DoubleSramTileCountOrThrow(outer_loop[outer_dimension], operation_name,
                               outer_dimension);
    inner_loop[inner_dimension] = next_size;
}

inline size_t SelectShrinkableSramDimension(
    const std::vector<uint32_t>& current_dimensions,
    const std::vector<uint32_t>& next_dimensions,
    const std::vector<size_t>& priority,
    const std::vector<uint32_t>& outer_loop,
    const std::string& operation_name, uint64_t required_bytes,
    uint64_t available_bytes, const std::string& detail = {}) {
    if (current_dimensions.size() != next_dimensions.size() ||
        current_dimensions.size() != outer_loop.size() ||
        priority.size() != current_dimensions.size()) {
        throw std::logic_error(operation_name +
                               " has inconsistent SRAM tiling dimensions");
    }

    size_t selected = current_dimensions.size();
    for (const size_t index : priority) {
        if (index >= current_dimensions.size()) {
            throw std::logic_error(operation_name +
                                   " has an invalid SRAM tiling priority");
        }
        if (next_dimensions[index] >= current_dimensions[index] ||
            !CanDoubleSramTileCount(outer_loop[index])) {
            continue;
        }
        if (selected == current_dimensions.size() ||
            current_dimensions[index] > current_dimensions[selected]) {
            selected = index;
        }
    }

    if (selected == current_dimensions.size()) {
        ThrowSramTileCannotFit(operation_name, required_bytes,
                               available_bytes, current_dimensions, detail);
    }
    return selected;
}

inline size_t SelectShrinkableSramDimensionWithCounts(
    const std::vector<uint32_t>& current_dimensions,
    const std::vector<uint32_t>& next_dimensions,
    const std::vector<size_t>& priority,
    const std::vector<uint32_t>& current_outer_loop,
    const std::vector<uint32_t>& next_outer_loop,
    const std::string& operation_name, uint64_t required_bytes,
    uint64_t available_bytes, const std::string& detail = {}) {
    if (current_dimensions.size() != next_dimensions.size() ||
        current_dimensions.size() != current_outer_loop.size() ||
        current_dimensions.size() != next_outer_loop.size() ||
        priority.size() != current_dimensions.size()) {
        throw std::logic_error(operation_name +
                               " has inconsistent SRAM tiling dimensions");
    }

    size_t selected = current_dimensions.size();
    for (const size_t index : priority) {
        if (index >= current_dimensions.size()) {
            throw std::logic_error(operation_name +
                                   " has an invalid SRAM tiling priority");
        }
        if (next_dimensions[index] >= current_dimensions[index] ||
            next_outer_loop[index] <= current_outer_loop[index]) {
            continue;
        }
        if (selected == current_dimensions.size() ||
            current_dimensions[index] > current_dimensions[selected]) {
            selected = index;
        }
    }

    if (selected == current_dimensions.size()) {
        ThrowSramTileCannotFit(operation_name, required_bytes,
                               available_bytes, current_dimensions, detail);
    }
    return selected;
}

inline void ApplySramDimensionSplit(
    std::vector<uint32_t>& current_dimensions,
    const std::vector<uint32_t>& next_dimensions,
    std::vector<uint32_t>& outer_loop, size_t dimension_index,
    const std::string& operation_name) {
    if (dimension_index >= current_dimensions.size() ||
        dimension_index >= next_dimensions.size() ||
        dimension_index >= outer_loop.size() ||
        next_dimensions[dimension_index] >=
            current_dimensions[dimension_index]) {
        throw std::logic_error(operation_name +
                               " attempted a non-progressing SRAM split");
    }
    DoubleSramTileCountOrThrow(outer_loop[dimension_index], operation_name,
                               dimension_index);
    current_dimensions[dimension_index] = next_dimensions[dimension_index];
}

inline void ApplySramDimensionSplitWithCount(
    std::vector<uint32_t>& current_dimensions,
    const std::vector<uint32_t>& next_dimensions,
    std::vector<uint32_t>& current_outer_loop,
    const std::vector<uint32_t>& next_outer_loop, size_t dimension_index,
    const std::string& operation_name) {
    if (dimension_index >= current_dimensions.size() ||
        dimension_index >= next_dimensions.size() ||
        dimension_index >= current_outer_loop.size() ||
        dimension_index >= next_outer_loop.size() ||
        next_dimensions[dimension_index] >= current_dimensions[dimension_index] ||
        next_outer_loop[dimension_index] <=
            current_outer_loop[dimension_index]) {
        throw std::logic_error(operation_name +
                               " attempted a non-progressing SRAM split");
    }
    current_outer_loop[dimension_index] = next_outer_loop[dimension_index];
    current_dimensions[dimension_index] = next_dimensions[dimension_index];
}

}  // namespace phsim
