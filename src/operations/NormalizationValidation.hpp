#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phsim {

namespace detail {

inline std::string FormatTensorDimensions(
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

}  // namespace detail

inline void ValidateLayerNormInputDimensions(
    const std::vector<uint32_t>& input_dimensions,
    const std::vector<uint32_t>& normalized_dimensions,
    const std::string& operation_name, size_t input_index) {
    if (normalized_dimensions.size() != 1) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name +
            "' currently supports a one-dimensional normalized shape, got " +
            detail::FormatTensorDimensions(normalized_dimensions));
    }
    if (input_dimensions.size() != 2) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name + "' input " +
            std::to_string(input_index) +
            " must have shape [tokens, hidden_size], got " +
            detail::FormatTensorDimensions(input_dimensions));
    }
    if (input_dimensions[0] == 0 || input_dimensions[1] == 0) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name + "' input " +
            std::to_string(input_index) +
            " dimensions must be positive, got " +
            detail::FormatTensorDimensions(input_dimensions));
    }
    if (input_dimensions[1] != normalized_dimensions[0]) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name + "' input " +
            std::to_string(input_index) +
            " hidden_size does not match gamma/beta (input=" +
            detail::FormatTensorDimensions(input_dimensions) +
            ", gamma/beta=" +
            detail::FormatTensorDimensions(normalized_dimensions) + ')');
    }
}

inline uint32_t ValidateLayerNormParameterDimensions(
    const std::vector<uint32_t>& gamma_dimensions,
    const std::vector<uint32_t>& beta_dimensions,
    const std::string& operation_name) {
    if (gamma_dimensions.size() != 1) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name +
            "' gamma must have one-dimensional shape [hidden_size], got " +
            detail::FormatTensorDimensions(gamma_dimensions));
    }
    if (gamma_dimensions != beta_dimensions) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name +
            "' gamma and beta shapes must match (gamma=" +
            detail::FormatTensorDimensions(gamma_dimensions) + ", beta=" +
            detail::FormatTensorDimensions(beta_dimensions) + ')');
    }

    if (gamma_dimensions[0] == 0) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name +
            "' gamma and beta hidden_size must be positive");
    }
    return gamma_dimensions[0];
}

inline uint32_t ValidateLayerNormParameterSizeBytes(
    uint32_t element_count, uint32_t precision_bytes,
    const std::string& operation_name) {
    if (precision_bytes == 0) {
        throw std::invalid_argument(
            "LayerNorm '" + operation_name +
            "' gamma/beta precision must be positive");
    }
    if (element_count >
        std::numeric_limits<uint32_t>::max() / precision_bytes) {
        throw std::overflow_error(
            "LayerNorm '" + operation_name +
            "' gamma/beta byte size exceeds uint32_t");
    }
    return element_count * precision_bytes;
}

}  // namespace phsim
