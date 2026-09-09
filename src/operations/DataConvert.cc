#include "DataConvert.h"
#include "SramTilingValidation.hpp"

namespace {
bool layout_has_physical_channel_index(AllocationScheme scheme) {
    switch (scheme) {
        case AllocationScheme::IANUS:
        case AllocationScheme::DASH:
            return true;
        case AllocationScheme::NPU:
        case AllocationScheme::NeuPIM:
        case AllocationScheme::AttAcc:
        case AllocationScheme::AttenPIM:
            return false;
    }
    return false;
}
}

DataConvert::DataConvert(std::string name, AllocationScheme src_scheme, AllocationScheme dst_scheme)
    : Operation(std::move(name)), _src_scheme(src_scheme), _dst_scheme(dst_scheme) {}

std::vector<Ptr<MyTensor>> DataConvert::get_my_outputs(std::vector<Ptr<MyTensor>> inputs,
                                                       TensorType output_tensor_type) {
    assert(inputs.size() == 1);
    assert(inputs[0]->_tensor_type == TensorType::WGT);
    assert(inputs[0]->get_dims().size() == 2);
    assert(output_tensor_type == TensorType::WGT);

    _batch_size = 1;
    _my_inputs = std::move(inputs);
    _my_outputs.resize(1);
    set_as_parent_tensor(_my_inputs);

    _my_outputs[0] = std::make_shared<MyTensor>(
        _name + "_output", _my_inputs[0]->get_dims(), output_tensor_type, false, _dst_scheme);

    calculate_my_loops();
    initialize_my_tiles();
    spdlog::debug("DataConvert operation {} converts layout {} -> {}", _name,
                 static_cast<int>(_src_scheme), static_cast<int>(_dst_scheme));
    return _my_outputs;
}

void DataConvert::calculate_my_loops() {
    auto dims = _my_inputs[0]->get_dims();
    const uint32_t M = dims[0];
    const uint32_t N = dims[1];

    _inner_loop.resize(2, 1);
    _outer_loop.resize(2, 1);
    _inner_loop[0] = std::min(M, _config.core_height);
    _inner_loop[1] = std::min(N, _config.core_width);

    const uint64_t available_sram_bytes =
        phsim::AvailablePingPongSramBytes(_config.spad_size);
    while (sram_size_needed() > available_sram_bytes && _inner_loop[0] > 1) {
        _inner_loop[0] = phsim::HalveSramTileDimensionOrThrow(
            _inner_loop[0], "DataConvert '" + _name + "'",
            sram_size_needed(), available_sram_bytes, _inner_loop);
    }
    while (sram_size_needed() > available_sram_bytes && _inner_loop[1] > 1) {
        _inner_loop[1] = phsim::HalveSramTileDimensionOrThrow(
            _inner_loop[1], "DataConvert '" + _name + "'",
            sram_size_needed(), available_sram_bytes, _inner_loop);
    }
    phsim::ValidateSramTileFits(
        sram_size_needed(), available_sram_bytes,
        "DataConvert '" + _name + "'", _inner_loop);

    _outer_loop[0] = std::ceil(static_cast<double>(M) / _inner_loop[0]);
    _outer_loop[1] = std::ceil(static_cast<double>(N) / _inner_loop[1]);
    spdlog::debug("DataConvert inner loop: {}, outer loop: {}", _inner_loop, _outer_loop);
}

void DataConvert::initialize_my_tiles() {
    int core_id = -1;
    for (uint32_t M = 0; M < _outer_loop[0]; ++M) {
        for (uint32_t N = 0; N < _outer_loop[1]; ++N) {
            core_id = (core_id + 1) % _config.num_cores;
            _tiles.push_back(initialize_my_instructions(M, N));
            _tiles.back().core_id = core_id;
        }
    }
}

Tile DataConvert::initialize_my_instructions(uint32_t M, uint32_t N) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .K = M,
        .accum = false,
    };

    auto dims = _my_inputs[0]->get_dims();
    const uint32_t row_begin = M * _inner_loop[0];
    const uint32_t row_end = std::min(row_begin + _inner_loop[0], dims[0]);
    const uint32_t col_begin = N * _inner_loop[1];
    const uint32_t col_end = std::min(col_begin + _inner_loop[1], dims[1]);

    std::vector<std::vector<uint32_t>> indexes;
    indexes.reserve((row_end - row_begin) * (col_end - col_begin));
    for (uint32_t row = row_begin; row < row_end; ++row) {
        for (uint32_t col = col_begin; col < col_end; ++col) {
            indexes.push_back({row, col});
        }
    }

    auto src_addrs = _my_inputs[0]->generate_addrs_based_on_indexes(indexes, _src_scheme);
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = SPAD_BASE,
        .size = static_cast<uint32_t>(indexes.size()) * _my_inputs[0]->_precision,
        .src_addrs = std::move(src_addrs),
        .operand_id = _INPUT_OPERAND,
        .per_ch_inst = layout_has_physical_channel_index(_src_scheme),
    });

    tile.instructions.push_back(Instruction{
        .opcode = Opcode::DATA_CONVERT,
        .dest_addr = ACCUM_SPAD_BASE,
        .size = static_cast<uint32_t>(indexes.size()),
        .src_addrs = std::vector<addr_type>{SPAD_BASE},
    });

    auto dst_addrs = _my_outputs[0]->generate_addrs_based_on_indexes(indexes, _dst_scheme);
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVOUT,
        .dest_addr = ACCUM_SPAD_BASE,
        .size = static_cast<uint32_t>(indexes.size()) * _my_outputs[0]->_precision,
        .src_addrs = std::move(dst_addrs),
        .operand_id = _OUTPUT_OPERAND,
        .per_ch_inst = layout_has_physical_channel_index(_dst_scheme),
    });

    return tile;
}

uint64_t DataConvert::sram_size_needed() {
    return 2ULL * _inner_loop[0] * _inner_loop[1] *
           MyAddressAllocator::precision_weight;
}
