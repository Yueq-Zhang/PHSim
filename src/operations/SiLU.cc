#include "SiLU.h"
#include "SramTilingValidation.hpp"

SiLU::SiLU(std::string name) : Operation(std::move(name)) {}

std::vector<Ptr<MyTensor>> SiLU::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    set_as_parent_tensor(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        _my_inputs[i] = inputs[i];
        auto input_dims = _my_inputs[i]->get_dims();
        spdlog::debug("SiLU input index: {} / input size: {}", i, inputs[i]->get_dims());
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, output_tensor_type, false);
    }

    calculate_my_loops();
    initialize_my_tiles();
    return _my_outputs;
}

void SiLU::calculate_my_loops() {
    std::vector<uint32_t> input_dims = {0, 0};
    for (uint32_t i = 0; i < _batch_size; i++) {
        if (_my_inputs[i]->get_dims()[0] != 1) {
            input_dims[0] += std::ceil(static_cast<double>(_my_inputs[i]->get_dims()[0]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::dram_channels;
        }
        else {
            input_dims[0] += _my_inputs[i]->get_dims()[0];
        }
        input_dims[1] = _my_inputs[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]);
    }

    _inner_loop.resize(2, 1);
    _outer_loop.assign(1, 1);
    _inner_loop[0] = input_dims[0];
    _inner_loop[1] = input_dims[1];

    const uint64_t available_sram_bytes =
        phsim::AvailablePingPongSramBytes(_config.spad_size);
    while (sram_size_needed() > available_sram_bytes) {
        phsim::HalveSramTileDimensionAndDoubleCount(
            _inner_loop, 0, _outer_loop, 0, "SiLU '" + _name + "'",
            sram_size_needed(), available_sram_bytes);
    }
}

uint64_t SiLU::sram_size_needed() {
    uint64_t n = _inner_loop[0];
    uint64_t k = _inner_loop[1];
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }
    return 2ULL * n * k * MyAddressAllocator::precision_activation;
}

void SiLU::initialize_my_tiles() {
    for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
        _tiles.push_back(initialize_my_instructions(N));
    }
}

Tile SiLU::initialize_my_instructions(uint32_t N) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    const uint32_t tensor_dim = _inner_loop[1];
    const uint32_t n_inner = _inner_loop[0];
    const uint32_t n_outer_offset = n_inner * N;
    const uint32_t N_max = _m_batch_dim.back();
    const uint32_t n_loop_size = (_my_inputs[0]->get_dims()[0] == 1) ? 1 : MyAddressAllocator::dram_channels;

    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    uint32_t batch_index = 0;
    uint32_t previous_batch_boundary = 0;
    uint32_t batch_boundary = _m_batch_dim[batch_index];

    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
        addr_type sram_activation_offset = sram_activation_base + n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;
        addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;

        uint32_t row_idx = n_outer_offset + n_inner_offset;
        if (row_idx >= N_max) {
            continue;
        }

        while (row_idx >= batch_boundary) {
            previous_batch_boundary = _m_batch_dim[batch_index];
            batch_index++;
            batch_boundary = _m_batch_dim[batch_index];
        }

        std::vector<std::vector<uint32_t>> activation_indexes;
        for (uint32_t row = row_idx; row < row_idx + n_loop_size; row++) {
            if (row - previous_batch_boundary >= _my_inputs[batch_index]->get_dims()[0]) {
                continue;
            }
            for (uint32_t k = 0; k < tensor_dim; ++k) {
                activation_indexes.push_back({row - previous_batch_boundary, k});
            }
        }

        auto activation_addrs = _my_inputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
        if (!activation_addrs.empty()) {
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_activation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(activation_addrs),
                .operand_id = _INPUT_OPERAND,
            });
        }

        tile.instructions.push_back(Instruction{
            .opcode = Opcode::SILU,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)activation_indexes.size(),
            .src_addrs = std::vector<addr_type>{sram_activation_offset},
        });

        auto output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
            .src_addrs = std::move(output_addrs),
            .operand_id = _OUTPUT_OPERAND,
        });
    }

    return tile;
}
