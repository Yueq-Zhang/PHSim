#include "Mul.h"

Mul::Mul(std::string name) : Operation(std::move(name)) {}

std::vector<Ptr<MyTensor>> Mul::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    assert(inputs.size() % 2 == 0);
    _batch_size = inputs.size() / 2;
    _my_inputs_1.resize(_batch_size);
    _my_inputs_2.resize(_batch_size);
    _my_outputs.resize(_batch_size);

    _num_inputs = 2;
    _my_inputs = inputs;
    set_as_parent_tensor(inputs);

    for (size_t i = 0; i < _batch_size; ++i) {
        _my_inputs_1[i] = inputs[i];
        _my_inputs_2[i] = inputs[i + _batch_size];
        assert(_my_inputs_1[i]->get_dims() == _my_inputs_2[i]->get_dims());
        auto input_dims = _my_inputs_1[i]->get_dims();
        spdlog::info("Mul input index: {} / input size: {}", i, inputs[i]->get_dims());
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, output_tensor_type, false);
    }

    calculate_my_loops();
    initialize_my_tiles();
    return _my_outputs;
}

void Mul::calculate_my_loops() {
    std::vector<uint32_t> input_dims = {0, 0};
    for (uint32_t i = 0; i < _batch_size; i++) {
        if (_my_inputs_1[i]->get_dims()[0] != 1) {
            input_dims[0] += std::ceil(static_cast<double>(_my_inputs_1[i]->get_dims()[0]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::dram_channels;
        }
        else {
            input_dims[0] += _my_inputs_1[i]->get_dims()[0];
        }
        input_dims[1] = _my_inputs_1[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]);
    }

    _inner_loop.resize(2, 1);
    _outer_loop.assign(1, 1);
    _inner_loop[0] = input_dims[0];
    _inner_loop[1] = input_dims[1];

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
    spdlog::info("Mul for {} batches, inner loop: {}, outer loop: {}", _batch_size, _inner_loop, _outer_loop);
}

void Mul::initialize_my_tiles() {
    for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
        _tiles.push_back(initialize_my_instructions(N));
    }
}

Tile Mul::initialize_my_instructions(uint32_t N) {
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
    const uint32_t n_loop_size = (_my_inputs_1[0]->get_dims()[0] == 1) ? 1 : MyAddressAllocator::dram_channels;

    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    uint32_t batch_index = 0;
    uint32_t previous_batch_boundary = 0;
    uint32_t batch_boundary = _m_batch_dim[batch_index];

    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
        addr_type sram_activation0_offset = sram_activation_base + 2 * n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;
        addr_type sram_activation1_offset = sram_activation_base + (2 * n_inner_offset + 1) * tensor_dim * MyAddressAllocator::precision_activation;
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
            if (row - previous_batch_boundary >= _my_inputs_1[batch_index]->get_dims()[0]) {
                continue;
            }
            for (uint32_t k = 0; k < tensor_dim; ++k) {
                activation_indexes.push_back({row - previous_batch_boundary, k});
            }
        }

        auto activation_addrs = _my_inputs_1[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_activation0_offset,
            .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
            .src_addrs = std::move(activation_addrs),
            .operand_id = _INPUT_OPERAND,
        });

        activation_addrs = _my_inputs_2[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_activation1_offset,
            .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
            .src_addrs = std::move(activation_addrs),
            .operand_id = _INPUT_OPERAND,
        });

        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MUL,
            .dest_addr = sram_accumulation_offset,
            .size = (uint32_t)activation_indexes.size(),
            .src_addrs = std::vector<addr_type>{sram_activation0_offset, sram_activation1_offset},
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

uint32_t Mul::sram_size_needed() {
    auto n = _inner_loop[0];
    auto k = _inner_loop[1];
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }
    return 3 * n * k * MyAddressAllocator::precision_activation;
}
