#include "RMSNorm.h"

RMSNorm::RMSNorm(std::string name, std::vector<Ptr<MyTensor>> weights) : Operation(name) {
    assert(!weights.empty());
    _my_weights.resize(1);
    _my_weights[0] = weights[0];
    _weight_dim = weights[0]->get_dims();

    _prod_weight_dim = 1;
    for (auto weight : _weight_dim) {
        _prod_weight_dim *= weight;
    }
}

std::vector<Ptr<MyTensor>> RMSNorm::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    set_as_parent_tensor(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        _my_inputs[i] = inputs[i];
        auto input_dims = _my_inputs[i]->get_dims();
        spdlog::info("RMSNorm input index: {} / input size: {}", i, inputs[i]->get_dims());

        auto input_dim_riter = input_dims.rbegin();
        for (auto weight_dim_riter = _weight_dim.rbegin(); weight_dim_riter != _weight_dim.rend(); ++weight_dim_riter) {
            assert(input_dim_riter != input_dims.rend());
            assert((*weight_dim_riter) == (*input_dim_riter));
            input_dim_riter++;
        }
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, output_tensor_type, false);
    }

    calculate_my_loops();
    initialize_my_tiles();

    return _my_outputs;
}

void RMSNorm::initialize_my_tiles() {
    for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
        _tiles.push_back(initialize_my_instructions(N));
    }
}

Tile RMSNorm::initialize_my_instructions(uint32_t N) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    uint32_t weight_count = 1;
    for (auto weight : _weight_dim) {
        weight_count *= weight;
    }
    uint32_t N_max = _m_batch_dim.back();

    auto n_inner = _inner_loop[0];
    auto n_outer_offset = n_inner * N;

    addr_type sram_gamma_base = SPAD_BASE;
    addr_type sram_activation_base = sram_gamma_base + weight_count * MyAddressAllocator::precision_weight;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    auto gamma_tensor = _my_weights[0];

    std::string gamma_info = fmt::format("Load gamma for RMSNorm computation");
    std::vector<addr_type> gamma_addrs = gamma_tensor->get_all_addrs();
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_gamma_base,
        .size = gamma_tensor->_dims[0] * gamma_tensor->_precision,
        .src_addrs = std::move(gamma_addrs),
        .operand_id = _INPUT_OPERAND + 1,
        .inst_information = gamma_info,
    });

    uint32_t batch_index = 0;
    uint32_t previous_batch_boundary = 0;
    uint32_t batch_boundary = _m_batch_dim[batch_index];

    if (_my_inputs[0]->get_dims()[0] == 1) {
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
            addr_type sram_activation_offset = sram_activation_base + n_inner_offset * weight_count * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * weight_count * MyAddressAllocator::precision_activation;

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
            for (uint32_t k = 0; k < weight_count; ++k) {
                activation_indexes.push_back({row_idx - previous_batch_boundary, k});
            }
            std::vector<addr_type> activation_addrs = _my_inputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);

            if (activation_addrs.empty()) {
                spdlog::info("zero load for RMSNorm activation / activation tensor dim: {}", _my_inputs[batch_index]->get_dims());
            } else {
                std::string movin_info = fmt::format("Load Activation of row {} for RMSNorm computation", row_idx);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_activation_offset,
                    .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                    .src_addrs = std::move(activation_addrs),
                    .operand_id = _INPUT_OPERAND,
                    .inst_information = movin_info,
                });
            }

            std::string rmsnorm_info = fmt::format("RMSNorm computation for row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::RMSNORM,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_gamma_base},
                .inst_information = rmsnorm_info,
            });

            std::vector<addr_type> output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
            std::string movout_info = fmt::format("Save the RMSNorm computation result of row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information = movout_info,
            });
        }
    } else {
        uint32_t n_loop_size = MyAddressAllocator::dram_channels;
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
            addr_type sram_activation_offset = sram_activation_base + n_inner_offset * weight_count * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * weight_count * MyAddressAllocator::precision_activation;

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
                for (uint32_t k = 0; k < weight_count; ++k) {
                    activation_indexes.push_back({row - previous_batch_boundary, k});
                }
            }

            std::vector<addr_type> activation_addrs = _my_inputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);

            if (activation_addrs.empty()) {
                spdlog::info("zero load for RMSNorm activation / activation tensor dim: {}", _my_inputs[batch_index]->get_dims());
            } else {
                std::string movin_info = fmt::format("Load Activation of row {} for RMSNorm computation", row_idx);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_activation_offset,
                    .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                    .src_addrs = std::move(activation_addrs),
                    .operand_id = _INPUT_OPERAND,
                    .inst_information = movin_info,
                });
            }

            std::string rmsnorm_info = fmt::format("RMSNorm computation for row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::RMSNORM,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_gamma_base},
                .inst_information = rmsnorm_info,
            });

            std::vector<addr_type> output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
            std::string movout_info = fmt::format("Save the RMSNorm computation result of row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information = movout_info,
            });
        }
    }
    return tile;
}

void RMSNorm::calculate_my_loops() {
    std::vector<uint32_t> input_dims = {0, 0};
    for (int i = 0; i < _batch_size; i++) {
        if (_my_inputs[i]->get_dims()[0] != 1) {
            input_dims[0] = input_dims[0] + std::ceil(static_cast<double>(_my_inputs[i]->get_dims()[0]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::dram_channels;
        } else {
            input_dims[0] = input_dims[0] + _my_inputs[i]->get_dims()[0];
        }
        input_dims[1] = _my_inputs[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]);
    }

    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);

    _inner_loop[0] = input_dims[0];

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }

    spdlog::info("RMSNorm for {} batches, inner loop: {}, outer loop: {}", _batch_size, _inner_loop, _outer_loop);
}

uint32_t RMSNorm::sram_size_needed() {
    auto n = _inner_loop[0];
    auto k = _prod_weight_dim;
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }
    return n * 3 * k * MyAddressAllocator::precision_activation + k * MyAddressAllocator::precision_weight;
}
