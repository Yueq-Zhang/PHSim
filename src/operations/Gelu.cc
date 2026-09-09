#include "Gelu.h"
#include "SramTilingValidation.hpp"

Gelu::Gelu(std::string name) : Operation(name) {
}

std::vector<Ptr<MyTensor>> Gelu::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    // multi input batches
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    set_as_parent_tensor(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        _my_inputs[i] = inputs[i];
        auto input_dims = _my_inputs[i]->get_dims();
        spdlog::debug("GeLU input index: {} / input size: {}", i, inputs[i]->get_dims());
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, TensorType::ACT, false);
    }
    calculate_my_loops();
    initialize_my_tiles();
    return _my_outputs;
}


void Gelu::calculate_my_loops() {
    std::vector<uint32_t> input_dims = {0, 0};
    for (int i=0; i<_batch_size; i++) {
        if (_my_inputs[i]->get_dims()[0] != 1) {  // LayerNorm for GEMM
            input_dims[0] = input_dims[0] + std::ceil(static_cast<double>(_my_inputs[i]->get_dims()[0]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::dram_channels;
        }
        else {
            input_dims[0] = input_dims[0] + _my_inputs[i]->get_dims()[0];
        }
        input_dims[1] = _my_inputs[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]); // 对应了各个Batch的所存在的区间
    }

    _inner_loop.resize(2);
    _outer_loop.assign(1, 1);

    _inner_loop[0] = input_dims[0];
    _inner_loop[1] = input_dims[1];

    const uint64_t available_sram_bytes =
        phsim::AvailablePingPongSramBytes(_config.spad_size);
    while (my_sram_size_needed() > available_sram_bytes) {
        phsim::HalveSramTileDimensionAndDoubleCount(
            _inner_loop, 0, _outer_loop, 0, "GELU '" + _name + "'",
            my_sram_size_needed(), available_sram_bytes);
    }

    /*
    std::vector<uint32_t> input_dim = _my_inputs[0]->get_dims();
    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);

    _prod_batches = 1;
    for (size_t i = 0; i + 1 < input_dim.size(); i++) {
        _prod_batches *= input_dim[i];
    }
    _inner_loop[0] = _prod_batches;

    while (my_sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
    */
}


uint64_t Gelu::my_sram_size_needed() {
    uint64_t n = _inner_loop[0];
    uint64_t k = _inner_loop[1];
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }
    return 2ULL * n * k * MyAddressAllocator::precision_activation;
}


void Gelu::initialize_my_tiles() {
    for (uint32_t N = 0; N < _outer_loop[0]; ++N) {
        if (defer_decode_pruning_compile()) {
            Tile tile{
                .status = Tile::Status::INITIALIZED,
                .optype = get_name(),
                .operation_id = _id,
                .batch = N,
                .K = 0,
                .accum = false,
            };
            tile.deferred_compile = true;
            tile.materializer = [this, N](Tile& target) {
                target = initialize_my_instructions(N);
            };
            _tiles.push_back(std::move(tile));
        } else {
            _tiles.push_back(initialize_my_instructions(N));
        }
    }
}


Tile Gelu::initialize_my_instructions(uint32_t N) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    uint32_t N_max = _m_batch_dim.back();
    uint32_t weight_count = _my_inputs[0]->get_dims().back();

    auto n_inner = _inner_loop[0];
    auto n_outer_offset = n_inner * N;

    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    uint32_t batch_index = 0;
    uint32_t previous_batch_boundary = 0;
    uint32_t batch_boundary = _m_batch_dim[batch_index];

    const uint32_t loop_size = _config.vector_core_width;

    auto activation_tensor = _my_inputs[0];
    auto output_tensor = _my_outputs[0];

    if (_my_inputs[0]->get_dims()[0] == 1) {
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
            addr_type sram_activation_offset = sram_activation_base + n_inner_offset * weight_count * _config.precision;
            addr_type sram_accumulation_offset =  sram_accumulation_base + n_inner_offset * weight_count * _config.precision;

            // -- activation --
            uint32_t row_idx = n_outer_offset + n_inner_offset;
            if (row_idx >= N_max) {
                continue;
            }

            while (row_idx >= batch_boundary) { // 当前的m_index，找到对应的boundary, 对应的batch index就是相关的Tensor
                previous_batch_boundary = _m_batch_dim[batch_index];
                batch_index++;
                batch_boundary = _m_batch_dim[batch_index];
            }

            std::vector<std::vector<uint32_t>> activation_indexes;
            // 每加载一行，做一次计算过程
            for (uint32_t k = 0; k < weight_count; ++k) {
                activation_indexes.push_back({row_idx - previous_batch_boundary, k});
            }

            std::vector<addr_type> activation_addrs = _my_inputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
            if (activation_addrs.empty()) {
                spdlog::debug("zero load for activation m: {} {} / k: {} {} / activation tensor dim: {}", _my_inputs[batch_index]->get_dims());
            }
            else {
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_activation_offset,
                    .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                    .src_addrs = std::move(activation_addrs),
                    .operand_id = _INPUT_OPERAND,
                });
            }

            // -- compute --
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::GELU,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset},
            });
            // -- save outputs --
            std::vector<addr_type> output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);  // 由于输入与输出规模相同，因此计算之后相同的索引也存在相同的输出位置
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
            });
        }
    }
    else {
        uint32_t n_loop_size = MyAddressAllocator::dram_channels;
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
            addr_type sram_activation_offset = sram_activation_base + n_inner_offset * weight_count * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * weight_count * MyAddressAllocator::precision_activation;

            // Find Activation tensor
            uint32_t row_idx = n_outer_offset + n_inner_offset;
            if (row_idx >= N_max) {
                continue;
            }

            while (row_idx >= batch_boundary) { // 当前的m_index，找到对应的boundary, 对应的batch index就是相关的Tensor
                previous_batch_boundary = _m_batch_dim[batch_index];
                batch_index++;
                batch_boundary = _m_batch_dim[batch_index];
            }

            // -- load activation --
            std::vector<std::vector<uint32_t>> activation_indexes;
            // 每加载dram channel行，做一次计算过程
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
                spdlog::debug("zero load for activation m: {} {} / k: {} {} / activation tensor dim: {}", _my_inputs[batch_index]->get_dims());
            }
            else {
                std::string movin_info = fmt::format("Load Activation of row {} for LayerNorm computation", row_idx);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_activation_offset,
                    .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                    .src_addrs = std::move(activation_addrs),
                    .operand_id = _INPUT_OPERAND,
                    .inst_information = movin_info,
                });
            }

            // -- compute --
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::GELU,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset},
            });
            // -- save outputs --
            std::vector<addr_type> output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);  // 由于输入与输出规模相同，因此计算之后相同的索引也存在相同的输出位置
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
            });
        }
    }
    // spdlog::debug("{} instructions generated from tile {}", tile.instructions.size(), tile.optype); spdlog::debug("outer loop {}, inner loop {}", _outer_loop, _inner_loop);
    return tile;
}
