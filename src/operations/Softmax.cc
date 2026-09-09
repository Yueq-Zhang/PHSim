#include "Softmax.h"
#include "SramTilingValidation.hpp"

Softmax::Softmax(std::string name) : Operation(name) {
    // assume as dim = -1
    // _inputs.resize(1);
}


std::vector<Ptr<MyTensor>> Softmax::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) { // 此时有几个Input即为几组batch，开展各Batch分别的计算过程
    // 多个Batch输入
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    set_as_parent_tensor(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        _my_inputs[i] = inputs[i];
        auto input_dims = _my_inputs[i]->get_dims();
        spdlog::debug("Softmax input index: {} / input size: {}", i, inputs[i]->get_dims());
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, TensorType::ACT, false);
    }
    calculate_my_loops();
    /*
    _batch_size = inputs.size();  // 之前的写法，默认是一个三维的情况，这边都改成2维
    _my_outputs.resize(_batch_size);
    // assert(inputs.size() == 1);
    _my_inputs = inputs;

    for (int i = 0; i < _batch_size; i++) {
        std::vector<uint32_t> input_dim = inputs[i]->get_dims();
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dim, TensorType::ACT, false);
    }
    // spdlog::debug("softmax batch_size: {}", _batch_size);
    */

    initialize_my_tiles();
    return _my_outputs;
}

void Softmax::calculate_my_loops() {
    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);

    for (int batch=0; batch<_batch_size; batch++) {
        _inner_loop.assign(2, 1);
        _outer_loop.assign(1, 1);
        std::vector<uint32_t> input_dims(_my_inputs[batch]->get_dims());

        for (size_t i = 0; i + 1 < input_dims.size(); i++) {
            _inner_loop[0] *= input_dims[i];
        }
        _inner_loop[1] = input_dims.back();

        const uint64_t available_sram_bytes =
            phsim::AvailablePingPongSramBytes(_config.spad_size);
        while (my_sram_size_needed() > available_sram_bytes) {
            phsim::HalveSramTileDimensionAndDoubleCount(
                _inner_loop, 0, _outer_loop, 0,
                "Softmax '" + _name + "'", my_sram_size_needed(),
                available_sram_bytes,
                "batch=" + std::to_string(batch));
        }

        _inner_loop_softmax.push_back(_inner_loop);
        _outer_loop_softmax.push_back(_outer_loop);
        spdlog::debug("Softmax for batch for Batch {} operation with inner loop: {}, outer loop: {}", batch, _inner_loop, _outer_loop);
    }
}

void Softmax::initialize_my_tiles() {
    for (int B = 0; B < _batch_size; B++) {
        for (uint32_t N = 0; N < _outer_loop_softmax[B][0]; ++N) {
            if (_config.compile_time_tile_pruning &&
                _config.accelerate_ctrl &&
                _config.accelerate_method == "Proportional") {
                _tiles.push_back(make_deferred_tile(N, B));
            } else {
                _tiles.push_back(initialize_my_instructions(N, B));
            }
        }
    }
}

Tile Softmax::make_deferred_tile(uint32_t N, uint32_t req_idx) {
    Tile tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .head_index = 0,
        .N = N,
        .K = 0,
        .M = 0,
        .accum = false,
        .skip = false,
        .pim_tile = false,
    };
    tile.deferred_compile = true;
    tile.materializer = [this, N, req_idx](Tile& target) {
        target = initialize_my_instructions(N, req_idx);
    };
    return tile;
}


uint64_t Softmax::my_sram_size_needed() {
    uint64_t n = _inner_loop[0];
    uint64_t k = _inner_loop[1];
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }

    return n * (2ULL * k + 1ULL) *
           MyAddressAllocator::precision_activation;
}


Tile Softmax::initialize_my_instructions(uint32_t N, uint32_t Batch) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    // base on the inner loop, initialize instructions
    uint32_t softmax_size = _inner_loop_softmax[Batch][1];
    auto n_inner = _inner_loop_softmax[Batch][0];
    auto n_outer_offset = n_inner * N;

    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.vector_core_width;

    auto activation_tensor = _my_inputs[Batch];
    auto output_tensor = _my_outputs[Batch];

    auto input_dim = activation_tensor->get_dims();

    if ((input_dim.size() == 3 && input_dim[1] == 1) or (input_dim.size() == 2 && input_dim[0] == 1) or input_dim.size() == 1) {
        // single token operation
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
            addr_type sram_activation_offset = sram_activation_base + n_inner_offset * softmax_size * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * softmax_size * MyAddressAllocator::precision_activation;

            // -- activation --
            uint32_t row_idx = n_outer_offset + n_inner_offset;
            std::vector<std::vector<uint32_t>> activation_indexes;
            if (input_dim.size() == 3) {  // 三维的Activation
                uint32_t h = std::floor(static_cast<double>(row_idx) / input_dim[1]);
                uint32_t m = row_idx % input_dim[1];
                for (uint32_t k = 0; k < softmax_size; ++k) {
                    activation_indexes.push_back({h, m, k});
                }
            }
            else {  // 默认为二维的Activation
                for (uint32_t k = 0; k < softmax_size; ++k) {
                    activation_indexes.push_back({row_idx, k});
                }
            }

            std::vector<addr_type> activation_addrs = activation_tensor->generate_addrs_based_on_indexes(activation_indexes);

            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_activation_offset,
                .size = (uint32_t)activation_indexes.size() * activation_tensor->_precision,
                .src_addrs = std::move(activation_addrs),
                .operand_id = _INPUT_OPERAND,
            });

            // -- compute --
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::SOFTMAX,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset},
            });

            // -- save outputs --
            auto output_addrs = output_tensor->generate_addrs_based_on_indexes(activation_indexes);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * output_tensor->_precision,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
            });
        }
    }
    else {
        uint32_t n_loop_size = MyAddressAllocator::dram_channels;
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
            addr_type sram_activation_offset = sram_activation_base + n_inner_offset * softmax_size * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * softmax_size * MyAddressAllocator::precision_activation;

            // -- activation --
            uint32_t row_idx = n_outer_offset + n_inner_offset;
            std::vector<std::vector<uint32_t>> activation_indexes;

            for (uint32_t row = row_idx; row < row_idx + n_loop_size; row++) {
                if (input_dim.size() == 3) {  // 三维的Activation
                    uint32_t h = std::floor(static_cast<double>(row) / input_dim[1]);
                    uint32_t m = row % input_dim[1];
                    for (uint32_t k = 0; k < softmax_size; ++k) {
                        activation_indexes.push_back({h, m, k});
                    }
                }
                else {  // 默认为二维的Activation
                    for (uint32_t k = 0; k < softmax_size; ++k) {
                        activation_indexes.push_back({row, k});
                    }
                }
            }

            std::vector<addr_type> activation_addrs = activation_tensor->generate_addrs_based_on_indexes(activation_indexes);

            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_activation_offset,
                .size = (uint32_t)activation_indexes.size() * activation_tensor->_precision,
                .src_addrs = std::move(activation_addrs),
                .operand_id = _INPUT_OPERAND,
            });

            // -- compute --
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::SOFTMAX,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset},
            });

            // -- save outputs --
            auto output_addrs = output_tensor->generate_addrs_based_on_indexes(activation_indexes);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * output_tensor->_precision,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
            });
        }
    }


    return tile;
}
