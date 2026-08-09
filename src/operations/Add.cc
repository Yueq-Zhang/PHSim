#include "Add.h"

Add::Add(std::string name) : Operation(name) {}

std::vector<Ptr<MyTensor>> Add::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {

    assert(inputs.size() % 2 == 0);
    // 多个Batch输入
    _batch_size = inputs.size()/2;
    _my_inputs_1.resize(inputs.size()/2);
    _my_inputs_2.resize(inputs.size()/2);
    _my_outputs.resize(inputs.size()/2);

    _num_inputs = 2;

    _my_inputs = inputs;

    set_as_parent_tensor(inputs);
    for (size_t i = 0; i < _batch_size; ++i) {
        _my_inputs_1[i] = inputs[i];
        _my_inputs_2[i] = inputs[i+_batch_size];
        assert(_my_inputs_1[i]->get_dims() == _my_inputs_2[i]->get_dims());
        auto input_dims = _my_inputs_1[i]->get_dims();
        spdlog::info("Add input index: {} / input size: {}", i, inputs[i]->get_dims());
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, output_tensor_type, false);
    }

    calculate_my_loops();
    initialize_my_tiles();

    return _my_outputs;
}

void Add::calculate_my_loops() {

    std::vector<uint32_t> input_dims = {0, 0};
    for (int i=0; i<_batch_size; i++) {
        if (_my_inputs_1[i]->get_dims()[0] != 1) {  // LayerNorm for GEMM
            input_dims[0] = input_dims[0] + std::ceil(static_cast<double>(_my_inputs_1[i]->get_dims()[0]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::dram_channels;
        }
        else {
            input_dims[0] = input_dims[0] + _my_inputs_1[i]->get_dims()[0];
        }
        input_dims[1] = _my_inputs_1[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]); // 对应了各个Batch的所存在的区间
    }

    _inner_loop.resize(2,1);
    _outer_loop.assign(1, 1);

    _inner_loop[0] = input_dims[0];
    _inner_loop[1] = input_dims[1];

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
    spdlog::info("Add for {} batches, inner loop: {}, outer loop: {}", _batch_size, _inner_loop, _outer_loop);
    /*
    _prod_batches = 1;
    for (size_t i = 0; i + 1 < _input_dim.size(); i++) {
        _prod_batches *= _input_dim[i];
    }
    _inner_loop[0] = _prod_batches;

    while (sram_size_needed() > _config.spad_size KB / 2) {
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
    */
}

void Add::initialize_my_tiles() {
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

Tile Add::initialize_my_instructions(uint32_t N) {

    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false
    };

    uint32_t tensor_dim = _inner_loop[1]; // 表示每一个向量的长度

    // base on the inner loop, initialize instructions
    auto n_inner = _inner_loop[0];
    auto n_outer_offset = n_inner * N;

    uint32_t N_max = _m_batch_dim.back();

    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.vector_core_width;   // 每个vector core的长度为作为一次计算的loop size，每个tile计算n_inner数量的token

    uint32_t batch_index = 0;
    uint32_t previous_batch_boundary = 0;
    uint32_t batch_boundary = _m_batch_dim[batch_index];


    if (_my_inputs_1[0]->get_dims()[0] == 1) {
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
            addr_type sram_activation0_offset = sram_activation_base + 2 * n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;
            addr_type sram_activation1_offset = sram_activation_base + (2 * n_inner_offset + 1) * tensor_dim * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;

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

            std::vector<std::vector<uint32_t>> activation_indexes;
            // 每加载一行，做一次计算过程
            for (uint32_t k = 0; k < tensor_dim; ++k) {
                activation_indexes.push_back({row_idx - previous_batch_boundary, k});
            }

            std::vector<addr_type> activation_addrs;
            activation_addrs = _my_inputs_1[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
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

            // -- compute --
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::ADD,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation0_offset, sram_activation1_offset},
            });

            // -- save outputs --
            auto output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
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
            addr_type sram_activation0_offset = sram_activation_base + 2 * n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;
            addr_type sram_activation1_offset = sram_activation_base + (2 * n_inner_offset + 1) * tensor_dim * MyAddressAllocator::precision_activation;
            addr_type sram_accumulation_offset = sram_accumulation_base + n_inner_offset * tensor_dim * MyAddressAllocator::precision_activation;

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

            std::vector<std::vector<uint32_t>> activation_indexes;
            // 每加载dram channel 行，做一次计算过程
            for (uint32_t row = row_idx; row < row_idx + n_loop_size; row++) {
                if (row - previous_batch_boundary >= _my_inputs_1[batch_index]->get_dims()[0]) {
                    continue;
                }
                for (uint32_t k = 0; k < tensor_dim; ++k) {
                    activation_indexes.push_back({row - previous_batch_boundary, k});
                }
            }

            /*
            for (uint32_t k = 0; k < tensor_dim; ++k) {
                activation_indexes.push_back({row_idx - previous_batch_boundary, k});
            }
            */

            std::vector<addr_type> activation_addrs;
            activation_addrs = _my_inputs_1[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
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

            // -- compute --
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::ADD,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation0_offset, sram_activation1_offset},
            });

            // -- save outputs --
            auto output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
            });
        }

    }
    return tile;
}


uint32_t Add::sram_size_needed() {
    auto n = _inner_loop[0];
    auto k = _inner_loop[1];
    if (k % _config.core_width != 0) {
        k += _config.core_width - k % _config.core_width;
    }

    return 3 * n * k * _config.precision;  // 两个输入，一个输出，一共×3
}
