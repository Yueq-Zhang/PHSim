#include "LayerNorm.h"
#include "NormalizationValidation.hpp"
#include "SramTilingValidation.hpp"

#include <limits>

LayerNorm::LayerNorm(std::string name, std::vector<Ptr<MyTensor>> weights) : Operation(name) {
    if (weights.size() != 2) {
        throw std::invalid_argument(
            "LayerNorm '" + _name +
            "' requires exactly two parameters (gamma and beta), got " +
            std::to_string(weights.size()));
    }
    if (weights[0] == nullptr || weights[1] == nullptr) {
        throw std::invalid_argument(
            "LayerNorm '" + _name +
            "' gamma and beta parameters must not be null");
    }
    if (weights[0]->_tensor_type != TensorType::WGT ||
        weights[1]->_tensor_type != TensorType::WGT) {
        throw std::invalid_argument(
            "LayerNorm '" + _name +
            "' gamma and beta parameters must be weight tensors");
    }

    const auto gamma_dimensions = weights[0]->get_dims();
    const auto beta_dimensions = weights[1]->get_dims();
    const uint32_t parameter_element_count =
        phsim::ValidateLayerNormParameterDimensions(
            gamma_dimensions, beta_dimensions, _name);
    if (weights[0]->_precision != weights[1]->_precision) {
        throw std::invalid_argument(
            "LayerNorm '" + _name +
            "' gamma and beta parameters must use the same precision");
    }
    const uint32_t parameter_size_bytes =
        phsim::ValidateLayerNormParameterSizeBytes(
            parameter_element_count, weights[0]->_precision, _name);

    assert(weights.size() == 2);
    assert(weights[0] != nullptr && weights[1] != nullptr);
    assert(weights[0]->_tensor_type == TensorType::WGT);
    assert(weights[1]->_tensor_type == TensorType::WGT);
    assert(gamma_dimensions == beta_dimensions);
    assert(weights[0]->_precision == weights[1]->_precision);

    _my_weights = std::move(weights);
    _weight_dim = gamma_dimensions;
    _prod_weight_dim = parameter_element_count;
    _weight_size_bytes = parameter_size_bytes;

    /*
    _my_inputs.resize(3);
    assert(weights[0]->get_dims() == weights[1]->get_dims());
    _weight_dim = weights[0]->get_dims();
    _my_inputs[1] = weights[0];
    _my_inputs[2] = weights[1];
    */
}


std::vector<Ptr<MyTensor>> LayerNorm::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    // 多个Batch输入
    if (inputs.empty()) {
        throw std::invalid_argument(
            "LayerNorm '" + _name + "' requires at least one input tensor");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == nullptr) {
            throw std::invalid_argument(
                "LayerNorm '" + _name + "' input " + std::to_string(i) +
                " is null");
        }
        phsim::ValidateLayerNormInputDimensions(
            inputs[i]->get_dims(), _weight_dim, _name, i);
    }

    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    set_as_parent_tensor(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        _my_inputs[i] = inputs[i];
        auto input_dims = _my_inputs[i]->get_dims();
        spdlog::debug("Layernorm input index: {} / input size: {}", i, inputs[i]->get_dims());

        assert(input_dims.size() == 2);
        assert(_weight_dim.size() == 1);
        assert(input_dims[1] == _weight_dim[0]);
        _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", input_dims, output_tensor_type, false);
    }

    calculate_my_loops();  // 将Lin×K个输入token按照输入token维度的方向基于SRAM容量完成分组，inner_loop代表一次可以进行多少token的layernorm操作，outer_loop代表基于输入量分组所产生的Batch数量
    initialize_my_tiles(); // 将每一个batch作为一个tile，首个batch前计算前加载Layernorm的beta和gamma，对于每一个token的计算过程为：读取input、计算、写出output

    // spdlog::debug("MyTensor LayerNorm Input dims: {}, Output dims {}", input_dims, output->get_dims());

    return _my_outputs;
    /*
    _outputs.resize(1);
    _my_inputs[0] = inputs[0];

    auto input_dims = inputs[0]->get_dims();
    auto input_dim_riter = input_dims.rbegin();

    for (auto weight_dim_riter = _weight_dim.rbegin(); weight_dim_riter != _weight_dim.rend(); ++weight_dim_riter) {
        assert((*weight_dim_riter) == (*input_dim_riter));
        assert(input_dim_riter != input_dims.rend());
        input_dim_riter++;
    }

    auto output = std::make_shared<MyTensor>(_name + "_output", input_dims, output_tensor_type, false);  // Layernorm计算得到的输入与输出维度相同
    _my_outputs = {output};

    calculate_my_loops();  // 将Lin×K个输入token按照输入token维度的方向基于SRAM容量完成分组，inner_loop代表一次可以进行多少token的layernorm操作，outer_loop代表基于输入量分组所产生的Batch数量
    initialize_my_tiles(); // 将每一个batch作为一个tile，首个batch前计算前加载Layernorm的beta和gamma，对于每一个token的计算过程为：读取input、计算、写出output

    spdlog::debug("MyTensor LayerNorm Input dims: {}, Output dims {}", input_dims, output->get_dims());

    return _my_outputs;
    */
}


void LayerNorm::initialize_my_tiles() {
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


Tile LayerNorm::initialize_my_instructions(uint32_t N) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = N,
        .K = 0,
        .accum = false,
    };

    const uint32_t weight_count = _prod_weight_dim;
    uint32_t N_max = _m_batch_dim.back();

    auto n_inner = _inner_loop[0];  // 在当前每个batch的大小
    auto n_outer_offset = n_inner * N;

    addr_type sram_gamma_base = SPAD_BASE;
    addr_type sram_beta_base = sram_gamma_base + _weight_size_bytes;
    addr_type sram_activation_base = sram_beta_base + _weight_size_bytes;
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    auto gamma_tensor = _my_weights[0];

    // LayerNorm 计算相关的权值数据载入
    std::string movin_info = fmt::format("Load gamma for LayerNorm computation");
    std::vector<addr_type> gamma_addrs = gamma_tensor->get_all_addrs();
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_gamma_base,
        // assume broadcasting bias is available inside the npu
        .size = _weight_size_bytes,
        .src_addrs = std::move(gamma_addrs),
        .operand_id = _INPUT_OPERAND + 1,
        .inst_information = movin_info,
    });

    if (_my_weights.size()==2) {
        auto beta_tensor = _my_weights[1];
        std::string movin_info = fmt::format("Load beta for LayerNorm computation");
        std::vector<addr_type> beta_addrs = beta_tensor->get_all_addrs();
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_beta_base,
            // assume broadcasting bias is available inside the npu
            .size = _weight_size_bytes,
            .src_addrs = std::move(beta_addrs),
            .operand_id = _INPUT_OPERAND + 2,
            .inst_information = movin_info,
        });
    }

    const uint32_t loop_size = _config.vector_core_width;    // 每个vector core的长度为作为一次计算的loop size，每个tile计算n_inner数量的token

    uint32_t batch_index = 0;
    uint32_t previous_batch_boundary = 0;
    uint32_t batch_boundary = _m_batch_dim[batch_index];

    if (_my_inputs[0]->get_dims()[0] == 1) {
        // per token operation in decode stage
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; ++n_inner_offset) {
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
            // 每加载一行，做一次计算过程
            for (uint32_t k = 0; k < weight_count; ++k) {
                activation_indexes.push_back({row_idx - previous_batch_boundary, k});
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
            std::string layernorm_info = fmt::format("LayerNorm computation for row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::LAYERNORM,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_gamma_base, sram_beta_base},
                .inst_information = layernorm_info,
            });

            // -- save outputs --
            std::vector<addr_type> output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);  // 由于输入与输出规模相同，因此计算之后相同的索引也存在相同的输出位置
            std::string movout_info = fmt::format("Save the LayerNorm computation result of row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information =movout_info,
            });
        }
    }
    else {
        uint32_t n_loop_size = MyAddressAllocator::dram_channels;
        if (n_loop_size == 0U || n_inner % n_loop_size != 0U ||
            n_outer_offset % n_loop_size != 0U) {
            throw std::logic_error(
                "LayerNorm '" + _name +
                "' Prefill tile is not aligned to its channel-row group");
        }
        assert(n_inner % n_loop_size == 0U);
        assert(n_outer_offset % n_loop_size == 0U);
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
            if (static_cast<uint64_t>(row_idx) + n_loop_size >
                batch_boundary) {
                throw std::logic_error(
                    "LayerNorm '" + _name +
                    "' channel-row group crosses a batch boundary");
            }
            assert(static_cast<uint64_t>(row_idx) + n_loop_size <=
                   batch_boundary);
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
            std::string layernorm_info = fmt::format("LayerNorm computation for row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::LAYERNORM,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size(),
                .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_gamma_base, sram_beta_base},
                .inst_information = layernorm_info,
            });

            // -- save outputs --
            std::vector<addr_type> output_addrs = _my_outputs[batch_index]->generate_addrs_based_on_indexes(activation_indexes);  // 由于输入与输出规模相同，因此计算之后相同的索引也存在相同的输出位置
            std::string movout_info = fmt::format("Save the LayerNorm computation result of row {}", row_idx);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_accumulation_offset,
                .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information =movout_info,
            });
        }
    }
    return tile;
}


void LayerNorm::calculate_my_loops() {
    // weight_dim 已知
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

    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);

    const bool channel_grouped_prefill =
        _my_inputs[0]->get_dims()[0] != 1U;
    const uint32_t row_alignment =
        channel_grouped_prefill ? MyAddressAllocator::dram_channels : 1U;
    if (row_alignment == 0U || input_dims[0] % row_alignment != 0U) {
        throw std::logic_error(
            "LayerNorm '" + _name +
            "' has an invalid channel-row tiling alignment");
    }

    const uint32_t total_tile_units = input_dims[0] / row_alignment;
    uint32_t inner_tile_units = total_tile_units;
    _inner_loop[0] = input_dims[0];
    const uint64_t available_spad_bytes =
        phsim::AvailablePingPongSramBytes(_config.spad_size);
    const uint64_t available_accum_spad_bytes =
        phsim::AvailablePingPongSramBytes(_config.accum_spad_size);

    while (spad_size_needed() > available_spad_bytes ||
           accum_spad_size_needed() > available_accum_spad_bytes) {
        const uint64_t required_spad_bytes = spad_size_needed();
        const uint64_t required_accum_spad_bytes =
            accum_spad_size_needed();
        if (inner_tile_units <= 1U) {
            throw std::invalid_argument(
                "LayerNorm '" + _name +
                "' minimum tile cannot fit in SRAM: SPAD requires " +
                std::to_string(required_spad_bytes) +
                " bytes but provides " +
                std::to_string(available_spad_bytes) +
                " bytes; Accum SPAD requires " +
                std::to_string(required_accum_spad_bytes) +
                " bytes but provides " +
                std::to_string(available_accum_spad_bytes) +
                " bytes; hidden_size=" +
                std::to_string(_prod_weight_dim) +
                ", spad_size=" + std::to_string(_config.spad_size) +
                " KiB, accum_spad_size=" +
                std::to_string(_config.accum_spad_size) +
                " KiB, row_alignment=" +
                std::to_string(row_alignment));
        }

        phsim::DoubleSramTileCountOrThrow(
            _outer_loop[0], "LayerNorm '" + _name + "'", 0);
        inner_tile_units = phsim::CalculateSramTileDimension(
            total_tile_units, _outer_loop[0]);
        _inner_loop[0] = inner_tile_units * row_alignment;
    }

    assert(_inner_loop[0] % row_alignment == 0U);

    spdlog::debug("LayerNorm for {} batches, inner loop: {}, outer loop: {}", _batch_size, _inner_loop, _outer_loop);

    /*
    std::vector<uint32_t> input_dim = _my_inputs[0]->get_dims();
    _inner_loop.resize(1);
    _outer_loop.assign(1, 1);
    _prod_batches = 1;
    for (size_t i = 0; i + _weight_dim.size() < input_dim.size(); i++) {
        _prod_batches *= input_dim[i];
    }
    _inner_loop[0] = _prod_batches;  // 当前layer norm的输入就是[_prob_batched, Demb]，将除了最后一维Demb之外，前面所有的维度都打散，推理过程即发生 _prod_batches次layernorm计算

    while (sram_size_needed() > _config.spad_size KB / 2) {  // 之后，基于片上SRAM的大小，确定每次最多能载入多少个token，做layernorm计算过程
        _outer_loop[0] *= 2;
        _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
    }
    spdlog::debug("LayerNorm inner loop: {}, outer loop: {}", _inner_loop, _outer_loop);
    */
}


uint64_t LayerNorm::aligned_hidden_size() const {
    uint64_t k = _prod_weight_dim;
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;   // 基于向量单元的计算长度完成补齐
    }
    return k;
}


uint64_t LayerNorm::spad_size_needed() const {
    const uint64_t activation_bytes =
        static_cast<uint64_t>(_inner_loop[0]) * aligned_hidden_size() *
        MyAddressAllocator::precision_activation;

    // Normal SPAD stores the input activation plus gamma and beta.
    return activation_bytes + 2ULL * _weight_size_bytes;
}


uint64_t LayerNorm::accum_spad_size_needed() const {
    const uint64_t activation_bytes =
        static_cast<uint64_t>(_inner_loop[0]) * aligned_hidden_size() *
        MyAddressAllocator::precision_activation;

    // Accum SPAD stores the output activation and the LayerNorm reduction
    // workspace. Keep the previous conservative two-activation accounting,
    // but validate it against the actual accumulation capacity.
    return 2ULL * activation_bytes;
}
