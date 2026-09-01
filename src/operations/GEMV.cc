#include "GEMV.h"

#include <boost/mpl/vector/vector0.hpp>
#include <boost/mpl/vector/vector0.hpp>


namespace {
void append_rope_if_enabled(std::deque<Instruction>& instructions, bool apply_rope,
                            addr_type sram_addr, uint32_t element_count,
                            const std::string& info) {
    if (!apply_rope || element_count == 0) {
        return;
    }
    instructions.push_back(Instruction{
        .opcode = Opcode::ROPE,
        .dest_addr = sram_addr,
        .size = element_count,
        .src_addrs = std::vector<addr_type>{sram_addr},
        .src_from_accum = true,
        .inst_information = info,
    });
}
}  // namespace
GEMV::GEMV(std::string name, std::vector<Ptr<MyTensor>> weights) : Operation(name) {
    _my_weights.resize(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        _my_weights[i] = weights[i];
    }
    matrix_tensor_type = weights[0]->_tensor_type;
    /*
    if (weights.size() == 2) {
        _my_inputs.resize(3);
        _my_inputs[1] = weights[0];
        _my_inputs[2] = weights[1];
        _matrix_dim = _my_inputs[1]->get_dims();
    } else if (weights.size() == 1) {
        _my_inputs.resize(2);
        _my_inputs[1] = weights[0];
        _matrix_dim = _my_inputs[1]->get_dims();
    }
    else {
        spdlog::info("current GEMV don't has input");
    }
    matrix_tensor_type = weights[0]->_tensor_type;
    */
    cache_append = false;
}

std::vector<Ptr<MyTensor>> GEMV::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    if (matrix_tensor_type == TensorType::WGT or matrix_tensor_type == TensorType::ACT) {
        set_as_parent_tensor(inputs);
        _matrix_dim = _my_weights[0]->get_dims();
        for (size_t i = 0; i < inputs.size(); ++i) {
            _my_inputs[i] = inputs[i];
            auto input_dims = _my_inputs[i]->get_dims();
            assert(*input_dims.rbegin() == *(_matrix_dim.rbegin() + 1));
            spdlog::info("GEMM input index: {} / input size: {}", i, inputs[i]->get_dims());
            // 计算output dim
            std::vector<uint32_t> output_dims = {0, 0};
            *(output_dims.rbegin() + 1) = *(_my_inputs[i]->get_dims().rbegin() + 1);
            *output_dims.rbegin() = *_matrix_dim.rbegin();
            _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
        }
        calculate_my_loops();  // Calculate tile size based on SRAM
        initialize_my_tiles(); // Initialize the tiles for GEMV
    }
    else if (matrix_tensor_type == TensorType::KCache or matrix_tensor_type == TensorType::VCache) {
        set_as_parent_tensor(inputs);   // 构建一个当前operation的shared point, 添加到input tensor的子节点中
        assert(output_tensor_type == TensorType::ACT);
        for (size_t i = 0; i < inputs.size(); ++i) {
            _my_inputs[i] = inputs[i];
        }
        std::vector<uint32_t> output_dims;
        if (matrix_tensor_type == TensorType::KCache) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto Q_dims = _my_inputs[i]->get_dims();
                auto K_dims = _my_weights[i]->get_dims();
                output_dims={MyAddressAllocator::h, Q_dims[0], K_dims[0]};
                _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
                spdlog::info("The dimension of Batch {} KCache QKT operation: input Tensor {} and {}, output Tensor {}",i, Q_dims,K_dims,output_dims);
            }
        }
        else if (matrix_tensor_type == TensorType::VCache) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto S_dims = _my_inputs[i]->get_dims();
                auto V_dims = _my_weights[i]->get_dims();
                output_dims={S_dims[S_dims.size() - 2], MyAddressAllocator::h * MyAddressAllocator::d_k};
                _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
                spdlog::info("The dimension of Batch {} KCache SV operation: input Tensor {} and {}, output Tensor {}",i, S_dims,V_dims,output_dims);
            }
        }
        calculate_attention_loops();  // 基于SRAM的大小，计算inner_loop 和 outer_loop, 存储在op中; 每个Loop优先完成单个Attention的计算过程
        initialize_my_tiles(); // 之后开始完成tile的初始化过程
    }
    return _my_outputs;
}

std::vector<Ptr<MyTensor>> GEMV::kvcache_append(std::vector<Ptr<MyTensor>> inputs, std::vector<Ptr<MyTensor>> kvcaches, TensorType output_tensor_type) {
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(_batch_size);
    cache_append = true;
    assert(inputs.size() == kvcaches.size());
    set_as_parent_tensor(inputs);

    auto matrix_tensor = _my_weights[0];
    _matrix_dim = matrix_tensor->get_dims();
    assert(matrix_tensor->_tensor_type == TensorType::WGT);

    for (size_t i = 0; i < _batch_size; ++i) {
        _my_inputs[i] = inputs[i];
        assert(_my_inputs[i]->get_dims().back() == _matrix_dim[0]);  // 这里的matrix是weight
        std::vector<uint32_t> output_dim = {_my_inputs[i]->get_dims()[0], _matrix_dim[1]};

        // Take updated KVCaches as the output
        _my_outputs[i] = kvcaches[i];
        if (_my_outputs[i]->Cache_length + output_dim[0] > _my_outputs[i]->Cache_capacity) {
            _my_outputs[i]->cache_append();
        }

        // 对当前的KVCache有效行数完成实现更新
        _my_outputs[i]->Cache_length += output_dim[0];
        _my_outputs[i]->Cache_capacity -= output_dim[0];

        if (_my_outputs[0]->_tensor_type == TensorType::KCache) {
            spdlog::info("GEMV Result is append to Batch {} KCaches with cache length = {}", i, _my_outputs[i]->Cache_length);
        }
        else if (_my_outputs[0]->_tensor_type == TensorType::VCache) {
            spdlog::info("GEMV Result is append to Batch {} VCaches with cache length = {}", i, _my_outputs[i]->Cache_length);
        }
        else {
            throw std::runtime_error("Invalid Number of PIM-GEMV Inputs");
        }
    }
    calculate_my_loops();
    initialize_my_tiles();
    cache_append = false;  // cache_append操作结束
    return _my_outputs;
}


void GEMV::calculate_my_loops() {

    std::vector<uint32_t> input_dims = {0, 0};  // 多个Batch拼接得到的完整维度
    for (int i=0; i<_batch_size; i++) {
        input_dims[0] = input_dims[0] + _my_inputs[i]->get_dims()[0];
        input_dims[1] = _my_inputs[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]); // 对应了各个Batch的所存在的区间
    }

    _inner_loop.resize(3);  // M, K, N
    assert(input_dims.back() == _matrix_dim[0]);
    uint32_t M = input_dims[input_dims.size() - 2];
    uint32_t K = input_dims.back();
    uint32_t N = _matrix_dim.back();

    if (_batch_size > 1) {
        spdlog::info("GEMV Operation for {} batches with concat size of {}", _batch_size, M);
    }
    else {
        spdlog::info("GEMV Operation for single batches with concat size of {}",M);
    }

    _outer_loop.assign(3, 1);

    _inner_loop[0] = M;
    _inner_loop[1] = std::ceil(((double)K / _outer_loop[1]) / _config.vector_core_width) * _config.vector_core_width;
    _inner_loop[2] = N;

    while (sram_size_needed() > _config.spad_size KB / 2) {
        auto max_el = max_element(_inner_loop.begin(), _inner_loop.end());
        int max_value_index = max_el - _inner_loop.begin();
        _outer_loop[max_value_index] *= 2;

        if (max_value_index == 0) {
            _inner_loop[0] = std::ceil((double)M / _outer_loop[0]);
        }
        else if (max_value_index == 1) {
            _inner_loop[1] = std::ceil(((double)K / _outer_loop[1]) / _config.vector_core_width) * _config.vector_core_width;
        }
        else if (max_value_index == 2){
            // _inner_loop[2] = std::ceil((double)N / _outer_loop[2]);
            _inner_loop[2] = std::ceil((static_cast<double>(N) / _outer_loop[2]) / _config.core_width) * _config.core_width; // inner_loop基于Tile大小向上取整, n方向单位为Core_width
        }
        else {
            throw std::runtime_error("GEMV::calculate_my_loops: invalid value index");
        }
    }

    assert(_inner_loop[0] * _outer_loop[0] >= M);
    assert(_inner_loop[1] * _outer_loop[1] >= K);
    assert(_inner_loop[2] * _outer_loop[2] >= N);

    // number of outer_loop based on the inner loop and data dimension
    _outer_loop[0] = std::ceil(double(M)/_inner_loop[0]);
    _outer_loop[1] = std::ceil(double(K)/_inner_loop[1]);
    _outer_loop[2] = std::ceil(double(N)/_inner_loop[2]);
    spdlog::info("GEMV for {} Batches inner loop: {}, outer loop: {}", _batch_size, _inner_loop, _outer_loop);
}

void GEMV::initialize_my_tiles() {
    int core_id = -1; // Core mapping from 0

    if (matrix_tensor_type == TensorType::WGT or matrix_tensor_type == TensorType::ACT) {
        uint32_t M_max = _m_batch_dim.back();
        uint32_t K_max = _my_weights[0]->_dims[0];
        uint32_t N_max = _my_weights[0]->_dims[1];
        for (uint32_t M = 0; M < _outer_loop[0]; ++M) {
            if (M * _inner_loop[0] >= M_max){continue;}
            for (uint32_t N = 0; N < _outer_loop[2]; ++N) {
                if (N * _inner_loop[2] >= N_max) {continue;}
                for (uint32_t K = 0; K < _outer_loop[1]; ++K) {
                    if (K * _inner_loop[1] >= K_max) {continue;}
                    if (K==0) { core_id = (core_id + 1) % _config.num_cores; }  // if K==0, mapping the accumulation process to the same core
                    const bool should_store =
                        (K + 1) * _inner_loop[1] >= K_max;
                    const bool defer_qkv_decode_compile =
                        _config.decode_pruning_enabled &&
                        _config.decode_pruning_compile_context &&
                        !_config.virtual_mem_hash_enable &&
                        (get_name().find(".attn.QGen") != std::string::npos ||
                         get_name().find(".attn.KGen") != std::string::npos ||
                         get_name().find(".attn.VGen") != std::string::npos ||
                         get_name().find(".attn.proj") != std::string::npos);
                    if (defer_decode_pruning_compile() ||
                        defer_qkv_decode_compile) {
                        Tile tile{
                            .status = Tile::Status::INITIALIZED,
                            .optype = get_name(),
                            .operation_id = _id,
                            .batch = 0,
                            .N = N,
                            .K = K,
                            .M = M,
                            .accum = K != 0,
                        };
                        tile.deferred_compile = true;
                        tile.materializer =
                            [this, M, K, N, should_store](Tile& target) {
                                target = initialize_my_instructions_tile(
                                    M, K, N, should_store);
                            };
                        _tiles.push_back(std::move(tile));
                    } else {
                        _tiles.push_back(initialize_my_instructions_tile(
                            M, K, N, should_store));
                    }
                    _tiles.back().core_id = core_id;
                }
            }
        }
    }
    else if (matrix_tensor_type == TensorType::KCache or matrix_tensor_type == TensorType::VCache) {
        for (uint32_t batch = 0; batch < _batch_size; batch++) {
            for (uint32_t H = 0; H < MyAddressAllocator::h; ++H) { // Attention Head
                for (uint32_t M = 0; M < _outer_loop_attn[batch][0]; ++M) {
                    for (uint32_t N = 0; N < _outer_loop_attn[batch][2]; ++N) {
                        if ((batch * MyAddressAllocator::h + H) % (matrix_tensor_type == TensorType::KCache ? MyAddressAllocator::parallel_KCache_head_per_channel : MyAddressAllocator::parallel_VCache_head_per_channel) == 0) {
                            core_id = (core_id + 1) % _config.num_cores; // if K==0, mapping the accumulation process to the same core
                        }
                        for (uint32_t K = 0; K < _outer_loop_attn[batch][1]; ++K) {
                            _tiles.push_back(initialize_my_attention_instructions(batch, H, M, K, N, K + 1 == _outer_loop_attn[batch][1]));
                            _tiles.back().core_id = core_id;
                        }
                    }
                }
            }
        }
    }
}


void GEMV::calculate_attention_loops() {
    std::string gemv_attetion_type;  // 需要计算每个Head的尺寸
    uint32_t M, K, N;
    _inner_loop.resize(3);  // M, K, N

    for (int i=0; i<_batch_size; i++) {
        std::vector<uint32_t> input0_dims(_my_inputs[i]->get_dims());
        std::vector<uint32_t> input1_dims(_my_weights[i]->get_dims());
        _outer_loop.assign(3, 1);

        if (matrix_tensor_type == TensorType::KCache) {
            // K Cache 计算下的单个Head
            M = input0_dims[input0_dims.size() - 2]; // vector输入，M=1
            K = MyAddressAllocator::d_k;
            N = input1_dims[input1_dims.size() - 2]; // Lt
            spdlog::info("Current Attention Operation for QKT, GEMV: M = {}, K = {}, N = {}", M, K, N);
        }
        else if(matrix_tensor_type == TensorType::VCache) {
            M = input0_dims[input0_dims.size() - 2];
            K = input1_dims[input1_dims.size() - 2];
            N = MyAddressAllocator::d_k;
            spdlog::info("Current Attention Operation for SV, GEMV: M = {}, K = {}, N = {}", M, K, N);
        }
        else {
            assert(0);
        }

        // 分配的计算内容，按照单个的Attention head计算流程进行inner loop和outer Loop的划分
        _inner_loop[0] = M;
        _inner_loop[1] = std::ceil(((double)K / _outer_loop[1]) / _config.vector_core_width) * _config.vector_core_width;  // 向量计算的K方向，基于向量单元尺寸实现对齐
        _inner_loop[2] = N;

        while (sram_size_needed() > _config.spad_size KB / 2) {
            auto max_el = max_element(_inner_loop.begin(), _inner_loop.end());
            int max_value_index = max_el - _inner_loop.begin();
            _outer_loop[max_value_index] *= 2;

            if (max_value_index == 0) {
                _inner_loop[0] = std::ceil((double)M / _outer_loop[0]);
            }
            else if (max_value_index == 1) {
                _inner_loop[1] = std::ceil(((double)K / _outer_loop[1]) / _config.vector_core_width) * _config.vector_core_width;
            }
            else if (max_value_index == 2){
                _inner_loop[2] = std::ceil((double)N / _outer_loop[2]);
            }
            else {
                throw std::runtime_error("GEMV::calculate_my_loops: invalid value index");
            }
        }

        assert(_inner_loop[0] * _outer_loop[0] >= M);
        assert(_inner_loop[1] * _outer_loop[1] >= K);
        assert(_inner_loop[2] * _outer_loop[2] >= N);

        // number of outer_loop based on the inner loop and data dimension
        _outer_loop[0] = std::ceil(double(M)/_inner_loop[0]);
        _outer_loop[1] = std::ceil(double(K)/_inner_loop[1]);
        _outer_loop[2] = std::ceil(double(N)/_inner_loop[2]);
        spdlog::info("GEMV inner loop: {}, outer loop: {}", _inner_loop, _outer_loop);

        spdlog::info("GEMV for Batch {} attention operation {} with inner loop: {}, outer loop: {}", i,gemv_attetion_type, _inner_loop, _outer_loop);
        _inner_loop_attn.push_back(_inner_loop);
        _outer_loop_attn.push_back(_outer_loop);
    }
}


Tile GEMV::initialize_my_instructions(uint32_t M, uint32_t K, uint32_t N, bool should_store) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,   // 若进行MatMul计算过程中，K不为零，则说明需要进行一定的累加处理过程
    };

    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop[0];  // M-axis L2 tile size
    auto k_inner = _inner_loop[1];  // K-axis L2 tile size
    auto n_inner = _inner_loop[2];  // N-axis L2 tile size

    // 计算出每一块个Tile的对应位置
    auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
    auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
    auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

    auto vector_tensor = _my_inputs[0];
    auto matrix_tensor = _my_inputs[1];
    auto output_tensor = _my_outputs[0];

    uint32_t M_max = vector_tensor->_dims[0];
    uint32_t K_max = matrix_tensor->_dims[0];
    uint32_t N_max = matrix_tensor->_dims[1];

    // load tile to spad Memory in order vector / weight, L2 Tile
    addr_type sram_vector_base = SPAD_BASE;
    addr_type sram_matrix_base = SPAD_BASE + m_inner * k_inner * vector_tensor->_precision; // vector_space = m_inner * k_inner * _config.precision
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.vector_core_width;

    if (_my_inputs.size() == 3 && K == 0) {  // For Bias
        // spdlog::info("Load bias element from {} to {} for N = {} outer loop", n_outer_offset, n_outer_offset + n_inner, N);
        auto bias_tensor =_my_inputs[2];

        std::vector<std::vector<uint32_t>> bias_indexes;
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
            // n_inner_offset: L1 tile start index in each L2 tile
            for (uint32_t n_loop = 0; n_loop < loop_size; ++n_loop) {
                if (n_outer_offset + n_inner_offset + n_loop >= bias_tensor->_dims[0]) {continue;}
                bias_indexes.push_back(std::vector<uint32_t>{n_outer_offset + n_inner_offset + n_loop});
            }
        }

        auto bias_addrs = bias_tensor->generate_addrs_based_on_indexes(bias_indexes);
        if (bias_addrs.empty()) {
            spdlog::info("zero load for activation n: {} / bias tensor dim: {}", n_outer_offset, bias_tensor->get_dims());
        } else {
            std::string movin_info = fmt::format("Load Bias with the begin n = {}-{}", bias_indexes.front()[0], bias_indexes.back()[0]);
            // spdlog::info("{} addrs have been generated for {} / bias tensor with dims {}", bias_addrs.size(), n_outer_offset, bias_tensor->get_dims());
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_accumulation_base,
                .size = (uint32_t)bias_indexes.size() * bias_tensor->_precision,
                .src_addrs = std::move(bias_addrs),
                .operand_id = _INPUT_OPERAND + 2,
                .inst_information = movin_info,
            });
            sram_accumulation_base = sram_accumulation_base + static_cast<uint32_t>(bias_indexes.size()) * bias_tensor->_precision;
        }
    }

    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset++) {
        for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
            for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset++) {

                uint32_t m_tile_index = m_outer_offset + m_inner_offset;
                uint32_t k_tile_index = std::ceil(static_cast<double>(k_outer_offset + k_inner_offset) / _config.vector_core_width);
                uint32_t n_tile_index = n_outer_offset + n_inner_offset;

                // SRAM act L1 tile offset
                addr_type sram_vector_offset = sram_vector_base + (m_inner_offset * k_inner + k_inner_offset) * vector_tensor->_precision; // 对于向量数据的存储
                // SRAM wgt L1 tile offset
                addr_type sram_matrix_offset = sram_matrix_base + (n_inner_offset * k_inner + k_inner_offset) * matrix_tensor->_precision; // 矩阵数据的按列存储
                // SRAM out L1 tile offset
                addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner) * output_tensor->_precision;

                // -- vector --
                if (n_inner_offset == 0) {
                    std::vector<std::vector<uint32_t>> vector_indexes;
                    for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                        if (m_outer_offset + m_inner_offset >= M_max or k_outer_offset + k_inner_offset + k_loop >= K_max) {
                            continue;
                        }
                        vector_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset + k_loop});
                    }
                    auto vector_addrs = vector_tensor->generate_addrs_based_on_indexes(vector_indexes);

                    if (vector_addrs.empty()) {
                        spdlog::info(
                            "zero load for vector m: {} {} / k: "
                            "{} {} / activation tensor dim: {}",
                            m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset,
                            vector_tensor->get_dims());
                    } else {
                        std::string movin_info = fmt::format("Load Vector with the begin m = {}-{} and k = {}-{}",
                        vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1]);
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_vector_offset,
                            .size = (uint32_t)vector_indexes.size() * vector_tensor->_precision,
                            .src_addrs = std::move(vector_addrs),
                            .operand_id = _INPUT_OPERAND,
                            .inst_information = movin_info,
                        });
                    }
                }

                // -- Matrix --
                if (m_inner_offset == 0) {
                    std::vector<std::vector<uint32_t>> matrix_indexes;
                    for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                        if (k_outer_offset + k_inner_offset + k_loop >= K_max or n_outer_offset + n_inner_offset >= N_max) {continue;}
                        matrix_indexes.push_back(std::vector<uint32_t>{k_outer_offset + k_inner_offset + k_loop, n_outer_offset + n_inner_offset});
                    }

                    auto matrix_addrs = matrix_tensor->generate_addrs_based_on_indexes(matrix_indexes);
                    if (matrix_addrs.empty()) {
                        spdlog::info(
                            "operation name : {} / "
                            "zero load for weight k: {} {} / n: {} {} "
                            "/ weight tensor dim: {} / is transposed: {}",
                            get_name(), k_outer_offset, k_inner_offset, n_outer_offset,
                            n_inner_offset, matrix_tensor->get_dims(),
                            matrix_tensor->_is_transposed);
                    }
                    else {
                        std::string movin_info = fmt::format("Load matrix with the begin index {} and end index {}",
                            matrix_indexes.front(), matrix_indexes.back());
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_matrix_offset,
                            .size = (uint32_t)matrix_indexes.size() * matrix_tensor->_precision,
                            .src_addrs = std::move(matrix_addrs),
                            .operand_id = _INPUT_OPERAND + 1,
                            .inst_information = movin_info,
                        });
                    }
                }

                // -- compute --
                if ((m_outer_offset + m_inner_offset) < M_max and(k_outer_offset + k_inner_offset) < K_max and (n_outer_offset + n_inner_offset) < N_max) {
                    std::string gemv_info = fmt::format("GEMV computation for L1 tile: m = {}, k = {}, n = {}", m_tile_index, k_tile_index, n_tile_index);
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::GEMV,
                        .dest_addr = sram_accumulation_offset,
                        .size = output_tensor->_precision,  // GEMV的计算结果就是一个值
                        .src_addrs = std::vector<addr_type>{sram_vector_offset, sram_matrix_offset},
                        .tile_m = m_tile_index,
                        .tile_k = k_tile_index,
                        .tile_n = n_tile_index,
                        .inst_information = gemv_info,
                    });
                }
                else {
                    spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                    spdlog::info("vector dims {}, Matrix dim {}", vector_tensor->get_dims(), matrix_tensor->get_dims());
                }

                if (should_store && (k_inner_offset + loop_size >= k_inner) && (n_inner_offset + 1 >= n_inner)) {
                    // 当前GEMV Tile执行完所有N完成回存的最后一次K方向累加时，将计算结果回存
                    std::vector<std::vector<uint32_t>> output_indexes;
                    for (int n_loop = 0; n_loop < n_inner; n_loop++) {
                        if (m_outer_offset + m_inner_offset >= M_max or n_outer_offset + n_loop >= N_max) {
                            continue;
                        }
                        output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset, n_outer_offset + n_loop});
                    }
                    auto output_addrs = output_tensor->generate_addrs_based_on_indexes(output_indexes);
                    if (!output_addrs.empty()) {
                        append_rope_if_enabled(tile.instructions, _apply_rope, sram_accumulation_offset,
                                               (uint32_t)output_indexes.size(),
                                               "Apply RoPE before storing GEMV projection");
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVOUT,
                            .dest_addr = sram_accumulation_offset,
                            .size = (uint32_t)output_indexes.size() * output_tensor->_precision,
                            .src_addrs = std::move(output_addrs),
                            .operand_id = _OUTPUT_OPERAND,
                        });
                    }
                }
            }
        }
    }
    return tile;
}


Tile GEMV::initialize_my_instructions_tile(uint32_t M, uint32_t K, uint32_t N, bool should_store) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,   // 若进行MatMul计算过程中，K不为零，则说明需要进行一定的累加处理过程
    };

    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop[0];  // M-axis L2 tile size
    auto k_inner = _inner_loop[1];  // K-axis L2 tile size
    auto n_inner = _inner_loop[2];  // N-axis L2 tile size

    // 计算出每一块个Tile的对应位置
    auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
    auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
    auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

    auto vector_tensors = _my_inputs;
    auto matrix_tensor = _my_weights[0];
    auto output_tensors = _my_outputs;
    /*
    auto vector_tensor = _my_inputs[0];
    auto matrix_tensor = _my_inputs[1];
    auto output_tensor = _my_outputs[0];
    */
    uint32_t M_max = _m_batch_dim.back();
    uint32_t K_max = matrix_tensor->_dims[0];
    uint32_t N_max = matrix_tensor->_dims[1];

    // load tile to SPM in order vector / weight  L2 Tile，Weight与Activation在NPU的SRAM中的存放位置的计算
    addr_type sram_vector_base = SPAD_BASE;
    addr_type sram_matrix_base = SPAD_BASE + m_inner * k_inner * vector_tensors[0]->_precision;  // 加载vector所占的空间 = m_inner * k_inner * _config.precision
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t m_loop_size = 1;
    const uint32_t k_loop_size = _config.vector_core_width;
    const uint32_t n_loop_size = _config.core_width; // 这里选择对于n的加载以此规模为单位，与MatMul计算过程中的数据排布相契合

    if (_my_inputs.size() == 3) {  // 对于Bias的加载
        auto bias_tensor =_my_inputs[2];
        sram_accumulation_base = sram_accumulation_base + n_inner * bias_tensor->_precision;
        // spdlog::info("Load bias element from {} to {} for N = {} outer loop", n_outer_offset, n_outer_offset + n_inner, N);
        if (K == 0){
            // 构建二维数组，用于存储计算过程中所需的indexes, 由于Bias的规模较小，直接完成所有的Bias加载
            std::vector<std::vector<uint32_t>> bias_indexes;
            for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += k_loop_size) {
                // n_inner_offset: L1 tile start index in each L2 tile
                for (uint32_t n_loop = 0; n_loop < k_loop_size; ++n_loop) {
                    if (n_outer_offset + n_inner_offset + n_loop >= bias_tensor->_dims[0]) {continue;}
                    bias_indexes.push_back(std::vector<uint32_t>{n_outer_offset + n_inner_offset + n_loop});
                }
            }

            auto bias_addrs = bias_tensor->generate_addrs_based_on_indexes(bias_indexes);
            if (bias_addrs.empty()) {
                spdlog::info("zero load for activation n: {} / bias tensor dim: {}", n_outer_offset, bias_tensor->get_dims());
            }
            else {
                std::string movin_info = fmt::format("Load Bias with the begin n = {}-{}", bias_indexes.front()[0], bias_indexes.back()[0]);
                // spdlog::info("{} addrs have been generated for {} / bias tensor with dims {}", bias_addrs.size(), n_outer_offset, bias_tensor->get_dims());
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = ACCUM_SPAD_BASE,
                    .size = (uint32_t)bias_indexes.size() * bias_tensor->_precision,
                    .src_addrs = std::move(bias_addrs),
                    .operand_id = _INPUT_OPERAND + 2,
                    .inst_information = movin_info,
                });
            }
        }
    }


    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
        for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += k_loop_size) {
            for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += m_loop_size) {

                // 一次循环中，这个block中所有的N和M都可以参与运算
                uint32_t m_tile_index = m_outer_offset + m_inner_offset;
                uint32_t k_tile_index = std::ceil(static_cast<double>(k_outer_offset + k_inner_offset) / _config.vector_core_width);
                uint32_t n_tile_index = std::ceil((n_outer_offset + n_inner_offset) / n_loop_size);

                // SRAM act L1 tile offset
                addr_type sram_vector_offset = sram_vector_base + (m_inner_offset * k_inner + k_inner_offset) *  vector_tensors[0]->_precision; // 对于向量数据的存储
                // SRAM wgt L1 tile offset
                addr_type sram_matrix_offset = sram_matrix_base + (n_inner_offset * k_inner + k_inner_offset) * matrix_tensor->_precision; // 矩阵数据的按列存储
                // SRAM out L1 tile offset
                addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner) * MyAddressAllocator::precision_psum;

                // -- vector -- 完成Vector计算元素的加载
                if (n_inner_offset == 0) {
                    std::map<uint32_t, std::vector<std::vector<uint32_t>>> vector_indexes;
                    uint32_t batch_index = 0;
                    uint32_t previous_batch_boundary = 0;
                    uint32_t batch_boundary = _m_batch_dim[batch_index];

                    for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                        uint32_t vector_m_index = m_outer_offset + m_inner_offset + m_loop;
                        if (vector_m_index >= M_max) {
                            continue;
                        }
                        while (vector_m_index >= batch_boundary) { // 当前的m_index，找到对应的boundary, 对应的batch index就是相关的Tensor
                            previous_batch_boundary = _m_batch_dim[batch_index];
                            batch_index++;
                            batch_boundary = _m_batch_dim[batch_index];
                        }
                        for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                            uint32_t vector_k_index = k_outer_offset + k_inner_offset + k_loop;
                            if (vector_k_index >= K_max) {
                                continue;  // 超过边界跳过
                            }
                            vector_indexes[batch_index].push_back(std::vector<uint32_t>{vector_m_index - previous_batch_boundary, vector_k_index});
                        }
                    }

                    std::vector<addr_type> vector_addrs;
                    uint32_t vector_index_size = 0;
                    if (vector_indexes.empty()) {
                        spdlog::info("No valid activation tiles to load.");
                    }
                    else {
                        for (const auto& [batch_id, indexes] : vector_indexes) {
                            // 假设你有 per-batch tensor 列表：_activation_batch_tensors[batch_id]
                            vector_index_size += indexes.size();
                            auto addrs = vector_tensors[batch_id]->generate_addrs_based_on_indexes(indexes);
                            vector_addrs.insert(vector_addrs.end(), addrs.begin(), addrs.end());
                        }
                    }

                    if (vector_addrs.empty()) {
                        spdlog::info("zero load for vector m outer offset {} and inner offset {} / k outer offset {} and inner offset {} ",
                            m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset);
                    } else {
                        std::string movin_info = fmt::format("Load Vector with the begin m = {}-{} and k = {}-{}",
                            m_outer_offset + m_inner_offset, std::min(m_outer_offset + m_inner_offset + m_loop_size, M_max),
                            k_outer_offset + k_inner_offset, std::min( k_outer_offset + k_inner_offset + k_loop_size, M_max));
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_vector_offset,
                            .size = (uint32_t)vector_indexes.size() *  vector_tensors[0]->_precision,
                            .src_addrs = std::move(vector_addrs),
                            .operand_id = _INPUT_OPERAND,
                            .inst_information = movin_info,
                        });
                    }
                }

                // -- Matrix --  进行矩阵元素的加载，这里考虑每次加载以单个Tile的size为单位进行，以保证Matrix的加载效率
                if (m_inner_offset == 0) {
                    std::vector<std::vector<uint32_t>> matrix_indexes;
                    for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
                        for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                            if (k_outer_offset + k_inner_offset + k_loop >= K_max or n_outer_offset + n_inner_offset + n_loop >= N_max) {
                                continue;
                            }
                            matrix_indexes.push_back(std::vector<uint32_t>{k_outer_offset + k_inner_offset + k_loop, n_outer_offset + n_inner_offset + n_loop});
                        }
                    }
                    auto matrix_addrs = matrix_tensor->generate_addrs_based_on_indexes(matrix_indexes);
                    if (matrix_addrs.empty()) {
                        spdlog::info(
                            "operation name : {} / "
                            "zero load for weight k: {} {} / n: {} {} "
                            "/ weight tensor dim: {} / is transposed: {}",
                            get_name(), k_outer_offset, k_inner_offset, n_outer_offset,
                            n_inner_offset, matrix_tensor->get_dims(),
                            matrix_tensor->_is_transposed);
                    }
                    else {
                        std::string movin_info = fmt::format("Load matrix with the begin index {} and end index {}",
                            matrix_indexes.front(), matrix_indexes.back());
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_matrix_offset,
                            .size = (uint32_t)matrix_indexes.size() * matrix_tensor->_precision,
                            .src_addrs = std::move(matrix_addrs),
                            .operand_id = _INPUT_OPERAND + 1,
                            .inst_information = movin_info,
                        });
                    }
                }

                // -- compute --
                if ((m_outer_offset + m_inner_offset) < M_max and(k_outer_offset + k_inner_offset) < K_max and (n_outer_offset + n_inner_offset) < N_max) {
                    std::string gemv_info = fmt::format("GEMV computation for L1 tile: m = {}, k = {}, n = {}", m_tile_index, k_tile_index, n_tile_index);
                    tile.instructions.push_back(Instruction{
                        .opcode = Opcode::GEMV,
                        .dest_addr = sram_accumulation_offset,
                        .size = k_loop_size * n_loop_size,  // GEMV的计算结果就是向量中的n个元素
                        .src_addrs = std::vector<addr_type>{sram_vector_offset, sram_matrix_offset},
                        .tile_m = m_tile_index,
                        .tile_k = k_tile_index,
                        .tile_n = n_tile_index,
                        .inst_information = gemv_info,
                    });
                }
                else {
                    spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                }

                // -- store --
                if (should_store && (k_inner_offset + k_loop_size >= k_inner)) {
                    std::vector<addr_type> output_addrs;
                    uint32_t output_index_size = 0;
                    if (cache_append) {
                        uint32_t n_begin = n_outer_offset + n_inner_offset;
                        uint32_t n_end = n_outer_offset + n_inner_offset + n_loop_size;
                        uint32_t head_index_begin = n_begin/MyAddressAllocator::d_k;
                        uint32_t head_index_end = n_end/MyAddressAllocator::d_k;

                        for (uint32_t head_index = head_index_begin; head_index <= head_index_end; head_index++) {

                            uint32_t batch_index = 0;
                            uint32_t previous_batch_boundary = 0;
                            uint32_t batch_boundary = _m_batch_dim[batch_index];

                            std::vector<std::vector<uint32_t>> output_indexes;
                            for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                                uint32_t output_m_index = m_outer_offset + m_inner_offset + m_loop;
                                if (output_m_index >= M_max) {
                                    continue;
                                }
                                while (output_m_index >= batch_boundary) {
                                    previous_batch_boundary = _m_batch_dim[batch_index];
                                    batch_index++;
                                    batch_boundary = _m_batch_dim[batch_index];
                                }

                                for (uint32_t k=0; k<MyAddressAllocator::d_k; k++) {
                                    if (head_index * MyAddressAllocator::d_k + k >= N_max
                                        or head_index * MyAddressAllocator::d_k + k >= n_outer_offset + n_inner_offset + n_loop_size) {
                                        continue;
                                    }
                                    output_indexes.push_back({m_outer_offset + m_inner_offset + m_loop + output_tensors[batch_index]->Cache_length, head_index * MyAddressAllocator::d_k + k});
                                }
                            }
                            output_index_size += output_indexes.size();

                            auto output_addrs_head = output_tensors[batch_index]->generate_addrs_based_on_indexes(output_indexes);
                            for (auto output_addr_head : output_addrs_head) {
                                output_addrs.push_back(output_addr_head);
                            }
                        }
                    }
                    else {
                        // 当前GEMV Tile执行完所有N完成回存的最后一次K方向累加时, 计算结果回存, 正常结果计算，非KV Cache更新
                        std::map<uint32_t, std::vector<std::vector<uint32_t>>> output_indexes;
                        uint32_t batch_index = 0;
                        uint32_t previous_batch_boundary = 0;
                        uint32_t batch_boundary = _m_batch_dim[batch_index];
                        for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                            uint32_t output_m_index = m_outer_offset + m_inner_offset + m_loop;
                            if (output_m_index >= M_max) {
                                continue;
                            }
                            while (output_m_index >= batch_boundary) {
                                previous_batch_boundary = _m_batch_dim[batch_index];
                                batch_index++;
                                batch_boundary = _m_batch_dim[batch_index];
                            }
                            for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
                                uint32_t output_n_index = n_outer_offset + n_inner_offset + n_loop;
                                if (output_n_index >= N_max) {
                                    continue;
                                }
                                output_indexes[batch_index].push_back(std::vector<uint32_t>{output_m_index - previous_batch_boundary, output_n_index});
                            }
                        }

                        if (output_indexes.empty()) {
                            spdlog::info("No valid output tiles to store.");
                        } else {
                            for (const auto& [batch_id, indexes] : output_indexes) {
                                output_index_size += indexes.size();
                                auto addrs = output_tensors[batch_id]->generate_addrs_based_on_indexes(indexes);
                                output_addrs.insert(output_addrs.end(), addrs.begin(), addrs.end());
                            }
                        }
                        /*
                        std::vector<std::vector<uint32_t>> output_indexes;
                        for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
                            for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= M_max or n_outer_offset + n_inner_offset + n_loop >= N_max) {
                                    continue;
                                }
                                output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, n_outer_offset + n_inner_offset + n_loop});
                            }
                        }
                        output_addrs = output_tensor->generate_addrs_based_on_indexes(output_indexes);
                        */
                    }

                    if (!output_addrs.empty()) {
                        append_rope_if_enabled(tile.instructions, _apply_rope, sram_accumulation_offset,
                                               output_index_size,
                                               "Apply RoPE before storing GEMV projection");
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVOUT,
                            .dest_addr = sram_accumulation_offset,
                            .size = _inner_loop[0] * _matrix_dim[1] * output_tensors[0]->_precision,
                            .src_addrs = std::move(output_addrs),
                            .operand_id = _OUTPUT_OPERAND,
                            .per_ch_inst = (output_tensors[0]->_tensor_type == TensorType::KCache or output_tensors[0]->_tensor_type == TensorType::VCache),
                        });
                    }
                }
            }
        }
    }
    return tile;
}


Tile GEMV::initialize_my_attention_instructions(uint32_t B, uint32_t head_index, uint32_t M, uint32_t K, uint32_t N, bool should_store) {

    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = B,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,   // 若进行MatMul计算过程中，K不为零，则说明需要进行一定的累加处理过程
    };

    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop_attn[B][0];  // M-axis L2 tile size
    auto k_inner = _inner_loop_attn[B][1];  // K-axis L2 tile size
    auto n_inner = _inner_loop_attn[B][2];  // N-axis L2 tile size

    auto vector_tensor = _my_inputs[B];
    auto matrix_tensor = _my_weights[B];
    auto output_tensor = _my_outputs[B];

    // 设定进行GEMV的计算单位
    const uint32_t m_loop_size = 1;
    const uint32_t k_loop_size = _config.vector_core_width;
    const uint32_t n_loop_size = _config.core_width;

    if (matrix_tensor->_tensor_type == TensorType::KCache) {  // QKT的GEMV计算过程, Head index用于完成Q和K的索引生成
        uint32_t q_head_index = head_index;
        uint32_t kv_head_index = MyAddressAllocator::get_kv_head_index(q_head_index);

        uint32_t M_max = vector_tensor->_dims[0];
        uint32_t K_max = MyAddressAllocator::d_k;
        uint32_t N_max = matrix_tensor->_dims[0];

        // 计算出L2 Tile的对应位置
        auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
        auto q_k_outer_offset = q_head_index * MyAddressAllocator::d_k + k_inner * K;  // Q hidden slice for q head
        auto kcache_k_outer_offset = kv_head_index * MyAddressAllocator::d_k + k_inner * K;  // K cache slice for mapped kv head
        auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

        // load tile to SPM in order vector / weight  加载的L2 Tile，Weight与Activation在NPU的SRAM中的存放位置的计算
        addr_type sram_vector_base = SPAD_BASE;
        addr_type sram_matrix_base = SPAD_BASE + m_inner * k_inner * vector_tensor->_precision; // 加载vector所占的空间 = m_inner * k_inner * _config.precision
        addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
            for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += k_loop_size) {
                for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += m_loop_size) {

                    // 一次循环中，这个block中所有的N和M都可以参与运算
                    uint32_t m_tile_index = m_outer_offset + m_inner_offset;
                    uint32_t k_tile_index = std::ceil(static_cast<double>(q_k_outer_offset + k_inner_offset) / _config.vector_core_width);
                    uint32_t n_tile_index = std::ceil((n_outer_offset + n_inner_offset) / n_loop_size);

                    // SRAM act L1 tile offset
                    addr_type sram_vector_offset = sram_vector_base + (m_inner_offset * k_inner + k_inner_offset) * vector_tensor->_precision; // 对于向量数据的存储
                    // SRAM wgt L1 tile offset
                    addr_type sram_matrix_offset = sram_matrix_base + (n_inner_offset * k_inner + k_inner_offset) * matrix_tensor->_precision; // 矩阵数据的按列存储
                    // SRAM out L1 tile offset
                    addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * MyAddressAllocator::precision_psum;

                    // -- vector --
                    if (n_inner_offset == 0) {  // 在GEMV计算开始前，首先完成针对Vector计算元素的加载
                        std::vector<std::vector<uint32_t>> vector_indexes;
                        for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                            for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= M_max or k_inner * K + k_inner_offset + k_loop >= K_max) {
                                    continue;
                                }
                                vector_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, q_k_outer_offset + k_inner_offset + k_loop});
                            }
                        }
                        auto vector_addrs = vector_tensor->generate_addrs_based_on_indexes(vector_indexes);
                        if (vector_addrs.empty()) {
                            spdlog::info("zero load for vector m: {} {} / k: {} {} / activation tensor dim: {}",
                                m_outer_offset, m_inner_offset, q_k_outer_offset, k_inner_offset, vector_tensor->get_dims());
                        } else {
                            // spdlog::info("QKT GEMV operation for the Q Vector m = {}-{} and k = {}-{}",vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load Vector with the begin m = {}-{} and k = {}-{}",
                            vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1]);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_vector_offset,
                                .size = (uint32_t)vector_indexes.size() * vector_tensor->_precision,
                                .src_addrs = std::move(vector_addrs),
                                .operand_id = _INPUT_OPERAND,
                                .inst_information = movin_info,
                            });
                        }
                    }

                    // -- Matrix --  进行矩阵元素的加载，这里考虑每次加载以单个Tile的size为单位进行，以保证Matrix的加载效率，在QKT计算过程中，K做了转置，这里通过控制index来实现
                    if (m_inner_offset == 0) {
                        std::vector<std::vector<uint32_t>> matrix_indexes;
                        for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
                            for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                                uint32_t matrix_n_index = n_outer_offset + n_inner_offset + n_loop;
                                uint32_t matrix_k_index = kcache_k_outer_offset + k_inner_offset + k_loop;
                                if (matrix_n_index >= matrix_tensor->_dims[0] or
                                    k_inner * K + k_inner_offset + k_loop >= MyAddressAllocator::d_k or
                                    matrix_k_index >= matrix_tensor->_dims[1]) {
                                    continue;
                                }
                                matrix_indexes.push_back(std::vector<uint32_t>{matrix_n_index, matrix_k_index});
                            }
                        }
                        auto matrix_addrs = matrix_tensor->generate_addrs_based_on_indexes(matrix_indexes);
                        if (matrix_addrs.empty()) {
                            spdlog::info(
                                "operation name : {} / zero load for weight k: {} {} / n: {} {} / weight tensor dim: {} / is transposed: {}",
                                get_name(), kcache_k_outer_offset,
                                k_inner_offset, n_outer_offset,
                                n_inner_offset, matrix_tensor->get_dims(),
                                matrix_tensor->_is_transposed);
                        }
                        else {
                            // spdlog::info("QKT GEMV operation for the K Cache with n = {}-{}, k = {}-{}", matrix_indexes.front()[0], matrix_indexes.back()[0], matrix_indexes.front()[1], matrix_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load matrix with the begin index {} and end index {}",matrix_indexes.front(), matrix_indexes.back());
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_matrix_offset,
                                .size = (uint32_t)matrix_indexes.size() * matrix_tensor->_precision,
                                .src_addrs = std::move(matrix_addrs),
                                .operand_id = _INPUT_OPERAND + 1,
                                .per_ch_inst = true,
                                .inst_information = movin_info,
                            });
                        }
                    }

                    // -- compute --
                    if ((m_outer_offset + m_inner_offset) < M_max and (k_inner * K + k_inner_offset) < K_max and (n_outer_offset + n_inner_offset) < N_max) {
                        std::string gemv_info = fmt::format("GEMV computation for L1 tile: m = {}, k = {}, n = {}", m_tile_index, k_tile_index, n_tile_index);
                        // spdlog::info(gemv_info);
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::GEMV,
                            .dest_addr = sram_accumulation_offset,
                            .size = k_loop_size * n_loop_size,  // GEMV的计算结果就是向量中的n个元素
                            .src_addrs = std::vector<addr_type>{sram_vector_offset, sram_matrix_offset},
                            .tile_m = m_tile_index,
                            .tile_k = k_tile_index,
                            .tile_n = n_tile_index,
                            .inst_information = gemv_info,
                        });
                    }
                    else {
                        spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, q_k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                        spdlog::info("vector dims {}, Matrix dim {}", vector_tensor->get_dims(), matrix_tensor->get_dims());
                    }

                    // -- store --
                    if (should_store && (k_inner_offset + k_loop_size >= k_inner)) {
                        // 当前GEMV Tile执行完所有N完成回存的最后一次K方向累加时，将计算结果回存，对应规模为 m × n
                        std::vector<std::vector<uint32_t>> output_indexes;
                        for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
                            for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= M_max or n_outer_offset + n_inner_offset + n_loop >= N_max) {
                                    continue;
                                }
                                output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, n_outer_offset + n_inner_offset + n_loop});
                            }
                        }
                        auto output_addrs = output_tensor->generate_addrs_based_on_indexes_attention(output_indexes, head_index);
                        if (!output_addrs.empty()) {
                            std::string movout_info = fmt::format("Store QKT result of Batch{} head {} with M = {}-{}, N = {}-{}", B, head_index, output_indexes.front()[0], output_indexes.back()[0], output_indexes.front()[1], output_indexes.back()[1]);
                            // spdlog::info(movout_info);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVOUT,
                                .dest_addr = sram_accumulation_offset,
                                .size = (uint32_t)output_indexes.size() * output_tensor->_precision,
                                .src_addrs = std::move(output_addrs),
                                .operand_id = _OUTPUT_OPERAND,
                                .inst_information = movout_info,
                            });
                        }
                    }
                }
            }
        }
    }
    else if (matrix_tensor->_tensor_type == TensorType::VCache) {
        uint32_t q_head_index = head_index;
        uint32_t kv_head_index = MyAddressAllocator::get_kv_head_index(q_head_index);
        uint32_t M_max = vector_tensor->_dims[1]; // Lin
        uint32_t K_max = matrix_tensor->_dims[0]; // Lin
        uint32_t N_max = MyAddressAllocator::d_k; // d_k

        // 计算出每一块个Tile的对应位置
        auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
        auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
        auto output_n_outer_offset = n_inner * N + q_head_index * MyAddressAllocator::d_k;  // Output hidden slice for q head
        auto vcache_n_outer_offset = n_inner * N + kv_head_index * MyAddressAllocator::d_k;  // V cache slice for mapped kv head

        // load tile to SPM in order vector / weight  加载的L2 Tile，Weight与Activation在NPU的SRAM中的存放位置的计算
        addr_type sram_vector_base = SPAD_BASE;
        addr_type sram_matrix_base = SPAD_BASE + m_inner * k_inner * vector_tensor->_precision; // 加载vector所占的空间 = m_inner * k_inner * _config.precision
        addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += n_loop_size) {
            for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += k_loop_size) {
                for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += m_loop_size) {

                    uint32_t m_tile_index = m_outer_offset + m_inner_offset;
                    uint32_t k_tile_index = std::ceil(static_cast<double>(k_outer_offset + k_inner_offset) / _config.vector_core_width);
                    uint32_t n_tile_index = std::ceil((output_n_outer_offset + n_inner_offset) / n_loop_size);

                    // SRAM act L1 tile offset
                    addr_type sram_vector_offset = sram_vector_base + (m_inner_offset * k_inner + k_inner_offset) * vector_tensor->_precision; // 对于向量数据的存储
                    // SRAM wgt L1 tile offset
                    addr_type sram_matrix_offset = sram_matrix_base + (n_inner_offset * k_inner + k_inner_offset) * matrix_tensor->_precision; // 矩阵数据的按列存储
                    // SRAM out L1 tile offset
                    addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * MyAddressAllocator::precision_psum;

                    // -- vector --  (S)  // S是三维情况，同时存在各Attention head分离，因此索引与地址生成与QKT Output Index的情况类似
                    if (n_inner_offset == 0) {  // 在GEMV计算开始前，首先完成针对Vector计算元素的加载
                        std::vector<std::vector<uint32_t>> vector_indexes;
                        for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                            for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= M_max or k_outer_offset + k_inner_offset + k_loop >= K_max) {
                                    continue;
                                }
                                vector_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, k_outer_offset + k_inner_offset + k_loop});
                            }
                        }
                        auto vector_addrs = vector_tensor->generate_addrs_based_on_indexes_attention(vector_indexes, head_index);
                        if (vector_addrs.empty()) {
                            spdlog::info("zero load for vector m: {} {} / k: {} {} / activation tensor dim: {}",
                                m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset, vector_tensor->get_dims());
                        } else {
                            // spdlog::info("SV GEMV operation for the S Vector m = {}-{} and k = {}-{}",vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load Vector with the begin m = {}-{} and k = {}-{}",
                            vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1]);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_vector_offset,
                                .size = (uint32_t)vector_indexes.size() * vector_tensor->_precision,
                                .src_addrs = std::move(vector_addrs),
                                .operand_id = _INPUT_OPERAND,
                                .inst_information = movin_info,
                            });
                        }
                    }

                    // Weight (V Cache), V Cache的加载基于[Lin, dk]，对应的索引没有变化
                    if (m_inner_offset == 0) {
                        std::vector<std::vector<uint32_t>> weight_indexes;
                        for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                            for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {  // 在KCache存储，转置计算之前，n方向为行，k方向为列
                                uint32_t weight_k_index = k_outer_offset + k_inner_offset + k_loop;
                                uint32_t weight_n_index = vcache_n_outer_offset + n_inner_offset + n_loop;
                                if (weight_k_index >= matrix_tensor->_dims[0] or
                                    n_inner_offset + n_loop >= MyAddressAllocator::d_k or
                                    weight_n_index >= matrix_tensor->_dims[1]) {
                                    continue;
                                }
                                weight_indexes.push_back(std::vector<uint32_t>{weight_k_index, weight_n_index});
                            }
                        }
                        auto weight_addrs = matrix_tensor->generate_addrs_based_on_indexes(weight_indexes); // 对于K Cache，地址生成过程中可以基于index计算所属Head
                        if (weight_addrs.empty()) {
                            spdlog::info("operation name : {} / zero load for weight k: {} {} / n: {} {} / weight tensor dim: {} / is transposed: {}",
                                get_name(), k_outer_offset, k_inner_offset, vcache_n_outer_offset,n_inner_offset, matrix_tensor->get_dims(),matrix_tensor->_is_transposed);
                        }
                        else {
                            // spdlog::info("SV operation for the V Cache with k = {}-{}, n = {}-{}", weight_indexes.front()[0], weight_indexes.back()[0], weight_indexes.front()[1], weight_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load V Cache with the begin index {} and end index {}", weight_indexes.front(), weight_indexes.back());
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_matrix_offset,
                                .size = (uint32_t)weight_indexes.size() * MyAddressAllocator::precision_weight,
                                .src_addrs = std::move(weight_addrs),
                                .operand_id = _INPUT_OPERAND + 1,
                                .per_ch_inst = true,
                                .inst_information = movin_info,
                            });
                        }
                    }

                    // -- compute --
                    if ((m_outer_offset + m_inner_offset) < M_max and (k_outer_offset + k_inner_offset) < K_max and (n_inner * N + n_inner_offset) < N_max) {
                        std::string gemv_info = fmt::format("GEMV computation for L1 tile: m = {}, k = {}, n = {}", m_tile_index, k_tile_index, n_tile_index);
                        //spdlog::info(gemv_info);
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::GEMV,
                            .dest_addr = sram_accumulation_offset,
                            .size = k_loop_size * n_loop_size,  // GEMV的计算结果就是向量中的n个元素
                            .src_addrs = std::vector<addr_type>{sram_vector_offset, sram_matrix_offset},
                            .tile_m = m_tile_index,
                            .tile_k = k_tile_index,
                            .tile_n = n_tile_index,
                            .inst_information = gemv_info,
                        });
                    }
                    else {
                        spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, output_n_outer_offset + n_inner_offset);
                        spdlog::info("vector dims {}, Matrix dim {}", vector_tensor->get_dims(), matrix_tensor->get_dims());
                    }

                    // -- Store --
                    if (should_store && k_inner_offset + k_loop_size >= k_inner) {
                        std::vector<std::vector<uint32_t>> output_indexes;
                        for (int m_loop = 0; m_loop < m_loop_size; m_loop++) {
                            for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= vector_tensor->_dims[0] or output_n_outer_offset + n_inner_offset + n_loop >= output_tensor->_dims[1]) {
                                    continue;
                                }
                                output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, output_n_outer_offset + n_inner_offset + n_loop});
                            }
                        }

                        auto output_addrs = output_tensor->generate_addrs_based_on_indexes_attention(output_indexes, head_index);
                        if (!output_addrs.empty()) {
                            std::string movout_info = fmt::format("Store QKT result of head {} with M = {}-{}, N = {}-{}", head_index, output_indexes.front()[0], output_indexes.back()[0], output_indexes.front()[1], output_indexes.back()[1]);
                            // spdlog::info(movout_info);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVOUT,
                                .dest_addr = sram_accumulation_offset,
                                .size = (uint32_t)output_indexes.size() * output_tensor->_precision,
                                .src_addrs = std::move(output_addrs),
                                .operand_id = _OUTPUT_OPERAND,
                                .inst_information = movout_info
                            });
                        }
                    }
                }
            }
        }
    }
    return tile;
}





uint32_t GEMV::sram_size_needed() {
    auto m = _inner_loop[0];
    auto k = _inner_loop[1];
    if (k % _config.vector_core_width != 0) {
        k += _config.vector_core_width - k % _config.vector_core_width;
    }
    auto n = _inner_loop[2];

    return (m * k) * _my_inputs[0]->_precision + (k * n) * _my_weights[0]->_precision;  // vector + Matrix
    /*
    if (_my_inputs.size() == 3) {
        auto bias_dim = _my_inputs[2]->get_dims()[0];
        return (m * k) * _my_inputs[0]->_precision + (k * n) * _my_inputs[1]->_precision + bias_dim * _my_inputs[2]->_precision;
    }
    else {
        return (m * k) * _my_inputs[0]->_precision + (k * n) * _my_inputs[1]->_precision;
    }
    */
}
