#include "GEMM.h"

#include <boost/core/demangle.hpp>

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
GEMM::GEMM(std::string name, std::vector<Ptr<MyTensor>> weights) : Operation(name) {  // initialize based on Mytensor
    _my_weights.resize(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        _my_weights[i] = weights[i];
    }
    matrix_tensor_type = weights[0]->_tensor_type;
    /*
    if (weights.size() == 2) {
        // assert(weights.size() == 2);
        _my_inputs.resize(3);
        _my_inputs[1] = weights[0];
        _my_inputs[2] = weights[1];
    } else if (weights.size() == 1) {
        _my_inputs.resize(2);
        _my_inputs[1] = weights[0];
    }
    */

    _is_transposed = false;
}


std::vector<Ptr<MyTensor>> GEMM::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {

    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    if (matrix_tensor_type == TensorType::WGT or matrix_tensor_type == TensorType::ACT) {
        set_as_parent_tensor(inputs);  // Constrict a shared point of current operation, add to the child nodes of input tensors
        auto weight_dims = _my_weights[0]->get_dims();
        for (size_t i = 0; i < inputs.size(); ++i) {
            _my_inputs[i] = inputs[i];
            // 验证input dim
            auto input_dims = _my_inputs[i]->get_dims();
            assert(*input_dims.rbegin() == *(weight_dims.rbegin() + 1)); // validate input dimension.
            // spdlog::info("GEMM input index: {} / input size: {}", i, inputs[i]->get_dims());
            // 计算output dim
            std::vector<uint32_t> output_dims = {0, 0};
            *(output_dims.rbegin() + 1) = *(_my_inputs[i]->get_dims().rbegin() + 1);
            *output_dims.rbegin() = *weight_dims.rbegin();
            _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
        }

        //calculate_my_loops();  // 基于SRAM的大小，计算inner_loop 和 outer_loop, 存储在op中
        calculate_my_loops_prime();

        initialize_my_tiles(); // 之后开始完成tile的初始化过程

        //spdlog::info("GEMM operation with {} batches of total [{}, {}] activation and weight [{}, {}]", _batch_size,
        //     _m_batch_dim.back(), _my_weights[0]->get_dims()[0], _my_weights[0]->get_dims()[0], _my_weights[0]->get_dims()[1]);
        /*
        uint32_t input_dim = _my_inputs[0]->get_dims()[0];
        auto input0_dims = _my_inputs[0]->get_dims();
        auto input1_dims = _my_inputs[1]->get_dims();
        assert(*input0_dims.rbegin() == *(input1_dims.rbegin() + 1));

        // 选择维度数量较大的输入作为输出维度的基础, 基于矩阵乘法规则(M, K) × (K, N) → (M, N)设置输出维度
        auto larger_dim = input0_dims.size() > input1_dims.size() ? input0_dims : input1_dims;
        std::vector<uint32_t> output_dims(larger_dim.begin(), larger_dim.end());
        *(output_dims.rbegin() + 1) =
            *(input0_dims.rbegin() + 1);                // Set (M, x) in GEMM (M, K) x (K, N).
        *output_dims.rbegin() = *input1_dims.rbegin();  // Set (x, N) in GEMM (M, K) x (K, N)

        _my_outputs[0] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);

        calculate_my_loops();  // 基于SRAM的大小，计算inner_loop 和 outer_loop, 存储在op中
        initialize_my_tiles(); // 之后开始完成tile的初始化过程

        spdlog::info("GEMM = input0 : {} * input1: {} = output0 : {}", input0_dims, input1_dims, output_dims);
        // spdlog::info("outer loop : {} / inner loop : {}", _outer_loop, _inner_loop);
        */
    }
    else if (matrix_tensor_type == TensorType::KCache or matrix_tensor_type == TensorType::VCache) {
        set_as_parent_tensor(inputs);   // 构建一个当前operation的shared point, 添加到input tensor的子节点中
        assert(output_tensor_type == TensorType::ACT);
        for (size_t i = 0; i < inputs.size(); ++i) {
            _my_inputs[i] = inputs[i];
            // spdlog::info("GEMM input index: {} / input size: {}", i, inputs[i]->get_dims());
        }

        if (matrix_tensor_type == TensorType::KCache) { // QKT进行MatMul的计算的结果是三维情况，[Head, lin, lin]
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto Q_dims = _my_inputs[i]->get_dims();
                auto K_dims = _my_weights[i]->get_dims();
                std::vector<uint32_t> output_dims = {MyAddressAllocator::h, Q_dims[0], K_dims[0]};
                _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
                // spdlog::info("The dimension of Batch {} QKT operation: input Tensor {} and {}, output Tensor {}",i, Q_dims,K_dims,output_dims);
            }
        }
        else if (matrix_tensor_type == TensorType::VCache) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto S_dims = _my_inputs[i]->get_dims();
                auto V_dims = _my_weights[i]->get_dims();
                std::vector<uint32_t> output_dims = {S_dims[S_dims.size() - 2], MyAddressAllocator::h * MyAddressAllocator::d_k};
                _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
                // spdlog::info("The dimension of Batch {} SV operation: input Tensor {} and {}, output Tensor {}",i, S_dims,V_dims,output_dims);
            }
        }
        calculate_attention_loops();  // 基于SRAM的大小，计算inner_loop 和 outer_loop, 存储在op中; 每个Loop优先完成单个Attention的计算过程
        initialize_my_tiles(); // 之后开始完成tile的初始化过程
        // spdlog::info("outer loop : {} / inner loop : {}", _outer_loop, _inner_loop);
        /*
        auto input0_dims = _my_inputs[0]->get_dims();
        auto input1_dims = _my_inputs[1]->get_dims();

        std::vector<uint32_t> output_dims;
        if (matrix_tensor_type == TensorType::KCache) {  // QKT进行MatMul的计算的结果是三维情况，[Head, lin, lin]
            output_dims={MyAddressAllocator::h, input0_dims[0], input1_dims[0]};   //
            _my_outputs[0] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
            spdlog::info("The dimension of QKT operation: input Tensor {} and {}, output Tensor {}",input0_dims,input1_dims,output_dims);
        }
        else if (matrix_tensor_type == TensorType::VCache) { // SV的输入是三维 + 2维，计算的结果是二维情况 [lin, Demb]
            output_dims={input0_dims[input0_dims.size() - 2], MyAddressAllocator::h * MyAddressAllocator::d_k};
            _my_outputs[0] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
            spdlog::info("The dimension of SV operation: input Tensor {} and {}, output Tensor {}",input0_dims,input1_dims,output_dims);
        }
        */
    }
    spdlog::info("The Compilation for Operation {} is Finished", _name);

     return _my_outputs;
}


void GEMM::initialize_my_tiles() {
    // Here, B does not refer to batch_size,
    // but rather to the outer dimensions of the tensor in GEMM,
    // such as b*h in [b, h, l, d_k].


    if (matrix_tensor_type == TensorType::WGT or matrix_tensor_type == TensorType::ACT) {
        int core_id = -1; // Core mapping from 0

        uint32_t M_max = _m_batch_dim.back();
        uint32_t K_max = _my_weights[0]->_dims[0];
        uint32_t N_max = _my_weights[0]->_dims[1];
        for (uint32_t M = 0; M < _outer_loop[0]; ++M) {
            if (M * _inner_loop[0] >= M_max){continue;}
            for (uint32_t N = 0; N < _outer_loop[2]; ++N) {
                if (N * _inner_loop[2] >= N_max) {continue;}
                for (uint32_t K = 0; K < _outer_loop[1]; ++K) {
                    if (K * _inner_loop[1] >= K_max) {continue;}
                    // [0,K-1] Accumulate, Kth output result , 优先加载前M行K列的Activation，与N列K行的的权值，进行计算过程，生成输出的M行，N列数据，
                    if (K==0) {
                        core_id = (core_id + 1) % _config.num_cores;  // if K==0, mapping the accumulation process to the same core
                    }
                    const bool should_store =
                        (K + 1) * _inner_loop[1] >= K_max;
                    if (_config.compile_time_tile_pruning &&
                        _config.accelerate_ctrl &&
                        _config.accelerate_method == "Proportional") {
                        _tiles.push_back(
                            make_deferred_gemm_tile(0, M, K, N, should_store));
                    } else {
                        _tiles.push_back(initialize_my_DASH_instructions(
                            0, M, K, N, should_store));
                    }
                    /*
                    if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                        // _tiles.push_back(initialize_my_DASH_instructions(B, M, K, N, K + 1 == _outer_loop[1]))

                    }
                    else {
                        //_tiles.push_back(initialize_my_instructions(B, M, K, N, K + 1 == _outer_loop[1]));
                        _tiles.push_back(initialize_my_instructions(0, M, K, N, (K+1) * _inner_loop[1] >= K_max));
                    }
                    */
                    _tiles.back().core_id = core_id;
                }
            }
        }
    }
    else if (matrix_tensor_type == TensorType::KCache or matrix_tensor_type == TensorType::VCache) {
        int core_id = -1;
        for (uint32_t batch = 0; batch < _batch_size; batch++) {
            for (uint32_t H = 0; H < MyAddressAllocator::h; ++H) {  // Attention Head
                if ((batch * MyAddressAllocator::h + H) % (matrix_tensor_type == TensorType::KCache ? MyAddressAllocator::parallel_KCache_head_per_channel : MyAddressAllocator::parallel_VCache_head_per_channel) == 0) {
                    core_id = (core_id + 1) % _config.num_cores;  // if K==0, mapping the accumulation process to the same core
                }
                for (uint32_t M = 0; M < _outer_loop_attn[batch][0]; ++M) {
                    for (uint32_t N = 0; N < _outer_loop_attn[batch][2]; ++N) {
                        for (uint32_t K = 0; K < _outer_loop_attn[batch][1]; ++K) {
                            const bool should_store =
                                K + 1 == _outer_loop_attn[batch][1];
                            if (_config.compile_time_tile_pruning &&
                                _config.accelerate_ctrl &&
                                _config.accelerate_method == "Proportional") {
                                _tiles.push_back(make_deferred_attention_tile(
                                    batch, H, M, K, N, should_store));
                            } else {
                                _tiles.push_back(
                                    initialize_my_attention_instructions(
                                        batch, H, M, K, N, should_store));
                            }
                            _tiles.back().core_id = core_id;
                        }
                    }
                }
                // spdlog::info("Tiles for Batch {} head {} is finished", batch, H);
            }
        }
    }
    // spdlog::info("GEMM operation::initialize {} tiles for outer_loop {}", _tiles.size(), _outer_loop);
}


Tile GEMM::initialize_my_instructions(uint32_t B, uint32_t M, uint32_t K, uint32_t N, bool should_store) {
    // Initialize an L2 Tile
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
    // 生成一个tile，其中B-N-K-M分别代表了MatMul计算过程中的Batch-output-inner-input维度
    addr_type act_addr = 0;
    addr_type wgt_addr = 0;

    // base on the inner loop, initialize instructions
    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop[0];  // M-axis L2 tile size
    auto k_inner = _inner_loop[1];  // K-axis L2 tile size
    auto n_inner = _inner_loop[2];  // N-axis L2 tile size

    // 计算出每一块个Tile的对应位置
    auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
    auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
    auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

    // load tile to SPM in order ACT / WGT  L2 Tile，Weight与Activation在NPU的SRAM中的存放位置的计算
    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_weight_base = SPAD_BASE + m_inner * k_inner * _config.precision;  // 加载激活值所占的空间 = m_inner * k_inner * _config.precision
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.core_width;

    auto activation_tensor = _my_inputs[0];
    auto weight_tensor = _my_inputs[1];
    auto output_tensor = _my_outputs[0];

    uint32_t M_max = activation_tensor->_dims[0];
    uint32_t K_max = weight_tensor->_dims[0];
    uint32_t N_max = weight_tensor->_dims[1];

    if (_is_transposed) {
        std::swap(activation_tensor, weight_tensor);  // swap用于交换两个对象的值，之后将activation_tensor与weight_tensor均进行转换
        activation_tensor->set_transposed();
        weight_tensor->set_transposed();
    }

    // Flatten 3D matrix to 2D for GEMM computation
    auto batch_index = std::vector<uint32_t>();
    if (_my_inputs[0]->get_dims().size() == 3) {
        batch_index.push_back(B);
    }

    // -- bias --
    // 如果有bias(相应的weight)，且K=0(即刚开始计算), K来自outer loop，需要首先读取Bias Tensor的数据，生成相应的MOVIN指令
    if (_my_inputs.size() == 3 && K == 0) {
        // spdlog::info("Load bias element from {} to {} for N = {} outer loop", n_outer_offset, n_outer_offset + n_inner, N);
        auto bias_tensor =_my_inputs[2];
        // 构建二维数组，用于存储计算过程中所需的indexes, 由于Bias的规模较小，直接完成所有的Bias加载
        std::vector<std::vector<uint32_t>> bias_indexes;
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
            // n_inner_offset: L1 tile start index in each L2 tile
            for (uint32_t n_loop = 0; n_loop < loop_size; ++n_loop) {
                if (n_outer_offset + n_inner_offset + n_loop >= bias_tensor->_dims[0]) {continue;}
                bias_indexes.push_back(std::vector<uint32_t>{n_outer_offset + n_inner_offset + n_loop});
            }
        }

        // 将index进行计算，转换为加载地址
        auto bias_addrs = bias_tensor->generate_addrs_based_on_indexes(bias_indexes);

        if (bias_addrs.empty()) {
            spdlog::info("zero load for activation n: {} / bias tensor dim: {}", n_outer_offset, bias_tensor->get_dims());
        } else {
            std::string movin_info = fmt::format("Load Bias with the n = {}-{}",bias_indexes.front()[0], bias_indexes.back()[0]);
            // spdlog::info("{} addrs have been generated for {} / bias tensor with dims {}", bias_addrs.size(), n_outer_offset, bias_tensor->get_dims());
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_accumulation_base + n_inner * MyAddressAllocator::precision_weight,
                .size = (uint32_t)bias_indexes.size() * MyAddressAllocator::precision_weight,
                .src_addrs = std::move(bias_addrs),
                .operand_id = _INPUT_OPERAND + 2,
                .inst_information = movin_info,
            });
        }
    }

    // L1 tile 中含有的inner tile的嵌套为M、K、N共三层，对应了两个输入矩阵的分块
    // 基于嵌套计算出每一个L1 Tile大小的Activate和Weight以及计算结果的SRAM Offset
    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
        for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
            for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += loop_size) {
                uint32_t m_tile_index =  std::ceil((double)(m_outer_offset + m_inner_offset) / _config.core_height);
                uint32_t k_tile_index =  std::ceil((double)(k_outer_offset + k_inner_offset) / _config.core_width);
                uint32_t n_tile_index =  std::ceil((double)(n_outer_offset + n_inner_offset) / _config.core_width);

                // SRAM act L1 tile offset
                addr_type sram_activation_offset = sram_activation_base + (m_inner_offset * k_inner + k_inner_offset) * MyAddressAllocator::precision_activation;
                // SRAM wgt L1 tile offset
                addr_type sram_weight_offset = sram_weight_base + (k_inner_offset * n_inner + n_inner_offset) * MyAddressAllocator::precision_weight;
                // SRAM out L1 tile offset
                addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * MyAddressAllocator::precision_psum;

                // -- activation --
                if (n_inner_offset == 0) {
                    std::vector<std::vector<uint32_t>> activation_indexes;
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                        for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                            if (m_outer_offset + m_inner_offset + m_loop >= M_max or k_outer_offset + k_inner_offset + k_loop >= K_max) {continue;}
                            activation_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, k_outer_offset + k_inner_offset + k_loop});
                        }
                    }

                    auto activation_addrs = activation_tensor->generate_addrs_based_on_indexes(activation_indexes);
                    if (activation_addrs.empty()) {
                        spdlog::info(
                            "zero load for activation m: {} {} / k: "
                            "{} {} / activation tensor dim: {}",
                            m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset,
                            activation_tensor->get_dims());
                    } else {
                        std::string movin_info = fmt::format("Load Activation with the begin m = {}-{} and k = {}-{}",activation_indexes.front()[0], activation_indexes.back()[0], activation_indexes.front()[1], activation_indexes.back()[1]);
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_activation_offset,
                            .size = (uint32_t) activation_indexes.size() * MyAddressAllocator::precision_activation,
                            .src_addrs = std::move(activation_addrs),
                            .operand_id = _INPUT_OPERAND,
                            .inst_information = movin_info,
                        });
                    }
                }

                // -- weight --
                if (m_inner_offset == 0) {
                    std::vector<std::vector<uint32_t>> weight_indexes;
                    for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                            if (k_outer_offset + k_inner_offset + k_loop >= K_max or n_outer_offset + n_inner_offset + n_loop >= N_max) {continue;}
                            weight_indexes.push_back(std::vector<uint32_t>{k_outer_offset + k_inner_offset + k_loop, n_outer_offset + n_inner_offset + n_loop});
                        }
                    }

                    auto weight_addrs = weight_tensor->generate_addrs_based_on_indexes(weight_indexes);

                    if (weight_addrs.empty()) {
                        spdlog::info(
                            "operation name : {} / "
                            "zero load for weight k: {} {} / n: {} {} "
                            "/ weight tensor dim: {} / is transposed: {}",
                            get_name(), k_outer_offset, k_inner_offset, n_outer_offset,
                            n_inner_offset, weight_tensor->get_dims(),
                            weight_tensor->_is_transposed);
                    }
                    else {
                        std::string movin_info = fmt::format("Load weight with the begin index {} and end index {}", weight_indexes.front(), weight_indexes.back());
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_weight_offset,
                            .size = (uint32_t)weight_indexes.size() * MyAddressAllocator::precision_weight,
                            .src_addrs = std::move(weight_addrs),
                            .operand_id = _INPUT_OPERAND + 1,
                            .inst_information = movin_info,
                        });
                    }
                }

                // -- compute --
                if ((m_outer_offset + m_inner_offset) < M_max and(k_outer_offset + k_inner_offset) < K_max and (n_outer_offset + n_inner_offset) < N_max) {
                    std::string gemm_info = fmt::format("GEMM computation for L1 tile: m = {}, k = {}, n = {}", m_tile_index, k_tile_index, n_tile_index);

                    tile.instructions.push_back(Instruction{
                        .opcode = (m_inner_offset == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM),
                        .dest_addr = sram_accumulation_offset,
                        // xxx : fixed to systolic array size 8
                        .size = loop_size,
                        // what does src_addrs do in computation instructions?
                        // read Core::can_issue_compute.
                        // checks if it's loaded to sram.
                        .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_weight_offset},
                        .tile_m = m_tile_index,
                        .tile_k = k_tile_index,
                        .tile_n = n_tile_index,
                        .inst_information = gemm_info,
                    });
                }
                else {
                    spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                    spdlog::info("activation dims {}, weight dim {}", activation_tensor->get_dims(), weight_tensor->get_dims());
                }

                // -- store --
                // when iterating inner_loop k times,
                // store L1 tile to output
                if (should_store && (k_inner_offset + loop_size >= k_inner)) { // 当前Tile的最后一次K方向计算累加
                    // 生成output index
                    std::vector<std::vector<uint32_t>> output_indexes;  // 这里修改了生成index的顺序
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                            if (m_outer_offset + m_inner_offset + m_loop >= weight_tensor->_dims[0] or n_outer_offset + n_inner_offset + n_loop >= weight_tensor->_dims[1]) {
                                continue;
                            }
                            output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, n_outer_offset + n_inner_offset + n_loop});
                        }
                    }
                    auto output_addrs = output_tensor->generate_addrs_based_on_indexes(output_indexes);

                    if (!output_addrs.empty()) {
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVOUT,
                            .dest_addr = sram_accumulation_offset,
                            .size = (uint32_t)output_indexes.size() * MyAddressAllocator::precision_activation,
                            .src_addrs = std::move(output_addrs),
                            .operand_id = _OUTPUT_OPERAND,
                        });
                    }
                }
            }
        }
    }

    if (_is_transposed) {
        activation_tensor->unset_transposed();
        weight_tensor->unset_transposed();
    }

    // spdlog::info("{} instructions generated from tile {}",
    // t.instructions.size(), t.optype); spdlog::info("outer loop {}, inner loop
    // {}", _outer_loop, _inner_loop);

    return tile;
}


Tile GEMM::initialize_my_DASH_instructions(uint32_t B, uint32_t M, uint32_t K, uint32_t N, bool should_store) {

    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = B,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,   // if K != 0, the result need to accumulated
    };

    addr_type act_addr = 0;
    addr_type wgt_addr = 0;

    // base on the inner loop, initialize instructions
    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop[0];  // M-axis L2 tile size
    auto k_inner = _inner_loop[1];  // K-axis L2 tile size
    auto n_inner = _inner_loop[2];  // N-axis L2 tile size

    // 计算出每一块个Tile的对应位置
    auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
    auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
    auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

    // load tile to SPM in order ACT / WGT  计算出Activation，Weight计算过程中在SRAM中的存放位置
    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_weight_base = SPAD_BASE + m_inner * k_inner * MyAddressAllocator::precision_activation;  // size of Activation = m_inner * k_inner * _config.precision
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.core_width;
    // 参与MatMul计算的Weight与Activation, 在多个Batch的情况下，对应的Activate Tensor在迭代计算过程中会存在改变的可能，需要基于M来确定索引
    //************************************************
    // 最重要的步骤，基于现有的每一个Tensor的规模，基于索引映射到tensor的相对位置，获得Tensor的加载地址，在边界处也要进行处理，从而在跨Batch计算的过程中可以正确的取出对应的Tensor

    auto activation_tensors= _my_inputs;
    auto weight_tensor = _my_weights[0];
    auto output_tensors = _my_outputs;

    // auto activation_tensor = _my_inputs[0];
    // auto weight_tensor = _my_weights[0];
    // auto output_tensor = _my_outputs[0];

    uint32_t M_max = _m_batch_dim.back();  // activation_tensor->_dims[0];
    uint32_t K_max = weight_tensor->_dims[0];
    uint32_t N_max = weight_tensor->_dims[1];

    // 对于K=0的Tile，初始阶段需要加载Bias
    if (_my_weights.size() == 2 && K == 0) {
        //spdlog::info("Load n = {}-{} Bias for N = {} outer loop", n_outer_offset, n_outer_offset + n_inner, N);
        auto bias_tensor =_my_weights[1];
        std::vector<std::vector<uint32_t>> bias_indexes;  // 按照顺序，生成相应的index
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) { // n_inner_offset: L1 tile start index in each L2 tile
            for (uint32_t n_loop = 0; n_loop < loop_size; ++n_loop) {
                uint32_t bias_index = n_outer_offset + n_inner_offset + n_loop;
                if (bias_index >= bias_tensor->_dims[0]) {  // 由于基于tile划分，因此可能会出现补零，这种情况下跳过
                    continue;
                }
                bias_indexes.push_back(std::vector<uint32_t>{bias_index}); // 存储Index的形式
            }
        }

        // 将index进行计算，转换为加载地址
        auto bias_addrs = bias_tensor->generate_addrs_based_on_indexes(bias_indexes);

        if (bias_addrs.empty() && !should_store) {
            spdlog::info("zero load for activation n: {} / bias tensor dim: {}", n_outer_offset, bias_tensor->get_dims());
        } else {
            std::string movin_info = fmt::format("Load Bias with the n = {}-{}",bias_indexes.front()[0], bias_indexes.back()[0]);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_accumulation_base,  // 将bias加载至sram_accumulation_base中，用于完成累加操作
                .size = (uint32_t)bias_indexes.size() * MyAddressAllocator::precision_weight,
                .src_addrs = std::move(bias_addrs),
                .operand_id = _INPUT_OPERAND + 2,
                .inst_information = movin_info
            });
        }
    }

    // 计算过程，N为最外侧,计算优先级为M，K，N，加载Activation与weight
    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
        for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
            for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += loop_size) {

                // spdlog::info("Current m_offset = {}, k_offset = {}, n_offset = {}", m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                uint32_t m_tile_index =  std::ceil((double)(m_outer_offset + m_inner_offset) / _config.core_height);
                uint32_t k_tile_index =  std::ceil((double)(k_outer_offset + k_inner_offset) / _config.core_width);
                uint32_t n_tile_index =  std::ceil((double)(n_outer_offset + n_inner_offset) / _config.core_width);

                // SRAM activation L1 tile offset
                addr_type sram_activation_offset = sram_activation_base + (m_inner_offset * k_inner + k_inner_offset) * MyAddressAllocator::precision_activation;
                // SRAM wgt L1 tile offset
                addr_type sram_weight_offset = sram_weight_base + (k_inner_offset * n_inner + n_inner_offset) * MyAddressAllocator::precision_weight;
                // SRAM output result L1 tile offset
                addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * _my_outputs[0]->_precision;  // 输出位宽与数据类型相关

                // -- activation -- if inner offset == 0 && k inner offset == 0, load m tiles Activation
                if (n_inner_offset == 0) {
                    // During the n_inner tile iterations (to prevent duplication), add the MOVIN instruction only in the first inner loop.
                    std::map<uint32_t, std::vector<std::vector<uint32_t>>> activation_indexes;
                    uint32_t batch_index = 0;
                    uint32_t previous_batch_boundary = 0;
                    uint32_t batch_boundary = _m_batch_dim[batch_index];
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {  // 这里需要判断的内容：m对应的Activation的位置是否超过边界, 并划分到不同的batch
                        uint32_t activation_m_index = m_outer_offset + m_inner_offset + m_loop;
                        if (activation_m_index >= M_max) {
                            continue;
                        }
                        while (activation_m_index >= batch_boundary) { // 当前的m_index，找到对应的boundary, 对应的batch index就是相关的Tensor
                            previous_batch_boundary = _m_batch_dim[batch_index];
                            batch_index++;
                            batch_boundary = _m_batch_dim[batch_index];
                        }
                        for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                            uint32_t activation_k_index = k_outer_offset + k_inner_offset + k_loop;
                            if (activation_k_index >= K_max) {
                                continue;  // 超过边界跳过
                            }
                            activation_indexes[batch_index].push_back(std::vector<uint32_t>{activation_m_index - previous_batch_boundary, activation_k_index});
                        }
                    }

                    /*
                    if (activation_indexes.size()>1) {
                        spdlog::info("Load Activation with the begin m = {}-{} and k = {}-{}",
                        m_outer_offset + m_inner_offset, std::min(m_outer_offset + m_inner_offset + loop_size, M_max),
                        k_outer_offset + k_inner_offset, std::min( k_outer_offset + k_inner_offset + loop_size, K_max));
                    }
                    */
                    std::vector<addr_type> activation_addrs;
                    uint32_t activation_index_size = 0;
                    if (activation_indexes.empty()) {
                        spdlog::info("No valid activation tiles to load.");
                    }
                    else {
                        for (const auto& [batch_id, indexes] : activation_indexes) {
                            // 假设你有 per-batch tensor 列表：_activation_batch_tensors[batch_id]
                            activation_index_size += indexes.size();
                            auto addrs = activation_tensors[batch_id]->generate_addrs_based_on_indexes(indexes);
                            activation_addrs.insert(activation_addrs.end(), addrs.begin(), addrs.end());
                        }
                    }

                    // auto activation_addrs = activation_tensors[batch_index]->generate_addrs_based_on_indexes(activation_indexes);

                    if (activation_addrs.empty()) {
                        spdlog::info("zero load for activation m: {}, {} / k: {}, {}",m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset);
                        // assert(0);
                    } else {
                        // spdlog::info("Load Activation with the begin m = {}-{} and k = {}-{}",activation_indexes.front()[0], activation_indexes.back()[0], activation_indexes.front()[1], activation_indexes.back()[1]);
                        std::string movin_info = fmt::format("Load Activation with the begin m = {}-{} and k = {}-{}",
                            m_outer_offset + m_inner_offset, std::min(m_outer_offset + m_inner_offset + loop_size, M_max),
                            k_outer_offset + k_inner_offset, std::min( k_outer_offset + k_inner_offset + loop_size, M_max));

                        spdlog::debug("Generate {} instructions for activation load", activation_addrs.size());
                        // MyAddressAllocator::check_addrs(activation_addrs);

                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_activation_offset,
                            .size = activation_index_size * MyAddressAllocator::precision_activation,
                            .src_addrs = std::move(activation_addrs),
                            .operand_id = _INPUT_OPERAND,
                            .inst_information = movin_info
                        });
                    }
                }

                // -- weight --
                if (m_inner_offset == 0) {
                    // During the m_inner tile iterations (to prevent duplication),
                    // add the MOVIN instruction only in the first inner loop.
                    // std::vector<addr_type> weight_addrs;
                    // 生成weight加载的index，每一个小的tile中，index的生成按照列优先的顺序，按行生成相应的地址.
                    std::vector<std::vector<uint32_t>> weight_indexes;
                    if (weight_tensor->_tensor_type == TensorType::KCache) {
                        for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                            for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                                if (k_outer_offset + k_inner_offset + k_loop >= weight_tensor->_dims[1] or n_outer_offset + n_inner_offset + n_loop >= weight_tensor->_dims[0]) {
                                    continue;
                                }
                                weight_indexes.push_back(std::vector<uint32_t>{n_outer_offset + n_inner_offset + n_loop, k_outer_offset + k_inner_offset + k_loop});
                            }
                        }
                    }
                    else {
                        for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                            for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                                if (k_outer_offset + k_inner_offset + k_loop >= weight_tensor->_dims[0] or n_outer_offset + n_inner_offset + n_loop >= weight_tensor->_dims[1]) {
                                    continue;
                                }
                                weight_indexes.push_back(std::vector<uint32_t>{k_outer_offset + k_inner_offset + k_loop, n_outer_offset + n_inner_offset + n_loop});
                            }
                        }
                    }

                    auto weight_addrs = weight_tensor->generate_addrs_based_on_indexes(weight_indexes);
                    if (weight_addrs.empty()) {
                        spdlog::info(
                            "operation name : {} / zero load for weight k: {} {} / n: {} {} / weight tensor dim: {} / is transposed: {}",
                            get_name(), k_outer_offset, k_inner_offset, n_outer_offset,
                            n_inner_offset, weight_tensor->get_dims(),
                            weight_tensor->_is_transposed);
                        //spdlog::info("inner loop {}, outer loop {}, act_size {}, wgt_size {}", _inner_loop, _outer_loop, activation_tensor->get_dims(), weight_tensor->get_dims());
                    }
                    else {
                        // spdlog::info("Load weight with the begin index {} and end index {}", weight_indexes.front(), weight_indexes.back());
                        // spdlog::info("Load weight with K = {}-{}, N = {}-{}", weight_indexes.front()[0], weight_indexes.back()[0], weight_indexes.front()[1], weight_indexes.back()[1]);
                        std::string movin_info = fmt::format("Load weight with the begin index {} and end index {}", weight_indexes.front(), weight_indexes.back());
                        spdlog::debug("Generate {} addrs for weight load", weight_addrs.size());
                        // MyAddressAllocator::check_addrs(weight_addrs);
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_weight_offset,
                            .size = (uint32_t)weight_indexes.size() * MyAddressAllocator::precision_weight,
                            .src_addrs = std::move(weight_addrs),
                            .operand_id = _INPUT_OPERAND + 1,
                            .per_ch_inst = (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS && MyAddressAllocator::IANUS_channel_parallel==false),
                            .inst_information = movin_info
                        });
                    }
                }

                // -- compute --
                if ((m_outer_offset + m_inner_offset) < M_max and(k_outer_offset + k_inner_offset) < K_max and (n_outer_offset + n_inner_offset) < N_max) {
                    std::string gemm_info = fmt::format("GEMM Computation tile index: M={}, K={}, N={}", m_tile_index, k_tile_index, n_tile_index);
                    tile.instructions.push_back(Instruction{
                        .opcode = (m_inner_offset == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM),  // M=0的情况下，需要一个更长的启动延迟(K的预加载)，其余的M可以对于加载的weight进行复用
                        .dest_addr = sram_accumulation_offset,
                        .size = loop_size,
                        .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_weight_offset}, //
                        .tile_m = m_tile_index,
                        .tile_k = k_tile_index,
                        .tile_n = n_tile_index,
                        .inst_information = gemm_info,
                    });
                }
                else {
                    spdlog::info("Computation jump: M_max = {}, K_max = {}, N_max = {}, current m_start = {}, k_start = {}, n_start = {}", M_max, K_max, N_max,
                    m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                    // spdlog::info("activation dims {}, weight dim {}", activation_tensor->get_dims(), weight_tensor->get_dims());
                }

                // -- store --
                // when iterating inner_loop k times, should store L1 tile to output
                // 仍然可能存在由WK和WV参与，输出结果为KV Cache的情况，对应生成的offset可能存在一定的区别 -- 最好都能集成在Mytensor中，基于输入的index计算tile的index完成计算过程
                // 可以基于Tile的offset完成，比较好统一进行修改
                if (should_store && k_inner_offset + loop_size >= k_inner) {  // should store代表最后一层output
                    std::map<uint32_t, std::vector<std::vector<uint32_t>>> output_indexes;
                    uint32_t batch_index = 0;
                    uint32_t previous_batch_boundary = 0;
                    uint32_t batch_boundary = _m_batch_dim[batch_index];
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                        uint32_t output_m_index = m_outer_offset + m_inner_offset + m_loop;
                        if (output_m_index >= M_max) {
                            continue;
                        }
                        while (output_m_index >= batch_boundary) {
                            previous_batch_boundary = _m_batch_dim[batch_index];
                            batch_index++;
                            batch_boundary = _m_batch_dim[batch_index];
                        }
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                            uint32_t output_n_index = n_outer_offset + n_inner_offset + n_loop;
                            if (output_n_index >= N_max) {
                                continue;
                            }
                            output_indexes[batch_index].push_back(std::vector<uint32_t>{output_m_index - previous_batch_boundary, output_n_index});
                        }
                    }

                    if (output_indexes.size() > 1) {
                        // spdlog::info("Load Activation with the begin m = {}-{} and n = {}-{}",
                        // m_outer_offset + m_inner_offset, std::min(m_outer_offset + m_inner_offset + loop_size, M_max),
                        // n_outer_offset + n_inner_offset, std::min( n_outer_offset + n_inner_offset + loop_size, N_max));
                        // spdlog::info("Previous batch boundary is {}", previous_batch_boundary);
                    }

                    // Output Address generation
                    std::vector<addr_type> output_addrs;
                    uint32_t output_index_size = 0;
                    if (output_indexes.empty()) {
                        spdlog::info("No valid output tiles to store.");
                    } else {
                        for (const auto& [batch_id, indexes] : output_indexes) {
                            output_index_size += indexes.size();
                            auto addrs = output_tensors[batch_id]->generate_addrs_based_on_indexes(indexes);
                            output_addrs.insert(output_addrs.end(), addrs.begin(), addrs.end());
                        }
                    }

                    if (!output_addrs.empty()) {
                        std::string movout_info = fmt::format("Store GEMM result (multi-batch) with M = {}-{}, N = {}-{}",
                        m_outer_offset + m_inner_offset, std::min(m_outer_offset + m_inner_offset + loop_size, M_max),
                        n_outer_offset + n_inner_offset, std::min(n_outer_offset + n_inner_offset + loop_size, N_max));

                        append_rope_if_enabled(tile.instructions, _apply_rope, sram_accumulation_offset,
                                               output_index_size,
                                               "Apply RoPE before storing GEMM projection");
                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVOUT,
                            .dest_addr = sram_accumulation_offset,
                            .size = output_index_size * output_tensors[0]->_precision,
                            .src_addrs = std::move(output_addrs),
                            .operand_id = _OUTPUT_OPERAND,
                            .per_ch_inst = (output_tensors[0]->_tensor_type == TensorType::KCache or output_tensors[0]->_tensor_type == TensorType::VCache),
                            .inst_information = movout_info,
                        });
                    }
                }
            }
        }
    }
    return tile;
}


Tile GEMM::initialize_my_attention_instructions(uint32_t B, uint32_t head_index, uint32_t M, uint32_t K, uint32_t N, bool should_store) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = B,
        .head_index = head_index,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,   // if K != 0, the result need to accumulated
    };

    addr_type act_addr = 0;
    addr_type wgt_addr = 0;

    // base on the inner loop, initialize instructions
    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop_attn[B][0];  // M-axis L2 tile size
    auto k_inner = _inner_loop_attn[B][1];  // K-axis L2 tile size
    auto n_inner = _inner_loop_attn[B][2];  // N-axis L2 tile size

    // load tile to SPM in order ACT / WGT  计算出Activation，Weight计算过程中在SRAM中的存放位置
    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_weight_base = SPAD_BASE + m_inner * k_inner * MyAddressAllocator::precision_activation;  // size of Activation = m_inner * k_inner * _config.precision
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.core_width;

    // 参与MatMul计算的Weight与Activation 需要判断是QKT还是SV，之后生成相应的计算约束
    auto activation_tensor = _my_inputs[B];
    auto weight_tensor = _my_weights[B];
    auto output_tensor = _my_outputs[B];

    // Head index被利用在k的索引生成
    if (weight_tensor->_tensor_type == TensorType::KCache) {  // QKT
        // spdlog::info("Initialize the QKT computation of head {}", head_index);
        uint32_t q_head_index = head_index;
        uint32_t kv_head_index = MyAddressAllocator::get_kv_head_index(q_head_index);

        uint32_t M_max = activation_tensor->_dims[0]; // Lin
        uint32_t K_max = MyAddressAllocator::d_k; // d_k
        uint32_t N_max = weight_tensor->_dims[0]; // Lin

        // 计算出每一块个Tile的对应位置
        auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
        auto q_k_outer_offset = k_inner * K + q_head_index * MyAddressAllocator::d_k;  // Q hidden slice for q head
        auto kcache_k_outer_offset = k_inner * K + kv_head_index * MyAddressAllocator::d_k;  // K cache slice for mapped kv head
        auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

        // Tile生成的单位按照head循环进行，基于head index实现定位
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
            for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
                for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += loop_size) {
                    uint32_t m_tile_index = std::ceil((double)(m_outer_offset + m_inner_offset) / _config.core_height);
                    uint32_t k_tile_index = std::ceil((double)(q_k_outer_offset + k_inner_offset) / _config.core_width);
                    uint32_t n_tile_index = std::ceil((double)(n_outer_offset + n_inner_offset) / _config.core_width);

                    addr_type sram_activation_offset = sram_activation_base + (m_inner_offset * k_inner + k_inner_offset) * activation_tensor->_precision;
                    addr_type sram_weight_offset = sram_weight_base + (k_inner_offset * n_inner + n_inner_offset) * weight_tensor->_precision;
                    addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * output_tensor->_precision;

                    // Activation (Q)
                    if (n_inner_offset == 0) {
                        std::vector<std::vector<uint32_t>> activation_indexes;
                        for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                            for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                                uint32_t activation_m_index = m_outer_offset + m_inner_offset + m_loop;
                                uint32_t activation_k_index = q_k_outer_offset + k_inner_offset + k_loop;
                                if (activation_m_index >= activation_tensor->_dims[0] or activation_k_index >= activation_tensor->_dims[1]) {
                                    continue;  // TODO:: 超过每一个注意力头的边界跳过
                                }
                                activation_indexes.push_back(std::vector<uint32_t>{activation_m_index, activation_k_index});
                            }
                        }

                        auto activation_addrs = activation_tensor->generate_addrs_based_on_indexes(activation_indexes);
                        if (activation_addrs.empty()) {
                            spdlog::info("zero load for Q m: {} {} / k: {} {} / activation tensor dim: {}",m_outer_offset, m_inner_offset, q_k_outer_offset, k_inner_offset, activation_tensor->get_dims());
                        } else {
                            // spdlog::info("QKT operation for the Q Matrix m = {}-{} and k = {}-{}",activation_indexes.front()[0], activation_indexes.back()[0], activation_indexes.front()[1], activation_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load Q matrix with the begin index {} and end index {}", activation_indexes.front(), activation_indexes.back());
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_activation_offset,
                                .size = (uint32_t)activation_indexes.size() * activation_tensor->_precision,
                                .src_addrs = std::move(activation_addrs),
                                .operand_id = _INPUT_OPERAND,
                                .inst_information = movin_info
                            });
                        }
                    }

                    // Weight (K Cache), 对于K的加载过程不考虑，仍然按照[Lin, dk]将相应的K Cache加载进来即可，对应k的取值对应k的offset，n的取值对应Lout
                    if (m_inner_offset == 0) {
                        std::vector<std::vector<uint32_t>> weight_indexes;
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {  // 在KCache存储，转置计算之前，n方向为行，k方向为列
                            for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                                uint32_t weight_n_index = n_outer_offset + n_inner_offset + n_loop;
                                uint32_t weight_k_index = kcache_k_outer_offset + k_inner_offset + k_loop;
                                if (weight_n_index >= weight_tensor->_dims[0] or
                                    k_inner * K + k_inner_offset + k_loop >= MyAddressAllocator::d_k or
                                    weight_k_index >= weight_tensor->_dims[1]) {
                                    continue;
                                }
                                weight_indexes.push_back(std::vector<uint32_t>{weight_n_index, weight_k_index});
                            }
                        }
                        auto weight_addrs = weight_tensor->generate_addrs_based_on_indexes(weight_indexes); // 对于K Cache，地址生成过程中可以基于index计算所属Head
                        if (weight_addrs.empty()) {
                            spdlog::info("operation name : {} / zero load for weight k: {} {} / n: {} {} / weight tensor dim: {} / is transposed: {}",
                                get_name(), kcache_k_outer_offset, k_inner_offset, n_outer_offset,n_inner_offset, weight_tensor->get_dims(),weight_tensor->_is_transposed);
                        }
                        else {
                            // spdlog::info("QKT operation for the K Cache with n = {}-{}, k = {}-{}", weight_indexes.front()[0], weight_indexes.back()[0], weight_indexes.front()[1], weight_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load K Cache with the begin index {} and end index {}", weight_indexes.front(), weight_indexes.back());
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_weight_offset,
                                .size = (uint32_t)weight_indexes.size() * MyAddressAllocator::precision_weight,
                                .src_addrs = std::move(weight_addrs),
                                .operand_id = _INPUT_OPERAND + 1,
                                .per_ch_inst = true,
                                .inst_information = movin_info,
                            });
                        }
                    }

                    // -- compute --
                    if ((m_outer_offset + m_inner_offset) < M_max and (k_inner * K + k_inner_offset) < K_max and (n_outer_offset + n_inner_offset) < N_max) {
                        std::string gemm_info = fmt::format("GEMM Computation tile index: M={}, K={}, N={}", m_tile_index, k_tile_index, n_tile_index);
                        // spdlog::info(gemm_info);
                        tile.instructions.push_back(Instruction{
                            .opcode = (m_inner_offset == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM),
                            .dest_addr = sram_accumulation_offset,
                            .size = loop_size,
                            .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_weight_offset},
                            .tile_m = m_tile_index,
                            .tile_k = k_tile_index,
                            .tile_n = n_tile_index,
                            .inst_information = gemm_info,
                        });
                    }
                    else {
                        spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, q_k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                        spdlog::info("activation dims {}, weight dim {}", activation_tensor->get_dims(), weight_tensor->get_dims());
                    }

                    // -- Store --
                    if (should_store && k_inner_offset + loop_size >= k_inner) {
                        std::vector<std::vector<uint32_t>> output_indexes;
                        for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                            for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= activation_tensor->_dims[0] or n_outer_offset + n_inner_offset + n_loop >= N_max) {
                                    continue;
                                }
                                output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, n_outer_offset + n_inner_offset + n_loop});
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
    else if (weight_tensor->_tensor_type == TensorType::VCache) {
        // spdlog::info("Initialize the SV computation of head {}", head_index);
        uint32_t q_head_index = head_index;
        uint32_t kv_head_index = MyAddressAllocator::get_kv_head_index(q_head_index);
        uint32_t M_max = activation_tensor->_dims[1]; // Lin
        uint32_t K_max = weight_tensor->_dims[0]; // Lin
        uint32_t N_max = MyAddressAllocator::d_k; // d_k

        // 计算出每一块个Tile的对应位置
        auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
        auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
        auto output_n_outer_offset = n_inner * N + q_head_index * MyAddressAllocator::d_k;  // Output hidden slice for q head
        auto vcache_n_outer_offset = n_inner * N + kv_head_index * MyAddressAllocator::d_k;  // V cache slice for mapped kv head

        // Tile生成的单位按照head循环进行，基于head index实现定位
        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
            for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
                for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += loop_size) {
                    uint32_t m_tile_index =  std::ceil((double)(m_outer_offset + m_inner_offset) / _config.core_height);
                    uint32_t k_tile_index =  std::ceil((double)(k_outer_offset + k_inner_offset) / _config.core_width);
                    uint32_t n_tile_index =  std::ceil((double)(output_n_outer_offset + n_inner_offset) / _config.core_width);

                    addr_type sram_activation_offset = sram_activation_base + (m_inner_offset * k_inner + k_inner_offset) * activation_tensor->_precision;
                    addr_type sram_weight_offset = sram_weight_base + (k_inner_offset * n_inner + n_inner_offset) * weight_tensor->_precision;
                    addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * output_tensor->_precision;

                    // Activation (S)  // S是三维情况，同时存在各Attention head分离，因此索引与地址生成与QKT Output Index的情况类似
                    if (n_inner_offset == 0) {
                        std::vector<std::vector<uint32_t>> activation_indexes;
                        for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                            for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                                uint32_t activation_m_index = m_outer_offset + m_inner_offset + m_loop;
                                uint32_t activation_k_index = k_outer_offset + k_inner_offset + k_loop;
                                if (activation_m_index >= activation_tensor->_dims[1] or activation_k_index >= activation_tensor->_dims[1]) {
                                    continue;  // TODO:: 超过每一个注意力头的边界跳过
                                }
                                activation_indexes.push_back(std::vector<uint32_t>{activation_m_index, activation_k_index});
                            }
                        }
                        auto activation_addrs = activation_tensor->generate_addrs_based_on_indexes_attention(activation_indexes, head_index);
                        if (activation_addrs.empty()) {
                            spdlog::info("zero load for Q m: {} {} / k: {} {} / activation tensor dim: {}",m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset, activation_tensor->get_dims());
                        } else {
                            //spdlog::info("SV operation for the S Matrix m = {}-{} and k = {}-{}",activation_indexes.front()[0], activation_indexes.back()[0], activation_indexes.front()[1], activation_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load S matrix for head {} with the begin index {} and end index {}", head_index, activation_indexes.front(), activation_indexes.back());
                            // spdlog::info(movin_info);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_activation_offset,
                                .size = (uint32_t)activation_indexes.size() * activation_tensor->_precision,
                                .src_addrs = std::move(activation_addrs),
                                .operand_id = _INPUT_OPERAND,
                                .inst_information = movin_info
                            });
                        }
                    }

                    // Weight (V Cache), V Cache的加载基于[Lin, dk]，对应的索引没有变化
                    if (m_inner_offset == 0) {
                        std::vector<std::vector<uint32_t>> weight_indexes;
                        for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                            for (int n_loop = 0; n_loop < loop_size; n_loop++) {  // 在KCache存储，转置计算之前，n方向为行，k方向为列
                                uint32_t weight_k_index = k_outer_offset + k_inner_offset + k_loop;
                                uint32_t weight_n_index = vcache_n_outer_offset + n_inner_offset + n_loop;
                                if (weight_k_index >= weight_tensor->_dims[0] or
                                    n_inner_offset + n_loop >= MyAddressAllocator::d_k or
                                    weight_n_index >= weight_tensor->_dims[1]) {
                                    continue;
                                }
                                weight_indexes.push_back(std::vector<uint32_t>{weight_k_index, weight_n_index});
                            }
                        }
                        auto weight_addrs = weight_tensor->generate_addrs_based_on_indexes(weight_indexes); // 对于K Cache，地址生成过程中可以基于index计算所属Head
                        if (weight_addrs.empty()) {
                            spdlog::info("operation name : {} / zero load for weight k: {} {} / n: {} {} / weight tensor dim: {} / is transposed: {}",
                                get_name(), k_outer_offset, k_inner_offset, vcache_n_outer_offset,n_inner_offset, weight_tensor->get_dims(),weight_tensor->_is_transposed);
                        }
                        else {
                            // spdlog::info("SV operation for the V Cache with k = {}-{}, n = {}-{}", weight_indexes.front()[0], weight_indexes.back()[0], weight_indexes.front()[1], weight_indexes.back()[1]);
                            std::string movin_info = fmt::format("Load V Cache for head {} with the begin index {} and end index {}", head_index, weight_indexes.front(), weight_indexes.back());
                            // spdlog::info(movin_info);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVIN,
                                .dest_addr = sram_weight_offset,
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
                        std::string gemm_info = fmt::format("GEMM Computation tile index: M={}, K={}, N={}", m_tile_index, k_tile_index, n_tile_index);
                        // spdlog::info(gemm_info);
                        tile.instructions.push_back(Instruction{
                            .opcode = (m_inner_offset == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM),
                            .dest_addr = sram_accumulation_offset,
                            .size = loop_size,
                            .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_weight_offset},
                            .tile_m = m_tile_index,
                            .tile_k = k_tile_index,
                            .tile_n = n_tile_index,
                            .inst_information = gemm_info,
                        });
                    }
                    else {
                        spdlog::info("Computation jump: current m_start = {}, k_start = {}, n_start = {}",  m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, output_n_outer_offset + n_inner_offset);
                        spdlog::info("activation dims {}, weight dim {}", activation_tensor->get_dims(), weight_tensor->get_dims());
                    }

                    // -- Store --
                    if (should_store && k_inner_offset + loop_size >= k_inner) {  // && (k_inner_offset + loop_size >= k_inner) // 这里在should store的最后一次K循环, 完成了K方向的所有计算，将计算的PSUM结果输出
                        std::vector<std::vector<uint32_t>> output_indexes;
                        for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                            for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                                if (m_outer_offset + m_inner_offset + m_loop >= activation_tensor->_dims[1] or output_n_outer_offset + n_inner_offset + n_loop >= output_tensor->_dims[1]) {
                                    continue;
                                }
                                output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, output_n_outer_offset + n_inner_offset + n_loop});
                            }
                        }
                        // 将index进行计算，转换为加载地址
                        auto gen_output_addrs = output_tensor->generate_addrs_based_on_indexes_attention(output_indexes, head_index);
                        if (!gen_output_addrs.empty()) {
                            std::string movout_info = fmt::format("Store SV result of head {} with M = {}-{}, N = {}-{}", head_index, output_indexes.front()[0], output_indexes.back()[0], output_indexes.front()[1], output_indexes.back()[1]);
                            // spdlog::info(movout_info);
                            tile.instructions.push_back(Instruction{
                                .opcode = Opcode::MOVOUT,
                                .dest_addr = sram_accumulation_offset,
                                .size = (uint32_t)output_indexes.size() * output_tensor->_precision,
                                .src_addrs = std::move(gen_output_addrs),
                                .operand_id = _OUTPUT_OPERAND,
                                .inst_information = movout_info,
                            });
                        }
                    }
                }
            }
        }
    }
    else {
        throw std::runtime_error("Unsupported tensor type for Atttention operation");
    }
    return tile;
}



Tile GEMM::initialize_my_DASH_instructions_activation_first(uint32_t B, uint32_t M, uint32_t K, uint32_t N, bool should_store) {

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

    addr_type act_addr = 0;
    addr_type wgt_addr = 0;

    // base on the inner loop, initialize instructions
    // _inner_loop means L2 tile size， Tile的分块后，每一块的大小
    auto m_inner = _inner_loop[0];  // M-axis L2 tile size
    auto k_inner = _inner_loop[1];  // K-axis L2 tile size
    auto n_inner = _inner_loop[2];  // N-axis L2 tile size
    // 计算出每一块个Tile的对应位置
    auto m_outer_offset = m_inner * M;  // M-axis L2 tile idx
    auto k_outer_offset = k_inner * K;  // K-axis L2 tile idx
    auto n_outer_offset = n_inner * N;  // N-axis L2 tile idx

    // load tile to SPM in order ACT / WGT  Tile计算过程中，Activation，Weight在NPU的SRAM中的存放位置
    addr_type sram_activation_base = SPAD_BASE;
    addr_type sram_weight_base = SPAD_BASE + m_inner * k_inner * MyAddressAllocator::precision_activation;  // 加载计算过程中的Activation所占的空间 = m_inner * k_inner * _config.precision
    addr_type sram_accumulation_base = ACCUM_SPAD_BASE;

    const uint32_t loop_size = _config.core_width;

    // 参与MatMul计算的Weight与Activation
    auto activation_tensor = _my_inputs[0];
    auto weight_tensor = _my_inputs[1];
    auto output_tensor = _my_outputs[0];

    uint32_t tile_m;
    uint32_t tile_k;
    uint32_t tile_n;

    // Batch的部分之后可能需要，但现在用不上
    auto batch_index = std::vector<uint32_t>();
    if (_my_inputs[0]->get_dims().size() == 3) {
        batch_index.push_back(B);
    }

    // 对于K=0的Tile，初始阶段需要加载Bias
    if (_my_inputs.size() == 3 && K == 0) {
        spdlog::info("Load n = {}-{} Bias for N = {} outer loop", n_outer_offset, n_outer_offset + n_inner, N);
        auto bias_tensor =_my_inputs[2];

        std::vector<std::vector<uint32_t>> bias_indexes;  // 按照顺序，生成相应的index

        for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) { // n_inner_offset: L1 tile start index in each L2 tile
            for (uint32_t n_loop = 0; n_loop < loop_size; ++n_loop) {
                uint32_t bias_index = n_outer_offset + n_inner_offset + n_loop;
                if (bias_index >= bias_tensor->_dims[0]) {  // 由于基于tile划分，因此可能会出现补零，这种情况下跳过
                    continue;
                }
                bias_indexes.push_back(std::vector<uint32_t>{bias_index}); // 存储Index的形式
            }
        }

        // 将index进行计算，转换为加载地址
        auto bias_addrs = bias_tensor->generate_addrs_based_on_indexes(bias_indexes);
        if (bias_addrs.size() == 0) {
            spdlog::info("zero load for activation n: {} / bias tensor dim: {}", n_outer_offset, bias_tensor->get_dims());
            assert(0);
        } else {
            // spdlog::info("{} addrs have been generated for {} / bias tensor with dims {}", bias_addrs.size(), n_outer_offset, bias_tensor->get_dims());
            spdlog::info("Load Bias with N = {}-{}", bias_indexes.front(), bias_indexes.back());
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = sram_accumulation_base,  // 将bias加载至sram_accumulation_base中，用于完成累加操作
                .size = (uint32_t)bias_indexes.size() * MyAddressAllocator::precision_weight,
                .src_addrs = std::move(bias_addrs),
                .operand_id = _INPUT_OPERAND + 2,
            });
        }
    }

    // 计算过程，N为最外侧,计算优先级为M，K，N，加载Activation
    for (uint32_t n_inner_offset = 0; n_inner_offset < n_inner; n_inner_offset += loop_size) {
        for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += loop_size) {
            for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += loop_size) {
                // spdlog::info("Current m_offset = {}, k_offset = {}, n_offset = {}", m_outer_offset + m_inner_offset, k_outer_offset + k_inner_offset, n_outer_offset + n_inner_offset);
                // SRAM activation L1 tile offset
                addr_type sram_activation_offset = sram_activation_base + (m_inner_offset * k_inner + k_inner_offset) * MyAddressAllocator::precision_activation;
                // SRAM wgt L1 tile offset
                addr_type sram_weight_offset = sram_weight_base + (k_inner_offset * n_inner + n_inner_offset) * MyAddressAllocator::precision_weight;
                // SRAM output result L1 tile offset
                addr_type sram_accumulation_offset = sram_accumulation_base + (m_inner_offset * n_inner + n_inner_offset) * output_tensor->_precision;  // 输出位宽与数据类型相关

                // -- activation -- if inner offset == 0 && k inner offset == 0, load m tiles Activation
                if (n_inner_offset == 0 && k_inner_offset == 0) {  // 在这里进行了修改，每一个tile计算过程中, 对于Activation沿K方向进行连续的加载，从而进一步降低换行开销

                    tile_m = 0;
                    tile_k = 0;
                    // During the n_inner tile iterations (to prevent duplication), add the MOVIN instruction only in the first inner loop.
                    std::vector<std::vector<uint32_t>> activation_indexes;

                    // Activation一次加载量 = core_width × k_inner, 即尽量沿着K inner连续做加载
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                        for (int k_loop = 0; k_loop < k_inner; k_loop++) {
                            uint32_t activation_m_index = m_outer_offset + m_inner_offset + m_loop;
                            uint32_t activation_k_index = k_outer_offset + k_inner_offset + k_loop;

                            if (activation_m_index >= activation_tensor->_dims[0] or activation_k_index >= activation_tensor->_dims[1]) {
                                continue;  // 超过边界跳过
                            }
                            activation_indexes.push_back(std::vector<uint32_t>{activation_m_index, activation_k_index});

                            uint32_t burst_offset = (activation_m_index * activation_tensor->_dims[1] + activation_k_index)
                                * activation_tensor->_precision / MyAddressAllocator::memory_burst_size;

                            // save maximum tile_m, tile_k value
                            tile_m = m_loop + 1;
                            tile_k = k_loop + 1;
                        }
                    }
                    // 将index进行计算，转换为加载地址

                    auto activation_addrs = activation_tensor->generate_addrs_based_on_indexes(activation_indexes);

                    if (activation_addrs.size() == 0) {
                        spdlog::info(
                            "zero load for activation m: {} {} / k: {} {} / activation tensor dim: {}",
                            m_outer_offset, m_inner_offset, k_outer_offset, k_inner_offset,
                            activation_tensor->get_dims());
                        assert(0);
                    } else {
                        // spdlog::info("Load Activation with the begin m = {}-{} and k = {}-{}", activation_indexes.front()[0], activation_indexes.back()[0], activation_indexes.front()[1], activation_indexes.back()[1]);
                        // "Load weight with K = {}-{}, N = {}-{}", weight_indexes.front()[0], weight_indexes.front()[1], weight_indexes.back()[0], weight_indexes.back()[1]
                        tile.instructions.push_back(Instruction{
                        .opcode = Opcode::MOVIN,
                        .dest_addr = sram_activation_offset,
                        .size = (uint32_t)activation_indexes.size() * MyAddressAllocator::precision_activation,
                        .src_addrs = std::move(activation_addrs),
                        .operand_id = _INPUT_OPERAND});
                    }
                }

                // -- weight --
                if (m_inner_offset == 0) {
                    tile_n = 0;
                    // During the m_inner tile iterations (to prevent duplication),
                    // add the MOVIN instruction only in the first inner loop.
                    // std::vector<addr_type> weight_addrs;
                    // 生成weight加载的index，每一个小的tile中，index的生成按照列优先的顺序，按行生成相应的地址.
                    std::vector<std::vector<uint32_t>> weight_indexes;
                    for (int k_loop = 0; k_loop < loop_size; k_loop++) {
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                            if (k_outer_offset + k_inner_offset + k_loop >= weight_tensor->_dims[0] or n_outer_offset + n_inner_offset + n_loop >= weight_tensor->_dims[1]) {
                                continue;
                            }
                            weight_indexes.push_back(std::vector<uint32_t>{k_outer_offset + k_inner_offset + k_loop, n_outer_offset + n_inner_offset + n_loop});
                            tile_n = n_loop + 1;
                        }
                    }

                    auto gen_weight_addrs = weight_tensor->generate_addrs_based_on_indexes(weight_indexes);

                    if (gen_weight_addrs.size() == 0) {
                        spdlog::info(
                            "operation name : {} / zero load for weight k: {} {} / n: {} {} / weight tensor dim: {} / is transposed: {}",
                            get_name(), k_outer_offset, k_inner_offset, n_outer_offset,
                            n_inner_offset, weight_tensor->get_dims(),
                            weight_tensor->_is_transposed);
                        spdlog::info("inner loop {}, outer loop {}, act_size {}, wgt_size {}",
                                     _inner_loop, _outer_loop, activation_tensor->get_dims(),
                                     weight_tensor->get_dims());
                        assert(0);
                    } else {
                        // spdlog::info("Load weight with the begin index {} and end index {}", weight_indexes.front(), weight_indexes.back());
                        spdlog::info("Load weight with K = {}-{}, N = {}-{}", weight_indexes.front()[0], weight_indexes.back()[0], weight_indexes.front()[1], weight_indexes.back()[1]);
                        // MyAddressAllocator::check_addrs(gen_weight_addrs);

                        tile.instructions.push_back(Instruction{
                            .opcode = Opcode::MOVIN,
                            .dest_addr = sram_weight_offset,
                            .size = (uint32_t)weight_indexes.size() * MyAddressAllocator::precision_weight,
                            .src_addrs = std::move(gen_weight_addrs),
                            .operand_id = _INPUT_OPERAND + 1,
                        });
                    }
                }

                // -- compute --
                tile.instructions.push_back(Instruction{
                    .opcode = (m_inner_offset == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM),
                    .dest_addr = sram_accumulation_offset,
                    // xxx : fixed to systolic array size 8
                    .size = loop_size,
                    // what does src_addrs do in computation instructions?
                    // read Core::can_issue_compute.
                    // checks if it's loaded to sram.
                    .src_addrs = std::vector<addr_type>{sram_activation_offset, sram_weight_offset},

                    .tile_m = tile_m,
                    .tile_k = tile_k,
                    .tile_n = tile_n,
                });

                // -- store --
                // when iterating inner_loop k times, should store L1 tile to output
                // 仍然可能存在由WK和WV参与，输出结果为KV Cache的情况，对应生成的offset可能存在一定的区别 -- 最好都能集成在Mytensor中，基于输入的index计算tile的index完成计算过程
                // 可以基于Tile的offset完成，比较好统一进行修改
                if (should_store && (k_inner_offset + loop_size >= k_inner)) {  // should store代表最后一层output
                    std::vector<std::vector<uint32_t>> output_indexes;
                    for (int m_loop = 0; m_loop < loop_size; m_loop++) {
                        for (int n_loop = 0; n_loop < loop_size; n_loop++) {
                            if (m_outer_offset + m_inner_offset + m_loop >= activation_tensor->_dims[0] or n_outer_offset + n_inner_offset + n_loop >= weight_tensor->_dims[1]) {
                                continue;
                            }
                            output_indexes.push_back(std::vector<uint32_t>{m_outer_offset + m_inner_offset + m_loop, n_outer_offset + n_inner_offset + n_loop});
                        }
                    }
                    // 将index进行计算，转换为加载地址
                    auto output_addrs = output_tensor->generate_addrs_based_on_indexes(output_indexes);
                    spdlog::info("Store GEMM result with M = {}-{}, N = {}-{}", output_indexes.front()[0], output_indexes.back()[0], output_indexes.front()[1], output_indexes.back()[1]);
                    // MyAddressAllocator::check_addrs(output_addrs);
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
    return tile;
}

void GEMM::calculate_my_loops() {
    std::vector<uint32_t> matrix_dims(_my_weights[0]->get_dims());
    std::vector<uint32_t> input_dims = {0, 0};
    for (int i=0; i<_batch_size; i++) {
        input_dims[0] = input_dims[0] + _my_inputs[i]->get_dims()[0];
        input_dims[1] = _my_inputs[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]); // 对应了各个Batch的所存在的区间
    }
    // std::vector<uint32_t> input0_dims(_my_inputs[0]->get_dims());
    // std::vector<uint32_t> input1_dims(_my_inputs[1]->get_dims());
    // m,k @ k,n
    _inner_loop.resize(3);  // M, K, N
    assert(input_dims.back() == matrix_dims[0]);
    uint32_t M = input_dims[input_dims.size() - 2];
    uint32_t K = input_dims.back();
    uint32_t N = matrix_dims.back();

    if (_batch_size > 1) {
        spdlog::info("GEMM Operation for {} batches with concat size of {}", _batch_size, M);
    }
    else {
        spdlog::info("GEMM Operation for single batches with concat size of {}",M);
    }

    _outer_loop.assign(3, 1);

    // _outer_loop[2] = _config.num_cores;  // 需要保证每个core中的tile相同

    // 计算过程中需要考虑向上取整的问题
    _inner_loop[0] = std::ceil((static_cast<double>(M) / _outer_loop[0]) / _config.core_height) * _config.core_height;  // inner_loop基于Tile大小向上取整, m方向单位为Core_height
    _inner_loop[1] = std::ceil((static_cast<double>(K) / _outer_loop[1]) / _config.core_width) * _config.core_width;  // inner_loop基于Tile大小向上取整, k方向单位为Core_width
    _inner_loop[2] = std::ceil((static_cast<double>(N) / _outer_loop[2]) / _config.core_width) * _config.core_width; // inner_loop基于Tile大小向上取整, n方向单位为Core_width

    // todo: future work, consider broadcasting.
    // currently, it just assumes that the feature size of the smaller
    // dimensions are included to larger dimensions.

    // larger_dims: [768, 2304] => _prod_batches: 1
    // larger_dims: [1, 12, 64, 15] => _prod_batches: 12
    // Calculate the number of iterations for GEMM by multiplying all dimensions except the last two.

    while (sram_size_needed() > _config.spad_size KB / 2)  // double ping-pong buffer
    {
        // max_element return iterator
        // divide max_element dimension to 1/2,
        // increment outer_loop to 1
        // find the maximum dimension of [M, K, N]
        // Reduce the dimension by half (rounded up) and double of the outer loop count.
        // Repeat this process until the SRAM usage of tile meets requirements

        // auto max_el = max_element(_inner_loop.begin(), _inner_loop.end());
        // int max_value_index = max_el - _inner_loop.begin();

        std::vector<int> priority = {2, 0, 1}; // 优先 dim2, dim0, dim1
        int max_value_index = priority[0];
        for (int idx : priority) {
            if (_inner_loop[idx] > _inner_loop[max_value_index]) {
                max_value_index = idx;
            }
        }
        _outer_loop[max_value_index] *= 2;

        if (max_value_index == 0) {
            _inner_loop[0] = std::ceil((static_cast<double>(M) / _outer_loop[0]) / _config.core_height) * _config.core_height;
        }
        else if (max_value_index == 1) {
            _inner_loop[1] = std::ceil((static_cast<double>(K) / _outer_loop[1]) / _config.core_height) * _config.core_height;
        }
        else if (max_value_index == 2){
            _inner_loop[2] = std::ceil((static_cast<double>(N) / _outer_loop[2]) / _config.core_width) * _config.core_width;
        }
        else {
            throw std::runtime_error("GEMM::calculate_my_loops: invalid value index");
        }
        /*
        _outer_loop[max_el - _inner_loop.begin()] *= 2;
        *max_el = ((*max_el) & 1) + ((*max_el) >> 1);  // ceil(*max_el / 2)
        */
        assert(_inner_loop[0] * _outer_loop[0] >= M);
        assert(_inner_loop[1] * _outer_loop[1] >= K);
        assert(_inner_loop[2] * _outer_loop[2] >= N);
    }

    if (_is_transposed) {
        std::reverse(_inner_loop.begin(), _inner_loop.end());
        std::reverse(_outer_loop.begin(), _outer_loop.end());
    }

    // number of outer_loop based on the inner loop and data dimension
    _outer_loop[0] = std::ceil(double(M)/_inner_loop[0]);
    _outer_loop[1] = std::ceil(double(K)/_inner_loop[1]);
    _outer_loop[2] = std::ceil(double(N)/_inner_loop[2]);

    spdlog::info("GEMM inner loop: {}, outer loop: {}", _inner_loop, _outer_loop);
    // todo: if _inner_loop cannot fill the sram, extra batching is needed for
    // spdlog::info("sram utilization of tile {}: {}", get_name(), (float)sram_size_needed() / (float)_config.spad_size);
}


void GEMM::calculate_my_loops_prime() {
    std::vector<uint32_t> matrix_dims(_my_weights[0]->get_dims());
    std::vector<uint32_t> input_dims = {0, 0};

    for (int i = 0; i < _batch_size; i++) {
        input_dims[0] += _my_inputs[i]->get_dims()[0];
        input_dims[1] = _my_inputs[i]->get_dims()[1];
        _m_batch_dim.push_back(input_dims[0]);
    }

    // M, K, N
    _inner_loop.resize(3);
    assert(input_dims.back() == matrix_dims[0]);

    uint32_t M = input_dims[input_dims.size() - 2];
    uint32_t K = input_dims.back();
    uint32_t N = matrix_dims.back();

    if (_batch_size > 1) {
        spdlog::info("GEMM Operation for {} batches with concat size of {}", _batch_size, M);
    } else {
        spdlog::info("GEMM Operation for single batch with size {}", M);
    }

    _outer_loop.assign(3, 1);

    // 初始inner_loop（按tile对齐）
    _inner_loop[0] = std::ceil((double)M / _config.core_height) * _config.core_height;
    _inner_loop[1] = std::ceil((double)K / _config.core_width) * _config.core_width;
    _inner_loop[2] = std::ceil((double)N / _config.core_width) * _config.core_width;

    // -------- 最大质因子函数 --------
    auto get_largest_prime_factor = [](uint32_t x) -> uint32_t {
        uint32_t largest = 1;

        while (x % 2 == 0) {
            largest = 2;
            x /= 2;
        }

        for (uint32_t i = 3; i * i <= x; i += 2) {
            while (x % i == 0) {
                largest = i;
                x /= i;
            }
        }

        if (x > 2) largest = x;

        return largest;
    };

    // -------- Tile调整循环 --------
    while (sram_size_needed() > _config.spad_size KB / 2) {
        /*
        auto max_el = std::max_element(_inner_loop.begin(), _inner_loop.end());
        int index = max_el - _inner_loop.begin();
        */
        std::vector<int> priority = {0, 2, 1}; // 优先 dim2, dim0, dim1
        int max_value_index = priority[0];
        for (int idx : priority) {
            if (_inner_loop[idx] > _inner_loop[max_value_index]) {
                max_value_index = idx;
            }
        }
        int index = max_value_index;

        if (index == 0) {
            // ===== M维：保持二分策略 =====
            _outer_loop[0] *= 2;
            _inner_loop[0] = std::ceil((double)M / _outer_loop[0] / _config.core_height) * _config.core_height;
        }
        else {
            // ===== K / N维：最大质因子策略 =====
            uint32_t dim_size = (index == 1) ? K : N;
            // 当前tile尺寸
            uint32_t current_tile = dim_size / _outer_loop[index];
            if (current_tile == 0) current_tile = 1;
            uint32_t factor = get_largest_prime_factor(current_tile);
            // 防止卡死（例如=1）
            if (factor <= 1) factor = 2;
            _outer_loop[index] *= factor;
            // 最小粒度约束
            uint32_t base;
            if (index == 1) {
                base = _config.core_width;   // K维
            } else {
                base = _config.core_height;  // N维
            }
            _inner_loop[index] =  std::max((uint32_t)(std::ceil((double)dim_size / _outer_loop[index] / base) * base),base);
        }

        // 安全检查
        assert(_inner_loop[0] * _outer_loop[0] >= M);
        assert(_inner_loop[1] * _outer_loop[1] >= K);
        assert(_inner_loop[2] * _outer_loop[2] >= N);
    }

    // 转置处理
    if (_is_transposed) {
        std::reverse(_inner_loop.begin(), _inner_loop.end());
        std::reverse(_outer_loop.begin(), _outer_loop.end());
    }

    // 最终outer_loop修正（精确覆盖）
    _outer_loop[0] = std::ceil((double)M / _inner_loop[0]);
    _outer_loop[1] = std::ceil((double)K / _inner_loop[1]);
    _outer_loop[2] = std::ceil((double)N / _inner_loop[2]);

    spdlog::info("GEMM inner loop: {}, outer loop: {}, Spad Size: {} KB", _inner_loop, _outer_loop, sram_size_needed()/1024);
}


void GEMM::calculate_attention_loops() {
    // 每一个Batch分别计算生成
    std::string gemm_attetion_type;
    uint32_t M, K, N;
    _inner_loop.resize(3);

    for (int i=0; i<_batch_size; i++) {
        std::vector<uint32_t> input0_dims(_my_inputs[0]->get_dims());
        std::vector<uint32_t> input1_dims(_my_weights[0]->get_dims());

        _outer_loop.assign(3, 1);
        if (_my_weights[0]->_tensor_type == TensorType::KCache) {
            M = input0_dims[input0_dims.size() - 2]; // Lin
            K = MyAddressAllocator::d_k;
            N = input1_dims[input1_dims.size() - 2]; // Lin
            gemm_attetion_type = "QKT";
            spdlog::info("Current Attention Operation for QKT, GEMM: M = {}, K = {}, N = {}", M, K, N);
        }
        else if(_my_weights[0]->_tensor_type == TensorType::VCache) {
            M = input0_dims[input0_dims.size() - 2];
            K = input1_dims[input1_dims.size() - 2];
            N = MyAddressAllocator::d_k;
            gemm_attetion_type = "SV";
            spdlog::info("Current Attention Operation for SV, GEMM: M = {}, K = {}, N = {}", M, K, N);
        }
        else {
            assert(0);
        }

        // 计算过程中需要考虑向上取整的问题
        _inner_loop[0] = std::ceil((static_cast<double>(M) / _outer_loop[0]) / _config.core_height) * _config.core_height;  // inner_loop基于Tile大小向上取整
        _inner_loop[1] = std::ceil((static_cast<double>(K) / _outer_loop[1]) / _config.core_height) * _config.core_height;  // inner_loop基于Tile大小向上取整
        _inner_loop[2] = std::ceil((static_cast<double>(N) / _outer_loop[2]) / _config.core_width) * _config.core_width;

        while (sram_size_needed() > _config.spad_size KB / 2) {  // double ping-pong buffer
            // max_element return iterator
            // divide max_element dimension to 1/2,
            // increment outer_loop to 1
            // find the maximum dimension of [M, K, N]
            // Reduce the dimension by half (rounded up) and double of the outer loop count.
            // Repeat this process until the SRAM usage of tile meets requirements

            auto max_el = max_element(_inner_loop.begin(), _inner_loop.end());
            int max_value_index = max_el - _inner_loop.begin();
            _outer_loop[max_value_index] *= 2;  // 这里使outer loop每次+2，可能需要修改逻辑将×3的部分提取出来

            if (max_value_index == 0) {
                _inner_loop[0] = std::ceil(((double)M / _outer_loop[0]) / _config.core_height) * _config.core_height;
            }
            else if (max_value_index == 1) {
                _inner_loop[1] = std::ceil(((double)K / _outer_loop[1]) / _config.core_height) * _config.core_height;
            }
            else if (max_value_index == 2){
                _inner_loop[2] = std::ceil(((double)N / _outer_loop[2]) / _config.core_width) * _config.core_width;
            }
            else {
                throw std::runtime_error("GEMM::calculate_my_loops: invalid value index");
            }
            assert(_inner_loop[0] * _outer_loop[0] >= M);
            assert(_inner_loop[1] * _outer_loop[1] >= K);
            assert(_inner_loop[2] * _outer_loop[2] >= N);
        }
        spdlog::info("GEMM for Batch {} attention operation {} with inner loop: {}, outer loop: {}", i, gemm_attetion_type, _inner_loop, _outer_loop);
        // push back current batch, copy value, has no influence
        _inner_loop_attn.push_back(_inner_loop);
        _outer_loop_attn.push_back(_outer_loop);
    }


    /*
    // 基于input dims=0，计算出来目前是QKT还是SV计算
    std::vector<uint32_t> input0_dims(_my_inputs[0]->get_dims());
    std::vector<uint32_t> input1_dims(_my_inputs[1]->get_dims());

    // 需要算出每个Head的尺寸
    _inner_loop.resize(3);  // M, K, N
    uint32_t M, K, N;
    if (_my_inputs[1]->_tensor_type == TensorType::KCache) {
        // K Cache 计算下的单个Head
        M = input0_dims[input0_dims.size() - 2]; // Lin
        K = MyAddressAllocator::d_k;
        N = input1_dims[input1_dims.size() - 2]; // Lin
        matmul_attetion_type = "QKT";
        spdlog::info("Current Attention Operation for QKT, GEMM: M = {}, K = {}, N = {}", M, K, N);
    }
    else if(_my_inputs[1]->_tensor_type == TensorType::VCache) {
        M = input0_dims[input0_dims.size() - 2];
        K = input1_dims[input1_dims.size() - 2];
        N = MyAddressAllocator::d_k;
        matmul_attetion_type = "SV";
        spdlog::info("Current Attention Operation for SV, GEMM: M = {}, K = {}, N = {}", M, K, N);
    }
    else {
        assert(0);
    }

    // 分配的计算内容，按照单个的Attention head计算流程进行inner loop和outer Loop的划分
    _outer_loop.assign(3, 1);

    // 计算过程中需要考虑向上取整的问题
    _inner_loop[0] = std::ceil((static_cast<double>(M) / _outer_loop[0]) / _config.core_height) * _config.core_height;  // inner_loop基于Tile大小向上取整
    _inner_loop[1] = std::ceil((static_cast<double>(K) / _outer_loop[1]) / _config.core_height) * _config.core_height;  // inner_loop基于Tile大小向上取整
    _inner_loop[2] = std::ceil((static_cast<double>(N) / _outer_loop[2]) / _config.core_width) * _config.core_width;

    while (sram_size_needed() > _config.spad_size KB / 2) {  // double ping-pong buffer
        // max_element return iterator
        // divide max_element dimension to 1/2,
        // increment outer_loop to 1
        // find the maximum dimension of [M, K, N]
        // Reduce the dimension by half (rounded up) and double of the outer loop count.
        // Repeat this process until the SRAM usage of tile meets requirements

        auto max_el = max_element(_inner_loop.begin(), _inner_loop.end());
        int max_value_index = max_el - _inner_loop.begin();
        _outer_loop[max_value_index] *= 2;

        if (max_value_index == 0) {
            _inner_loop[0] = std::ceil(((double)M / _outer_loop[0]) / _config.core_height) * _config.core_height;
        }
        else if (max_value_index == 1) {
            _inner_loop[1] = std::ceil(((double)K / _outer_loop[1]) / _config.core_height) * _config.core_height;
        }
        else if (max_value_index == 2){
            _inner_loop[2] = std::ceil(((double)N / _outer_loop[2]) / _config.core_width) * _config.core_width;
        }
        else {
            throw std::runtime_error("GEMM::calculate_my_loops: invalid value index");
        }

        assert(_inner_loop[0] * _outer_loop[0] >= M);
        assert(_inner_loop[1] * _outer_loop[1] >= K);
        assert(_inner_loop[2] * _outer_loop[2] >= N);
    }

    spdlog::info("GEMM for attention operation {} with inner loop: {}, outer loop: {}", matmul_attetion_type, _inner_loop, _outer_loop);
    */
}


// bias is loaded to the accumulation space
uint32_t GEMM::sram_size_needed() {
    // 计算过程中，各计算维度需要对齐到硬件核宽，确保数据通路效率.
    auto n = _inner_loop[0];
    if (n % _config.core_width != 0) {
        n += _config.core_width - n % _config.core_width;
    }
    auto k = _inner_loop[1];
    if (k % _config.core_width != 0) {
        k += _config.core_width - k % _config.core_width;
    }
    auto m = _inner_loop[2];
    if (m % _config.core_height != 0) {
        m += _config.core_height - m % _config.core_height;
    }

    if (_my_weights.size() == 2) {
        auto bias_dim = _my_weights[1]->get_dims()[0];
        return (n * k) * _my_inputs[0]->_precision + (k * m) * _my_weights[0]->_precision + bias_dim * _my_weights[1]->_precision;
    }
    else {
        return (n * k) * _my_inputs[0]->_precision + (k * m) * _my_weights[0]->_precision;
    }
    // return (n * k + k * m + m * n) * _config.precision;
}

Tile GEMM::make_deferred_gemm_tile(uint32_t B, uint32_t M, uint32_t K,
                                   uint32_t N, bool should_store) {
    Tile tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = B,
        .head_index = 0,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,
        .skip = false,
        .pim_tile = false,
    };
    tile.deferred_compile = true;
    tile.materializer =
        [this, B, M, K, N, should_store](Tile& target) {
            target = initialize_my_DASH_instructions(
                B, M, K, N, should_store);
        };
    return tile;
}

Tile GEMM::make_deferred_attention_tile(uint32_t B, uint32_t head_index,
                                        uint32_t M, uint32_t K, uint32_t N,
                                        bool should_store) {
    Tile tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = B,
        .head_index = head_index,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,
        .skip = false,
        .pim_tile = false,
    };
    tile.deferred_compile = true;
    tile.materializer =
        [this, B, head_index, M, K, N, should_store](Tile& target) {
            target = initialize_my_attention_instructions(
                B, head_index, M, K, N, should_store);
        };
    return tile;
}
