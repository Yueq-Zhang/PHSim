#include "PIMGEMV.h"

namespace {
constexpr uint32_t kInstrAddrGranularityBytes = 1024;

inline uint32_t CeilDivU64(uint64_t x, uint64_t y) {
    return static_cast<uint32_t>((x + y - 1) / y);
}

inline uint32_t BytesToInstrChunks(uint64_t bytes) {
    return CeilDivU64(bytes, kInstrAddrGranularityBytes);
}

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
inline void AssertNoOverlap1KB(const std::vector<addr_type> &addrs, const std::string &tag) {
    std::unordered_set<addr_type> unique_addrs;
    unique_addrs.reserve(addrs.size());

    for (auto addr : addrs) {
        if (!unique_addrs.insert(addr).second) {
            spdlog::warn("[{}] duplicated dram address detected: {}", tag, addr);
        }
    }
}

inline void PIM_COMP_AddressMerge(const std::vector<addr_type> &addrs, std::deque<Instruction> &inst_q, Opcode opcode,
    addr_type dest_addr, uint32_t chunks, addr_type src_addr, uint32_t operand_id, const std::string &inst_info = "") {

    // generate one instruction every 1KB
    uint32_t hash_offset = MyAddressAllocator::AddrGranularity_Hash_Bytes / MyAddressAllocator::dram_burst_size;
    addr_type hash_merge_start_addr = 0;  // Start address of the current COMP hash group.
    uint32_t comp_inst_num = 0;  // Number of COMP bursts in the current group.

    for (auto addr : addrs) {
        auto current_burst_addr = MyAddressAllocator::get_data_offset(addr, hash_offset);
        if (addr == addrs.front()) {  // first addr, push to address
            hash_merge_start_addr = current_burst_addr;
            comp_inst_num++;
        }
        else if (hash_merge_start_addr != current_burst_addr) { // once all the address are in the trace ,
            // Emit an instruction for the completed address group.
            inst_q.push_back(Instruction{
                .opcode = opcode,
                .dest_addr = dest_addr,
                .size = comp_inst_num * MyAddressAllocator::burst_length,
                .src_addrs = std::vector<addr_type>{hash_merge_start_addr},
                .operand_id = operand_id,
                .inst_information = inst_info,
                .logical_start_addr = src_addr,
                .exec_len_bytes = kInstrAddrGranularityBytes,  // One execution unit per hash group.
            });

            // Start tracking the next address group.
            hash_merge_start_addr = current_burst_addr;
            comp_inst_num = 1;
        }
        else if (addr == addrs.back()){ // last trace generate the last instruction
            comp_inst_num++;
            // Emit the final instruction group.
            inst_q.push_back(Instruction{
                .opcode = opcode,
                .dest_addr = dest_addr,
                .size = comp_inst_num * MyAddressAllocator::burst_length,
                .src_addrs = std::vector<addr_type>{hash_merge_start_addr},
                .operand_id = operand_id,
                .inst_information = inst_info,
                .logical_start_addr = src_addr,
                .exec_len_bytes = kInstrAddrGranularityBytes,
            });
        }
    }
}

inline void PushRepeatedPIMInstructions(std::deque<Instruction> &inst_q, Opcode opcode,
                                        addr_type dest_addr, uint32_t chunks, addr_type src_addr,
                                        uint32_t operand_id, const std::string &inst_info = "") {
    for (uint32_t i = 0; i < chunks; i++) {
        inst_q.push_back(Instruction{
            .opcode = opcode,
            .dest_addr = dest_addr,
            .size = kInstrAddrGranularityBytes,
            .src_addrs = std::vector<addr_type>{src_addr},
            .operand_id = operand_id,
            .inst_information = inst_info,
            .logical_start_addr = src_addr,
            .exec_len_bytes = kInstrAddrGranularityBytes,
        });
    }
}

inline void PushAddressPIMInstructions(std::deque<Instruction> &inst_q, Opcode opcode,
                                       addr_type dest_addr, const std::vector<addr_type> &src_addrs,
                                       uint32_t operand_id, const std::string &inst_info = "") {
    for (auto addr : src_addrs) {
        inst_q.push_back(Instruction{
            .opcode = opcode,
            .dest_addr = dest_addr,
            .size = kInstrAddrGranularityBytes,
            .src_addrs = std::vector<addr_type>{addr},
            .operand_id = operand_id,
            .inst_information = inst_info,
            .logical_start_addr = addr,
            .exec_len_bytes = kInstrAddrGranularityBytes,
        });
    }
}

void PushHashedPIMCompInstructions(
    std::deque<Instruction>& inst_q, addr_type dest_addr,
    const std::vector<addr_type>& src_addrs, uint32_t operand_id,
    const std::string& inst_info = "") {
    const auto groups = PIMHashAddressing::group_comp_addresses(
        src_addrs, MyAddressAllocator::AddrGranularity_Hash_Bytes,
        MyAddressAllocator::dram_burst_size);
    for (const auto& group : groups) {
        const uint64_t group_size_bytes =
            static_cast<uint64_t>(group.burst_count) *
            MyAddressAllocator::dram_burst_size;
        if (group_size_bytes > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "PIM hash instruction size exceeds the supported range");
        }
        inst_q.push_back(Instruction{
            .opcode = Opcode::PIM_COMP_HASH,
            .dest_addr = dest_addr,
            .size = static_cast<uint32_t>(group_size_bytes),
            .src_addrs = std::vector<addr_type>{group.aligned_start_address},
            .operand_id = operand_id,
            .inst_information = inst_info,
            .logical_start_addr = group.aligned_start_address,
            .exec_len_bytes = static_cast<uint32_t>(group_size_bytes),
        });
    }
}
}  // namespace


PIMGEMV::PIMGEMV(std::string name, std::vector<Ptr<MyTensor>> weights) : Operation(name) {
    _my_weights.resize(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        _my_weights[i] = weights[i];
    }
    matrix_tensor_type = weights[0]->_tensor_type;
    _matrix_dim = weights[0]->get_dims();

    _prod_batches = 0;

    head_compute_iteration = 1;
    head_per_iteration = 1;
    pim_gemv_granularity = _config.pim_PE_num;

    pim_global_input_buffer_size = _config.pim_input_buffer_size;
    if (PIM_Parameters::dual_bank == true) {
        pim_output_buffer_size = _config.pim_output_buffer_size / 2;
    }
    else {
        pim_output_buffer_size = _config.pim_output_buffer_size;
    }

    if (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS) {
        column_interleave = 1;
        weight_rows_per_bank_row = MyAddressAllocator::page_size_bytes / MyAddressAllocator::precision_weight;
    }
    else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
        column_interleave = MyAddressAllocator::weight_interleave_columns;  // Weight columns interleaved in each bank.
        weight_rows_per_bank_row = MyAddressAllocator::weight_rows_per_bank_row;
    }

    cache_append = false;
}

std::vector<Ptr<MyTensor>> PIMGEMV::get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type) {
    set_as_parent_tensor(inputs);
    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());

    if (matrix_tensor_type == TensorType::WGT or matrix_tensor_type == TensorType::ACT) {
        _matrix_dim = _my_weights[0]->get_dims();
        for (size_t i = 0; i < inputs.size(); ++i) {
            _my_inputs[i] = inputs[i];
            auto input_dims = _my_inputs[i]->get_dims();
            assert(*input_dims.rbegin() == *(_matrix_dim.rbegin() + 1));
            spdlog::info("GEMM input index: {} / input size: {}", i, inputs[i]->get_dims());
            // Compute output dimensions.
            std::vector<uint32_t> output_dims = {0, 0};
            *(output_dims.rbegin() + 1) = *(_my_inputs[i]->get_dims().rbegin() + 1);
            *output_dims.rbegin() = *_matrix_dim.rbegin();
            _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
        }
    }
    else if (matrix_tensor_type == TensorType::KCache or matrix_tensor_type == TensorType::VCache){
        set_as_parent_tensor(inputs);
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
                spdlog::info("The dimension of Batch {} QKT operation: input Tensor {} and {}, output Tensor {}",i, Q_dims,K_dims,output_dims);
            }
        }
        else if (matrix_tensor_type == TensorType::VCache) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto S_dims = _my_inputs[i]->get_dims();
                auto V_dims = _my_weights[i]->get_dims();
                output_dims={S_dims[S_dims.size() - 2], MyAddressAllocator::h * MyAddressAllocator::d_k};
                _my_outputs[i] = std::make_shared<MyTensor>(_name + "_output", output_dims, output_tensor_type, false);
                spdlog::info("The dimension of Batch {} SV operation: input Tensor {} and {}, output Tensor {}",i, S_dims,V_dims,output_dims);
            }
        }
    }
    calculate_my_loops();
    initialize_my_tiles();
    return _my_outputs;
}


std::vector<Ptr<MyTensor>> PIMGEMV::kvcache_append(std::vector<Ptr<MyTensor>> inputs, std::vector<Ptr<MyTensor>> kvcaches, TensorType output_tensor_type) {

    _batch_size = inputs.size();
    _my_inputs.resize(inputs.size());
    _my_outputs.resize(inputs.size());
    cache_append = true;
    assert(inputs.size() == kvcaches.size());
    set_as_parent_tensor(inputs);

    auto matrix_tensor = _my_weights[0];
    _matrix_dim = matrix_tensor->get_dims();
    assert(matrix_tensor->_tensor_type == TensorType::WGT);

    for (size_t i = 0; i < _batch_size; ++i) {
        _my_inputs[i] = inputs[i];
        assert(_my_inputs[i]->get_dims().back() == _matrix_dim[0]);  // Weight Matrix
        std::vector<uint32_t> output_dim = {_my_inputs[i]->get_dims()[0], _matrix_dim[1]};

        // Take updated KVCaches as the output
        _my_outputs[i] = kvcaches[i];
        if (_my_outputs[i]->Cache_length + output_dim[0] > _my_outputs[i]->Cache_capacity) {
            _my_outputs[i]->cache_append();
        }

        // Update the valid KVCache Capacity
        _my_outputs[i]->Cache_length += output_dim[0];
        _my_outputs[i]->Cache_capacity -= output_dim[0];

        if (_my_outputs[0]->_tensor_type == TensorType::KCache) {
            spdlog::info("GEMV Result is append to Batch {} KCaches with cache length = {}", i, _my_outputs[0]->Cache_length);
        }
        else if (_my_outputs[0]->_tensor_type == TensorType::VCache) {
            spdlog::info("GEMV Result is append to Batch {} VCaches with cache length = {}", i, _my_outputs[0]->Cache_length);
        }
        else {
            throw std::runtime_error("Invalid Number of PIM-GEMV Inputs");
        }
    }
    calculate_my_loops();
    initialize_my_tiles();
    cache_append = false;  // cache_append operation finished
    return _my_outputs;
}


void PIMGEMV::calculate_my_loops() {
    // The granularity of pim gemv computation is the rows of current each bank row with current interleaving factor
    if (matrix_tensor_type == TensorType::WGT) {
        std::vector<uint32_t> input_dims = {0, 0};
        for (int i=0; i<_batch_size; i++) {
            input_dims[0] = input_dims[0] + _my_inputs[i]->get_dims()[0];
            input_dims[1] = _my_inputs[i]->get_dims()[1];
            _m_batch_dim.push_back(input_dims[0]); // m range of each batch
        }
        _prod_batches = _m_batch_dim.back();
        // The m dimension accumulates the rows of all batches.
        _inner_loop.resize(2);
        _outer_loop.assign(1, 1);
        _inner_loop[0] = _prod_batches;
        _inner_loop[1] = input_dims[1];

        while (sram_size_needed() > _config.spad_size KB / 2) {  // PIM
            _outer_loop[0] *= 2;
            _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
        }

        spdlog::info("For NPU side PIM_GEMV operation, inner_loop: {}, outer_loop {}", _inner_loop, _outer_loop);
        pim_gemv_granularity = std::min( _config.pim_PE_num, MyAddressAllocator::dram_burst_size / MyAddressAllocator::precision_weight);
        // pim_gemv_granularity = weight_rows_per_bank_row;

        // Tile for PIM calculation
        _pim_inner_loop.resize(3);
        _pim_outer_loop.assign(3, 1);

        _pim_inner_loop[0] = _inner_loop[0];
        _pim_inner_loop[1] = std::ceil(static_cast<double>(_inner_loop[1]) / pim_gemv_granularity) * pim_gemv_granularity;
        _pim_inner_loop[2] = _matrix_dim[1];

        // Reject an unsupported Decode shape before checking the internal
        // tiling invariants below.  In Debug builds the invariants are asserts,
        // so validating the user-controlled dimensions first preserves the
        // actionable configuration diagnostic.
        const uint64_t complete_bank_group_width =
            static_cast<uint64_t>(column_interleave) *
            MyAddressAllocator::total_banks;
        if (complete_bank_group_width == 0 ||
            _matrix_dim[1] < complete_bank_group_width) {
            throw std::invalid_argument(fmt::format(
                "PIM Decode weight output dimension {} is too small for "
                "column_interleave={} and total_banks={}; require at least {} "
                "output elements or use a non-PIM Decode backend",
                _matrix_dim[1], column_interleave,
                MyAddressAllocator::total_banks,
                complete_bank_group_width));
        }

        // Split M and K based on the input buffer capacity, to make sure fully utilization of one column
        // Get the max batch size based on the column_interleave and output buffer size
        assert(column_interleave * MyAddressAllocator::precision_psum <= pim_output_buffer_size);
        while (_pim_inner_loop[0] * column_interleave * MyAddressAllocator::precision_psum > pim_output_buffer_size) {
            if (_pim_inner_loop[0] == 1) {
                break;
            }
            _pim_inner_loop[0] = _pim_inner_loop[0] / 2;
        }

        // The Batch size is defined by the input buffer size
        while (_pim_inner_loop[0] * weight_rows_per_bank_row * MyAddressAllocator::precision_activation > pim_global_input_buffer_size) {
            if (_pim_inner_loop[0] == 1) {
                break;
            }
            _pim_inner_loop[0] = _pim_inner_loop[0] / 2;
        }

        assert(_pim_inner_loop[0] >= 1);
        _pim_outer_loop[0] = _batch_size / _pim_inner_loop[0];

        spdlog::info("Current PIM Output Buffer size {} Byte, PIM Input Global Buffer size {} Byte, Corresponding to {} output {} inputs, "
            "column_interleave = {}, weight_rows_per_bank_row = {}, Data in each Page is {}", pim_output_buffer_size, pim_global_input_buffer_size,
             pim_output_buffer_size/MyAddressAllocator::precision_weight, pim_global_input_buffer_size/MyAddressAllocator::precision_weight,
            column_interleave, weight_rows_per_bank_row, MyAddressAllocator::page_size_bytes / MyAddressAllocator::precision_weight);

        uint32_t _pim_gemv_granularity_per_row = MyAddressAllocator::page_size_bytes / MyAddressAllocator::precision_weight / column_interleave;

        // Obtain the maximum quality factor
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

        while (pim_input_buffer_size_needed() > _config.pim_input_buffer_size) {
            if (_pim_inner_loop[0] == 1) {
                uint32_t current_tile = _inner_loop[1] / _pim_outer_loop[1];
                if (current_tile == 0) current_tile = 1;
                uint32_t factor = get_largest_prime_factor(current_tile);
                if (factor <= 1) factor = 2;
                _pim_outer_loop[1] *= factor;
                _pim_inner_loop[1] = std::ceil((static_cast<double>(_inner_loop[1])/ _pim_outer_loop[1]) / pim_gemv_granularity) * pim_gemv_granularity;
            }
            else {
                if (_pim_inner_loop[1] > _pim_gemv_granularity_per_row) { 
                    uint32_t current_tile = _inner_loop[1] / _pim_outer_loop[1];
                    if (current_tile == 0) current_tile = 1;
                    uint32_t factor = get_largest_prime_factor(current_tile);
                    if (factor <= 1) factor = 2;
                    _pim_outer_loop[1] *= factor;
                    _pim_inner_loop[1] = std::ceil((static_cast<double>(_inner_loop[1])/ _pim_outer_loop[1]) / _pim_gemv_granularity_per_row) * _pim_gemv_granularity_per_row; 
                }
                else {
                    _pim_outer_loop[0] *= 2;
                    _pim_inner_loop[0] = std::ceil(static_cast<double>(_inner_loop[0])/ _pim_outer_loop[0]);
                }
            }
        }


        _pim_outer_loop[1] = std::ceil(static_cast<double>(_inner_loop[1])/_pim_inner_loop[1]); //
        if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
            assert(_pim_inner_loop[1] * column_interleave >= MyAddressAllocator::page_size_bytes / MyAddressAllocator::precision_weight);
        }
        uint32_t column_inner_loop_per_bank = std::floor(static_cast<double>(pim_output_buffer_size) / _my_outputs[0]->_precision / column_interleave) * column_interleave;

        _pim_inner_loop[2] = std::min(std::floor(static_cast<double>(_matrix_dim[1])/complete_bank_group_width) *column_interleave,
            static_cast<double>(column_inner_loop_per_bank)); 
        if (_pim_inner_loop[2] == 0) {
            throw std::invalid_argument(fmt::format(
                "PIM Decode cannot form a non-zero output tile for dimension "
                "{}, column_interleave={}, total_banks={}, and output buffer "
                "{} bytes",
                _matrix_dim[1], column_interleave,
                MyAddressAllocator::total_banks, pim_output_buffer_size));
        }
        
        _pim_outer_loop[2] = std::ceil(static_cast<double>(_matrix_dim[1])/(_pim_inner_loop[2] * MyAddressAllocator::total_banks));
        // _pim_inner_loop[2] = std::floor(static_cast<double>(pim_output_buffer_size) / _my_outputs[0]->_precision);
        // _pim_outer_loop[2] = std::ceil(static_cast<double>(_matrix_dim[1])/ _pim_inner_loop[2]);
        spdlog::info("For PIM side PIM_GEMV operation for Weight, inner_loop: {}, outer_loop {}. The actual inner_loop[2] with Total {} banks is = {}", _pim_inner_loop, _pim_outer_loop,
            MyAddressAllocator::total_banks, _pim_inner_loop[2] * MyAddressAllocator::total_banks);
    }
    else if (matrix_tensor_type == TensorType::KCache) {
        _inner_loop.resize(2);
        pim_gemv_granularity = MyAddressAllocator::dram_burst_size/_my_weights[0]->_precision;

        for (int i=0; i<_batch_size; i++) {
            std::vector<uint32_t> input0_dims(_my_inputs[i]->get_dims());
            std::vector<uint32_t> input1_dims(_my_weights[i]->get_dims());

            _outer_loop.assign(1, 1);
            _inner_loop[0] = input0_dims[0];
            _inner_loop[1] = input0_dims[1];

            while (sram_size_needed() > _config.spad_size KB / 2) {
                _outer_loop[0] *= 2;
                _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
            }
            spdlog::info("For NPU side PIM_GEMV operation, inner_loop: {}, outer_loop {}", _inner_loop, _outer_loop);
            spdlog::info("PIM GEMV operation for QKT with {} attention head and {} length of each head", MyAddressAllocator::h, MyAddressAllocator::d_k);
            auto kcache_length = input1_dims[0];
            head_per_iteration = std::ceil(static_cast<double>(MyAddressAllocator::allocated_KCache_head_per_iteration) / MyAddressAllocator::dram_channels); 
            head_compute_iteration = std::ceil((double)MyAddressAllocator::h / MyAddressAllocator::allocated_KCache_head_per_iteration);

            _pim_inner_loop.resize(3);
            _pim_outer_loop.assign(3, 1);
            _pim_inner_loop[0] = _inner_loop[0];
            _pim_inner_loop[1] = std::ceil(MyAddressAllocator::d_k / pim_gemv_granularity) * pim_gemv_granularity; 

            while (pim_buffer_size_needed() > _config.pim_input_buffer_size) {
                if (_pim_outer_loop[0] == 1) {
                    _pim_outer_loop[1] *= 2;
                    _pim_inner_loop[1] = std::ceil((static_cast<double>(_inner_loop[1])/ _pim_outer_loop[1]) / pim_gemv_granularity) * pim_gemv_granularity;
                }
                else {
                    _pim_outer_loop[0] *= 2;
                    _pim_inner_loop[0] = std::ceil(static_cast<double>(_inner_loop[0])/ _pim_outer_loop[0]);
                }
            }
            assert(_pim_inner_loop[1] == MyAddressAllocator::d_k);  // for PIM computation, the k_dim length of QKT operation is d_k
            // _pim_outer_loop[1] = std::ceil(static_cast<double>(_inner_loop[1]) / _pim_inner_loop[1]);
            _pim_inner_loop[2] = std::floor(static_cast<double>(pim_output_buffer_size) / _my_outputs[0]->_precision);
            _pim_outer_loop[2] = std::ceil(static_cast<double>(_matrix_dim[0]) / (_pim_inner_loop[2] * MyAddressAllocator::KCache_interleaved_banks_per_head));
            spdlog::info("For PIM side PIM_GEMV operation, inner_loop: {}, outer_loop {}", _pim_inner_loop, _pim_outer_loop);

            _inner_loop_attn.push_back(_inner_loop);
            _outer_loop_attn.push_back(_outer_loop);
            _pim_inner_loop_attn.push_back(_pim_inner_loop);
            _pim_outer_loop_attn.push_back(_pim_outer_loop);
        }
    }
    else if (matrix_tensor_type == TensorType::VCache) {
        _inner_loop.resize(2);
        pim_gemv_granularity = MyAddressAllocator::dram_burst_size/_my_weights[0]->_precision;

        for (int i=0; i<_batch_size; i++) {
            std::vector<uint32_t> input0_dims(_my_inputs[i]->get_dims());
            std::vector<uint32_t> input1_dims(_my_weights[i]->get_dims());
            _inner_loop[0] = input0_dims[1];
            _inner_loop[1] = input0_dims[2];
            _outer_loop.assign(1, 1);

            while (sram_size_needed() > _config.spad_size KB / 2) {
                _inner_loop[0] = (_inner_loop[0] & 1) + (_inner_loop[0] >> 1);
            }

            auto vcache_length = input0_dims[2];

            spdlog::info("For NPU side PIM_GEMV operation, inner_loop: {}, outer_loop {}", _inner_loop, _outer_loop);
            spdlog::info("PIM GEMV operation for SV with {} attention head and {} length of V Cache", MyAddressAllocator::h, vcache_length);

            head_per_iteration = std::ceil(static_cast<double>(MyAddressAllocator::allocated_VCache_head_per_iteration) / MyAddressAllocator::dram_channels);
            head_compute_iteration = std::ceil((double)MyAddressAllocator::h / MyAddressAllocator::allocated_VCache_head_per_iteration);

            pim_gemv_granularity = MyAddressAllocator::VCache_rows_per_bank_row;
            //
            _pim_inner_loop.resize(3);
            _pim_outer_loop.assign(3, 1);

            _pim_inner_loop[0] = _inner_loop[0];
            _pim_inner_loop[1] = std::ceil((double)vcache_length / pim_gemv_granularity) * pim_gemv_granularity;

            while (pim_buffer_size_needed() > _config.pim_input_buffer_size) {
                if (_pim_outer_loop[0] == 1) {  // if m can not divide, split K
                    _pim_outer_loop[1] *= 2;
                    _pim_inner_loop[1] = std::ceil((static_cast<double>(vcache_length)/ _pim_outer_loop[1]) / pim_gemv_granularity) * pim_gemv_granularity;
                }
                else {
                    _pim_outer_loop[0] *= 2;
                    _pim_inner_loop[0] = std::ceil(static_cast<double>(_inner_loop[0])/(_pim_outer_loop[0]));
                }
            }

            // Check the size of pim_outer_loop
            _pim_outer_loop[1] = std::ceil(static_cast<double>(vcache_length)/_pim_inner_loop[1]) ;
            // For the K cache, during the computation of each head, d_k columns arranged to different banks based on the output buffer and VCache interleave scale
            // If this condition is not satisfied, a single V cache head cannot be computed within one tile, leading to additional row switches over head
            assert(pim_output_buffer_size / _my_outputs[0]->_precision >= MyAddressAllocator::VCache_columns_per_bank);
            _pim_inner_loop[2] = MyAddressAllocator::VCache_columns_per_bank;
            _pim_outer_loop[2] = std::ceil((double)MyAddressAllocator::d_k / (MyAddressAllocator::VCache_columns_per_bank * MyAddressAllocator::VCache_interleaved_banks_per_head));
            spdlog::info("For PIM side PIM_GEMV operation, inner_loop: {}, outer_loop {}", _pim_inner_loop, _pim_outer_loop);

            // save batch information
            _inner_loop_attn.push_back(_inner_loop);
            _outer_loop_attn.push_back(_outer_loop);
            _pim_inner_loop_attn.push_back(_pim_inner_loop);
            _pim_outer_loop_attn.push_back(_pim_outer_loop);
        }
    }
    else {
        throw std::runtime_error("Unsupported matrix tensor type");
    }

}

uint32_t PIMGEMV::sram_size_needed() {
    // batch number and vector length of current tensor
    uint32_t m = _inner_loop[0];
    uint32_t k = _inner_loop[1];

    return  m * k * _my_inputs[0]->_precision;
}

uint32_t PIMGEMV::pim_buffer_size_needed() {
    // Constrained by the input buffer capacity
    if (matrix_tensor_type == TensorType::WGT) {
        return _pim_inner_loop[0] * _pim_inner_loop[1] * _my_inputs[0]->_precision;
    }
    else if (matrix_tensor_type == TensorType::KCache or matrix_tensor_type == TensorType::VCache) {
        return _pim_inner_loop[0] * _pim_inner_loop[1] * _my_inputs[0]->_precision * head_per_iteration;
    }
    else {
        throw std::runtime_error("Unsupported matrix tensor type");
    }
}


uint32_t PIMGEMV::pim_input_buffer_size_needed() {
    auto m = _pim_inner_loop[0];
    auto k = _pim_inner_loop[1];
    return m * k * _my_inputs[0]->_precision;
}

uint32_t PIMGEMV::pim_output_buffer_size_needed() {
    auto m = _pim_inner_loop[0];
    auto n = _pim_inner_loop[2];
    return m * n * _my_outputs[0]->_precision;
}

void PIMGEMV::initialize_my_tiles() {
    if (matrix_tensor_type == TensorType::WGT) {
        for (uint32_t npu_tile_index = 0; npu_tile_index < _outer_loop[0]; npu_tile_index++) {
            //  generate the MKN Loops, N is the outer loop, K is in the middle and M is in inner
            uint32_t M_max = _m_batch_dim.back();
            uint32_t K_max = _inner_loop[1];
            uint32_t N_max = _my_weights[0]->_dims[1];
            for (uint32_t M = 0; M < _pim_outer_loop[0]; ++M) {
                if (M * _pim_inner_loop[0] >= M_max){continue;}
                for (uint32_t N = 0; N < _pim_outer_loop[2]; ++N) {
                    if (N * _pim_inner_loop[2] * MyAddressAllocator::total_banks >= N_max) {continue;}
                    for (uint32_t K = 0; K < _pim_outer_loop[1]; ++K) {
                        if (K * _pim_inner_loop[1] >= K_max) {continue;}
                        const bool should_read =
                            (K + 1) * _pim_inner_loop[1] >= K_max;
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
                                .pim_tile = true,
                            };
                            tile.deferred_compile = true;
                            tile.materializer =
                                [this, npu_tile_index, M, K, N,
                                 should_read](Tile& target) {
                                    target = initialize_my_instructions(
                                        npu_tile_index, M, K, N,
                                        should_read);
                                };
                            _tiles.push_back(std::move(tile));
                        } else {
                            _tiles.push_back(initialize_my_instructions(
                                npu_tile_index, M, K, N, should_read));
                        }
                    }
                }
            }
        }
    }
    else if (matrix_tensor_type== TensorType::KCache or matrix_tensor_type == TensorType::VCache) {
        for (uint32_t batch = 0; batch < _batch_size; batch++) {
            for (uint32_t npu_tile_index = 0; npu_tile_index < _outer_loop_attn[batch][0]; npu_tile_index++) {
                for (uint32_t M = 0; M < _pim_outer_loop_attn[batch][0]; ++M) {  // the computation of different batch are split for attention computation
                    for (uint32_t head_iteration_index = 0; head_iteration_index < head_compute_iteration; head_iteration_index++) {
                        // After the predefined allocation, attention heads are processed in parallel, within each head is determined.
                        // Through parallel processing across multiple channels and iterative execution, the computation for all heads is completed.
                        for (uint32_t N = 0; N < _pim_outer_loop_attn[batch][2]; ++N) {
                            for (uint32_t K = 0; K < _pim_outer_loop_attn[batch][1]; ++K) {
                                _tiles.push_back(initialize_my_attention_instructions(batch, npu_tile_index, head_iteration_index, M, K, N, K + 1 == _pim_outer_loop_attn[batch][1]));
                            }
                        }
                    }
                }
            }
        }
    }
}


Tile PIMGEMV::initialize_my_instructions(uint32_t npu_tile_index, uint32_t M, uint32_t K, uint32_t N, bool should_read) {
    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,   // If and K!=0, means should have PSUM accumulation
        .pim_tile = true,  // If PIM Tile, only send to the first core for execution
    };

    auto m_inner = _pim_inner_loop[0];
    auto k_inner = _pim_inner_loop[1];
    auto n_inner = _pim_inner_loop[2];

    // calculate the loops
    auto m_outer_offset = npu_tile_index * _inner_loop[0] + m_inner * M;
    auto k_outer_offset = k_inner * K;
    auto n_outer_offset = n_inner * N * MyAddressAllocator::total_banks;

    auto vector_tensors = _my_inputs;
    auto matrix_tensor = _my_weights[0];
    auto output_tensors = _my_outputs;

    uint32_t M_max = _m_batch_dim.back();
    uint32_t K_max = matrix_tensor->_dims[0];
    uint32_t N_max = matrix_tensor->_dims[1];

    // For activation
    addr_type sram_vector_base = SPAD_BASE;
    addr_type sram_addr = sram_vector_base;

    // for results
    addr_type sram_acc_base = ACCUM_SPAD_BASE;
    addr_type sram_acc_addr = sram_acc_base;


    // m_loop_size = one input tensor for computation
    // n_loop_size = column interleave factor in each bank
    // k_loop_size = the supported K accumulation time per bank row
    const uint32_t m_loop_size = 1;
    const uint32_t n_loop_size = column_interleave;
    const uint32_t k_loop_size = std::min((MyAddressAllocator::BL_num_per_row / column_interleave) * (MyAddressAllocator::dram_burst_size/matrix_tensor->_precision), _pim_inner_loop[1]);

    // const uint32_t k_loop_size = _pim_inner_loop[1];
    // -- Bias --
    if (_my_inputs.size() == 3) {
        auto bias_tensor = _my_weights[1];
        sram_acc_addr = sram_acc_addr + bias_tensor->get_dims()[1] * bias_tensor->_precision;
        if (npu_tile_index == 0 && M == 0 && K == 0 && N == 0) {
            auto bias_addrs = bias_tensor->get_all_addrs();
            assert(bias_addrs.size() != 0);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVIN,
                .dest_addr = ACCUM_SPAD_BASE,
                .size = (uint32_t)bias_tensor->get_dims()[1] * bias_tensor->_precision,
                .src_addrs = std::move(bias_addrs),
                .operand_id = _INPUT_OPERAND + 2,
            });
        }
    }

    // -- Vector --
    if (M == 0 && K == 0 && N == 0) {
        std::map<uint32_t, std::vector<std::vector<uint32_t>>> vector_indexes;
        uint32_t batch_index = 0;
        uint32_t previous_batch_boundary = 0;
        uint32_t batch_boundary = _m_batch_dim[batch_index];

        for (uint32_t m_inner_offset = 0; m_inner_offset < _pim_inner_loop[0]; m_inner_offset++) {
            uint32_t vector_m_index = m_outer_offset + m_inner_offset;
            if (vector_m_index >= M_max) {
                continue;
            }
            while (vector_m_index >= batch_boundary) {
                previous_batch_boundary = _m_batch_dim[batch_index];
                batch_index++;
                batch_boundary = _m_batch_dim[batch_index];
            }
            for (uint32_t k_index = 0; k_index < K_max; k_index++) {
                vector_indexes[batch_index].push_back({m_outer_offset + m_inner_offset, k_index});
            }
        }

        std::vector<addr_type> vector_addrs;
        uint32_t vector_index_size = 0;
        if (vector_indexes.empty()) {
            spdlog::info("No valid activation tiles to load.");
        }
        else {
            for (const auto& [batch_id, indexes] : vector_indexes) {
                // _activation_batch_tensors[batch_id]
                vector_index_size += indexes.size();
                auto addrs = vector_tensors[batch_id]->generate_addrs_based_on_indexes(indexes);
                vector_addrs.insert(vector_addrs.end(), addrs.begin(), addrs.end());
            }
        }

        if (vector_addrs.empty()) {
            // spdlog::info("zero load for activation / activation tensor dim: {}", vector_tensor->get_dims());
        }
        else {
            tile.instructions.push_back(Instruction{
               .opcode = Opcode::MOVIN,
               .dest_addr = sram_addr,
               .size = (uint32_t)vector_index_size * vector_tensors[0]->_precision,  // (uint32_t)activation_indexes.size() * data_width
               .src_addrs = std::move(vector_addrs),
               .operand_id = _INPUT_OPERAND
           });
        }
    }

    // -- PHeader --  start the PIM Computation process, make sure the computation data in the SPAD
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::PIM_HEADER,
        .dest_addr = 0,
        .size = 0,
        .src_addrs = std::vector<addr_type>{sram_addr},
        .operand_id = _INPUT_OPERAND,
    });


    // -- PIM_GWRITE -- it is same as the previous GWRITE operation
    // uint32_t gwrite_chunks = BytesToInstrChunks(static_cast<uint64_t>(_pim_inner_loop[0]) * _pim_inner_loop[1] * vector_tensors[0]->_precision);
    // PushRepeatedPIMInstructions(tile.instructions, Opcode::PIM_GWRITE, 0, gwrite_chunks, sram_addr, _OUTPUT_OPERAND);
    uint32_t gwrite_burst_times = std::ceil(static_cast<double>(_pim_inner_loop[0] * _pim_inner_loop[1]) * vector_tensors[0]->_precision / MyAddressAllocator::dram_burst_size);
    tile.instructions.push_back(Instruction{
        .opcode = Opcode::PIM_GWRITE,
        .dest_addr = 0,
        .size = gwrite_burst_times * MyAddressAllocator::dram_burst_size,
        .src_addrs = std::vector<addr_type>(gwrite_burst_times,sram_addr),
        .operand_id = _OUTPUT_OPERAND,
    });


    // -- PIM_COMP --
    uint32_t tile_iteration_size = _pim_inner_loop[2] / n_loop_size;
    uint32_t tile_iteration_outer_offset = N * tile_iteration_size;

    std::vector<uint32_t> tile_rows;
    for (uint32_t tile_iteration_index = 0; tile_iteration_index < tile_iteration_size; tile_iteration_index++) {
        uint32_t tile_iteration = tile_iteration_outer_offset + tile_iteration_index;  // Tile index indicate the start row
        tile_rows.push_back(matrix_tensor->allocate_rows[tile_iteration]);
    }

    std::vector<uint32_t> pim_comp_burst_offset;
    std::vector<uint32_t> k_indexes;

    for (uint32_t k_inner_offset = 0; k_inner_offset < k_inner; k_inner_offset += k_loop_size) {
        for (uint32_t m_inner_offset = 0; m_inner_offset < m_inner; m_inner_offset += m_loop_size) {
            if (m_outer_offset + m_inner_offset >= M_max) {
                continue;
            }
            for (uint32_t k_loop = 0; k_loop < k_loop_size; k_loop++) {
                if (k_outer_offset + k_inner_offset + k_loop >= K_max) {
                    continue;
                }
                k_indexes.push_back(k_outer_offset + k_inner_offset + k_loop);
            }
        }
    }
    auto pim_comp_addrs = matrix_tensor->generate_pim_comp_addrs(tile_rows, k_indexes);

    if (MyAddressAllocator::virtual_mem_hash_enable) {  // two different method for allocation
        PushHashedPIMCompInstructions(
            tile.instructions, sram_addr, pim_comp_addrs, _INPUT_OPERAND);
        // -- PIM_READRES --  Read PSUM Result after every PIM Operation tile
        uint32_t pim_readers_burst_times =
            std::ceil(_pim_inner_loop[2] * MyAddressAllocator::banks_per_channel * output_tensors[0]->_precision / MyAddressAllocator::dram_burst_size);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_READRES,
            .dest_addr = sram_acc_addr,
            .size = pim_readers_burst_times * MyAddressAllocator::dram_burst_size,
            .src_addrs = std::vector<addr_type>(pim_readers_burst_times, 0),
            .operand_id = _OUTPUT_OPERAND,
        });
        /*
        if (should_read) {
            uint32_t pim_readers_chunks = BytesToInstrChunks(static_cast<uint64_t>(pim_output_buffer_size) * MyAddressAllocator::banks_per_channel);
            PushRepeatedPIMInstructions(tile.instructions, Opcode::PIM_READRES, sram_acc_addr, pim_readers_chunks, 0, _OUTPUT_OPERAND);
        }
        */
    }
    else {
        // Write the Operation data into global input buffer of each channel
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_COMP,
            .dest_addr = sram_addr,
            .size = (uint32_t)pim_comp_addrs.size() * MyAddressAllocator::dram_burst_size,
            .src_addrs = std::move(pim_comp_addrs),
            .operand_id = _INPUT_OPERAND,
        });

        // -- PIM_READRES -- NPU Read the Computation Result
        if (should_read) {
            uint32_t pim_readers_burst_times =
                std::ceil(_pim_inner_loop[2] * MyAddressAllocator::banks_per_channel * output_tensors[0]->_precision / MyAddressAllocator::dram_burst_size);
            tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_READRES,
            .dest_addr = sram_acc_addr,
            .size = pim_readers_burst_times * MyAddressAllocator::dram_burst_size,
            .src_addrs = std::vector<addr_type>(pim_readers_burst_times, 0),
            .operand_id = _OUTPUT_OPERAND,
            });
        }
    }

    // -- MOVOUT -- NPU write the PIM computation result back to DRAM
    if ((M+1)*_pim_inner_loop[0] >= _inner_loop[0] && (N+1) * _pim_inner_loop[2] * MyAddressAllocator::total_banks >= N_max && (K+1) * _pim_inner_loop[1] >= K_max) {
        std::vector<addr_type> output_addrs;
        uint32_t output_index_size = 0;
        if (cache_append) {
            uint32_t batch_index = 0;
            uint32_t previous_batch_boundary = 0;
            uint32_t batch_boundary = _m_batch_dim[batch_index];

            for (uint32_t output_m_index = npu_tile_index * _inner_loop[0]; output_m_index < (npu_tile_index + 1) * _inner_loop[0]; output_m_index++) {
                if (output_m_index >= M_max) {
                    continue;
                }
                while (output_m_index >= batch_boundary) {
                    previous_batch_boundary = _m_batch_dim[batch_index];
                    batch_index++;
                    batch_boundary = _m_batch_dim[batch_index];
                }
                for (uint32_t head_index = 0; head_index < MyAddressAllocator::h; head_index++) {
                    std::vector<std::vector<uint32_t>> output_indexes;
                    for (uint32_t k=0; k<MyAddressAllocator::d_k; k++) {
                        output_indexes.push_back({output_tensors[batch_index]->Cache_length, head_index * MyAddressAllocator::d_k + k});
                    }
                    output_index_size += output_indexes.size();

                    auto output_addrs_head = output_tensors[batch_index]->generate_addrs_based_on_indexes(output_indexes);
                    for (auto output_addr_head : output_addrs_head) {
                        output_addrs.push_back(output_addr_head);
                    }
                }
            }
        }
        else {
            std::map<uint32_t, std::vector<std::vector<uint32_t>>> output_indexes;
            uint32_t batch_index = 0;
            uint32_t previous_batch_boundary = 0;
            uint32_t batch_boundary = _m_batch_dim[batch_index];
            for (uint32_t output_m_index = npu_tile_index * _inner_loop[0]; output_m_index < (npu_tile_index + 1) * _inner_loop[0]; output_m_index++) {
                if (output_m_index >= M_max) {
                    continue;
                }
                while (output_m_index >= batch_boundary) {
                    previous_batch_boundary = _m_batch_dim[batch_index];
                    batch_index++;
                    batch_boundary = _m_batch_dim[batch_index];
                }
                for (uint32_t k = 0; k < _matrix_dim[1]; k++) {
                    // 1-D index
                    output_indexes[batch_index].push_back({output_m_index, k});
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
        }

        // MyAddressAllocator::check_addrs(output_addrs);
        std::string pim_movout_info = fmt::format("Store the PIM-GEMV Computation Result with {} times burst", output_addrs.size());
        spdlog::info(pim_movout_info);
        append_rope_if_enabled(tile.instructions, _apply_rope, sram_acc_addr,
                               output_index_size,
                               "Apply RoPE before storing PIM-GEMV projection");
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_acc_addr,
            .size = _inner_loop[0] * _matrix_dim[1] * output_tensors[0]->_precision,
            .src_addrs = std::move(output_addrs),
            .operand_id = _OUTPUT_OPERAND,
            .per_ch_inst = (output_tensors[0]->_tensor_type == TensorType::KCache or output_tensors[0]->_tensor_type == TensorType::VCache),
            .inst_information = pim_movout_info
        });
    }
    // spdlog::info("Tile initialized, NPU_tile_index {}, M = {}, N = {}, K = {}", npu_tile_index, M, N, K);
    return tile;
}


Tile PIMGEMV::initialize_my_attention_instructions(uint32_t B, uint32_t npu_tile_index, uint32_t head_iteration_index, uint32_t M, uint32_t K, uint32_t N, bool should_read) {

    auto tile = Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = get_name(),
        .operation_id = _id,
        .batch = 0,
        .N = N,
        .K = K,
        .M = M,
        .accum = K != 0,
        .pim_tile = true,
    };

    auto vector_tensor = _my_inputs[B];
    auto matrix_tensor = _my_weights[B];
    auto output_tensor = _my_outputs[B];

    addr_type sram_vector_base = SPAD_BASE;
    addr_type sram_addr = sram_vector_base;

    addr_type sram_acc_base = ACCUM_SPAD_BASE;
    addr_type sram_acc_addr = sram_acc_base;

    if (matrix_tensor->_tensor_type == TensorType::KCache) {
        auto m_inner = _pim_inner_loop_attn[B][0];
        auto k_inner = _pim_inner_loop_attn[B][1];
        auto n_inner= _pim_inner_loop_attn[B][2] * MyAddressAllocator::KCache_interleaved_banks_per_head;

        auto m_outer_offset = npu_tile_index * _inner_loop_attn[B][0] + m_inner * M;
        auto k_outer_offset = k_inner * K;
        auto n_outer_offset = n_inner * N;

        uint32_t M_max = vector_tensor->_dims[0];
        uint32_t K_max = MyAddressAllocator::d_k;
        uint32_t N_max = matrix_tensor->_dims[0]; // Lt

        // -- Vector --
        if (head_iteration_index == 0 && M == 0 && K == 0 && N == 0) {
            std::vector<std::vector<uint32_t>> vector_indexes;
            for (uint32_t m_inner_offset = 0; m_inner_offset < _pim_inner_loop_attn[B][0]; m_inner_offset++) {
                for (uint32_t k_index = 0; k_index < vector_tensor->_dims[1]; k_index++) {  // Load the entire Q
                    if (m_outer_offset + m_inner_offset >= M_max) {continue;}
                    vector_indexes.push_back({m_outer_offset + m_inner_offset, k_index});
                }
            }
            std::vector<addr_type> vector_addrs = vector_tensor->generate_addrs_based_on_indexes(vector_indexes);
            if (vector_addrs.empty()) {
                spdlog::info("zero load for activation / activation tensor dim: {}", vector_tensor->get_dims());
            }
            else {
                std::string pheader_info = fmt::format("Load Q Vector m = {}-{} and k = {}-{} for QKT PIM-GEMV operation",
                    vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1]);
                spdlog::info(pheader_info);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_addr,
                    .size = (uint32_t)vector_indexes.size() * vector_tensor->_precision,   // (uint32_t)activation_indexes.size() * data_width
                    .src_addrs = std::move(vector_addrs),
                    .operand_id = _INPUT_OPERAND,
                    .inst_information = pheader_info,
               });

            }
        }

        // -- PHeader --
        std::string pheader_info = fmt::format("Start QKT PIM-GEMV Computation for head interation index {} with N = {} - {}",
            head_iteration_index, N * _pim_inner_loop_attn[B][2] * MyAddressAllocator::KCache_interleaved_banks_per_head,
            std::max((N+1) * _pim_inner_loop_attn[B][2] * MyAddressAllocator::KCache_interleaved_banks_per_head - 1, K_max-1));
        spdlog::info(pheader_info);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_HEADER,
            .dest_addr = 0,
            .size = 0,
            .src_addrs = std::vector<addr_type>{sram_addr},
            .operand_id = _INPUT_OPERAND,
            .inst_information = pheader_info,
        });

        // -- PIM_GWRITE --
        if (_pim_outer_loop_attn[B][1] == 1) {
            if (K == 0 && N == 0) { //
                uint32_t gwrite_burst_times = std::ceil(static_cast<double>(_pim_inner_loop_attn[B][1] * head_per_iteration) * vector_tensor->_precision / MyAddressAllocator::dram_burst_size);  // Bursts needed to write the input vector to each channel's global buffer.
                std::string gwrite_info = fmt::format("Write Q Vector of head iteration {} with {} burst times to Global Input Buffer for QKT PIM-GEMV", head_iteration_index, gwrite_burst_times);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_GWRITE,
                    .dest_addr = 0,
                    .size = gwrite_burst_times * MyAddressAllocator::dram_burst_size,
                    .src_addrs = std::vector<addr_type>(gwrite_burst_times,sram_addr),
                    .operand_id = _OUTPUT_OPERAND,
                    .inst_information = gwrite_info,
                });
            }
        }
        else {
            uint32_t gwrite_burst_times = std::ceil(static_cast<double>(_pim_inner_loop_attn[B][1] * head_per_iteration) * vector_tensor->_precision / MyAddressAllocator::dram_burst_size);  // Bursts needed to write the input vector to each channel's global buffer.
            std::string gwrite_info = fmt::format("Write Q Vector of head iteration {} with {} burst times to Global Input Buffer for QKT PIM-GEMV", head_iteration_index, gwrite_burst_times);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_GWRITE,
                .dest_addr = 0,
                .size = gwrite_burst_times * MyAddressAllocator::dram_burst_size,
                .src_addrs = std::vector<addr_type>(gwrite_burst_times,sram_addr),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information = gwrite_info,
            });
        }

        // -- PIM_COMP -- Generate KCache Address
        uint32_t n_loop_size = n_inner;
        uint32_t k_loop_size = k_inner;

        std::vector<std::vector<uint32_t>> pim_comp_indexes;
        for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
            for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                if (k_inner * K + k_loop >= MyAddressAllocator::d_k or n_outer_offset + n_loop >= matrix_tensor->_dims[0]) {
                    continue;
                }
                pim_comp_indexes.push_back({k_outer_offset + k_loop, n_outer_offset + n_loop});
            }
        }
        auto pim_comp_addrs = matrix_tensor->generate_pim_comp_addrs_attention(head_iteration_index, pim_comp_indexes);

        std::string pim_comp_info = fmt::format("QKT PIM-GEMV operation with {} COMP instructions for the K Cache with head iteration index {}, k = {}-{}, n = {}-{}",
            pim_comp_addrs.size(), head_iteration_index,pim_comp_indexes.front()[0], pim_comp_indexes.back()[0],pim_comp_indexes.front()[1], pim_comp_indexes.back()[1]);
        // spdlog::info(pim_comp_info);

        if (MyAddressAllocator::virtual_mem_hash_enable) {
            PushHashedPIMCompInstructions(
                tile.instructions, sram_addr, pim_comp_addrs,
                _INPUT_OPERAND, pim_comp_info);
            uint32_t pim_readers_burst_times = std::ceil(pim_output_buffer_size * MyAddressAllocator::banks_per_channel / MyAddressAllocator::dram_burst_size);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_READRES,
                .dest_addr = sram_acc_addr,
                .size = pim_readers_burst_times * MyAddressAllocator::dram_burst_size,
                .src_addrs = std::vector<addr_type>(pim_readers_burst_times, 0),
                .operand_id = _OUTPUT_OPERAND,
            });
        }
        else {
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_COMP,
                .dest_addr = sram_addr,
                .size = (uint32_t)pim_comp_addrs.size() * MyAddressAllocator::dram_burst_size,
                .src_addrs = std::move(pim_comp_addrs),
                .operand_id = _INPUT_OPERAND,
                .inst_information = pim_comp_info,
            });

            // -- PIM_READRES --
            if (should_read) {
                std::string pim_readers_info = fmt::format("Readers For QKT Result for head iteration index {}, n = {}-{}",
                head_iteration_index, pim_comp_indexes.front()[1], pim_comp_indexes.back()[1]);
                spdlog::info(pim_readers_info);
                uint32_t pim_readers_burst_times = std::ceil(pim_output_buffer_size * MyAddressAllocator::banks_per_channel / MyAddressAllocator::dram_burst_size);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_READRES,
                    .dest_addr = sram_acc_addr,
                    .size = pim_readers_burst_times * MyAddressAllocator::dram_burst_size,
                    .src_addrs = std::vector<addr_type>(pim_readers_burst_times, 0),
                    .operand_id = _OUTPUT_OPERAND,
                    .inst_information = pim_readers_info,
                });
            }
        }

        // -- MOVOUT -- write all computation result from SPAD to DRAM
        if ((M+1) * _pim_inner_loop_attn[B][0] >= _inner_loop_attn[B][0] && head_iteration_index+1 >= head_compute_iteration && (N+1) * n_inner >= N_max && should_read) {
            // QKT
            std::vector<addr_type> output_addrs;
            std::vector<std::vector<uint32_t>> output_indexes;
            for (uint32_t m = npu_tile_index * _inner_loop_attn[B][0]; m < (npu_tile_index + 1) * _inner_loop_attn[B][0]; m++) {
                for (uint32_t n = 0; n < matrix_tensor->_dims[0]; n++) {
                    output_indexes.push_back({m, n});
                }
            }
            for (uint32_t head_index = 0; head_index < MyAddressAllocator::h; head_index++){
                auto QKT_output_addrs = output_tensor->generate_addrs_based_on_indexes_attention(output_indexes, head_index);
                for (auto QKT_output_addr : QKT_output_addrs) {
                    if (output_addrs.empty()) {
                        output_addrs.push_back(QKT_output_addr);
                    }
                    else if (QKT_output_addr != output_addrs.back()) {
                        output_addrs.push_back(QKT_output_addr);
                    }
                }
            }
            // std::vector<addr_type> output_addrs = output_tensor->generate_addrs_based_on_indexes(output_indexes);
            std::string pim_movout_info = fmt::format("Store the QKT Computation Result with {} times burst", output_addrs.size());
            spdlog::info(pim_movout_info);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_acc_addr,
                .size = (uint32_t)_inner_loop_attn[B][0] * matrix_tensor->_dims[0] * output_tensor->_precision,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information = pim_movout_info,
            });
        }
    }
    else if (matrix_tensor->_tensor_type == TensorType::VCache) {
        auto m_inner = _pim_inner_loop_attn[B][0];
        auto k_inner = _pim_inner_loop_attn[B][1];
        auto n_inner= _pim_inner_loop_attn[B][2] * MyAddressAllocator::KCache_interleaved_banks_per_head;

        auto m_outer_offset = npu_tile_index * _inner_loop_attn[B][0] + m_inner * M;
        auto k_outer_offset = k_inner * K;
        auto n_outer_offset = n_inner * N;

        uint32_t M_max = vector_tensor->_dims[1];
        uint32_t K_max = matrix_tensor->_dims[0];  // Lt
        uint32_t N_max = MyAddressAllocator::d_k;   // d_k

        // -- Vector --  load vector elements
        if (head_iteration_index == 0 && M == 0 && K == 0 && N == 0) {  // At the begin of NPU Tile, all Q are loaded into the NPU, and the write to PIM
            std::vector<std::vector<uint32_t>> vector_indexes;
            for (uint32_t m_inner_offset = 0; m_inner_offset < _pim_inner_loop_attn[B][0]; m_inner_offset++) {
                for (uint32_t k_index = 0; k_index < vector_tensor->_dims[2]; k_index++) {  // Load S Vector
                    if (m_outer_offset + m_inner_offset >= M_max) {continue;}
                    vector_indexes.push_back({m_outer_offset + m_inner_offset, k_index});
                }
            }
            std::vector<addr_type> vector_addrs;
            for (uint32_t head_index = 0; head_index < MyAddressAllocator::h; head_index++){
                auto S_input_addrs = vector_tensor->generate_addrs_based_on_indexes_attention(vector_indexes, head_index);
                for (auto QKT_output_addr : S_input_addrs) {
                    if (vector_addrs.empty()) {
                        vector_addrs.push_back(QKT_output_addr);
                    }
                    else if (QKT_output_addr != vector_addrs.back()) {
                        vector_addrs.push_back(QKT_output_addr);
                    }
                }
            }

            if (vector_addrs.empty()) {
                spdlog::info("zero load for activation / activation tensor dim: {}", vector_tensor->get_dims());
            }
            else {
                std::string pheader_info = fmt::format("Load S Vector m = {}-{} and k = {}-{} of {} heads for SV PIM-GEMV operation",
                    vector_indexes.front()[0], vector_indexes.back()[0], vector_indexes.front()[1], vector_indexes.back()[1], MyAddressAllocator::h);
                spdlog::info(pheader_info);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::MOVIN,
                    .dest_addr = sram_addr,
                    .size = (uint32_t)vector_indexes.size() * vector_tensor->_precision,   // (uint32_t)activation_indexes.size() * data_width
                    .src_addrs = std::move(vector_addrs),
                    .operand_id = _INPUT_OPERAND,
                    .inst_information = pheader_info,
               });
            }
        }

        // -- PHeader --  start PIM operation
        std::string pheader_info = fmt::format("Start SV PIM-GEMV Computation for head interation index {} with K = {} - {} and N = {} - {}",
            head_iteration_index, K * _pim_inner_loop_attn[B][1], std::max((K+1) * _pim_inner_loop_attn[B][1], K_max-1),
            N * _pim_inner_loop_attn[B][2] * MyAddressAllocator::VCache_interleaved_banks_per_head, (N+1) * _pim_inner_loop_attn[B][2] * MyAddressAllocator::VCache_interleaved_banks_per_head - 1);
        spdlog::info(pheader_info);
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_HEADER,
            .dest_addr = 0,
            .size = 0,
            .src_addrs = std::vector<addr_type>{sram_addr},
            .operand_id = _INPUT_OPERAND,
            .inst_information = pheader_info,
        });

        // -- PIM_GWRITE --
        if (_pim_outer_loop_attn[B][1] == 1) {
            if (K == 0 && N == 0) {
                uint32_t gwrite_burst_times = std::ceil(static_cast<double>(_pim_inner_loop_attn[B][1] * head_per_iteration) * vector_tensor->_precision / MyAddressAllocator::dram_burst_size);  // Bursts needed to write the input vector to each channel's global buffer.
                std::string gwrite_info = fmt::format("Write Q Vector of head iteration {} with {} burst times to Global Input Buffer for QKT PIM-GEMV", head_iteration_index, gwrite_burst_times);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_GWRITE,
                    .dest_addr = 0,
                    .size = gwrite_burst_times * MyAddressAllocator::dram_burst_size,
                    .src_addrs = std::vector<addr_type>(gwrite_burst_times,sram_addr),
                    .operand_id = _OUTPUT_OPERAND,
                    .inst_information = gwrite_info,
                });
            }
        }
        else {
            uint32_t gwrite_burst_times = std::ceil(static_cast<double>(_pim_inner_loop_attn[B][1] * head_per_iteration) * vector_tensor->_precision / MyAddressAllocator::dram_burst_size);  // Bursts needed to write the input vector to each channel's global buffer.
            std::string gwrite_info = fmt::format("Write Q Vector of head iteration {} with {} burst times to Global Input Buffer for QKT PIM-GEMV", head_iteration_index, gwrite_burst_times);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_GWRITE,
                .dest_addr = 0,
                .size = gwrite_burst_times * MyAddressAllocator::dram_burst_size,
                .src_addrs = std::vector<addr_type>(gwrite_burst_times,sram_addr),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information = gwrite_info,
            });
        }

        // -- PIM_COMP --
        // comput the index of K and N
        uint32_t n_loop_size = n_inner;
        uint32_t k_loop_size = k_inner;

        std::vector<std::vector<uint32_t>> pim_comp_indexes;
        for (int n_loop = 0; n_loop < n_loop_size; n_loop++) {
            for (int k_loop = 0; k_loop < k_loop_size; k_loop++) {
                if (k_inner * K + k_loop >= matrix_tensor->_dims[0] or n_outer_offset + n_loop >= MyAddressAllocator::d_k) {
                    continue;
                }
                pim_comp_indexes.push_back({k_outer_offset + k_loop, n_outer_offset + n_loop});
            }
        }
        auto pim_comp_addrs = matrix_tensor->generate_pim_comp_addrs_attention(head_iteration_index, pim_comp_indexes);

        std::string pim_comp_info = fmt::format("SV PIM-GEMV operation with {} COMP instructions for the V Cache with head iteration index {}, k = {}-{}, n = {}-{}",
            pim_comp_addrs.size(),head_iteration_index,
            pim_comp_indexes.front()[0], pim_comp_indexes.back()[0],
            pim_comp_indexes.front()[1], pim_comp_indexes.back()[1]);
        // spdlog::info(pim_comp_info);
        if (MyAddressAllocator::virtual_mem_hash_enable) {
            PushHashedPIMCompInstructions(
                tile.instructions, sram_addr, pim_comp_addrs,
                _INPUT_OPERAND, pim_comp_info);
            uint32_t pim_readers_burst_times = std::ceil(pim_output_buffer_size * MyAddressAllocator::banks_per_channel / MyAddressAllocator::dram_burst_size);
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_READRES,
                .dest_addr = sram_acc_addr,
                .size = pim_readers_burst_times * MyAddressAllocator::dram_burst_size,
                .src_addrs = std::vector<addr_type>(pim_readers_burst_times, 0),
                .operand_id = _OUTPUT_OPERAND,
            });
        }
        else {
            tile.instructions.push_back(Instruction{
               .opcode = Opcode::PIM_COMP,
               .dest_addr = sram_addr,
               .size = (uint32_t)pim_comp_addrs.size() * MyAddressAllocator::dram_burst_size,
               .src_addrs = std::move(pim_comp_addrs),
               .operand_id = _INPUT_OPERAND,
               .inst_information = pim_comp_info,
            });

            // -- PIM_READRES -- Host read the computation result
            if (should_read) {
                std::string pim_readers_info = fmt::format("Readers For SV Result for head iteration index {}, n = {}-{}",
                head_iteration_index, pim_comp_indexes.front()[1], pim_comp_indexes.back()[1]);
                spdlog::info(pim_readers_info);
                uint32_t pim_readers_burst_times = std::ceil(pim_output_buffer_size * MyAddressAllocator::banks_per_channel / MyAddressAllocator::dram_burst_size);
                tile.instructions.push_back(Instruction{
                    .opcode = Opcode::PIM_READRES,
                    .dest_addr = sram_acc_addr,
                    .size = pim_readers_burst_times * MyAddressAllocator::dram_burst_size,
                    .src_addrs = std::vector<addr_type>(pim_readers_burst_times, 0),
                    .operand_id = _OUTPUT_OPERAND,
                    .inst_information = pim_readers_info,
                });
            }
        }

        // -- MOVOUT -- last batch for the NPU tile, write the NPU result in SPAD back to DRAM
        if ((M+1) * _pim_inner_loop_attn[B][0] >= _inner_loop_attn[B][0] && head_iteration_index+1 >= head_compute_iteration && (N+1) * n_inner >= N_max && should_read) {
            std::vector<std::vector<uint32_t>> output_indexes;
            for (uint32_t m = npu_tile_index * _inner_loop_attn[B][0]; m < (npu_tile_index + 1) * _inner_loop_attn[B][0]; m++) {
                for (uint32_t n = 0; n < matrix_tensor->_dims[1]; n++) {
                    output_indexes.push_back({m, n});
                }
            }
            auto output_addrs = output_tensor->generate_addrs_based_on_indexes(output_indexes);
            std::string pim_movout_info = fmt::format("Store the SV Computation Result with {} times burst", output_addrs.size());
            spdlog::info(pim_movout_info);
            tile.instructions.push_back(Instruction{ // MOVOUT
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_acc_addr,
                .size = (uint32_t)_inner_loop_attn[B][0] * matrix_tensor->_dims[0] * output_tensor->_precision,
                .src_addrs = std::move(output_addrs),
                .operand_id = _OUTPUT_OPERAND,
                .inst_information = pim_movout_info,
            });
        }
    }
    return tile;
}
