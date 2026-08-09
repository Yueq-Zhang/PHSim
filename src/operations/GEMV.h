#pragma once

#include "Operation.h"
#include <fmt/format.h>

class GEMV : public Operation {
public:
    GEMV(std::string name, std::vector<Ptr<MyTensor>> weights);
    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type = TensorType::ACT);
    std::vector<Ptr<MyTensor>> kvcache_append(std::vector<Ptr<MyTensor>> inputs, std::vector<Ptr<MyTensor>> kvcaches, TensorType output_tensor_type = TensorType::ACT);

private:
    std::vector<uint32_t> _matrix_dim;
    std::vector<uint32_t> _vector_dim;

    uint32_t _prod_batches;
    uint32_t _prod_weight_dim;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    std::vector<std::vector<uint32_t>> _inner_loop_attn;
    std::vector<std::vector<uint32_t>> _outer_loop_attn;

    TensorType matrix_tensor_type;

    void calculate_my_loops();
    void initialize_my_tiles();
    Tile initialize_my_instructions(uint32_t M, uint32_t K, uint32_t N, bool should_store);

    Tile initialize_my_instructions_tile(uint32_t M, uint32_t K, uint32_t N, bool should_store);
    Tile initialize_my_attention_instructions(uint32_t B, uint32_t head_index, uint32_t M, uint32_t K, uint32_t N, bool should_store);

    void calculate_attention_loops();

    uint32_t sram_size_needed();

    bool cache_append;
};