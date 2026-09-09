#pragma once
#include "Operation.h"


class PIMGEMV : public Operation {
public:
    PIMGEMV(std::string name, std::vector<Ptr<MyTensor>> weights);
    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type = TensorType::ACT);
    std::vector<Ptr<MyTensor>> kvcache_append(std::vector<Ptr<MyTensor>> inputs, std::vector<Ptr<MyTensor>> kvcaches, TensorType output_tensor_type = TensorType::ACT);
    std::string get_optype() override { return "PIMGEMV"; }
    std::vector<uint32_t> get_inner_loop() override {
        return _pim_inner_loop;
    }
    std::vector<uint32_t> get_outer_loop() override {
        return _pim_outer_loop;
    }

private:
    std::vector<uint32_t> _vector_dim;
    std::vector<uint32_t> _matrix_dim;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    std::vector<uint32_t> _pim_inner_loop;
    std::vector<uint32_t> _pim_outer_loop;

    std::vector<std::vector<uint32_t>> _inner_loop_attn;
    std::vector<std::vector<uint32_t>> _outer_loop_attn;

    std::vector<std::vector<uint32_t>> _pim_inner_loop_attn;
    std::vector<std::vector<uint32_t>> _pim_outer_loop_attn;

    TensorType matrix_tensor_type;

    uint32_t _prod_batches;

    uint32_t column_interleave;
    uint32_t weight_rows_per_bank_row;

    uint64_t sram_size_needed();
    uint32_t pim_buffer_size_needed();
    uint32_t pim_input_buffer_size_needed();
    uint32_t pim_output_buffer_size_needed();

    void calculate_my_loops();
    void initialize_my_tiles();

    // 重写这两个函数
    Tile initialize_my_instructions(uint32_t npu_tile_index, uint32_t pim_tile_index);  // 针对weight的
    Tile initialize_my_instructions(uint32_t npu_tile_index, uint32_t head_tile_index, uint32_t pim_tile_index); // 针对KV cache计算的

    Tile initialize_my_instructions(uint32_t npu_tile_index, uint32_t M, uint32_t K, uint32_t N, bool should_read);
    Tile initialize_my_attention_instructions(uint32_t B, uint32_t npu_tile_index, uint32_t head_index, uint32_t M, uint32_t K, uint32_t N, bool should_read);

    uint32_t head_compute_iteration; // 执行PIM GEMV的过程中，head iteration的数量
    uint32_t head_per_iteration;
    uint32_t pim_gemv_granularity;  // PIM GEMV计算过程中的最小执行单位, 基于每次Burst的加载数量与计算量获得

    uint32_t pim_global_input_buffer_size;
    uint32_t pim_output_buffer_size;

    bool cache_append;
};
