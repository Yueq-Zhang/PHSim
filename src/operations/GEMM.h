#pragma once
#include "Operation.h"

class GEMM : public Operation {
   public:
    GEMM(std::string name, std::vector<Ptr<MyTensor>> weights); // 函数重载，基于My tensor完成所有运算
    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type); // 定义了MatMul计算输出的tensor类型

    std::string get_optype() override {
        if (matrix_tensor_type == TensorType::KCache || matrix_tensor_type == TensorType::VCache) return "GEMM_Att";
        return "GEMM";
    }

    void set_transposed() { _is_transposed = true; }
    // todo: add attributes
    // currently, values below are dummy.
    TensorType matrix_tensor_type;

    std::vector<uint32_t> get_inner_loop() {return _inner_loop; }
    std::vector<uint32_t> get_outer_loop() {return _outer_loop; }

   private:
    uint32_t _alpha;
    uint32_t _beta;
    bool _transA;
    bool _transB;
    bool _is_transposed;
    bool _is_gemv;

    // 3 dimensions, only for n, k, m
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    std::vector<std::vector<uint32_t>> _inner_loop_attn;
    std::vector<std::vector<uint32_t>> _outer_loop_attn;

    void calculate_my_loops();
    void calculate_attention_loops();
    void initialize_my_tiles();
    Tile make_deferred_gemm_tile(uint32_t B, uint32_t M, uint32_t K,
                                 uint32_t N, bool should_store);
    Tile make_deferred_attention_tile(uint32_t B, uint32_t head_index,
                                      uint32_t M, uint32_t K, uint32_t N,
                                      bool should_store);
    Tile initialize_my_instructions(uint32_t B, uint32_t N, uint32_t K, uint32_t M, bool should_store);

    Tile initialize_my_DASH_instructions(uint32_t B, uint32_t M, uint32_t K, uint32_t N, bool should_store);
    Tile initialize_my_attention_instructions(uint32_t B, uint32_t head_index, uint32_t M, uint32_t K, uint32_t N, bool should_store);


    Tile initialize_my_DASH_instructions_activation_first(uint32_t B, uint32_t M, uint32_t K, uint32_t N, bool should_store);

    uint32_t sram_size_needed();

    void calculate_my_loops_prime();
};
