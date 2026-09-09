#pragma once
#include "Operation.h"

class Softmax : public Operation {
   public:
    Softmax(std::string name);
    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type = TensorType::ACT);
    std::string get_optype() override { return "Softmax"; }

   private:
    uint32_t axis;
    uint32_t _batch_size;
    std::vector<uint32_t> _input_dim;
    uint32_t _prod_batches;
    // uint32_t softmax_size;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    std::vector<std::vector<uint32_t>> _inner_loop_softmax;
    std::vector<std::vector<uint32_t>> _outer_loop_softmax;

    void calculate_my_loops();
    void initialize_my_tiles();
    Tile make_deferred_tile(uint32_t N, uint32_t req_idx);
    Tile initialize_my_instructions(uint32_t N, uint32_t req_idx);
    uint64_t my_sram_size_needed();
};
