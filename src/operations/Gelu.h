#pragma once
#include "Operation.h"

class Gelu : public Operation {
   public:
    Gelu(std::string name);
    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type = TensorType::ACT);

   private:
    uint32_t _prod_batches;
    std::vector<uint32_t> _input_dim;
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_my_loops();
    void initialize_my_tiles();
    Tile initialize_my_instructions(uint32_t N);
    uint64_t my_sram_size_needed();
};
