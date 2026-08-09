#pragma once
#include "../tensor/MyTensor.hpp"
#include "Operation.h"

class Mul : public Operation {
   public:
    explicit Mul(std::string name);
    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type = TensorType::ACT);

   private:
    std::vector<Ptr<MyTensor>> _my_inputs_1;
    std::vector<Ptr<MyTensor>> _my_inputs_2;
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_my_loops();
    void initialize_my_tiles();
    Tile initialize_my_instructions(uint32_t N);
    uint32_t sram_size_needed();
};
