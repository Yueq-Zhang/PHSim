#pragma once

#include "Operation.h"

class DataConvert : public Operation {
   public:
    DataConvert(std::string name, AllocationScheme src_scheme, AllocationScheme dst_scheme);

    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs,
                                              TensorType output_tensor_type = TensorType::WGT) override;

   private:
    AllocationScheme _src_scheme;
    AllocationScheme _dst_scheme;
    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    void calculate_my_loops();
    void initialize_my_tiles();
    Tile initialize_my_instructions(uint32_t M, uint32_t N);
    uint32_t sram_size_needed();
};
