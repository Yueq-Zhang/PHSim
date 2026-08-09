#pragma once

#include "Operation.h"

// split input tensor to three
class Split : public Operation {
   public:
    Split(std::string name, std::vector<uint32_t> units, uint32_t dim);

    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs);

    // todo: add attributes
    // maybe the axis and number of splitting heads?
    // currently, it's only for last axis and the splitting heads is fixed to
    // three.
   private:
    std::vector<uint32_t> _units;
    uint32_t _sum;
    uint32_t _dim;
};