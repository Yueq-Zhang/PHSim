#pragma once
#include "../tensor/MyTensor.hpp"
#include "Operation.h"

class Concat : public Operation {
   public:
    Concat(std::string name, uint32_t dim);

    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs);

   private:
    uint32_t _dim;
};