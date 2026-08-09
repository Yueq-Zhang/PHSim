#pragma once

#include "Operation.h"

class Reshape : public Operation {
   public:
    Reshape(std::string name, std::vector<uint32_t> shape);

    std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs);

   private:
    std::vector<uint32_t> _shape;
};