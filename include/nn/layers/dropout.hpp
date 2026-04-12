#pragma once

#include "../core/module.hpp"
#include "../../tensor/tensor.hpp"
#include "../../tensor/prng.hpp"
#include <iostream>

namespace gradientcore {
namespace nn {

class Dropout : public Module {
private:
  float p;  

public:
  explicit Dropout(float dropout_prob = 0.5f) : p(dropout_prob) {
    if (p < 0.0f || p > 1.0f) {
      std::cerr << "Warning: Dropout probability should be between 0 and 1, got " 
                << p << std::endl;
    }
  }

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override;

  float get_dropout_prob() const { return p; }
};

} // namespace nn
} // namespace gradientcore
