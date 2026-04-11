#pragma once
#include "../autograd/autograd.hpp"
#include <vector>

namespace gradientcore {
namespace optim {

class SGD {
private:
  std::vector<autograd::Variable *> parameters;
  float learning_rate;

public:
  SGD(const std::vector<autograd::Variable *> &params, float lr);

  void step(Arena *temp_arena);

  void zero_grad();
};

} // namespace optim
} // namespace gradientcore
