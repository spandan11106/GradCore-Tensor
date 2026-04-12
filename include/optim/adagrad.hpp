#pragma once
#include "../autograd/autograd.hpp"
#include <vector>

namespace gradientcore {
namespace optim {

struct AdagradState {
  Tensor *sum_sq;
};

class Adagrad {
private:
  std::vector<autograd::Variable *> parameters;
  std::vector<AdagradState> states;
  float learning_rate;
  float epsilon;
  float weight_decay;

public:
  Adagrad(Arena *perm_arena, const std::vector<autograd::Variable *> &params,
          float lr = 0.01f, float eps = 1e-10f, float weight_decay = 0.0f);

  void step(Arena *temp_arena = nullptr);
  void zero_grad();
};

} // namespace optim
} // namespace gradientcore
