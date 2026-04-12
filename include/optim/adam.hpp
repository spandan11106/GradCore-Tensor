#pragma once
#include "../autograd/autograd.hpp"
#include <vector>

namespace gradientcore {
namespace optim {

struct AdamState {
  Tensor *m;
  Tensor *v;
};

class Adam {
private:
  std::vector<autograd::Variable *> parameters;
  std::vector<AdamState> states;
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  uint32_t t;

public:
  Adam(Arena *perm_arena, const std::vector<autograd::Variable *> &params,
       float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f,
       float eps = 1e-8f);

  void step(Arena *temp_arena = nullptr);
  void zero_grad();
};

} // namespace optim
} // namespace gradientcore
