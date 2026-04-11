#pragma once
#include "../autograd/autograd.hpp"
#include <deque>
#include <functional>
#include <vector>

namespace gradientcore {
namespace optim {

struct LBFGS_StepData {
  Tensor *s;
  Tensor *y;
  float rho;
};

class LBFGS {
private:
  std::vector<autograd::Variable *> parameters;
  Arena *perm_arena;

  uint32_t history_size;
  float learning_rate;
  float tolerance_grad;
  float tolerance_change;

  std::deque<LBFGS_StepData> history;
  Tensor *d; // Search direction

  uint64_t num_params;
  uint32_t n_iter;

public:
  LBFGS(Arena *perm_arena, const std::vector<autograd::Variable *> &params,
        float lr = 1.0f, uint32_t history_size = 20, float tol_grad = 1e-7f,
        float tol_change = 1e-9f);

  // L-BFGS requires a closure that evaluates the network and returns the loss.
  // This is because the line-search evaluates the network multiple times per
  // step.
  float step(Arena *temp_arena, std::function<float()> closure);

  void zero_grad();
};

} // namespace optim
} // namespace gradientcore
