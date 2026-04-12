#pragma once

#include "../core/module.hpp"
#include "../../tensor/tensor.hpp"
#include <cmath>
#include <iostream>

namespace gradientcore {
namespace nn {

class BatchNorm1d : public Module {
private:
  uint32_t num_features;
  float momentum;       
  float epsilon;        
  
  autograd::Variable *gamma;   
  autograd::Variable *beta;    
  
  Tensor *running_mean;
  Tensor *running_var;
  
  uint64_t num_batches_tracked;

public:
  BatchNorm1d(Arena *perm_arena, uint32_t num_features,
              float momentum = 0.1f, float epsilon = 1e-5f);

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override;

  uint32_t get_num_features() const { return num_features; }
  float get_momentum() const { return momentum; }
};

class BatchNorm2d : public Module {
private:
  uint32_t num_features;  
  float momentum;
  float epsilon;
  
  autograd::Variable *gamma;
  autograd::Variable *beta;
  
  Tensor *running_mean;
  Tensor *running_var;
  
  uint64_t num_batches_tracked;

public:
  BatchNorm2d(Arena *perm_arena, uint32_t num_features,
              float momentum = 0.1f, float epsilon = 1e-5f);

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override;

  uint32_t get_num_features() const { return num_features; }
};

} // namespace nn
} // namespace gradientcore
