#pragma once
#include "../tensor/prng.hpp"
#include "module.hpp"
#include <cmath>

namespace gradientcore {
namespace nn {

class Linear : public Module {
public:
  autograd::Variable *weight;
  autograd::Variable *bias;

  Linear(Arena *perm_arena, uint32_t in_features, uint32_t out_features,
         bool use_bias = true) {
    uint32_t w_shape[2] = {in_features, out_features};
    Tensor *w_tensor = tensor_create(perm_arena, 2, w_shape);

    float limit = std::sqrt(1.0f / in_features);
    for (uint64_t i = 0; i < w_tensor->size; i++) {
      w_tensor->storage->data[w_tensor->offset + i] =
          ((prng::randf() * 2.0f) - 1.0f) * limit;
    }
    weight = autograd::create_leaf(perm_arena, w_tensor, true);
    register_parameter(weight);

    if (use_bias) {
      uint32_t b_shape[2] = {1, out_features};
      Tensor *b_tensor = tensor_create_zeros(perm_arena, 2, b_shape);
      bias = autograd::create_leaf(perm_arena, b_tensor, true);
      register_parameter(bias);
    } else {
      bias = nullptr;
    }
  }

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    autograd::Variable *out = autograd::matmul(compute_arena, x, weight);
    if (bias) {
      out = autograd::add(compute_arena, out, bias);
    }
    return out;
  }
};

} // namespace nn
} // namespace gradientcore
