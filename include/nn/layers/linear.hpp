#pragma once
#include "../../tensor/prng.hpp"
#include "module.hpp"
#include <cmath>
#include <iostream>

namespace gradientcore {
namespace nn {

class Linear : public Module {
public:
  uint32_t in_features;
  uint32_t out_features;
  bool has_bias;
  autograd::Variable *weight;  
  autograd::Variable *bias;    

  Linear(Arena *perm_arena, uint32_t in_f, uint32_t out_f, bool use_bias = true)
      : in_features(in_f), out_features(out_f), has_bias(use_bias) {

    if (in_features == 0 || out_features == 0) {
      std::cerr << "Error: Linear layer dimensions must be > 0" << std::endl;
      std::cerr << "  in_features=" << in_features 
                << " out_features=" << out_features << std::endl;
      weight = nullptr;
      bias = nullptr;
      return;
    }

    uint32_t w_shape[2] = {in_features, out_features};
    Tensor *w_tensor = tensor_create(perm_arena, 2, w_shape);
    if (w_tensor == nullptr) {
      std::cerr << "Error: Failed to create weight tensor" << std::endl;
      weight = nullptr;
      bias = nullptr;
      return;
    }
    weight = autograd::create_leaf(perm_arena, w_tensor, true);
    register_parameter(weight);

    if (use_bias) {
      uint32_t b_shape[2] = {1, out_features};
      Tensor *b_tensor = tensor_create(perm_arena, 2, b_shape);
      if (b_tensor == nullptr) {
        std::cerr << "Error: Failed to create bias tensor" << std::endl;
        bias = nullptr;
        return;
      }
      bias = autograd::create_leaf(perm_arena, b_tensor, true);
      register_parameter(bias);
    } else {
      bias = nullptr;
    }

    reset_parameters();
  }

  void reset_parameters() {
    if (weight == nullptr || weight->data == nullptr) {
      std::cerr << "Error: Weight not initialized" << std::endl;
      return;
    }

    float bound = std::sqrt(1.0f / static_cast<float>(in_features));

    for (uint64_t i = 0; i < weight->data->size; i++) {
      weight->data->storage->data[weight->data->offset + i] =
          ((prng::randf() * 2.0f) - 1.0f) * bound;
    }

    if (bias != nullptr && bias->data != nullptr) {
      for (uint64_t i = 0; i < bias->data->size; i++) {
        bias->data->storage->data[bias->data->offset + i] =
            ((prng::randf() * 2.0f) - 1.0f) * bound;
      }
    }
  }

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    if (x == nullptr || x->data == nullptr) {
      std::cerr << "Error: Invalid input to Linear layer" << std::endl;
      return nullptr;
    }

    if (weight == nullptr || weight->data == nullptr) {
      std::cerr << "Error: Linear layer not properly initialized" << std::endl;
      return nullptr;
    }

    if (x->data->ndims != 2) {
      std::cerr << "Warning: Linear layer expects 2D input, got " 
                << x->data->ndims << "D" << std::endl;
    }

    if (x->data->ndims >= 2 && x->data->shape[1] != in_features) {
      std::cerr << "Error: Input feature dimension mismatch in Linear layer" << std::endl;
      std::cerr << "  Expected: " << in_features 
                << " Got: " << x->data->shape[1] << std::endl;
      return nullptr;
    }

    autograd::Variable *out = autograd::matmul(compute_arena, x, weight);

    if (has_bias && bias != nullptr) {
      out = autograd::add(compute_arena, out, bias);
    }
    return out;
  }

  void summary() override {
    std::cout << "Linear(in=" << in_features << ", out=" << out_features;
    if (has_bias) std::cout << ", bias=true";
    else std::cout << ", bias=false";
    std::cout << ")";
    std::cout << " [" << num_parameters() << " params]" << std::endl;
  }
};

} // namespace nn
} // namespace gradientcore
