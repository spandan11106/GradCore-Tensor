#include "../../../include/nn/layers/dropout.hpp"
#include "../../../include/tensor/prng.hpp"

namespace gradientcore {
namespace nn {

Dropout::Dropout(Arena *perm_arena, float p) : p(p) {
  if (p < 0.0f || p >= 1.0f) {
    std::cerr << "Error: Dropout probability must be in [0, 1)" << std::endl;
    this->p = 0.5f; // Default fallback
  }
}

autograd::Variable *Dropout::forward(Arena *compute_arena, autograd::Variable *x) {
  if (!x || !x->data) {
    std::cerr << "Error: Invalid input to Dropout" << std::endl;
    return nullptr;
  }

  if (p == 0.0f) {
    return x;
  }

  if (!_training) {
    return x;
  }

  uint32_t out_shape[4];
  uint32_t ndims = x->data->ndims;
  for (uint32_t i = 0; i < ndims; i++) {
    out_shape[i] = x->data->shape[i];
  }

  Tensor *out_tensor = tensor_create(compute_arena, ndims, out_shape);
  if (!out_tensor) {
    std::cerr << "Error: Failed to allocate output tensor for Dropout" << std::endl;
    return nullptr;
  }

  autograd::Variable *out = autograd::create_variable(compute_arena, out_tensor, true);
  
  float *in_data = x->data->storage->data + x->data->offset;
  float *out_data = out->data->storage->data + out->data->offset;

  uint64_t total_elements = 1;
  for (uint32_t i = 0; i < ndims; i++) {
    total_elements *= x->data->shape[i];
  }

  float scale = 1.0f / (1.0f - p);

  for (uint64_t i = 0; i < total_elements; i++) {
    float rand_val = prng::randf();
    if (rand_val < (1.0f - p)) {
      out_data[i] = in_data[i] * scale;
    } else {
      out_data[i] = 0.0f;
    }
  }

  return out;
}

} // namespace nn
} // namespace gradientcore
