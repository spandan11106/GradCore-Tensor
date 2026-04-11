#pragma once
#include "../../include/autograd/autograd.hpp"
#include <vector>

namespace gradientcore {
namespace optim {

inline uint64_t
get_total_params_size(const std::vector<autograd::Variable *> &params) {
  uint64_t total = 0;
  for (auto *p : params) {
    if (p->requires_grad)
      total += p->data->size;
  }
  return total;
}

inline void flatten_params(const std::vector<autograd::Variable *> &params,
                           Tensor *flat_out) {
  uint64_t offset = 0;
  for (auto *p : params) {
    if (!p->requires_grad)
      continue;
    std::memcpy(flat_out->storage->data + flat_out->offset + offset,
                p->data->storage->data + p->data->offset,
                p->data->size * sizeof(float));
    offset += p->data->size;
  }
}

inline void unflatten_params(Tensor *flat_in,
                             const std::vector<autograd::Variable *> &params) {
  uint64_t offset = 0;
  for (auto *p : params) {
    if (!p->requires_grad)
      continue;
    std::memcpy(p->data->storage->data + p->data->offset,
                flat_in->storage->data + flat_in->offset + offset,
                p->data->size * sizeof(float));
    offset += p->data->size;
  }
}

inline void flatten_grads(const std::vector<autograd::Variable *> &params,
                          Tensor *flat_out) {
  uint64_t offset = 0;
  for (auto *p : params) {
    if (!p->requires_grad || !p->grad)
      continue;
    std::memcpy(flat_out->storage->data + flat_out->offset + offset,
                p->grad->storage->data + p->grad->offset,
                p->grad->size * sizeof(float));
    offset += p->grad->size;
  }
}

inline float tensor_dot_1d(Tensor *a, Tensor *b) {
  float sum = 0.0f;
  for (uint64_t i = 0; i < a->size; i++) {
    sum += a->storage->data[a->offset + i] * b->storage->data[b->offset + i];
  }
  return sum;
}

} // namespace optim
} // namespace gradientcore
