#include "../../include/tensor/tensor.hpp"
#include <cstdint>

namespace gradientcore {

Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape) {
  if (arena == nullptr || ndims == 0 || ndims > MAX_TENSOR_DIMS)
    return nullptr;

  for (uint32_t i = 0; i < ndims; i++) {
    if (shape[i] == 0)
      return nullptr;
  }

  Tensor *t = arena->push<Tensor>();
  t->ndims = ndims;
  t->size = 1;
  t->offset = 0;

  for (uint32_t i = 0; i < ndims; i++) {
    t->shape[i] = shape[i];
    t->size *= shape[i];
  }

  t->strides[ndims - 1] = 1;
  for (int32_t i = ndims - 2; i >= 0; i--) {
    uint64_t s = (uint64_t)t->strides[i + 1] * t->shape[i + 1];
    t->strides[i] = (uint32_t)s;
  }

  t->storage = arena->push<TensorStorage>();
  t->storage->size = t->size;
  t->storage->data = arena->push_array<float>(t->size, false);

  return t;
}

Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape) {
  Tensor *t = tensor_create(arena, ndims, shape);
  if (t)
    tensor_fill(t, 0.0f);
  return t;
}

} // namespace gradientcore
