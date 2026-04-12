#include "tensor/tensor.hpp"
#include <cstdint>
#include <cstring>

namespace gradientcore {

Tensor *tensor_view(Arena *arena, const Tensor *src) {
  Tensor *t = arena->push<Tensor>();
  t->ndims = src->ndims;
  t->size = src->size;
  t->offset = src->offset;
  t->storage = src->storage;

  std::memcpy(t->shape, src->shape, sizeof(uint32_t) * MAX_TENSOR_DIMS);
  std::memcpy(t->strides, src->strides, sizeof(uint32_t) * MAX_TENSOR_DIMS);

  return t;
}

Tensor *tensor_reshape(Arena *arena, const Tensor *src, uint32_t ndims,
                       const uint32_t *shape) {
  if (ndims == 0 || ndims > MAX_TENSOR_DIMS)
    return nullptr;

  if (!tensor_is_contiguous(src))
    return nullptr;

  uint64_t new_size = 1;
  for (uint32_t i = 0; i < ndims; i++) {
    new_size *= shape[i];
  }
  if (new_size != src->size)
    return nullptr;

  Tensor *t = tensor_view(arena, src);
  t->ndims = ndims;

  for (uint32_t i = 0; i < ndims; i++) {
    t->shape[i] = shape[i];
  }
  t->strides[ndims - 1] = 1;
  for (int32_t i = (int32_t)ndims - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
  }

  return t;
}

Tensor *tensor_transpose(Arena *arena, const Tensor *src, uint32_t dim0,
                         uint32_t dim1) {
  if (dim0 >= src->ndims || dim1 >= src->ndims)
    return nullptr;

  Tensor *t = tensor_view(arena, src);

  uint64_t temp_shape = t->shape[dim0];
  t->shape[dim0] = t->shape[dim1];
  t->shape[dim1] = temp_shape;

  uint32_t temp_stride = t->strides[dim0];
  t->strides[dim0] = t->strides[dim1];
  t->strides[dim1] = temp_stride;

  return t;
}

} // namespace gradientcore
