#pragma once

#include "memory_cpu/arena.hpp"
#include <cstdint>

namespace gradientcore {

constexpr uint32_t MAX_TENSOR_DIMS = 10;

struct TensorStorage {
  float *data;
  uint64_t size;
};

struct Tensor {
  uint32_t ndims;
  uint32_t shape[MAX_TENSOR_DIMS];
  uint32_t strides[MAX_TENSOR_DIMS];
  uint64_t size;
  uint64_t offset; // Starting index in the data

  TensorStorage *storage;
};

// Creation
Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape);
Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape);

// Views
Tensor *tensor_view(Arena *arena, const Tensor *src);
Tensor *tensor_reshape(Arena *arena, const Tensor *src, uint32_t ndims,
                       const uint32_t *shape);
Tensor *tensor_transpose(Arena *arena, const Tensor *src, uint32_t dim0,
                         uint32_t dim1);

// Utils
uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices);
void tensor_clear(Tensor *t);
bool tensor_copy(Tensor *dst, const Tensor *src);
void tensor_fill(Tensor *t, float val);
bool tensor_is_contiguous(const Tensor *t);

// Arithmetics
bool tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
bool tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
bool tensor_mul(Tensor *out, const Tensor *a, const Tensor *b);
bool mat_mul(Tensor *out, const Tensor *a, const Tensor *b, bool zero_out,
             bool transpose_a, bool transpose_b);

} // namespace gradientcore
