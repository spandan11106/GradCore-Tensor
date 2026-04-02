#include "../../include/tensor/tensor.hpp"

namespace gradientcore {

uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices) {
  uint64_t flat_index = t->offset;
  for (uint32_t i = 0; i < t->ndims; i++) {
    flat_index += indices[i] * t->strides[i];
  }
  return flat_index;
}

bool tensor_is_contiguous(const Tensor *t) {
  uint64_t expected_stride = 1;
  for (int32_t i = t->ndims - 1; i >= 0; i--) {
    if (t->shape[i] == 1)
      continue; // Broadcasting dimensions don't break contiguity
    if (t->strides[i] != expected_stride)
      return false;
    expected_stride *= t->shape[i];
  }
  return true;
}

void tensor_clear(Tensor *t) {
  if (tensor_is_contiguous(t)) {
    std::memset(t->storage->data + t->offset, 0, t->size * sizeof(float));
  } else {
    tensor_fill(t, 0.0f);
  }
}

void tensor_fill(Tensor *t, float val) {
  if (tensor_is_contiguous(t)) {
    for (uint64_t i = 0; i < t->size; i++) {
      t->storage->data[t->offset + i] = val;
    }
  } else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    for (uint64_t i = 0; i < t->size; i++) {
      uint64_t flat_idx = tensor_get_flat_index(t, indices);
      t->storage->data[flat_idx] = val;

      for (int32_t d = t->ndims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < t->shape[d]) {
          break;
        }
        indices[d] = 0;
      }
    }
  }
}

bool tensor_copy(Tensor *dst, const Tensor *src) {
  if (dst == nullptr || src == nullptr)
    return false;

  if (dst->size != src->size)
    return false;

  if (dst->ndims != src->ndims)
    return false;

  for (uint32_t i = 0; i < dst->ndims; i++) {
    if (dst->shape[i] != src->shape[i])
      return false;
  }

  uint32_t indices[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < src->size; i++) {
    uint64_t src_idx = tensor_get_flat_index(src, indices);
    uint64_t dst_idx = tensor_get_flat_index(dst, indices);

    dst->storage->data[dst_idx] = src->storage->data[src_idx];

    for (int32_t d = src->ndims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < src->shape[d]) {
        break;
      }
      indices[d] = 0;
    }
  }

  return true;
}

} // namespace gradientcore
