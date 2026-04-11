#include "../../../include/tensor/tensor.hpp"
#include <cstdint>

namespace gradientcore {

bool tensor_mul(Tensor *out, const Tensor *a, const Tensor *b) {
  if (out == nullptr || a == nullptr || b == nullptr)
    return false;

  uint32_t expected_ndims;
  uint32_t expected_shape[MAX_TENSOR_DIMS];
  if (!tensor_check_broadcastable(a, b, &expected_ndims, expected_shape))
    return false;

  if (out->ndims != expected_ndims)
    return false;
  for (uint32_t i = 0; i < expected_ndims; i++) {
    if (out->shape[i] != expected_shape[i])
      return false;
  }

  bool exact_match = (a->ndims == b->ndims) && shape_match(a, b);

  if (exact_match && tensor_is_contiguous(a) && tensor_is_contiguous(b) &&
      tensor_is_contiguous(out)) {
    for (uint64_t i = 0; i < out->size; i++) {
      out->storage->data[out->offset + i] =
          a->storage->data[a->offset + i] * b->storage->data[b->offset + i];
    }
  } else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    for (uint64_t i = 0; i < out->size; i++) {
      uint64_t flat_idx_out = tensor_get_flat_index(out, indices);

      uint64_t flat_idx_a = a->offset;
      for (uint32_t d = 0; d < a->ndims; d++) {
        uint32_t out_d = out->ndims - a->ndims + d;
        uint32_t idx = indices[out_d];
        if (a->shape[d] == 1)
          idx = 0;
        flat_idx_a += idx * a->strides[d];
      }

      uint64_t flat_idx_b = b->offset;
      for (uint32_t d = 0; d < b->ndims; d++) {
        uint32_t out_d = out->ndims - b->ndims + d;
        uint32_t idx = indices[out_d];
        if (b->shape[d] == 1)
          idx = 0;
        flat_idx_b += idx * b->strides[d];
      }

      out->storage->data[flat_idx_out] =
          a->storage->data[flat_idx_a] * b->storage->data[flat_idx_b];

      for (int32_t d = out->ndims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < out->shape[d])
          break;
        indices[d] = 0;
      }
    }
  }

  return true;
}

} // namespace gradientcore
