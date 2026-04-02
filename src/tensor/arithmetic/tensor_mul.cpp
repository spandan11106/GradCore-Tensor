#include "../../../include/tensor/tensor.hpp"
#include <cstdint>

namespace gradientcore {

bool tensor_mul(Tensor *out, const Tensor *a, Tensor *b) {
  if (out == nullptr || a == nullptr || b == nullptr)
    return false;
  if (a->ndims != b->ndims || out->ndims != a->ndims)
    return false;

  for (uint32_t i = 0; i < a->ndims; i++) {
    if (a->shape[i] != b->shape[i] || out->shape[i] != a->shape[i])
      return false;
  }

  if (tensor_is_contiguous(a) && tensor_is_contiguous(b) &&
      tensor_is_contiguous(out)) {
    for (uint64_t i = 0; i < out->size; i++) {
      out->storage->data[out->offset + i] =
          a->storage->data[a->offset + i] * b->storage->data[b->offset + i];
    }
  } else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    for (uint64_t i = 0; i < out->size; i++) {
      uint64_t flat_idx_out = tensor_get_flat_index(out, indices);
      uint64_t flat_idx_a = tensor_get_flat_index(a, indices);
      uint64_t flat_idx_b = tensor_get_flat_index(b, indices);

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
