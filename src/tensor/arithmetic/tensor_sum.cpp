#include "../../../include/tensor/tensor.hpp"
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

float tensor_sum(Tensor *t) {
  if (t == nullptr || t->storage == nullptr || t->size == 0)
    return 0.0f;

  float total_sum = 0.0f;

  if (tensor_is_contiguous(t)) {
    const float *data = t->storage->data + t->offset;

#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_sum)
#endif

    for (uint64_t i = 0; i < t->size; i++) {
      total_sum += data[i];
    }
  }

  else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};

    for (uint64_t i = 0; i < t->size; i++) {
      uint64_t flat_idx = tensor_get_flat_index(t, indices);
      total_sum += t->storage->data[flat_idx];

      for (int32_t d = t->ndims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < t->shape[d]) {
          break;
        }
        indices[d] = 0;
      }
    }
  }

  return total_sum;
}

bool tensor_sum_to_shape(Tensor *out, const Tensor *in) {
  if (out == nullptr || in == nullptr)
    return false;

  if (shape_match(out, in)) {
    return tensor_copy(out, in);
  }

  tensor_clear(out);

  uint32_t in_coords[MAX_TENSOR_DIMS] = {0};

  for (uint64_t i = 0; i < in->size; i++) {
    uint64_t in_flat = tensor_get_flat_index(in, in_coords);

    uint32_t out_coords[MAX_TENSOR_DIMS] = {0};
    for (uint32_t out_d = 0; out_d < out->ndims; out_d++) {
      int32_t in_d = (int32_t)in->ndims - (int32_t)out->ndims + out_d;

      if (in_d >= 0) {
        out_coords[out_d] = (out->shape[out_d] == 1) ? 0 : in_coords[in_d];
      }
    }

    uint64_t out_flat = tensor_get_flat_index(out, out_coords);

    out->storage->data[out_flat] += in->storage->data[in_flat];

    for (int32_t d = in->ndims - 1; d >= 0; d--) {
      in_coords[d]++;
      if (in_coords[d] < in->shape[d])
        break;
      in_coords[d] = 0;
    }
  }

  return true;
}

} // namespace gradientcore
