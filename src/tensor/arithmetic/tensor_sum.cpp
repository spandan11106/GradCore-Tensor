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

} // namespace gradientcore
