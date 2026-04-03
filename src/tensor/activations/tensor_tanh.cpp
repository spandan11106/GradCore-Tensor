#include "../../../include/tensor/tensor.hpp"
#include <cmath>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_tanh(Tensor *out, const Tensor *in) {
  if (!out || !in || !shape_match(out, in))
    return false;

  if (tensor_is_contiguous(out) && tensor_is_contiguous(in)) {
#if defined(_OPENMP)
#include <omp.h>
#endif
    for (uint64_t i = 0; i < out->size; i++) {
      float val = in->storage->data[in->offset + i];
      out->storage->data[out->offset + i] = std::tanh(val);
    }
  }

  else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};
    for (uint64_t i = 0; i < out->size; i++) {
      uint64_t in_idx = tensor_get_flat_index(in, indices);
      uint64_t out_idx = tensor_get_flat_index(out, indices);

      float val = in->storage->data[in_idx];
      out->storage->data[out_idx] = tanh(val);

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
