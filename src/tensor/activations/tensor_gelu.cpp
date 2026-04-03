#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_gelu(Tensor *out, const Tensor *in) {
  if (!out || !in || !shape_match(out, in))
    return false;

  // Precomputed constants for the GELU approximation
  constexpr float SQRT_2_OVER_PI = 0.7978845608f;
  constexpr float COEF = 0.044715f;

  if (tensor_is_contiguous(out) && tensor_is_contiguous(in)) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t i = 0; i < out->size; i++) {
      float x = in->storage->data[in->offset + i];
      float cube = x * x * x;
      out->storage->data[out->offset + i] =
          0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + COEF * cube)));
    }
  } else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};
    for (uint64_t i = 0; i < out->size; i++) {
      uint64_t in_idx = tensor_get_flat_index(in, indices);
      uint64_t out_idx = tensor_get_flat_index(out, indices);

      float x = in->storage->data[in_idx];
      float cube = x * x * x;
      out->storage->data[out_idx] =
          0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + COEF * cube)));

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
