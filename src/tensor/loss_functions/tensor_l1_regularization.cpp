#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_l1_regularization(Tensor *out, const Tensor *weights,
                              Reduction reduction) {
  if (!out || !weights)
    return false;

  if (reduction == REDUCTION_NONE) {
    if (!shape_match(out, weights))
      return false;
  } else {
    if (out->size != 1)
      return false;
  }

  if (tensor_is_contiguous(weights) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(out))) {

    if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < weights->size; i++) {
        float w = weights->storage->data[weights->offset + i];
        out->storage->data[out->offset + i] = std::fabs(w);
      }
    } else {
      float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
      for (uint64_t i = 0; i < weights->size; i++) {
        float w = weights->storage->data[weights->offset + i];
        total_loss += std::fabs(w);
      }

      if (reduction == REDUCTION_MEAN) {
        total_loss /= static_cast<float>(weights->size);
      }
      out->storage->data[out->offset] = total_loss;
    }
  } else {
    if (reduction == REDUCTION_NONE) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < weights->size; i++) {
        uint64_t w_idx = tensor_get_flat_index(weights, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float w = weights->storage->data[w_idx];
        out->storage->data[o_idx] = std::fabs(w);

        for (int32_t d = weights->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < weights->shape[d])
            break;
          indices[d] = 0;
        }
      }
    } else {
      float total_loss = 0.0f;
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < weights->size; i++) {
        uint64_t w_idx = tensor_get_flat_index(weights, indices);

        float w = weights->storage->data[w_idx];
        total_loss += std::fabs(w);

        for (int32_t d = weights->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < weights->shape[d])
            break;
          indices[d] = 0;
        }
      }

      if (reduction == REDUCTION_MEAN) {
        total_loss /= static_cast<float>(weights->size);
      }
      out->storage->data[out->offset] = total_loss;
    }
  }

  return true;
}

} // namespace gradientcore
