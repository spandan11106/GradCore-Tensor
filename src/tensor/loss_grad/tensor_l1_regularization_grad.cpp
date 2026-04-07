#include "../../../include/tensor/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_l1_regularization_grad(Tensor *out, const Tensor *weights,
                                   const Tensor *grad, Reduction reduction) {
  if (!out || !weights || !grad)
    return false;

  if (!shape_match(out, weights))
    return false;

  if (reduction == REDUCTION_NONE) {
    if (!shape_match(weights, grad))
      return false;
  } else {
    if (grad->size != 1)
      return false;
  }

  float scale = 1.0f;
  if (reduction == REDUCTION_MEAN) {
    scale = 1.0f / static_cast<float>(weights->size);
  }

  if (tensor_is_contiguous(out) && tensor_is_contiguous(weights) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(grad))) {

    if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < out->size; i++) {
        float w = weights->storage->data[weights->offset + i];
        float g = grad->storage->data[grad->offset + i];

        float sign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        out->storage->data[out->offset + i] = g * scale * sign;
      }
    } else {
      float g = grad->storage->data[grad->offset];
      float final_scale = g * scale;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < out->size; i++) {
        float w = weights->storage->data[weights->offset + i];

        float sign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        out->storage->data[out->offset + i] = final_scale * sign;
      }
    }
  } else {
    if (reduction == REDUCTION_NONE) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < out->size; i++) {
        uint64_t w_idx = tensor_get_flat_index(weights, indices);
        uint64_t g_idx = tensor_get_flat_index(grad, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float w = weights->storage->data[w_idx];
        float g = grad->storage->data[g_idx];

        float sign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        out->storage->data[o_idx] = g * scale * sign;

        for (int32_t d = out->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < out->shape[d])
            break;
          indices[d] = 0;
        }
      }
    } else {
      float g = grad->storage->data[grad->offset];
      float final_scale = g * scale;

      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < out->size; i++) {
        uint64_t w_idx = tensor_get_flat_index(weights, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float w = weights->storage->data[w_idx];

        float sign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
        out->storage->data[o_idx] = final_scale * sign;

        for (int32_t d = out->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < out->shape[d])
            break;
          indices[d] = 0;
        }
      }
    }
  }

  return true;
}

} // namespace gradientcore
