#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_bce_with_logits_loss_grad(Tensor *out, const Tensor *logits,
                                      const Tensor *target, const Tensor *grad,
                                      Reduction reduction) {
  if (!out || !logits || !target || !grad)
    return false;
  if (!shape_match(logits, target) || !shape_match(out, logits))
    return false;

  if (reduction == REDUCTION_NONE) {
    if (!shape_match(logits, grad))
      return false;
  } else {
    if (grad->size != 1)
      return false;
  }

  float scale = 1.0f;
  if (reduction == REDUCTION_MEAN) {
    scale = 1.0f / static_cast<float>(logits->size);
  }

  if (tensor_is_contiguous(out) && tensor_is_contiguous(logits) &&
      tensor_is_contiguous(target) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(grad))) {

    if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < out->size; i++) {
        float x = logits->storage->data[logits->offset + i];
        float t = target->storage->data[target->offset + i];
        float g = grad->storage->data[grad->offset + i];

        float s = 1.0f / (1.0f + std::exp(-x));

        out->storage->data[out->offset + i] = g * scale * (s - t);
      }
    } else {
      float g = grad->storage->data[grad->offset];
      float final_scale = g * scale;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < out->size; i++) {
        float x = logits->storage->data[logits->offset + i];
        float t = target->storage->data[target->offset + i];

        float s = 1.0f / (1.0f + std::exp(-x));
        out->storage->data[out->offset + i] = final_scale * (s - t);
      }
    }
  }

  else {
    if (reduction == REDUCTION_NONE) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < out->size; i++) {
        uint64_t l_idx = tensor_get_flat_index(logits, indices);
        uint64_t t_idx = tensor_get_flat_index(target, indices);
        uint64_t g_idx = tensor_get_flat_index(grad, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float x = logits->storage->data[l_idx];
        float t = target->storage->data[t_idx];
        float g = grad->storage->data[g_idx];

        float s = 1.0f / (1.0f + std::exp(-x));
        out->storage->data[o_idx] = g * scale * (s - t);

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
        uint64_t l_idx = tensor_get_flat_index(logits, indices);
        uint64_t t_idx = tensor_get_flat_index(target, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float x = logits->storage->data[l_idx];
        float t = target->storage->data[t_idx];

        float s = 1.0f / (1.0f + std::exp(-x));
        out->storage->data[o_idx] = final_scale * (s - t);

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
