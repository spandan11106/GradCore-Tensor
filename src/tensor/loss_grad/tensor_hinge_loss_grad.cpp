#include "../../../include/tensor/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_hinge_loss_grad(Tensor *out, const Tensor *pred,
                            const Tensor *target, const Tensor *grad,
                            Reduction reduction) {
  if (!out || !pred || !target || !grad)
    return false;
  if (!shape_match(pred, target) || !shape_match(out, pred))
    return false;

  if (reduction == REDUCTION_NONE) {
    if (!shape_match(pred, grad))
      return false;
  } else {
    if (grad->size != 1)
      return false;
  }

  float scale = 1.0f;
  if (reduction == REDUCTION_MEAN) {
    scale = 1.0f / static_cast<float>(pred->size);
  }

  if (tensor_is_contiguous(out) && tensor_is_contiguous(pred) &&
      tensor_is_contiguous(target) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(grad))) {

    if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < out->size; i++) {
        float p = pred->storage->data[pred->offset + i];
        float t = target->storage->data[target->offset + i];
        float g = grad->storage->data[grad->offset + i];

        float margin = 1.0f - (t * p);
        float local_grad = (margin > 0.0f) ? -t : 0.0f;

        out->storage->data[out->offset + i] = g * scale * local_grad;
      }
    } else {
      float g = grad->storage->data[grad->offset];
      float final_scale = g * scale;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < out->size; i++) {
        float p = pred->storage->data[pred->offset + i];
        float t = target->storage->data[target->offset + i];

        float margin = 1.0f - (t * p);
        float local_grad = (margin > 0.0f) ? -t : 0.0f;

        out->storage->data[out->offset + i] = final_scale * local_grad;
      }
    }
  } else {
    if (reduction == REDUCTION_NONE) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < out->size; i++) {
        uint64_t p_idx = tensor_get_flat_index(pred, indices);
        uint64_t t_idx = tensor_get_flat_index(target, indices);
        uint64_t g_idx = tensor_get_flat_index(grad, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float p = pred->storage->data[p_idx];
        float t = target->storage->data[t_idx];
        float g = grad->storage->data[g_idx];

        float margin = 1.0f - (t * p);
        float local_grad = (margin > 0.0f) ? -t : 0.0f;

        out->storage->data[o_idx] = g * scale * local_grad;

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
        uint64_t p_idx = tensor_get_flat_index(pred, indices);
        uint64_t t_idx = tensor_get_flat_index(target, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float p = pred->storage->data[p_idx];
        float t = target->storage->data[t_idx];

        float margin = 1.0f - (t * p);
        float local_grad = (margin > 0.0f) ? -t : 0.0f;

        out->storage->data[o_idx] = final_scale * local_grad;

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
