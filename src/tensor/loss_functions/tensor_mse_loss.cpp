#include "../../../include/tensor/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_mse_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                     Reduction reduction) {
  if (!out || !pred || !target)
    return false;
  if (!shape_match(pred, target))
    return false;

  if (reduction == REDUCTION_NONE) {
    if (!shape_match(out, pred))
      return false;
  } else {
    // MEAN and SUM reductions result in a scalar (size 1)
    if (out->size != 1)
      return false;
  }

  // Fast Path: Contiguous Tensors
  if (tensor_is_contiguous(pred) && tensor_is_contiguous(target) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(out))) {

    if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < pred->size; i++) {
        float diff = pred->storage->data[pred->offset + i] -
                     target->storage->data[target->offset + i];
        out->storage->data[out->offset + i] = diff * diff;
      }
    } else {
      float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
      for (uint64_t i = 0; i < pred->size; i++) {
        float diff = pred->storage->data[pred->offset + i] -
                     target->storage->data[target->offset + i];
        total_loss += diff * diff;
      }

      if (reduction == REDUCTION_MEAN) {
        total_loss /= static_cast<float>(pred->size);
      }
      out->storage->data[out->offset] = total_loss;
    }
  }

  else {
    if (reduction == REDUCTION_NONE) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < pred->size; i++) {
        uint64_t p_idx = tensor_get_flat_index(pred, indices);
        uint64_t t_idx = tensor_get_flat_index(target, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float diff = pred->storage->data[p_idx] - target->storage->data[t_idx];
        out->storage->data[o_idx] = diff * diff;

        for (int32_t d = pred->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < pred->shape[d])
            break;
          indices[d] = 0;
        }
      }
    } else {
      float total_loss = 0.0f;
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < pred->size; i++) {
        uint64_t p_idx = tensor_get_flat_index(pred, indices);
        uint64_t t_idx = tensor_get_flat_index(target, indices);

        float diff = pred->storage->data[p_idx] - target->storage->data[t_idx];
        total_loss += diff * diff;

        for (int32_t d = pred->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < pred->shape[d])
            break;
          indices[d] = 0;
        }
      }

      if (reduction == REDUCTION_MEAN) {
        total_loss /= static_cast<float>(pred->size);
      }
      out->storage->data[out->offset] = total_loss;
    }
  }

  return true;
}

} // namespace gradientcore
