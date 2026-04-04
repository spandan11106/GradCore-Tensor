#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_bce_with_logits_loss(Tensor *out, const Tensor *logits,
                                 const Tensor *target, Reduction reduction) {
  if (!out || !logits || !target)
    return false;
  if (!shape_match(logits, target))
    return false;

  if (reduction == REDUCTION_NONE) {
    if (!shape_match(out, logits))
      return false;
  } else {
    if (out->size != 1)
      return false;
  }

  if (tensor_is_contiguous(logits) && tensor_is_contiguous(target) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(out))) {

    if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t i = 0; i < logits->size; i++) {
        float x = logits->storage->data[logits->offset + i];
        float y = target->storage->data[target->offset + i];

        float max_val = (x > 0.0f) ? x : 0.0f;
        out->storage->data[out->offset + i] =
            max_val - x * y + std::log1p(std::exp(-std::fabs(x)));
      }
    } else {
      float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
      for (uint64_t i = 0; i < logits->size; i++) {
        float x = logits->storage->data[logits->offset + i];
        float y = target->storage->data[target->offset + i];

        float max_val = (x > 0.0f) ? x : 0.0f;
        total_loss += max_val - x * y + std::log1p(std::exp(-std::fabs(x)));
      }

      if (reduction == REDUCTION_MEAN) {
        total_loss /= static_cast<float>(logits->size);
      }
      out->storage->data[out->offset] = total_loss;
    }
  } else {
    if (reduction == REDUCTION_NONE) {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < logits->size; i++) {
        uint64_t x_idx = tensor_get_flat_index(logits, indices);
        uint64_t y_idx = tensor_get_flat_index(target, indices);
        uint64_t o_idx = tensor_get_flat_index(out, indices);

        float x = logits->storage->data[x_idx];
        float y = target->storage->data[y_idx];

        float max_val = (x > 0.0f) ? x : 0.0f;
        out->storage->data[o_idx] =
            max_val - x * y + std::log1p(std::exp(-std::fabs(x)));

        for (int32_t d = logits->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < logits->shape[d])
            break;
          indices[d] = 0;
        }
      }
    } else {
      float total_loss = 0.0f;
      uint32_t indices[MAX_TENSOR_DIMS] = {0};
      for (uint64_t i = 0; i < logits->size; i++) {
        uint64_t x_idx = tensor_get_flat_index(logits, indices);
        uint64_t y_idx = tensor_get_flat_index(target, indices);

        float x = logits->storage->data[x_idx];
        float y = target->storage->data[y_idx];

        float max_val = (x > 0.0f) ? x : 0.0f;
        total_loss += max_val - x * y + std::log1p(std::exp(-std::fabs(x)));

        for (int32_t d = logits->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < logits->shape[d])
            break;
          indices[d] = 0;
        }
      }

      if (reduction == REDUCTION_MEAN) {
        total_loss /= static_cast<float>(logits->size);
      }
      out->storage->data[out->offset] = total_loss;
    }
  }

  return true;
}

} // namespace gradientcore
