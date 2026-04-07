#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_triplet_loss_grad(Tensor *out, const Tensor *anchor,
                              const Tensor *positive, const Tensor *negative,
                              const Tensor *grad, float margin,
                              Reduction reduction) {
  if (!out || !anchor || !positive || !negative || !grad)
    return false;

  if (!shape_match(anchor, positive) || !shape_match(anchor, negative) ||
      !shape_match(out, anchor))
    return false;
  if (anchor->ndims == 0)
    return false;

  uint32_t ndims = anchor->ndims;
  uint32_t F = anchor->shape[ndims - 1];
  uint64_t num_batches = anchor->size / F;

  if (reduction == REDUCTION_NONE) {
    if (grad->size != num_batches)
      return false;
  } else {
    if (grad->size != 1)
      return false;
  }

  float scale = 1.0f;
  if (reduction == REDUCTION_MEAN) {
    scale = 1.0f / static_cast<float>(num_batches);
  }

  constexpr float EPSILON = 1e-6f;

  if (tensor_is_contiguous(out) && tensor_is_contiguous(anchor) &&
      tensor_is_contiguous(positive) && tensor_is_contiguous(negative) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(grad))) {

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t b = 0; b < num_batches; b++) {
      float g_val = (reduction == REDUCTION_NONE)
                        ? grad->storage->data[grad->offset + b]
                        : grad->storage->data[grad->offset];
      float final_scale = g_val * scale;

      uint64_t feature_offset = b * F;

      float sq_d_ap = 0.0f;
      float sq_d_an = 0.0f;

      for (uint32_t f = 0; f < F; f++) {
        float a = anchor->storage->data[anchor->offset + feature_offset + f];
        float p =
            positive->storage->data[positive->offset + feature_offset + f];
        float n =
            negative->storage->data[negative->offset + feature_offset + f];

        float diff_ap = a - p;
        float diff_an = a - n;

        sq_d_ap += diff_ap * diff_ap;
        sq_d_an += diff_an * diff_an;
      }

      float d_ap = std::sqrt(std::max(sq_d_ap, EPSILON));
      float d_an = std::sqrt(std::max(sq_d_an, EPSILON));

      bool active = (d_ap - d_an + margin > 0.0f);

      for (uint32_t f = 0; f < F; f++) {
        if (!active) {
          out->storage->data[out->offset + feature_offset + f] = 0.0f;
        } else {
          float a = anchor->storage->data[anchor->offset + feature_offset + f];
          float p =
              positive->storage->data[positive->offset + feature_offset + f];
          float n =
              negative->storage->data[negative->offset + feature_offset + f];

          float grad_a = ((a - p) / d_ap) - ((a - n) / d_an);

          out->storage->data[out->offset + feature_offset + f] =
              final_scale * grad_a;
        }
      }
    }
  } else {
    for (uint64_t b = 0; b < num_batches; b++) {
      uint32_t batch_indices[MAX_TENSOR_DIMS] = {0};
      uint64_t temp_idx = b;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        batch_indices[d] = temp_idx % anchor->shape[d];
        temp_idx /= anchor->shape[d];
      }

      float g_val;
      if (reduction == REDUCTION_NONE) {
        uint64_t g_idx = tensor_get_flat_index(grad, batch_indices);
        g_val = grad->storage->data[g_idx];
      } else {
        g_val = grad->storage->data[grad->offset];
      }
      float final_scale = g_val * scale;

      float sq_d_ap = 0.0f;
      float sq_d_an = 0.0f;

      for (uint32_t f = 0; f < F; f++) {
        uint32_t feature_indices[MAX_TENSOR_DIMS];
        for (uint32_t d = 0; d < ndims - 1; d++)
          feature_indices[d] = batch_indices[d];
        feature_indices[ndims - 1] = f;

        uint64_t a_idx = tensor_get_flat_index(anchor, feature_indices);
        uint64_t p_idx = tensor_get_flat_index(positive, feature_indices);
        uint64_t n_idx = tensor_get_flat_index(negative, feature_indices);

        float a = anchor->storage->data[a_idx];
        float p = positive->storage->data[p_idx];
        float n = negative->storage->data[n_idx];

        float diff_ap = a - p;
        float diff_an = a - n;

        sq_d_ap += diff_ap * diff_ap;
        sq_d_an += diff_an * diff_an;
      }

      float d_ap = std::sqrt(std::max(sq_d_ap, EPSILON));
      float d_an = std::sqrt(std::max(sq_d_an, EPSILON));

      bool active = (d_ap - d_an + margin > 0.0f);

      for (uint32_t f = 0; f < F; f++) {
        uint32_t feature_indices[MAX_TENSOR_DIMS];
        for (uint32_t d = 0; d < ndims - 1; d++)
          feature_indices[d] = batch_indices[d];
        feature_indices[ndims - 1] = f;

        uint64_t o_idx = tensor_get_flat_index(out, feature_indices);

        if (!active) {
          out->storage->data[o_idx] = 0.0f;
        } else {
          uint64_t a_idx = tensor_get_flat_index(anchor, feature_indices);
          uint64_t p_idx = tensor_get_flat_index(positive, feature_indices);
          uint64_t n_idx = tensor_get_flat_index(negative, feature_indices);

          float a = anchor->storage->data[a_idx];
          float p = positive->storage->data[p_idx];
          float n = negative->storage->data[n_idx];

          float grad_a = ((a - p) / d_ap) - ((a - n) / d_an);
          out->storage->data[o_idx] = final_scale * grad_a;
        }
      }
    }
  }

  return true;
}

} // namespace gradientcore
