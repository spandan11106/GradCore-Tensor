#include "../../../include/tensor/tensor.hpp"
#include <algorithm>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_cosine_embedding_loss_grad(Tensor *out, const Tensor *x1,
                                       const Tensor *x2, const Tensor *target,
                                       const Tensor *grad, float margin,
                                       Reduction reduction) {
  if (!out || !x1 || !x2 || !target || !grad)
    return false;
  if (!shape_match(x1, x2) || !shape_match(out, x1))
    return false;
  if (x1->ndims == 0)
    return false;

  uint32_t ndims = x1->ndims;
  uint32_t F = x1->shape[ndims - 1];
  uint64_t num_batches = x1->size / F;

  if (target->size != num_batches)
    return false;

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

  constexpr float EPSILON = 1e-12f;

  if (tensor_is_contiguous(out) && tensor_is_contiguous(x1) &&
      tensor_is_contiguous(x2) && tensor_is_contiguous(target) &&
      (reduction != REDUCTION_NONE || tensor_is_contiguous(grad))) {

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t b = 0; b < num_batches; b++) {
      float t = target->storage->data[target->offset + b];

      float g_val = (reduction == REDUCTION_NONE)
                        ? grad->storage->data[grad->offset + b]
                        : grad->storage->data[grad->offset];
      float final_scale = g_val * scale;

      uint64_t feature_offset = b * F;

      float dot = 0.0f;
      float sq_norm_x1 = 0.0f;
      float sq_norm_x2 = 0.0f;

      for (uint32_t f = 0; f < F; f++) {
        float v1 = x1->storage->data[x1->offset + feature_offset + f];
        float v2 = x2->storage->data[x2->offset + feature_offset + f];
        dot += v1 * v2;
        sq_norm_x1 += v1 * v1;
        sq_norm_x2 += v2 * v2;
      }

      float norm_x1 = std::sqrt(std::max(sq_norm_x1, EPSILON));
      float norm_x2 = std::sqrt(std::max(sq_norm_x2, EPSILON));
      float safe_sq_norm_x1 = std::max(sq_norm_x1, EPSILON);
      float denom = norm_x1 * norm_x2;

      float cos_sim = dot / denom;

      bool active = false;
      float sign = 0.0f;

      if (t == 1.0f) {
        active = true;
        sign = -1.0f;
      } else if (t == -1.0f) {
        if (cos_sim > margin) {
          active = true;
          sign = 1.0f;
        }
      }

      for (uint32_t f = 0; f < F; f++) {
        if (!active) {
          out->storage->data[out->offset + feature_offset + f] = 0.0f;
        } else {
          float v1 = x1->storage->data[x1->offset + feature_offset + f];
          float v2 = x2->storage->data[x2->offset + feature_offset + f];

          // Derivative: (x2 - x1 * (dot / sq_norm_x1)) / (norm_x1 * norm_x2)
          float d_cos_sim_dx1 = (v2 - v1 * (dot / safe_sq_norm_x1)) / denom;

          out->storage->data[out->offset + feature_offset + f] =
              final_scale * sign * d_cos_sim_dx1;
        }
      }
    }
  } else {
    for (uint64_t b = 0; b < num_batches; b++) {
      uint32_t batch_indices[MAX_TENSOR_DIMS] = {0};
      uint64_t temp_idx = b;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        batch_indices[d] = temp_idx % x1->shape[d];
        temp_idx /= x1->shape[d];
      }

      uint64_t t_idx = tensor_get_flat_index(target, batch_indices);
      float t = target->storage->data[t_idx];

      float g_val;
      if (reduction == REDUCTION_NONE) {
        uint64_t g_idx = tensor_get_flat_index(grad, batch_indices);
        g_val = grad->storage->data[g_idx];
      } else {
        g_val = grad->storage->data[grad->offset];
      }
      float final_scale = g_val * scale;

      float dot = 0.0f;
      float sq_norm_x1 = 0.0f;
      float sq_norm_x2 = 0.0f;

      for (uint32_t f = 0; f < F; f++) {
        uint32_t feature_indices[MAX_TENSOR_DIMS];
        for (uint32_t d = 0; d < ndims - 1; d++)
          feature_indices[d] = batch_indices[d];
        feature_indices[ndims - 1] = f;

        uint64_t v1_idx = tensor_get_flat_index(x1, feature_indices);
        uint64_t v2_idx = tensor_get_flat_index(x2, feature_indices);

        float v1 = x1->storage->data[v1_idx];
        float v2 = x2->storage->data[v2_idx];
        dot += v1 * v2;
        sq_norm_x1 += v1 * v1;
        sq_norm_x2 += v2 * v2;
      }

      float norm_x1 = std::sqrt(std::max(sq_norm_x1, EPSILON));
      float norm_x2 = std::sqrt(std::max(sq_norm_x2, EPSILON));
      float safe_sq_norm_x1 = std::max(sq_norm_x1, EPSILON);
      float denom = norm_x1 * norm_x2;

      float cos_sim = dot / denom;

      bool active = false;
      float sign = 0.0f;

      if (t == 1.0f) {
        active = true;
        sign = -1.0f;
      } else if (t == -1.0f) {
        if (cos_sim > margin) {
          active = true;
          sign = 1.0f;
        }
      }

      for (uint32_t f = 0; f < F; f++) {
        uint32_t feature_indices[MAX_TENSOR_DIMS];
        for (uint32_t d = 0; d < ndims - 1; d++)
          feature_indices[d] = batch_indices[d];
        feature_indices[ndims - 1] = f;

        uint64_t o_idx = tensor_get_flat_index(out, feature_indices);

        if (!active) {
          out->storage->data[o_idx] = 0.0f;
        } else {
          uint64_t v1_idx = tensor_get_flat_index(x1, feature_indices);
          uint64_t v2_idx = tensor_get_flat_index(x2, feature_indices);
          float v1 = x1->storage->data[v1_idx];
          float v2 = x2->storage->data[v2_idx];

          float d_cos_sim_dx1 = (v2 - v1 * (dot / safe_sq_norm_x1)) / denom;
          out->storage->data[o_idx] = final_scale * sign * d_cos_sim_dx1;
        }
      }
    }
  }

  return true;
}

} // namespace gradientcore
