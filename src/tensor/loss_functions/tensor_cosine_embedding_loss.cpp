#include "../../../include/tensor/tensor.hpp"
#include <algorithm>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_cosine_embedding_loss(Tensor *out, const Tensor *x1,
                                  const Tensor *x2, const Tensor *target,
                                  float margin, Reduction reduction) {
  if (!out || !x1 || !x2 || !target)
    return false;
  if (!shape_match(x1, x2))
    return false;
  if (x1->ndims == 0)
    return false;

  uint32_t ndims = x1->ndims;
  uint32_t batch_ndims = ndims - 1;
  uint32_t D = x1->shape[batch_ndims];
  if (D == 0) return false;
  uint64_t num_batches = x1->size / D;

  if (target->size != num_batches)
    return false;

  if (reduction == REDUCTION_NONE) {
    if (out->size != num_batches)
      return false;
  } else {
    if (out->size != 1)
      return false;
  }

  constexpr float EPSILON = 1e-8f;

  uint64_t x1_stride_d = x1->strides[batch_ndims];
  uint64_t x2_stride_d = x2->strides[batch_ndims];

  uint64_t outer_x1_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_x2_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_t_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_o_strides[MAX_TENSOR_DIMS] = {0};

  for (uint32_t i = 0; i < batch_ndims; i++) {
    outer_x1_strides[i] = x1->strides[i];
    outer_x2_strides[i] = x2->strides[i];

    if (target->ndims == batch_ndims) {
      outer_t_strides[i] = target->strides[i];
      if (reduction == REDUCTION_NONE)
        outer_o_strides[i] = out->strides[i];
    }
  }

  if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t x1_offset = x1->offset;
      uint64_t x2_offset = x2->offset;
      uint64_t t_offset = target->offset;
      uint64_t o_offset = out->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)batch_ndims - 1; d >= 0; d--) {
        uint32_t coord = temp_idx % x1->shape[d];
        temp_idx /= x1->shape[d];

        x1_offset += coord * outer_x1_strides[d];
        x2_offset += coord * outer_x2_strides[d];

        if (target->ndims == batch_ndims) {
          t_offset += coord * outer_t_strides[d];
          o_offset += coord * outer_o_strides[d];
        }
      }

      if (target->ndims != batch_ndims) {
        t_offset = target->offset + batch * (target->ndims > 0 ? target->strides[0] : 1);
        o_offset = out->offset + batch * (out->ndims > 0 ? out->strides[0] : 1);
      }

      float dot = 0.0f, norm1_sq = 0.0f, norm2_sq = 0.0f;
      for (uint32_t i = 0; i < D; i++) {
        float v1 = x1->storage->data[x1_offset + i * x1_stride_d];
        float v2 = x2->storage->data[x2_offset + i * x2_stride_d];
        dot += v1 * v2;
        norm1_sq += v1 * v1;
        norm2_sq += v2 * v2;
      }

      float denom = std::sqrt(norm1_sq) * std::sqrt(norm2_sq);
      if (denom < EPSILON)
        denom = EPSILON;
      float cos_sim = dot / denom;

      float y = target->storage->data[t_offset];
      if (y > 0.0f) {
        out->storage->data[o_offset] = 1.0f - cos_sim;
      } else {
        out->storage->data[o_offset] = std::max(0.0f, cos_sim - margin);
      }
    }
  } else {
    float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t x1_offset = x1->offset;
      uint64_t x2_offset = x2->offset;
      uint64_t t_offset = target->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)batch_ndims - 1; d >= 0; d--) {
        uint32_t coord = temp_idx % x1->shape[d];
        temp_idx /= x1->shape[d];

        x1_offset += coord * outer_x1_strides[d];
        x2_offset += coord * outer_x2_strides[d];

        if (target->ndims == batch_ndims) {
          t_offset += coord * outer_t_strides[d];
        }
      }

      if (target->ndims != batch_ndims) {
        t_offset = target->offset + batch * (target->ndims > 0 ? target->strides[0] : 1);
      }

      float dot = 0.0f, norm1_sq = 0.0f, norm2_sq = 0.0f;
      for (uint32_t i = 0; i < D; i++) {
        float v1 = x1->storage->data[x1_offset + i * x1_stride_d];
        float v2 = x2->storage->data[x2_offset + i * x2_stride_d];
        dot += v1 * v2;
        norm1_sq += v1 * v1;
        norm2_sq += v2 * v2;
      }

      float denom = std::sqrt(norm1_sq) * std::sqrt(norm2_sq);
      if (denom < EPSILON)
        denom = EPSILON;
      float cos_sim = dot / denom;

      float y = target->storage->data[t_offset];
      if (y > 0.0f) {
        total_loss += 1.0f - cos_sim;
      } else {
        total_loss += std::max(0.0f, cos_sim - margin);
      }
    }

    if (reduction == REDUCTION_MEAN) {
      total_loss /= static_cast<float>(num_batches);
    }
    out->storage->data[out->offset] = total_loss;
  }

  return true;
}

} // namespace gradientcore
