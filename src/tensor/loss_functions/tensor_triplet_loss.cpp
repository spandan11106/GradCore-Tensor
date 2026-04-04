#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_triplet_loss(Tensor *out, const Tensor *anchor,
                         const Tensor *positive, const Tensor *negative,
                         float margin, Reduction reduction) {
  if (!out || !anchor || !positive || !negative)
    return false;

  // All three inputs must have the exact same shape
  if (!shape_match(anchor, positive) || !shape_match(anchor, negative))
    return false;
  if (anchor->ndims == 0)
    return false;

  uint32_t ndims = anchor->ndims;
  uint32_t batch_ndims = ndims - 1;
  uint32_t D = anchor->shape[batch_ndims];
  if (D == 0) return false;
  uint64_t num_batches = anchor->size / D;

  if (reduction == REDUCTION_NONE) {
    if (out->size != num_batches)
      return false;
  } else {
    if (out->size != 1)
      return false;
  }

  constexpr float EPSILON = 1e-6f;

  uint64_t a_stride_d = anchor->strides[batch_ndims];
  uint64_t p_stride_d = positive->strides[batch_ndims];
  uint64_t n_stride_d = negative->strides[batch_ndims];

  uint64_t outer_a_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_p_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_n_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_o_strides[MAX_TENSOR_DIMS] = {0};

  for (uint32_t i = 0; i < batch_ndims; i++) {
    outer_a_strides[i] = anchor->strides[i];
    outer_p_strides[i] = positive->strides[i];
    outer_n_strides[i] = negative->strides[i];
    if (reduction == REDUCTION_NONE) {
      outer_o_strides[i] = out->strides[i];
    }
  }

  if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t a_offset = anchor->offset;
      uint64_t p_offset = positive->offset;
      uint64_t n_offset = negative->offset;
      uint64_t o_offset = out->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)batch_ndims - 1; d >= 0; d--) {
        uint32_t coord = temp_idx % anchor->shape[d];
        temp_idx /= anchor->shape[d];

        a_offset += coord * outer_a_strides[d];
        p_offset += coord * outer_p_strides[d];
        n_offset += coord * outer_n_strides[d];
        o_offset += coord * outer_o_strides[d];
      }

      float dist_ap_sq = 0.0f;
      float dist_an_sq = 0.0f;

      for (uint32_t i = 0; i < D; i++) {
        float a_val = anchor->storage->data[a_offset + i * a_stride_d];
        float p_val = positive->storage->data[p_offset + i * p_stride_d];
        float n_val = negative->storage->data[n_offset + i * n_stride_d];

        float diff_ap = a_val - p_val;
        float diff_an = a_val - n_val;

        dist_ap_sq += diff_ap * diff_ap;
        dist_an_sq += diff_an * diff_an;
      }

      float dist_ap = std::sqrt(dist_ap_sq + EPSILON);
      float dist_an = std::sqrt(dist_an_sq + EPSILON);

      float loss_val = dist_ap - dist_an + margin;
      out->storage->data[o_offset] = (loss_val > 0.0f) ? loss_val : 0.0f;
    }
  } else {
    float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t a_offset = anchor->offset;
      uint64_t p_offset = positive->offset;
      uint64_t n_offset = negative->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)batch_ndims - 1; d >= 0; d--) {
        uint32_t coord = temp_idx % anchor->shape[d];
        temp_idx /= anchor->shape[d];

        a_offset += coord * outer_a_strides[d];
        p_offset += coord * outer_p_strides[d];
        n_offset += coord * outer_n_strides[d];
      }

      float dist_ap_sq = 0.0f;
      float dist_an_sq = 0.0f;

      for (uint32_t i = 0; i < D; i++) {
        float a_val = anchor->storage->data[a_offset + i * a_stride_d];
        float p_val = positive->storage->data[p_offset + i * p_stride_d];
        float n_val = negative->storage->data[n_offset + i * n_stride_d];

        float diff_ap = a_val - p_val;
        float diff_an = a_val - n_val;

        dist_ap_sq += diff_ap * diff_ap;
        dist_an_sq += diff_an * diff_an;
      }

      float dist_ap = std::sqrt(dist_ap_sq + EPSILON);
      float dist_an = std::sqrt(dist_an_sq + EPSILON);

      float loss_val = dist_ap - dist_an + margin;
      total_loss += (loss_val > 0.0f) ? loss_val : 0.0f;
    }

    if (reduction == REDUCTION_MEAN) {
      total_loss /= static_cast<float>(num_batches);
    }
    out->storage->data[out->offset] = total_loss;
  }

  return true;
}

} // namespace gradientcore
