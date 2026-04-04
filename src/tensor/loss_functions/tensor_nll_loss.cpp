#include "../../../include/tensor/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_nll_loss(Tensor *out, const Tensor *log_probs, const Tensor *target,
                     Reduction reduction) {
  if (!out || !log_probs || !target)
    return false;
  if (!shape_match(log_probs, target))
    return false;
  if (log_probs->ndims == 0)
    return false;

  uint32_t ndims = log_probs->ndims;
  uint32_t C = log_probs->shape[ndims - 1];
  if (C == 0) return false;
  uint64_t num_batches = log_probs->size / C;

  if (reduction == REDUCTION_NONE) {
    if (ndims > 1) {
      if (out->ndims != ndims - 1)
        return false;
      for (uint32_t i = 0; i < out->ndims; i++) {
        if (out->shape[i] != log_probs->shape[i])
          return false;
      }
    } else {
      if (out->size != 1)
        return false;
    }
  } else {
    if (out->size != 1)
      return false;
  }

  uint64_t lp_stride_c = log_probs->strides[ndims - 1];
  uint64_t t_stride_c = target->strides[ndims - 1];

  uint64_t outer_lp_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_t_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_o_strides[MAX_TENSOR_DIMS] = {0};

  for (uint32_t i = 0; i < ndims - 1; i++) {
    outer_lp_strides[i] = log_probs->strides[i];
    outer_t_strides[i] = target->strides[i];
    if (reduction == REDUCTION_NONE) {
      outer_o_strides[i] = out->strides[i];
    }
  }

  if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t lp_offset = log_probs->offset;
      uint64_t t_offset = target->offset;
      uint64_t o_offset = out->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        uint32_t coord = temp_idx % log_probs->shape[d];
        temp_idx /= log_probs->shape[d];

        lp_offset += coord * outer_lp_strides[d];
        t_offset += coord * outer_t_strides[d];
        o_offset += coord * outer_o_strides[d];
      }

      float batch_loss = 0.0f;
      for (uint32_t c = 0; c < C; c++) {
        float target_val = target->storage->data[t_offset + c * t_stride_c];
        float lp_val = log_probs->storage->data[lp_offset + c * lp_stride_c];
        batch_loss -= target_val * lp_val; // Note the minus sign
      }

      out->storage->data[o_offset] = batch_loss;
    }
  } else {
    float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t lp_offset = log_probs->offset;
      uint64_t t_offset = target->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        uint32_t coord = temp_idx % log_probs->shape[d];
        temp_idx /= log_probs->shape[d];

        lp_offset += coord * outer_lp_strides[d];
        t_offset += coord * outer_t_strides[d];
      }

      float batch_loss = 0.0f;
      for (uint32_t c = 0; c < C; c++) {
        float target_val = target->storage->data[t_offset + c * t_stride_c];
        float lp_val = log_probs->storage->data[lp_offset + c * lp_stride_c];
        batch_loss -= target_val * lp_val;
      }

      total_loss += batch_loss;
    }

    if (reduction == REDUCTION_MEAN) {
      total_loss /= static_cast<float>(num_batches);
    }
    out->storage->data[out->offset] = total_loss;
  }

  return true;
}

} // namespace gradientcore
