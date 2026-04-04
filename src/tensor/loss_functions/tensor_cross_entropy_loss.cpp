#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_cross_entropy_loss(Tensor *out, const Tensor *logits,
                               const Tensor *target, Reduction reduction) {
  if (!out || !logits || !target)
    return false;
  if (!shape_match(logits, target))
    return false;
  if (logits->ndims == 0)
    return false;

  uint32_t ndims = logits->ndims;
  uint32_t C = logits->shape[ndims - 1];
  if (C == 0) return false;
  uint64_t num_batches = logits->size / C;

  if (reduction == REDUCTION_NONE) {
    if (ndims > 1) {
      if (out->ndims != ndims - 1)
        return false;
      for (uint32_t i = 0; i < out->ndims; i++) {
        if (out->shape[i] != logits->shape[i])
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

  uint64_t l_stride_c = logits->strides[ndims - 1];
  uint64_t t_stride_c = target->strides[ndims - 1];

  uint64_t outer_l_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_t_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_o_strides[MAX_TENSOR_DIMS] = {0};

  for (uint32_t i = 0; i < ndims - 1; i++) {
    outer_l_strides[i] = logits->strides[i];
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
      uint64_t l_offset = logits->offset;
      uint64_t t_offset = target->offset;
      uint64_t o_offset = out->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        uint32_t coord = temp_idx % logits->shape[d];
        temp_idx /= logits->shape[d];

        l_offset += coord * outer_l_strides[d];
        t_offset += coord * outer_t_strides[d];
        o_offset += coord * outer_o_strides[d];
      }

      float max_logit = logits->storage->data[l_offset];
      for (uint32_t c = 1; c < C; c++) {
        float val = logits->storage->data[l_offset + c * l_stride_c];
        if (val > max_logit)
          max_logit = val;
      }

      float sum_exp = 0.0f;
      for (uint32_t c = 0; c < C; c++) {
        float val = logits->storage->data[l_offset + c * l_stride_c];
        sum_exp += std::exp(val - max_logit);
      }
      float log_sum_exp = max_logit + std::log(sum_exp);

      float batch_loss = 0.0f;
      for (uint32_t c = 0; c < C; c++) {
        float target_val = target->storage->data[t_offset + c * t_stride_c];
        float logit_val = logits->storage->data[l_offset + c * l_stride_c];
        // -target * (logit - log_sum_exp) == target * (log_sum_exp - logit)
        batch_loss += target_val * (log_sum_exp - logit_val);
      }

      out->storage->data[o_offset] = batch_loss;
    }
  } else {
    float total_loss = 0.0f;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_loss)
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t l_offset = logits->offset;
      uint64_t t_offset = target->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        uint32_t coord = temp_idx % logits->shape[d];
        temp_idx /= logits->shape[d];

        l_offset += coord * outer_l_strides[d];
        t_offset += coord * outer_t_strides[d];
      }

      float max_logit = logits->storage->data[l_offset];
      for (uint32_t c = 1; c < C; c++) {
        float val = logits->storage->data[l_offset + c * l_stride_c];
        if (val > max_logit)
          max_logit = val;
      }

      float sum_exp = 0.0f;
      for (uint32_t c = 0; c < C; c++) {
        float val = logits->storage->data[l_offset + c * l_stride_c];
        sum_exp += std::exp(val - max_logit);
      }
      float log_sum_exp = max_logit + std::log(sum_exp);

      float batch_loss = 0.0f;
      for (uint32_t c = 0; c < C; c++) {
        float target_val = target->storage->data[t_offset + c * t_stride_c];
        float logit_val = logits->storage->data[l_offset + c * l_stride_c];
        batch_loss += target_val * (log_sum_exp - logit_val);
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
