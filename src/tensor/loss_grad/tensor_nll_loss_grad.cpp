#include "../../../include/tensor/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_nll_loss_grad(Tensor *out, const Tensor *log_probs,
                          const Tensor *target, const Tensor *grad,
                          Reduction reduction) {
  if (!out || !log_probs || !target || !grad)
    return false;
  if (!shape_match(log_probs, target) || !shape_match(out, log_probs))
    return false;
  if (log_probs->ndims == 0)
    return false;

  uint32_t ndims = log_probs->ndims;
  uint32_t C = log_probs->shape[ndims - 1];
  uint64_t num_batches = log_probs->size / C;

  if (reduction == REDUCTION_NONE) {
    if (ndims > 1) {
      if (grad->ndims != ndims - 1)
        return false;
      for (uint32_t i = 0; i < grad->ndims; i++) {
        if (grad->shape[i] != log_probs->shape[i])
          return false;
      }
    } else {
      if (grad->size != 1)
        return false;
    }
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

  uint64_t t_stride_c = target->strides[ndims - 1];
  uint64_t o_stride_c = out->strides[ndims - 1];

  uint64_t outer_t_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_o_strides[MAX_TENSOR_DIMS] = {0};
  uint64_t outer_g_strides[MAX_TENSOR_DIMS] = {0};

  for (uint32_t i = 0; i < ndims - 1; i++) {
    outer_t_strides[i] = target->strides[i];
    outer_o_strides[i] = out->strides[i];
    if (reduction == REDUCTION_NONE) {
      outer_g_strides[i] = grad->strides[i];
    }
  }

  if (reduction == REDUCTION_NONE) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t t_offset = target->offset;
      uint64_t o_offset = out->offset;
      uint64_t g_offset = grad->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        uint32_t coord = temp_idx % log_probs->shape[d];
        temp_idx /= log_probs->shape[d];

        t_offset += coord * outer_t_strides[d];
        o_offset += coord * outer_o_strides[d];
        g_offset += coord * outer_g_strides[d];
      }

      float g_val = grad->storage->data[g_offset];
      float final_scale = g_val * scale;

      for (uint32_t c = 0; c < C; c++) {
        float target_val = target->storage->data[t_offset + c * t_stride_c];
        out->storage->data[o_offset + c * o_stride_c] =
            final_scale * (-target_val);
      }
    }
  } else {
    float g_val = grad->storage->data[grad->offset];
    float final_scale = g_val * scale;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t batch = 0; batch < num_batches; batch++) {
      uint64_t t_offset = target->offset;
      uint64_t o_offset = out->offset;

      uint64_t temp_idx = batch;
      for (int32_t d = (int32_t)ndims - 2; d >= 0; d--) {
        uint32_t coord = temp_idx % log_probs->shape[d];
        temp_idx /= log_probs->shape[d];

        t_offset += coord * outer_t_strides[d];
        o_offset += coord * outer_o_strides[d];
      }

      for (uint32_t c = 0; c < C; c++) {
        float target_val = target->storage->data[t_offset + c * t_stride_c];
        out->storage->data[o_offset + c * o_stride_c] =
            final_scale * (-target_val);
      }
    }
  }

  return true;
}

} // namespace gradientcore
