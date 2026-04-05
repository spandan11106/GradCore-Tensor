#include "../../../include/tensor/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_relu6_grad(Tensor *out, const Tensor *in, const Tensor *grad) {
  if (!out || !in || !grad)
    return false;
  if (!shape_match(out, in) || !shape_match(out, grad))
    return false;

  if (tensor_is_contiguous(out) && tensor_is_contiguous(in) &&
      tensor_is_contiguous(grad)) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t i = 0; i < out->size; i++) {
      float in_val = in->storage->data[in->offset + i];
      float grad_val = grad->storage->data[grad->offset + i];

      out->storage->data[out->offset + i] =
          (in_val > 0.0f && in_val <= 6.0f) ? grad_val : 0.0f;
    }
  }

  else {
    uint32_t indices[MAX_TENSOR_DIMS] = {0};
    for (uint64_t i = 0; i < out->size; i++) {
      uint64_t in_idx = tensor_get_flat_index(in, indices);
      uint64_t grad_idx = tensor_get_flat_index(grad, indices);
      uint64_t out_idx = tensor_get_flat_index(out, indices);

      float in_val = in->storage->data[in_idx];
      float grad_val = grad->storage->data[grad_idx];

      out->storage->data[out_idx] =
          (in_val > 0.0f && in_val <= 6.0f) ? grad_val : 0.0f;

      for (int32_t d = out->ndims - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < out->shape[d])
          break;
        indices[d] = 0;
      }
    }
  }

  return true;
}

} // namespace gradientcore
