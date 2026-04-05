#include "../../../include/tensor/tensor.hpp"
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_softmax_grad(Tensor *out, const Tensor *softmax_out,
                         const Tensor *grad, int32_t dim) {
  if (!out || !softmax_out || !grad)
    return false;
  if (!shape_match(out, softmax_out) || !shape_match(out, grad))
    return false;
  if (softmax_out->ndims == 0)
    return false;

  uint32_t ndims = softmax_out->ndims;

  if (dim < 0) {
    dim += ndims;
  }
  if (dim < 0 || dim >= ndims)
    return false;

  uint32_t N = softmax_out->shape[dim];
  uint64_t num_batches = softmax_out->size / N;

  uint64_t y_stride = softmax_out->strides[dim];
  uint64_t grad_stride = grad->strides[dim];
  uint64_t out_stride = out->strides[dim];

  uint32_t outer_shape[MAX_TENSOR_DIMS];
  uint64_t outer_y_strides[MAX_TENSOR_DIMS];
  uint64_t outer_grad_strides[MAX_TENSOR_DIMS];
  uint64_t outer_out_strides[MAX_TENSOR_DIMS];

  int32_t outer_idx = 0;
  for (uint32_t i = 0; i < ndims; i++) {
    if (i == static_cast<uint32_t>(dim))
      continue;
    outer_shape[outer_idx] = softmax_out->shape[i];
    outer_y_strides[outer_idx] = softmax_out->strides[i];
    outer_grad_strides[outer_idx] = grad->strides[i];
    outer_out_strides[outer_idx] = out->strides[i];
    outer_idx++;
  }

  int32_t outer_dims = ndims - 1;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (uint64_t batch = 0; batch < num_batches; batch++) {

    uint64_t y_row_offset = softmax_out->offset;
    uint64_t grad_row_offset = grad->offset;
    uint64_t out_row_offset = out->offset;

    uint64_t temp_idx = batch;
    for (int32_t d = outer_dims - 1; d >= 0; d--) {
      uint32_t coord = temp_idx % outer_shape[d];
      temp_idx /= outer_shape[d];

      y_row_offset += coord * outer_y_strides[d];
      grad_row_offset += coord * outer_grad_strides[d];
      out_row_offset += coord * outer_out_strides[d];
    }

    float sum_y_dy = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      float y_val = softmax_out->storage->data[y_row_offset + n * y_stride];
      float dy_val = grad->storage->data[grad_row_offset + n * grad_stride];
      sum_y_dy += y_val * dy_val;
    }

    for (uint32_t n = 0; n < N; n++) {
      float y_val = softmax_out->storage->data[y_row_offset + n * y_stride];
      float dy_val = grad->storage->data[grad_row_offset + n * grad_stride];

      out->storage->data[out_row_offset + n * out_stride] =
          y_val * (dy_val - sum_y_dy);
    }
  }

  return true;
}
} // namespace gradientcore
