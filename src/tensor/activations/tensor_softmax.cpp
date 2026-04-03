#include "../../../include/tensor/tensor.hpp"
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {

bool tensor_softmax(Tensor *out, const Tensor *in, int32_t dim) {
  if (!out || !in || !shape_match(out, in) || in->ndims == 0)
    return false;

  uint32_t ndims = in->ndims;

  if (dim < 0) {
    dim += ndims;
  }
  if (dim < 0 || dim >= ndims)
    return false;

  uint32_t N = in->shape[dim];
  uint64_t num_batches = in->size / N;

  uint64_t in_stride = in->strides[dim];
  uint64_t out_stride = out->strides[dim];

  uint32_t outer_shape[MAX_TENSOR_DIMS];
  uint64_t outer_in_strides[MAX_TENSOR_DIMS];
  uint64_t outer_out_strides[MAX_TENSOR_DIMS];

  int32_t outer_idx = 0;
  for (uint32_t i = 0; i < ndims; i++) {
    if (i == static_cast<uint32_t>(dim))
      continue;
    outer_shape[outer_idx] = in->shape[i];
    outer_in_strides[outer_idx] = in->strides[i];
    outer_out_strides[outer_idx] = out->strides[i];
    outer_idx++;
  }

  int32_t outer_dims = ndims - 1;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (uint64_t batch = 0; batch < num_batches; batch++) {

    uint64_t in_row_offset = in->offset;
    uint64_t out_row_offset = out->offset;
    uint64_t temp_idx = batch;

    for (int32_t d = outer_dims - 1; d >= 0; d--) {
      uint32_t coord = temp_idx % outer_shape[d];
      temp_idx /= outer_shape[d];

      in_row_offset += coord * outer_in_strides[d];
      out_row_offset += coord * outer_out_strides[d];
    }

    float max_val = in->storage->data[in_row_offset];
    for (uint32_t n = 1; n < N; n++) {
      float val = in->storage->data[in_row_offset + n * in_stride];
      if (val > max_val)
        max_val = val;
    }

    float sum_exp = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      float val = in->storage->data[in_row_offset + n * in_stride];
      float exp_val = std::exp(val - max_val); // Subtract max_val for stability
      out->storage->data[out_row_offset + n * out_stride] = exp_val;
      sum_exp += exp_val;
    }

    for (uint32_t n = 0; n < N; n++) {
      out->storage->data[out_row_offset + n * out_stride] /= sum_exp;
    }
  }

  return true;
}

} // namespace gradientcore
