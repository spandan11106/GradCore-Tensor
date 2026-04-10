#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *softmax(Arena *arena, Variable *in, int32_t dim) {
  Tensor *out_data = tensor_create_zeros(arena, in->data->ndims, in->data->shape);
  tensor_softmax(out_data, in->data, dim);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = in->requires_grad;
  out->is_leaf = false;
  out->metadata_float = static_cast<float>(dim);

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, out_data->ndims, out_data->shape);
    out->num_parents = 1;
    out->parents = arena->push_array<Edge>(1);
    out->parents[0] = {in};

    out->num_saved = 1;
    out->saved_tensors = arena->push_array<Tensor *>(1);
    out->saved_tensors[0] = out_data;

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent = self->parents[0].node;
      if (!parent->requires_grad)
        return;

      Tensor *local_grad = tensor_create_zeros(temp_arena, parent->grad->ndims,
                                               parent->grad->shape);

      int32_t dim = static_cast<int32_t>(self->metadata_float);
      tensor_softmax_grad(local_grad, self->saved_tensors[0], self->grad, dim);
      tensor_add(parent->grad, parent->grad, local_grad);
    };
  }

  return out;
}

} // namespace autograd
} // namespace gradientcore
