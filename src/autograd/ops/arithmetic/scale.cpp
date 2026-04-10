#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *scale(Arena *arena, Variable *in, float scale_factor) {
  Tensor *out_data = tensor_create_zeros(arena, in->data->ndims, in->data->shape);
  tensor_copy(out_data, in->data);
  tensor_scale(out_data, scale_factor);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = in->requires_grad;
  out->is_leaf = false;
  out->metadata_float = scale_factor;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, out_data->ndims, out_data->shape);
    out->num_parents = 1;
    out->parents = arena->push_array<Edge>(1);
    out->parents[0] = {in};

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent = self->parents[0].node;
      if (!parent->requires_grad)
        return;

      Tensor *local_grad = tensor_create_zeros(temp_arena, parent->grad->ndims,
                                               parent->grad->shape);
      tensor_copy(local_grad, self->grad);
      tensor_scale(local_grad, self->metadata_float);

      tensor_add(parent->grad, parent->grad, local_grad);
    };
  }

  return out;
}

} // namespace autograd
} // namespace gradientcore
