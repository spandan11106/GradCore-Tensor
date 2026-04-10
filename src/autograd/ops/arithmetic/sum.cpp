#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *sum(Arena *arena, Variable *in) {
  uint32_t scalar_shape[1] = {1};
  Tensor *out_data = tensor_create_zeros(arena, 1, scalar_shape);
  float sum_val = tensor_sum(in->data);
  out_data->storage->data[out_data->offset] = sum_val;

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = in->requires_grad;
  out->is_leaf = false;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, 1, scalar_shape);
    out->num_parents = 1;
    out->parents = arena->push_array<Edge>(1);
    out->parents[0] = {in};

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent = self->parents[0].node;
      if (!parent->requires_grad)
        return;

      // Gradient for sum is just the upstream gradient broadcast to all elements
      float grad_val = self->grad->storage->data[self->grad->offset];
      Tensor *local_grad =
          tensor_create_zeros(temp_arena, parent->grad->ndims, parent->grad->shape);
      tensor_fill(local_grad, grad_val);

      tensor_add(parent->grad, parent->grad, local_grad);
    };
  }

  return out;
}

} // namespace autograd
} // namespace gradientcore
