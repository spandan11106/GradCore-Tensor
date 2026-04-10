#include "../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *matmul(Arena *arena, Variable *a, Variable *b) {
  uint32_t out_shape[2] = {a->data->shape[0], b->data->shape[1]};
  Tensor *out_data = tensor_create_zeros(arena, 2, out_shape);

  mat_mul(out_data, a->data, b->data, true, false, false);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = a->requires_grad || b->requires_grad;
  out->is_leaf = false;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, out_data->ndims, out_data->shape);
    out->num_parents = 2;
    out->parents = arena->push_array<Edge>(2);
    out->parents[0] = {a};
    out->parents[1] = {b};

    out->num_saved = 2;
    out->saved_tensors = arena->push_array<Tensor *>(2);
    out->saved_tensors[0] = a->data;
    out->saved_tensors[1] = b->data;

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent_a = self->parents[0].node;
      Variable *parent_b = self->parents[1].node;
      Tensor *a_data = self->saved_tensors[0];
      Tensor *b_data = self->saved_tensors[1];

      if (parent_a->requires_grad) {
        Tensor *local_grad_a = tensor_create_zeros(
            temp_arena, parent_a->grad->ndims, parent_a->grad->shape);
        mat_mul(local_grad_a, self->grad, b_data, true, false, true);
        tensor_add(parent_a->grad, parent_a->grad, local_grad_a);
      }

      if (parent_b->requires_grad) {
        Tensor *local_grad_b = tensor_create_zeros(
            temp_arena, parent_b->grad->ndims, parent_b->grad->shape);
        mat_mul(local_grad_b, a_data, self->grad, true, true, false);
        tensor_add(parent_b->grad, parent_b->grad, local_grad_b);
      }
    };
  }
  return out;
}

} // namespace autograd
} // namespace gradientcore
