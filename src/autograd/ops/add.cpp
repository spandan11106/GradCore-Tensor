#include "../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *add(Arena *arena, Variable *a, Variable *b) {
  Tensor *out_data = tensor_create_zeros(arena, a->data->ndims, a->data->shape);
  tensor_add(out_data, a->data, b->data);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = a->requires_grad || b->requires_grad;
  out->is_leaf = false;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, out_data->ndims, out_data->shape);
    out->num_parents = 2;
    out->parents = arena->push<Edge>(2);
    out->parents[0] = {a};
    out->parents[1] = {b};

    out->backward_fn = [](Variable *self, Arena *arena) {
      Variable *parent_a = self->parents[0].node;
      Variable *parent_b = self->parents[1].node;

      if (parent_a->requires_grad) {
        tensor_add(parent_a->grad, parent_a->grad, self->grad);
      }
      if (parent_b->requires_grad) {
        tensor_add(parent_b->grad, parent_b->grad, self->grad);
      }
    };
  }

  return out;
}

} // namespace autograd
} // namespace gradientcore
