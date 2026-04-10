#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *hinge_loss(Arena *arena, Variable *pred, Variable *target,
                     Reduction reduction) {
  uint32_t scalar_shape[1] = {1};
  Tensor *out_data = tensor_create_zeros(arena, 1, scalar_shape);
  tensor_hinge_loss(out_data, pred->data, target->data, reduction);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = pred->requires_grad;
  out->is_leaf = false;
  out->reduction = reduction;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, 1, scalar_shape);
    out->num_parents = 1;
    out->parents = arena->push_array<Edge>(1);
    out->parents[0] = {pred};

    out->num_saved = 2;
    out->saved_tensors = arena->push_array<Tensor *>(2);
    out->saved_tensors[0] = pred->data;
    out->saved_tensors[1] = target->data;

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent = self->parents[0].node;
      if (!parent->requires_grad)
        return;

      Tensor *local_grad = tensor_create_zeros(temp_arena, parent->grad->ndims,
                                               parent->grad->shape);

      tensor_hinge_loss_grad(local_grad, self->saved_tensors[0],
                             self->saved_tensors[1], self->grad,
                             static_cast<Reduction>(self->reduction));

      tensor_add(parent->grad, parent->grad, local_grad);
    };
  }
  return out;
}

} // namespace autograd
} // namespace gradientcore
