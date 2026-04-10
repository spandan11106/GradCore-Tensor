#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *l1_regularization(Arena *arena, Variable *weights,
                            Reduction reduction) {
  uint32_t scalar_shape[1] = {1};
  Tensor *out_data = tensor_create_zeros(arena, 1, scalar_shape);
  tensor_l1_regularization(out_data, weights->data, reduction);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = weights->requires_grad;
  out->is_leaf = false;
  out->reduction = reduction;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, 1, scalar_shape);
    out->num_parents = 1;
    out->parents = arena->push_array<Edge>(1);
    out->parents[0] = {weights};

    out->num_saved = 1;
    out->saved_tensors = arena->push_array<Tensor *>(1);
    out->saved_tensors[0] = weights->data;

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent = self->parents[0].node;
      if (!parent->requires_grad)
        return;

      Tensor *local_grad = tensor_create_zeros(temp_arena, parent->grad->ndims,
                                               parent->grad->shape);

      tensor_l1_regularization_grad(local_grad, self->saved_tensors[0],
                                    self->grad,
                                    static_cast<Reduction>(self->reduction));

      tensor_add(parent->grad, parent->grad, local_grad);
    };
  }
  return out;
}

} // namespace autograd
} // namespace gradientcore
