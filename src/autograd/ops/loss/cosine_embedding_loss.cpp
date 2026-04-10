#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *cosine_embedding_loss(Arena *arena, Variable *x1, Variable *x2,
                                Variable *target, float margin,
                                Reduction reduction) {
  uint32_t scalar_shape[1] = {1};
  Tensor *out_data = tensor_create_zeros(arena, 1, scalar_shape);
  tensor_cosine_embedding_loss(out_data, x1->data, x2->data, target->data,
                               margin, reduction);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = x1->requires_grad || x2->requires_grad;
  out->is_leaf = false;
  out->reduction = reduction;
  out->metadata_float = margin;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, 1, scalar_shape);
    out->num_parents = 3;
    out->parents = arena->push_array<Edge>(3);
    out->parents[0] = {x1};
    out->parents[1] = {x2};
    out->parents[2] = {target};

    out->num_saved = 3;
    out->saved_tensors = arena->push_array<Tensor *>(3);
    out->saved_tensors[0] = x1->data;
    out->saved_tensors[1] = x2->data;
    out->saved_tensors[2] = target->data;

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent_x1 = self->parents[0].node;
      Variable *parent_x2 = self->parents[1].node;

      Tensor *local_grad_x1 = nullptr;
      if (parent_x1->requires_grad) {
        local_grad_x1 = tensor_create_zeros(temp_arena, parent_x1->grad->ndims,
                                            parent_x1->grad->shape);
      }

      Tensor *local_grad_x2 = nullptr;
      if (parent_x2->requires_grad) {
        local_grad_x2 = tensor_create_zeros(temp_arena, parent_x2->grad->ndims,
                                            parent_x2->grad->shape);
      }

      tensor_cosine_embedding_loss_grad(local_grad_x1, self->saved_tensors[0],
                                        self->saved_tensors[1],
                                        self->saved_tensors[2], self->grad,
                                        self->metadata_float,
                                        static_cast<Reduction>(self->reduction));

      if (parent_x1->requires_grad && local_grad_x1) {
        tensor_add(parent_x1->grad, parent_x1->grad, local_grad_x1);
      }
      if (parent_x2->requires_grad && local_grad_x2) {
        tensor_add(parent_x2->grad, parent_x2->grad, local_grad_x2);
      }
    };
  }
  return out;
}

} // namespace autograd
} // namespace gradientcore
