#include "../../../../include/autograd/autograd.hpp"

namespace gradientcore {
namespace autograd {

Variable *triplet_loss(Arena *arena, Variable *anchor, Variable *positive,
                       Variable *negative, float margin, Reduction reduction) {
  uint32_t scalar_shape[1] = {1};
  Tensor *out_data = tensor_create_zeros(arena, 1, scalar_shape);
  tensor_triplet_loss(out_data, anchor->data, positive->data, negative->data,
                      margin, reduction);

  Variable *out = arena->push<Variable>();
  out->data = out_data;
  out->requires_grad = anchor->requires_grad || positive->requires_grad ||
                       negative->requires_grad;
  out->is_leaf = false;
  out->reduction = reduction;
  out->metadata_float = margin;

  if (out->requires_grad) {
    out->grad = tensor_create_zeros(arena, 1, scalar_shape);
    out->num_parents = 3;
    out->parents = arena->push_array<Edge>(3);
    out->parents[0] = {anchor};
    out->parents[1] = {positive};
    out->parents[2] = {negative};

    out->num_saved = 3;
    out->saved_tensors = arena->push_array<Tensor *>(3);
    out->saved_tensors[0] = anchor->data;
    out->saved_tensors[1] = positive->data;
    out->saved_tensors[2] = negative->data;

    out->backward_fn = [](Variable *self, Arena *temp_arena) {
      Variable *parent_anchor = self->parents[0].node;
      Variable *parent_positive = self->parents[1].node;
      Variable *parent_negative = self->parents[2].node;

      Tensor *local_grad_anchor = nullptr;
      if (parent_anchor->requires_grad) {
        local_grad_anchor = tensor_create_zeros(
            temp_arena, parent_anchor->grad->ndims, parent_anchor->grad->shape);
      }

      Tensor *local_grad_positive = nullptr;
      if (parent_positive->requires_grad) {
        local_grad_positive = tensor_create_zeros(
            temp_arena, parent_positive->grad->ndims,
            parent_positive->grad->shape);
      }

      Tensor *local_grad_negative = nullptr;
      if (parent_negative->requires_grad) {
        local_grad_negative = tensor_create_zeros(
            temp_arena, parent_negative->grad->ndims,
            parent_negative->grad->shape);
      }

      tensor_triplet_loss_grad(
          local_grad_anchor, self->saved_tensors[0], self->saved_tensors[1],
          self->saved_tensors[2], self->grad, self->metadata_float,
          static_cast<Reduction>(self->reduction));

      if (parent_anchor->requires_grad && local_grad_anchor) {
        tensor_add(parent_anchor->grad, parent_anchor->grad, local_grad_anchor);
      }
      if (parent_positive->requires_grad && local_grad_positive) {
        tensor_add(parent_positive->grad, parent_positive->grad,
                   local_grad_positive);
      }
      if (parent_negative->requires_grad && local_grad_negative) {
        tensor_add(parent_negative->grad, parent_negative->grad,
                   local_grad_negative);
      }
    };
  }
  return out;
}

} // namespace autograd
} // namespace gradientcore
