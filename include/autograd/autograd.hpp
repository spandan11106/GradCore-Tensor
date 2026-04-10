#pragma once

#include "../tensor/tensor.hpp"

namespace gradientcore {
namespace autograd {
struct Variable;

struct Edge {
  Variable *node;
};

struct Variable {
  Tensor *data;
  Tensor *grad;
  bool requires_grad;
  bool is_leaf;

  Edge *parents;
  uint32_t num_parents;

  Tensor **saved_tensors;
  uint32_t num_saved;

  uint32_t reduction;  // For loss operations

  void (*backward_fn)(Variable *self, Arena *arena);
};

Variable *create_leaf(Arena *arena, Tensor *data, bool requires_grad);
void backward(Arena *arena, Variable *loss_node);

Variable *add(Arena *arena, Variable *a, Variable *b);
Variable *matmul(Arena *arena, Variable *a, Variable *b);
Variable *relu(Arena *arena, Variable *in);
Variable *mse_loss(Arena *arena, Variable *pred, Variable *target,
                   Reduction reduction);

} // namespace autograd
} // namespace gradientcore
