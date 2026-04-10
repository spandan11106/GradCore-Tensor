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

  uint32_t reduction;        // For loss operations
  float metadata_float;      // For parameters (delta, margin, alpha, scale, dim)

  void (*backward_fn)(Variable *self, Arena *arena);
};

Variable *create_leaf(Arena *arena, Tensor *data, bool requires_grad);
void backward(Arena *arena, Variable *loss_node);

// Arithmetic Operations
Variable *add(Arena *arena, Variable *a, Variable *b);
Variable *sub(Arena *arena, Variable *a, Variable *b);
Variable *mul(Arena *arena, Variable *a, Variable *b);
Variable *matmul(Arena *arena, Variable *a, Variable *b);
Variable *scale(Arena *arena, Variable *in, float scale_factor);
Variable *sum(Arena *arena, Variable *in);

// Activation Functions
Variable *relu(Arena *arena, Variable *in);
Variable *tanh(Arena *arena, Variable *in);
Variable *sigmoid(Arena *arena, Variable *in);
Variable *softmax(Arena *arena, Variable *in, int32_t dim = -1);
Variable *leaky_relu(Arena *arena, Variable *in, float alpha);
Variable *elu(Arena *arena, Variable *in, float alpha);
Variable *swish(Arena *arena, Variable *in);
Variable *gelu(Arena *arena, Variable *in);
Variable *relu6(Arena *arena, Variable *in);
Variable *hard_sigmoid(Arena *arena, Variable *in);
Variable *hard_swish(Arena *arena, Variable *in);
Variable *softplus(Arena *arena, Variable *in);

// Loss Functions 
Variable *mse_loss(Arena *arena, Variable *pred, Variable *target,
                   Reduction reduction);
Variable *l1_loss(Arena *arena, Variable *pred, Variable *target,
                  Reduction reduction);
Variable *bce_loss(Arena *arena, Variable *pred, Variable *target,
                   Reduction reduction);
Variable *bce_with_logits_loss(Arena *arena, Variable *logits,
                               Variable *target, Reduction reduction);
Variable *cross_entropy_loss(Arena *arena, Variable *logits, Variable *target,
                             Reduction reduction);
Variable *nll_loss(Arena *arena, Variable *log_probs, Variable *target,
                   Reduction reduction);
Variable *kl_div_loss(Arena *arena, Variable *pred, Variable *target,
                      Reduction reduction);
Variable *hinge_loss(Arena *arena, Variable *pred, Variable *target,
                     Reduction reduction);
Variable *huber_loss(Arena *arena, Variable *pred, Variable *target,
                     float delta, Reduction reduction);
Variable *cosine_embedding_loss(Arena *arena, Variable *x1, Variable *x2,
                                Variable *target, float margin,
                                Reduction reduction);
Variable *triplet_loss(Arena *arena, Variable *anchor, Variable *positive,
                       Variable *negative, float margin, Reduction reduction);
Variable *l2_loss(Arena *arena, Variable *weights, Reduction reduction);
Variable *l1_regularization(Arena *arena, Variable *weights,
                            Reduction reduction);

} // namespace autograd
} // namespace gradientcore
