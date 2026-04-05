#pragma once

#include "memory_cpu/arena.hpp"
#include <cstdint>

namespace gradientcore {

constexpr uint32_t MAX_TENSOR_DIMS = 10;

struct TensorStorage {
  float *data;
  uint64_t size;
};

struct Tensor {
  uint32_t ndims;
  uint32_t shape[MAX_TENSOR_DIMS];
  uint32_t strides[MAX_TENSOR_DIMS];
  uint64_t size;
  uint64_t offset; // Starting index in the data

  TensorStorage *storage;
};

// Creation
Tensor *tensor_create(Arena *arena, uint32_t ndims, const uint32_t *shape);
Tensor *tensor_create_zeros(Arena *arena, uint32_t ndims,
                            const uint32_t *shape);

// Views
Tensor *tensor_view(Arena *arena, const Tensor *src);
Tensor *tensor_reshape(Arena *arena, const Tensor *src, uint32_t ndims,
                       const uint32_t *shape);
Tensor *tensor_transpose(Arena *arena, const Tensor *src, uint32_t dim0,
                         uint32_t dim1);

// Utils
uint64_t tensor_get_flat_index(const Tensor *t, const uint32_t *indices);
void tensor_clear(Tensor *t);
bool tensor_copy(Tensor *dst, const Tensor *src);
void tensor_fill(Tensor *t, float val);
bool tensor_is_contiguous(const Tensor *t);
bool shape_match(const Tensor *a, const Tensor *b);

// Arithmetics
bool tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
bool tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
bool tensor_mul(Tensor *out, const Tensor *a, const Tensor *b);
bool mat_mul(Tensor *out, const Tensor *a, const Tensor *b, bool zero_out,
             bool transpose_a, bool transpose_b);
float tensor_sum(Tensor *t);
void tensor_scale(Tensor *t, float scale);

// Activation functions
bool tensor_relu(Tensor *out, const Tensor *in);
bool tensor_softmax(Tensor *out, const Tensor *in, int32_t dim = -1);
bool tensor_tanh(Tensor *out, const Tensor *in);
bool tensor_sigmoid(Tensor *out, const Tensor *in);
bool tensor_leaky_relu(Tensor *out, const Tensor *in, float alpha);
bool tensor_elu(Tensor *out, const Tensor *in, float alpha);
bool tensor_swish(Tensor *out, const Tensor *in);
bool tensor_gelu(Tensor *out, const Tensor *in);
bool tensor_relu6(Tensor *out, const Tensor *in);
bool tensor_hard_sigmoid(Tensor *out, const Tensor *in);
bool tensor_hard_swish(Tensor *out, const Tensor *in);
bool tensor_softplus(Tensor *out, const Tensor *in);

// Loss functions
enum Reduction { REDUCTION_NONE, REDUCTION_MEAN, REDUCTION_SUM };
bool tensor_mse_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                     Reduction reduction);
bool tensor_l1_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                    Reduction reduction);
bool tensor_huber_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                       float delta, Reduction reduction);
bool tensor_bce_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                     Reduction reduction);
bool tensor_bce_with_logits_loss(Tensor *out, const Tensor *logits,
                                 const Tensor *target, Reduction reduction);
bool tensor_cross_entropy_loss(Tensor *out, const Tensor *logits,
                               const Tensor *target, Reduction reduction);
bool tensor_nll_loss(Tensor *out, const Tensor *log_probs, const Tensor *target,
                     Reduction reduction);
bool tensor_kl_div_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                        Reduction reduction);
bool tensor_hinge_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                       Reduction reduction);
bool tensor_cosine_embedding_loss(Tensor *out, const Tensor *x1,
                                  const Tensor *x2, const Tensor *target,
                                  float margin, Reduction reduction);
bool tensor_triplet_loss(Tensor *out, const Tensor *anchor,
                         const Tensor *positive, const Tensor *negative,
                         float margin, Reduction reduction);
bool tensor_l2_loss(Tensor *out, const Tensor *weights, Reduction reduction);
bool tensor_l1_regularization(Tensor *out, const Tensor *weights,
                              Reduction reduction);

// Activation functions grad
bool tensor_relu_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_softmax_grad(Tensor *out, const Tensor *softmax_out,
                         const Tensor *grad, int32_t dim = -1);
bool tensor_tanh_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_sigmoid_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_leaky_relu_grad(Tensor *out, const Tensor *in, const Tensor *grad,
                            float alpha);
bool tensor_elu_grad(Tensor *out, const Tensor *in, const Tensor *grad,
                     float alpha);
bool tensor_swish_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_gelu_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_relu6_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_hard_sigmoid_grad(Tensor *out, const Tensor *in,
                              const Tensor *grad);
bool tensor_hard_swish_grad(Tensor *out, const Tensor *in, const Tensor *grad);
bool tensor_softplus_grad(Tensor *out, const Tensor *in, const Tensor *grad);

} // namespace gradientcore
