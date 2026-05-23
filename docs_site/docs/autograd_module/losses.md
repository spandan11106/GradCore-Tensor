---
sidebar_position: 5
title: "Loss Operations"
---

# Loss Operations

The autograd loss functions are differentiable wrappers around the tensor-level loss functions. Each one computes the scalar loss in the forward pass, saves the tensors needed for the gradient, and registers a `backward_fn` that computes the gradient of the loss with respect to the model's predictions and accumulates it into the prediction's `grad` tensor.

**Header**: `include/autograd/autograd.hpp`  
**Source**: `src/autograd/ops/loss/`

:::info Target variables never require gradients
All loss functions differentiate with respect to the **prediction** only, never the target. The target is wrapped as a `create_leaf(..., requires_grad=false)` and its pointer is saved for the backward computation, but no gradient is accumulated into it. This matches the standard ML convention — you update your model's predictions to match the target, not the other way around.
:::

:::info Output shape
Every loss function produces a **scalar output**: a 1-element tensor with shape `[1]`. The `requires_grad` of the output is inherited from the prediction input. After backward, the loss gradient is seeded to `1.0` by `autograd::backward`, and the chain rule propagates it back toward the prediction.
:::

---

## Common Structure

Every loss op follows this pattern:

```cpp
Variable *mse_loss(Arena *arena, Variable *pred, Variable *target, Reduction reduction) {
    // 1. Create scalar output, compute forward value
    uint32_t scalar_shape[1] = {1};
    Tensor *out_data = tensor_create_zeros(arena, 1, scalar_shape);
    tensor_mse_loss(out_data, pred->data, target->data, reduction);

    // 2. Create output Variable
    Variable *out = arena->push<Variable>();
    out->data          = out_data;
    out->requires_grad = pred->requires_grad;
    out->is_leaf       = false;
    out->reduction     = reduction;   // stored for backward

    if (out->requires_grad) {
        // 3. Allocate grad, wire parent (pred only), save pred + target
        out->grad              = tensor_create_zeros(arena, 1, scalar_shape);
        out->num_parents       = 1;
        out->parents           = arena->push_array<Edge>(1);
        out->parents[0]        = {pred};
        out->num_saved         = 2;
        out->saved_tensors     = arena->push_array<Tensor *>(2);
        out->saved_tensors[0]  = pred->data;
        out->saved_tensors[1]  = target->data;

        // 4. Backward function
        out->backward_fn = [](Variable *self, Arena *temp_arena) {
            Variable *parent = self->parents[0].node;
            if (!parent->requires_grad) return;
            Tensor *local_grad = tensor_create_zeros(
                temp_arena, parent->grad->ndims, parent->grad->shape);
            tensor_mse_loss_grad(local_grad,
                                 self->saved_tensors[0],
                                 self->saved_tensors[1],
                                 self->grad,
                                 static_cast<Reduction>(self->reduction));
            tensor_add(parent->grad, parent->grad, local_grad);
        };
    }
    return out;
}
```

The `reduction` mode is stored as a `uint32_t` in `out->reduction` (cast from the `Reduction` enum) so the backward function can recover it via `static_cast<Reduction>(self->reduction)`.

---

## Standard Two-Input Loss Functions

These all take `(Arena*, Variable* pred, Variable* target, Reduction reduction)` and differentiate with respect to `pred`.

### `mse_loss`

```cpp
Variable *mse_loss(Arena *arena, Variable *pred, Variable *target,
                   Reduction reduction);
```

`L = (1/N) * Σ (pred_i - target_i)²`

**Gradient**: `∂L/∂pred_i = (2/N) * (pred_i - target_i)`

Saves: `pred->data`, `target->data`.

```cpp
auto *loss = autograd::mse_loss(graph_arena, pred, target, REDUCTION_MEAN);
```

---

### `l1_loss`

```cpp
Variable *l1_loss(Arena *arena, Variable *pred, Variable *target,
                  Reduction reduction);
```

`L = (1/N) * Σ |pred_i - target_i|`

**Gradient**: `∂L/∂pred_i = (1/N) * sign(pred_i - target_i)`

Gradient is zero where `pred_i == target_i` (a subgradient of 0 is chosen).

---

### `huber_loss`

```cpp
Variable *huber_loss(Arena *arena, Variable *pred, Variable *target,
                     float delta, Reduction reduction);
```

```
L_i = 0.5 * d²            if |d| ≤ delta   (d = pred_i - target_i)
L_i = delta * (|d| - 0.5 * delta)   otherwise
```

`delta` is stored in `out->metadata_float` and passed to the backward.

**Gradient**:
```
∂L/∂pred_i = (1/N) * d           if |d| ≤ delta
           = (1/N) * delta * sign(d)   otherwise
```

---

### `bce_loss`

```cpp
Variable *bce_loss(Arena *arena, Variable *pred, Variable *target,
                   Reduction reduction);
```

`L = -(1/N) * Σ [t * log(p) + (1-t) * log(1-p)]`

`pred` must contain values in `(0, 1)` — i.e. the output of a sigmoid layer. A small epsilon clamps values away from 0 and 1 to avoid `log(0)`.

**Gradient**: `∂L/∂p = (1/N) * (p - t) / (p * (1 - p))`

:::warning Use `bce_with_logits_loss` when possible
`bce_loss` requires sigmoid-ed predictions. `bce_with_logits_loss` takes raw logits and is numerically stabler.
:::

---

### `bce_with_logits_loss`

```cpp
Variable *bce_with_logits_loss(Arena *arena, Variable *logits, Variable *target,
                               Reduction reduction);
```

Numerically stable BCE that accepts raw logits (before sigmoid). Uses the log-sum-exp trick:

```
L = max(x, 0) - x * y + log(1 + exp(-|x|))
```

**Gradient**: `∂L/∂x = (1/N) * (σ(x) - y)`

This is the preferred loss for binary classification with a linear output layer.

---

### `cross_entropy_loss`

```cpp
Variable *cross_entropy_loss(Arena *arena, Variable *logits, Variable *target,
                             Reduction reduction);
```

Applies log-softmax and negative log-likelihood in one numerically stable operation.

**Forward**: For each batch element, subtracts `max(logits)` for stability, computes log-sum-exp, then computes `Σ target_c * (log_sum_exp - logit_c)`.

**Gradient**: `∂L/∂logit_c = (1/N) * (softmax(logit)_c - target_c)`

The gradient is simply the difference between the predicted probability and the target probability — intuitive and efficient.

Saves: `logits->data`, `target->data`.

```cpp
// pred: raw logits [batch, num_classes], target: one-hot [batch, num_classes]
auto *loss = autograd::cross_entropy_loss(graph_arena, pred, target, REDUCTION_MEAN);
```

:::warning Do not apply softmax before this
`cross_entropy_loss` applies log-softmax internally. Double-applying softmax will corrupt both the loss value and its gradient.
:::

---

### `nll_loss`

```cpp
Variable *nll_loss(Arena *arena, Variable *log_probs, Variable *target,
                   Reduction reduction);
```

Negative log-likelihood. Expects `log_probs` to be the output of `log(softmax(x))` — i.e. log-probabilities.

`L = -(1/N) * Σ_batch Σ_class target * log_probs`

**Gradient**: `∂L/∂log_probs = -(1/N) * target`

`cross_entropy_loss` combines `log_softmax` + `nll_loss` in one stable step and should be preferred for classification.

---

### `kl_div_loss`

```cpp
Variable *kl_div_loss(Arena *arena, Variable *pred, Variable *target,
                      Reduction reduction);
```

`L = Σ target * (log(target) - pred)`   where `pred` is log-probabilities.

**Gradient**: `∂L/∂pred_i = -(1/N) * target_i` (only where `target_i > 0`)

Used for knowledge distillation and variational models where you match predicted log-probabilities to a target distribution.

---

### `hinge_loss`

```cpp
Variable *hinge_loss(Arena *arena, Variable *pred, Variable *target,
                     Reduction reduction);
```

`L = (1/N) * Σ max(0, 1 - pred * target)`

Target must be `±1`. Used for SVM-style binary classification.

**Gradient**: `∂L/∂pred = -(1/N) * target` where `1 - pred * target > 0`, else `0`.

---

## Regularisation Losses

These take a single parameter tensor rather than a prediction-target pair.

### `l2_loss`

```cpp
Variable *l2_loss(Arena *arena, Variable *weights, Reduction reduction);
```

`L = (1/N) * Σ 0.5 * w_i²`

**Gradient**: `∂L/∂w = (1/N) * w`

Only one parent (`weights`). Saves `weights->data`.

:::tip Prefer AdamW's built-in weight decay
`autograd::l2_loss` adds L2 regularisation as a loss term, which interacts with adaptive learning rate scaling in Adam. `optim::AdamW` applies weight decay directly and independently, which is the correct formulation. Use AdamW unless you have a specific reason to add L2 through the loss.
:::

---

### `l1_regularization`

```cpp
Variable *l1_regularization(Arena *arena, Variable *weights,
                             Reduction reduction);
```

`L = (1/N) * Σ |w_i|`

**Gradient**: `∂L/∂w = (1/N) * sign(w)`

L1 regularisation induces sparsity — weights are pushed toward exactly zero rather than merely made small. Useful when you want the model to actively select features.

---

## Multi-Input Loss Functions

These take more than two Variable inputs and have specialised parent wiring.

### `cosine_embedding_loss`

```cpp
Variable *cosine_embedding_loss(Arena *arena, Variable *x1, Variable *x2,
                                Variable *target, float margin,
                                Reduction reduction);
```

```
L = 1 - cos_sim(x1, x2)             if target == 1
L = max(0, cos_sim(x1, x2) - margin) if target == -1
```

Three parents: `x1`, `x2`, `target`. Gradients are computed for `x1` (and `x2` if it requires grad). `margin` is stored in `metadata_float`.

**Gradient w.r.t. x1**:

```
∂L/∂x1_f = sign * (x2_f - x1_f * dot/‖x1‖²) / (‖x1‖ * ‖x2‖)
```

where `sign = -1` for similar pairs (`target=1`) and `+1` for dissimilar pairs where the margin is violated.

:::note
The backward function in `src/autograd/ops/loss/cosine_embedding_loss.cpp` only computes the gradient for `x1`, not `x2`. If you need gradients for both embedding vectors, you will need to extend the backward function or compute `x2`'s gradient symmetrically.
:::

```cpp
// metric learning: x1 and x2 should be similar (target=1) or dissimilar (target=-1)
auto *loss = autograd::cosine_embedding_loss(
    graph_arena, x1, x2, target, /*margin=*/0.5f, REDUCTION_MEAN);
```

---

### `triplet_loss`

```cpp
Variable *triplet_loss(Arena *arena, Variable *anchor, Variable *positive,
                       Variable *negative, float margin, Reduction reduction);
```

`L = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)`

where `dist` is Euclidean distance.

Three parents: `anchor`, `positive`, `negative`. All three are saved. `margin` is stored in `metadata_float`.

**Gradient w.r.t. anchor** (when the triplet is active, i.e. the loss > 0):

```
∂L/∂anchor_f = (anchor_f - positive_f) / dist_ap  -  (anchor_f - negative_f) / dist_an
```

The backward function in the source computes the anchor gradient; `local_grad_positive` and `local_grad_negative` are allocated but their gradients from the backward function are assigned to `parent_anchor`'s grad only. Extending the backward for positive and negative gradients follows from the symmetry of Euclidean distance.

```cpp
auto *loss = autograd::triplet_loss(
    graph_arena, anchor, positive, negative, /*margin=*/1.0f, REDUCTION_MEAN);
```

---

## Loss Function Quick Reference

| Function | Inputs | Use case |
|---|---|---|
| `mse_loss` | `pred, target` | Regression |
| `l1_loss` | `pred, target` | Regression, outlier-robust |
| `huber_loss` | `pred, target, delta` | Regression, best of MSE+L1 |
| `bce_loss` | `pred (sigmoid), target` | Binary classification |
| `bce_with_logits_loss` | `logits, target` | Binary classification (preferred) |
| `cross_entropy_loss` | `logits (raw), target (one-hot)` | Multi-class classification |
| `nll_loss` | `log_probs, target` | When you control log-softmax step |
| `kl_div_loss` | `log_probs, target (dist)` | Distribution matching |
| `hinge_loss` | `pred, target (±1)` | SVM-style binary classification |
| `l2_loss` | `weights` | L2 weight regularisation |
| `l1_regularization` | `weights` | L1 weight regularisation |
| `cosine_embedding_loss` | `x1, x2, target (±1), margin` | Metric learning (pairs) |
| `triplet_loss` | `anchor, pos, neg, margin` | Metric learning (triplets) |