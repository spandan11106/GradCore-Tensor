---
sidebar_position: 1
title: "Loss Functions"
---

# Loss Functions

A loss function measures how wrong your model is. Training is the process of making it less wrong, one gradient at a time. Choosing the right loss function is not optional — using MSE for a classification problem is like using a ruler to measure temperature.

## Conventions

All loss functions share the same signature pattern:

```cpp
// Forward
bool tensor_<name>_loss(Tensor *out,
                         const Tensor *pred,    // model output
                         const Tensor *target,  // ground truth
                         Reduction reduction);

// Backward
bool tensor_<name>_loss_grad(Tensor *out,       // gradient w.r.t. pred
                              const Tensor *pred,
                              const Tensor *target,
                              const Tensor *grad, // upstream gradient
                              Reduction reduction);
```

### Reduction modes

```cpp
enum Reduction {
    REDUCTION_NONE,   // Output has same shape as inputs
    REDUCTION_MEAN,   // Output is a scalar: mean of all losses
    REDUCTION_SUM,    // Output is a scalar: sum of all losses
};
```

`REDUCTION_MEAN` is the most common choice — it normalises by the number of elements so the loss scale doesn't depend on batch size.

For the high-level `nn` API, the reduction is fixed at `REDUCTION_MEAN` (controlled by the `LossType` passed to `Model::compile()`).

---

## MSE Loss (Mean Squared Error)

```
L = (1/N) * Σ (pred_i - target_i)²
∂L/∂pred_i = (2/N) * (pred_i - target_i)
```

```cpp
bool tensor_mse_loss(Tensor *out, const Tensor *pred, const Tensor *target, Reduction);
bool tensor_mse_loss_grad(Tensor *out, const Tensor *pred, const Tensor *target, const Tensor *grad, Reduction);
```

**Use for**: Regression tasks where outliers should be penalised heavily (because the squared term amplifies large errors).

**Avoid when**: Your data has outliers that genuinely shouldn't dominate — use Huber instead.

**nn class**: `MSELoss`

---

## L1 Loss (Mean Absolute Error)

```
L = (1/N) * Σ |pred_i - target_i|
∂L/∂pred_i = (1/N) * sign(pred_i - target_i)
```

```cpp
bool tensor_l1_loss(Tensor *out, const Tensor *pred, const Tensor *target, Reduction);
bool tensor_l1_loss_grad(...);
```

**Use for**: Regression with outliers. The linear penalty is less sensitive to large errors than MSE.

**Gotcha**: The gradient is discontinuous at zero (a subgradient of 0 is used there). This rarely causes problems in practice.

**nn class**: `L1Loss`, also `MAELoss` (which is identical — two names, same implementation).

---

## Huber Loss

```
L_i = 0.5 * d²               if |d| ≤ δ
L_i = δ * (|d| - 0.5 * δ)    otherwise
where d = pred_i - target_i

∂L/∂pred_i = d               if |d| ≤ δ
           = δ * sign(d)      otherwise
```

```cpp
bool tensor_huber_loss(Tensor *out, const Tensor *pred, const Tensor *target,
                        float delta, Reduction);
bool tensor_huber_loss_grad(..., float delta, Reduction);
```

**Use for**: Regression with outliers. Huber is the best of both worlds: quadratic (MSE) for small errors, linear (L1) for large ones. The `delta` parameter controls the transition point.

The California Housing tutorial uses `delta = 1.0` (the default) with house prices scaled to the [0, 5] range.

import useBaseUrl from '@docusaurus/useBaseUrl';

**nn class**: `HuberLoss(float delta = 1.0f)`, exposed as `LossType::HUBER` in the enum.

<iframe 
  src={useBaseUrl('/visualizer/loss_graph.html')} 
  width="100%" 
  height="700px" 
  style={{border: 'none', borderRadius: '12px'}}
  title="Loss Function Visualization"
></iframe>

As you can see in the chart above...

## BCE Loss (Binary Cross-Entropy)

```
L = -(1/N) * Σ [t * log(p) + (1-t) * log(1-p)]
∂L/∂p = (1/N) * (p - t) / (p * (1 - p))
```

```cpp
bool tensor_bce_loss(Tensor *out, const Tensor *pred, const Tensor *target, Reduction);
bool tensor_bce_loss_grad(...);
```

**Inputs**: `pred` must be in (0, 1) — pass sigmoid outputs, not logits. A small epsilon clamps values away from 0 and 1 to avoid `log(0) = -inf`.

**Use for**: Binary classification where the output layer applies sigmoid.

**nn class**: `BCELoss`

---

## BCE with Logits Loss

```
L = max(x, 0) - x*y + log(1 + exp(-|x|))
  = log(1 + exp(x)) - x*y     (numerically stable form)
```

```cpp
bool tensor_bce_with_logits_loss(Tensor *out, const Tensor *logits, const Tensor *target, Reduction);
bool tensor_bce_with_logits_loss_grad(...);
```

**Inputs**: `logits` are raw (unbounded) model outputs — **not** after sigmoid.

**Advantage over BCE**: Numerically more stable. When logits are very positive, `exp(large)` overflows; the log-sum-exp formulation avoids this.

```
∂L/∂x = σ(x) - y
```

**Use for**: Binary classification with a linear output layer (no sigmoid).

**nn class**: `BCEWithLogitsLoss`

---

## Cross-Entropy Loss

```
L = -(1/N) * Σ_batch Σ_class  target * log(softmax(logits))

Numerically:
  log_sum_exp = max(logits) + log(Σ exp(logits - max(logits)))
  L_i         = Σ_c target_c * (log_sum_exp - logits_c)

∂L/∂logits_c = (1/N) * (softmax(logits)_c - target_c)
```

```cpp
bool tensor_cross_entropy_loss(Tensor *out, const Tensor *logits, const Tensor *target, Reduction);
bool tensor_cross_entropy_loss_grad(...);
```

**Inputs**: `logits` are raw scores (before softmax). `target` is a one-hot encoded distribution. The function applies log-softmax internally.

This is the loss used in the MNIST tutorial. It:
1. Subtracts `max(logits)` for numerical stability.
2. Computes log-sum-exp.
3. Computes cross-entropy against the target distribution.

**Use for**: Multi-class classification.

:::warning Don't double-softmax
`tensor_cross_entropy_loss` already applies softmax. Do not add a `nn::Softmax` layer to your model when using this loss.
:::

**nn class**: `CrossEntropyLoss`, exposed as `LossType::CROSS_ENTROPY`.

---

## NLL Loss (Negative Log-Likelihood)

```
L = -(1/N) * Σ_batch Σ_class  target * log_probs
∂L/∂log_probs = -(1/N) * target
```

```cpp
bool tensor_nll_loss(Tensor *out, const Tensor *log_probs, const Tensor *target, Reduction);
bool tensor_nll_loss_grad(...);
```

**Inputs**: `log_probs` must be **log-probabilities** (output of `log(softmax(x))`). `CrossEntropyLoss` is essentially `NLLLoss(LogSoftmax(logits), target)` combined.

**Use for**: When you want explicit control over the log-softmax step, or when working with models that output log-probabilities directly.

**nn class**: `NLLLoss`

---

## KL Divergence Loss

```
L = Σ target * (log(target) - pred)
  = Σ target * log(target/exp(pred))
```

```cpp
bool tensor_kl_div_loss(Tensor *out, const Tensor *pred, const Tensor *target, Reduction);
bool tensor_kl_div_loss_grad(...);
```

**Inputs**: `pred` is expected to be log-probabilities (i.e. after `log_softmax`). `target` is a probability distribution.

**Use for**: Knowledge distillation, variational autoencoders, any task where you're measuring the divergence between two distributions.

**Gradient**: `∂L/∂pred_i = -target_i / N` (only contributes where `target > 0`).

**nn class**: `KLDivLoss`

---

## Hinge Loss

```
L = (1/N) * Σ max(0, 1 - pred * target)
∂L/∂pred = -target / N    if 1 - pred * target > 0
          = 0               otherwise
```

```cpp
bool tensor_hinge_loss(Tensor *out, const Tensor *pred, const Tensor *target, Reduction);
bool tensor_hinge_loss_grad(...);
```

**Inputs**: `pred` is the model output (can be any real number). `target` is ±1 (not 0/1).

**Use for**: Support Vector Machines (SVMs) and binary classification tasks. The "margin" in "maximum margin classifier" comes from this loss.

**nn class**: `HingeLoss`

---

## L2 Loss (Weight Regularization)

```
L = (1/N) * Σ 0.5 * w_i²
∂L/∂w_i = w_i / N
```

```cpp
bool tensor_l2_loss(Tensor *out, const Tensor *weights, Reduction);
bool tensor_l2_loss_grad(Tensor *out, const Tensor *weights, const Tensor *grad, Reduction);
```

L2 regularisation on model weights. Note: in the `nn` API, weight regularisation is handled by the `AdamW` optimizer's `weight_decay` parameter (decoupled from the gradient), which is generally preferable to adding it as a loss term.

**nn class**: `L2Loss`

---

## L1 Regularization

```
L = (1/N) * Σ |w_i|
∂L/∂w_i = sign(w_i) / N
```

```cpp
bool tensor_l1_regularization(Tensor *out, const Tensor *weights, Reduction);
bool tensor_l1_regularization_grad(...);
```

L1 regularisation induces sparsity — it pushes weights toward exactly zero, unlike L2 which only makes them small. Useful when you want the model to actively select features.

**nn class**: `L2Loss` (note: L1 regularisation is also exposed via `LossFunction` but is not in the `LossType` enum — use directly).

---

## Cosine Embedding Loss

```
L = 1 - cosine_similarity(x1, x2)     if y = 1
L = max(0, cosine_similarity(x1, x2) - margin)   if y = -1
```

```cpp
bool tensor_cosine_embedding_loss(Tensor *out, const Tensor *x1, const Tensor *x2,
                                   const Tensor *target, float margin, Reduction);
```

**Use for**: Learning embeddings where similar pairs should be close (cosine distance ≈ 0) and dissimilar pairs should be far apart (cosine distance > margin).

**nn class**: `CosineEmbeddingLoss` (use `forward_triplet(x1, x2, target)` — the standard `forward` is a placeholder that warns).

---

## Triplet Loss

```
L = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)
```

Where `dist` is Euclidean distance.

```cpp
bool tensor_triplet_loss(Tensor *out, const Tensor *anchor, const Tensor *positive,
                          const Tensor *negative, float margin, Reduction);
```

**Use for**: Metric learning — learning an embedding space where the same class clusters together and different classes are separated. Each training example is a triplet: an anchor, a positive (same class), and a negative (different class).

**nn class**: `TripletLoss` (use `forward_triplet(anchor, positive, negative)`).

---

## Choosing a Loss Function

| Task | Recommended Loss |
|---|---|
| Binary classification | `BCEWithLogitsLoss` (output: linear) or `BCELoss` (output: sigmoid) |
| Multi-class classification | `CrossEntropyLoss` |
| Regression, no outliers | `MSELoss` |
| Regression, outliers present | `HuberLoss` |
| Regression, heavy outliers | `L1Loss` |
| Embedding / metric learning | `TripletLoss` or `CosineEmbeddingLoss` |
| Knowledge distillation | `KLDivLoss` |
| SVM-style classification | `HingeLoss` |
