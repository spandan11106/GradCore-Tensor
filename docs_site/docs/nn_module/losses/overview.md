---
sidebar_position: 1
title: "Loss Functions"
---

# Loss Functions (`nn` layer)

The `nn` loss classes are the user-facing wrappers over the tensor-level `tensor_*_loss` functions. They inherit from `LossFunction`, hold a `Reduction` mode, and expose a single `forward()` method. They also work as callables via `operator()`.

**Header**: `include/nn/losses/loss.hpp`

:::info What they call
Each `nn` loss class's `forward` calls the corresponding `autograd::*_loss` function, which calls `tensor_*_loss` and wires up a backward node. The `nn` layer itself contains no mathematical logic — it's a configuration object that packages the right reduction mode with the right underlying operation.
:::

---

## `LossFunction` — The Base Class

```cpp
class LossFunction {
public:
    Reduction reduction;

    virtual autograd::Variable *forward(Arena *arena,
                                        autograd::Variable *pred,
                                        autograd::Variable *target) = 0;

    autograd::Variable *operator()(Arena *arena,
                                   autograd::Variable *pred,
                                   autograd::Variable *target);
};
```

All loss classes inherit from `LossFunction`. The `operator()` overload lets you call a loss object like a function:

```cpp
nn::MSELoss criterion;
auto* loss = criterion(graph_arena, pred, target);
// equivalent to:
auto* loss = criterion.forward(graph_arena, pred, target);
```

---

## When to Use Each Loss

| Task | Recommended class |
|---|---|
| Multi-class classification | `CrossEntropyLoss` |
| Binary classification (logit output) | `BCEWithLogitsLoss` |
| Binary classification (sigmoid output) | `BCELoss` |
| Regression, standard | `MSELoss` |
| Regression, outlier-robust | `HuberLoss` |
| Regression, heavy outliers | `L1Loss` / `MAELoss` |
| Metric learning (pairs) | `CosineEmbeddingLoss` |
| Metric learning (triplets) | `TripletLoss` |
| SVM-style binary | `HingeLoss` |
| Distribution matching | `KLDivLoss` |
| Weight regularisation | `L2Loss` |

---

## `nn::CrossEntropyLoss`

```cpp
class CrossEntropyLoss : public LossFunction;
// Constructor: CrossEntropyLoss(Reduction red = REDUCTION_MEAN)
```

Combines log-softmax and negative log-likelihood into a single numerically stable operation. **This is the loss to use for multi-class classification.**

```cpp
nn::CrossEntropyLoss criterion;
// pred:   [batch, num_classes]  — raw logits, no softmax applied
// target: [batch, num_classes]  — one-hot encoded
auto* loss = criterion(graph_arena, pred, target);
```

The `LossType::CROSS_ENTROPY` enum value in `nn::Model::compile()` uses this class.

**Do not apply softmax before this loss.** It applies softmax internally.

---

## `nn::MSELoss`

```cpp
class MSELoss : public LossFunction;
// Constructor: MSELoss(Reduction red = REDUCTION_MEAN)
```

`L = (1/N) * Σ (pred_i - target_i)²`

The standard regression loss. Penalises large errors quadratically, making it sensitive to outliers.

```cpp
nn::MSELoss criterion;
// pred:   [batch, 1]   — scalar regression output
// target: [batch, 1]   — ground-truth values
auto* loss = criterion(graph_arena, pred, target);
```

`LossType::MSE` in `Model::compile()`.

---

## `nn::L1Loss`

```cpp
class L1Loss : public LossFunction;
// Constructor: L1Loss(Reduction red = REDUCTION_MEAN)
```

`L = (1/N) * Σ |pred_i - target_i|`

Linear penalty. More robust to outliers than MSE.

---

## `nn::MAELoss`

```cpp
class MAELoss : public LossFunction;
// Constructor: MAELoss(Reduction red = REDUCTION_MEAN)
```

Mean Absolute Error. **Identical to `L1Loss`** — both call `autograd::l1_loss` internally. Two names for user convenience; use whichever is clearer in context.

---

## `nn::HuberLoss`

```cpp
class HuberLoss : public LossFunction;
// Constructor: HuberLoss(float delta = 1.0f, Reduction red = REDUCTION_MEAN)
```

Quadratic for small errors (`|d| ≤ delta`), linear for large errors. The best of MSE and L1. The California Housing tutorial uses this:

```cpp
model.compile(nn::OptimizerType::ADAMW,
              nn::LossType::HUBER,      // ← HuberLoss(delta=1.0)
              0.001f, 200, 128);
```

`LossType::HUBER` in `Model::compile()` creates `HuberLoss` with `delta = 1.0f`.

For a custom delta, construct directly and use `Trainer` instead of `Model`:

```cpp
nn::HuberLoss criterion(2.0f);   // delta = 2.0
```

---

## `nn::BCELoss`

```cpp
class BCELoss : public LossFunction;
// Constructor: BCELoss(Reduction red = REDUCTION_MEAN)
```

`L = -(1/N) * Σ [t*log(p) + (1-t)*log(1-p)]`

For binary classification where the model output has already been passed through **sigmoid** (i.e. `pred ∈ (0, 1)`).

```cpp
// model output: sigmoid(logits)
nn::Sigmoid sig;
auto* prob = sig.forward(graph_arena, logits);
nn::BCELoss criterion;
auto* loss = criterion(graph_arena, prob, target);
```

`LossType::BCE` in `Model::compile()`.

---

## `nn::BCEWithLogitsLoss`

```cpp
class BCEWithLogitsLoss : public LossFunction;
// Constructor: BCEWithLogitsLoss(Reduction red = REDUCTION_MEAN)
```

Same as BCELoss but accepts raw **logits** (not sigmoid-ed). Numerically more stable because it uses the log-sum-exp trick internally. **Prefer this over `BCELoss` + separate sigmoid.**

```cpp
// model output: raw logits (no sigmoid)
nn::BCEWithLogitsLoss criterion;
auto* loss = criterion(graph_arena, logits, target);
```

`LossType::BCE_WITH_LOGITS` in `Model::compile()`.

---

## `nn::NLLLoss`

```cpp
class NLLLoss : public LossFunction;
// Constructor: NLLLoss(Reduction red = REDUCTION_MEAN)
```

Negative log-likelihood. Expects `pred` to be **log-probabilities** (output of `log(softmax(x))`). `CrossEntropyLoss` is essentially `NLLLoss(LogSoftmax(logits), target)` combined.

---

## `nn::KLDivLoss`

```cpp
class KLDivLoss : public LossFunction;
// Constructor: KLDivLoss(Reduction red = REDUCTION_MEAN)
```

`L = Σ target * (log(target) - pred)`

Expects `pred` as log-probabilities. Use for knowledge distillation or training distributions (VAEs, diffusion models).

---

## `nn::HingeLoss`

```cpp
class HingeLoss : public LossFunction;
// Constructor: HingeLoss(Reduction red = REDUCTION_MEAN)
```

`L = (1/N) * Σ max(0, 1 - pred * target)`

SVM-style loss. `target` must be `±1` (not `0/1`).

---

## `nn::L2Loss`

```cpp
class L2Loss : public LossFunction;
// Constructor: L2Loss(Reduction red = REDUCTION_MEAN)
```

`L = (1/N) * Σ 0.5 * w²`

L2 weight regularisation as a loss term. Generally prefer `AdamW`'s built-in `weight_decay` parameter for regularisation instead.

---

## `nn::CosineEmbeddingLoss`

```cpp
class CosineEmbeddingLoss : public LossFunction;
// Constructor: CosineEmbeddingLoss(float margin = 0.0f,
//                                  Reduction red = REDUCTION_MEAN)
```

For learning embeddings where pairs of inputs should be similar (`target=1`) or dissimilar (`target=-1`). Use the `forward_triplet` method, not `forward`:

```cpp
nn::CosineEmbeddingLoss criterion(0.5f);   // margin = 0.5
auto* loss = criterion.forward_triplet(graph_arena, x1, x2, target);
```

Calling `forward` (the base class override) prints a warning and returns `nullptr`.

---

## `nn::TripletLoss`

```cpp
class TripletLoss : public LossFunction;
// Constructor: TripletLoss(float margin = 1.0f,
//                          Reduction red = REDUCTION_MEAN)
```

`L = max(0, dist(anchor, pos) - dist(anchor, neg) + margin)`

For metric learning with triplets of (anchor, positive, negative) samples. Use `forward_triplet`:

```cpp
nn::TripletLoss criterion(1.0f);
auto* loss = criterion.forward_triplet(graph_arena, anchor, positive, negative);
```

---

## Using Loss Functions Directly (Outside `Model`)

If you're using `Trainer` directly or writing a custom training loop:

```cpp
nn::CrossEntropyLoss criterion;

// In your training loop:
auto* pred = model_seq->forward(graph_arena, x);
auto* loss = criterion.forward(graph_arena, pred, y);

float loss_val = loss->data->storage->data[loss->data->offset];

optimizer.zero_grad();
autograd::backward(graph_arena, loss);
optimizer.step(graph_arena);
```

For the high-level path, see [Model](../models/model) and [Trainer](../training/trainer).