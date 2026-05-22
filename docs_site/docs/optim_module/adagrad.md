---
sidebar_position: 6
title: "Adagrad"
---

# `optim::Adagrad`

Adagrad (Adaptive Gradient Algorithm) accumulates the sum of all squared gradients seen so far and divides the current gradient by its square root. Parameters that receive large gradients frequently get progressively smaller effective learning rates; parameters with sparse, infrequent gradients retain a larger effective rate. This makes Adagrad especially well-suited for tasks with naturally sparse gradient signals.

**Header**: `include/optim/adagrad.hpp`  
**Namespace**: `gradientcore::optim`

:::info What it calls
`Adagrad::step` reads from `grad` and `sum_sq` (the cumulative squared gradient accumulator), updates `sum_sq` by adding `g²`, then applies the scaled update to `data`. All operations are in-place. No temporary allocations needed.
:::

---

## Update Rule

```
G_t = G_{t-1} + g_t²           # cumulative sum of squared gradients
w_t = w_{t-1} - lr * g_t / (√G_t + ε)
```

With optional coupled weight decay:

```
g_t = ∇L(w) + λ * w
G_t = G_{t-1} + g_t²
w_t = w_{t-1} - lr * g_t / (√G_t + ε)
```

The critical difference from RMSprop: Adagrad accumulates **all** historical squared gradients, while RMSprop uses an exponential moving average. This means Adagrad's effective learning rate decreases monotonically and never recovers — once a parameter has seen enough large gradients, its effective step size will approach zero and it will essentially stop updating.

---

## Constructor

```cpp
optim::Adagrad(Arena *perm_arena,
               const std::vector<autograd::Variable *> &params,
               float lr           = 0.01f,
               float eps          = 1e-10f,
               float weight_decay = 0.0f);
```

| Parameter | Default | Description |
|---|---|---|
| `perm_arena` | — | Permanent arena. The `sum_sq` accumulator is allocated here. |
| `params` | — | Learnable parameters from `model->parameters()`. |
| `lr` | `0.01` | Global learning rate (the initial effective rate before any accumulation). |
| `eps` | `1e-10` | Stability constant. Smaller than Adam's default because `G_t` grows over time. |
| `weight_decay` | `0.0` | Coupled L2 regularisation coefficient. |

```cpp
auto params = seq->parameters();
optim::Adagrad adagrad(perm_arena, params);              // defaults
optim::Adagrad adagrad(perm_arena, params, 0.1f);        // higher initial lr
```

Notice that `eps = 1e-10` (not `1e-8`) — the smaller value is appropriate because `G_t` accumulates over many steps and is generally much larger than `v_t` in Adam or RMSprop.

### Memory overhead

One `sum_sq` tensor per parameter: `N * sizeof(float)` bytes on `perm_arena`. Same as RMSprop.

---

## Methods

### `step(Arena *temp_arena = nullptr)`

```cpp
void step(Arena *temp_arena = nullptr);
```

Applies one Adagrad update. `temp_arena` is unused.

For each trainable parameter:

```cpp
for each element k:
    g = p->grad[k]

    // Optional coupled weight decay
    if (weight_decay != 0.0f):
        g += weight_decay * p->data[k]

    // Accumulate squared gradient (never resets)
    sum_sq[k] += g * g

    // Apply scaled update
    p->data[k] -= lr * g / (sqrt(sum_sq[k]) + eps)
```

The `sum_sq` accumulator is never reset — it grows monotonically throughout the entire training run.

### `zero_grad()`

```cpp
void zero_grad();
```

Zeroes every parameter's `grad` tensor.

---

## The Dying Learning Rate Problem

Because `G_t = G_0 + G_1 + G_2 + …` grows without bound, the effective step size `lr / √G_t` decreases monotonically towards zero. For a parameter that receives gradients of roughly constant magnitude `g`, the effective learning rate at step `t` is approximately:

```
effective_lr ≈ lr / (√(t * g²) + ε) = lr / (g * √t)
```

This decays as `1/√t`. After enough steps, updates become so small that parameters effectively freeze. This can be desirable (training "naturally terminates") or problematic (parameters freeze before converging).

**RMSprop** was invented specifically to fix this by using an exponential moving average instead of an infinite sum, allowing the effective learning rate to recover after periods of small gradients.

---

## Full Example

```cpp
auto* perm  = Arena::create(MiB(1024), MiB(64), true);
auto* graph = Arena::create(MiB(512),  MiB(32), true);

nn::Sequential seq;
// ... add layers ...

// Adagrad with a high initial lr (it will decay naturally)
optim::Adagrad adagrad(perm, seq.parameters(), 0.1f);
nn::CrossEntropyLoss criterion;
nn::Trainer<optim::Adagrad, nn::CrossEntropyLoss> trainer(
    &seq, &adagrad, &criterion, graph);

TrainingStats stats = trainer.fit(X_train, Y_train, 50, 64);
```

---

## When to Use Adagrad

Adagrad is the right choice for:

- **Sparse features** — tasks like natural language processing with bag-of-words representations, where most word embedding parameters receive a gradient on only a fraction of batches. Parameters for rare words keep a high effective learning rate while common words' parameters are damped appropriately.
- **Short training runs** — the dying learning rate problem only becomes critical over many epochs. For fast prototyping with few epochs, Adagrad's simplicity is an advantage.
- **Tasks where stopping early is fine** — Adagrad's natural learning rate decay acts as an implicit schedule, so you don't need to implement one separately.

For long training runs or dense gradient tasks (standard feedforward networks, CNNs), prefer Adam, AdamW, or RMSprop — all of which avoid the monotonic decay problem.