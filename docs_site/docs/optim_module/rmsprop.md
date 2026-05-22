---
sidebar_position: 5
title: "RMSprop"
---

# `optim::RMSprop`

RMSprop (Root Mean Square Propagation) divides the learning rate by a running average of recent squared gradient magnitudes. This normalises the effective step size across parameters — parameters with historically large gradients get smaller steps, and parameters with small gradients get larger ones. Unlike Adam, RMSprop does not maintain a first moment estimate, making it lighter on memory and slightly simpler.

**Header**: `include/optim/rmsprop.hpp`  
**Namespace**: `gradientcore::optim`

:::info What it calls
`RMSprop::step` reads from `grad` and `v` (the squared gradient accumulator) and writes to both `v` and `data`. All operations are in-place on the parameter tensors. No temporary allocations are needed.
:::

---

## Update Rule

```
v_t = α * v_{t-1} + (1 - α) * g_t²           # running average of squared gradients
w_t = w_{t-1} - lr * g_t / (√v_t + ε)
```

With optional weight decay (coupled, added to the gradient before the update):

```
g_t = ∇L(w) + λ * w                           # L2 regularised gradient
v_t = α * v_{t-1} + (1 - α) * g_t²
w_t = w_{t-1} - lr * g_t / (√v_t + ε)
```

Note that RMSprop's weight decay is **coupled** — it is added to the gradient and then scaled by the adaptive denominator. This differs from AdamW's decoupled approach. For clean weight decay, use AdamW.

---

## Constructor

```cpp
optim::RMSprop(Arena *perm_arena,
               const std::vector<autograd::Variable *> &params,
               float lr           = 0.01f,
               float alpha        = 0.99f,
               float eps          = 1e-8f,
               float weight_decay = 0.0f);
```

| Parameter | Default | Description |
|---|---|---|
| `perm_arena` | — | Permanent arena. The `v` state tensor is allocated here per parameter. |
| `params` | — | Learnable parameters from `model->parameters()`. |
| `lr` | `0.01` | Learning rate. RMSprop typically uses a larger default than Adam. |
| `alpha` | `0.99` | Smoothing factor for the squared gradient running average. |
| `eps` | `1e-8` | Stability constant. Prevents division by zero when gradients are tiny. |
| `weight_decay` | `0.0` | Coupled L2 regularisation coefficient. `0.0` disables it. |

```cpp
auto params = seq->parameters();
optim::RMSprop rms(perm_arena, params);                       // defaults
optim::RMSprop rms(perm_arena, params, 0.001f);               // smaller lr
optim::RMSprop rms(perm_arena, params, 0.01f, 0.95f);         // faster adaptation
```

### Memory overhead

One `v` tensor per parameter: `N * sizeof(float)` bytes on `perm_arena`. Half the memory of Adam, which maintains both `m` and `v`.

---

## Methods

### `step(Arena *temp_arena = nullptr)`

```cpp
void step(Arena *temp_arena = nullptr);
```

Applies one RMSprop update. `temp_arena` is unused.

For each trainable parameter:

```cpp
for each element k:
    g = p->grad[k]

    // Optional coupled weight decay
    if (weight_decay != 0.0f):
        g += weight_decay * p->data[k]

    // Update running squared gradient average
    v[k] = alpha * v[k] + (1 - alpha) * g * g

    // Apply update
    p->data[k] -= lr * g / (sqrt(v[k]) + eps)
```

### `zero_grad()`

```cpp
void zero_grad();
```

Zeroes every parameter's `grad` tensor.

---

## The `alpha` Parameter

`alpha` controls how much historical squared gradient information is retained:

| `alpha` | Effective window | Effect |
|---|---|---|
| `0.9` | ~10 steps | Fast adaptation, more noise in the effective LR |
| `0.99` (default) | ~100 steps | Smooth adaptation, stable across most tasks |
| `0.999` | ~1000 steps | Very slow adaptation — effectively constant LR early on |

A larger `alpha` makes the denominator change more slowly, giving the effective learning rate more inertia. A smaller `alpha` lets the optimizer react quickly to sudden changes in gradient magnitude — useful for non-stationary problems.

---

## Full Example

```cpp
auto* perm  = Arena::create(MiB(1024), MiB(64), true);
auto* graph = Arena::create(MiB(512),  MiB(32), true);

nn::Sequential seq;
// ... add layers ...

optim::RMSprop rms(perm, seq.parameters(), 0.001f, 0.99f);
nn::HuberLoss criterion(2.0f);
nn::Trainer<optim::RMSprop, nn::HuberLoss> trainer(&seq, &rms, &criterion, graph);

TrainingStats stats = trainer.fit_dataloader(loader, 100);
```

---

## When to Use RMSprop

RMSprop is a strong choice when:

- **Training RNNs or sequences** — RMSprop was originally proposed by Geoffrey Hinton for this purpose. The absence of a first moment estimate avoids some of the instability that momentum can introduce in recurrent settings.
- **Non-stationary objectives** — problems where the gradient distribution changes during training. RMSprop adapts its effective learning rate relatively quickly compared to Adam.
- **Memory-constrained settings** — RMSprop uses half the state memory of Adam (`v` only, no `m`).

For standard feedforward networks and CNNs, Adam or AdamW will typically converge faster. RMSprop and Adam differ mainly in whether they include the first moment (momentum) term — Adam does, RMSprop does not.