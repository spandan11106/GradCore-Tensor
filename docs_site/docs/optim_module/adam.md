---
sidebar_position: 3
title: "Adam"
---

# `optim::Adam`

Adam (Adaptive Moment Estimation) maintains a running estimate of the first moment (mean) and second moment (uncentred variance) of the gradients, using them to scale the learning rate individually for each parameter. The result is an optimizer that adapts to the local curvature of the loss landscape and typically requires far less tuning than SGD.

**Header**: `include/optim/adam.hpp`  
**Namespace**: `gradientcore::optim`

:::info What it calls
`Adam::step` reads and writes six float tensors per parameter: `data`, `grad`, `m` (first moment), and `v` (second moment). No tensor utility functions are called for the contiguous fast path — the update is a tight loop of scalar operations directly on the data pointers. The non-contiguous fallback uses `tensor_get_flat_index`.
:::

---

## Update Rule

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           # first moment (mean)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²           # second moment (variance)

m̂_t = m_t / (1 - β₁ᵗ)                          # bias-corrected first moment
v̂_t = v_t / (1 - β₂ᵗ)                          # bias-corrected second moment

w_t = w_{t-1} - lr * m̂_t / (√v̂_t + ε)
```

The bias correction `1 / (1 - βᵗ)` compensates for the fact that both `m` and `v` are initialised to zero — without it, the early updates would be pulled towards zero and training would be slow to start.

---

## Constructor

```cpp
optim::Adam(Arena *perm_arena,
            const std::vector<autograd::Variable *> &params,
            float lr    = 0.001f,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float eps   = 1e-8f);
```

| Parameter | Default | Description |
|---|---|---|
| `perm_arena` | — | Permanent arena. Moment tensors (`m`, `v`) are allocated here. |
| `params` | — | Learnable parameters. Get via `model->parameters()`. |
| `lr` | `0.001` | Learning rate. Start here; rarely needs changing. |
| `beta1` | `0.9` | Exponential decay for first moment. Controls how much past gradient direction matters. |
| `beta2` | `0.999` | Exponential decay for second moment. Controls how quickly the per-parameter learning rate adapts. |
| `eps` | `1e-8` | Denominator stability term. Prevents division by zero when `v` is near zero. |

The constructor allocates two zero-initialised tensors — `m` and `v` — for each parameter that has `requires_grad = true`. Parameters without gradients get `nullptr` state entries.

```cpp
auto params = seq->parameters();
optim::Adam adam(perm_arena, params);                        // defaults
optim::Adam adam(perm_arena, params, 0.0005f);              // lower lr
optim::Adam adam(perm_arena, params, 0.001f, 0.9f, 0.98f); // slower v decay
```

### Memory overhead

For a model with `N` trainable parameters, Adam allocates `2 * N * sizeof(float)` bytes on `perm_arena` for moment tensors. For the MNIST MLP (101 770 parameters), this is about 815 KB — negligible on modern hardware.

---

## Methods

### `step(Arena *temp_arena = nullptr)`

```cpp
void step(Arena *temp_arena = nullptr);
```

Applies one Adam update step. `temp_arena` is accepted for API consistency with SGD but is not used — Adam's update is entirely in-place and requires no temporary allocations.

For each trainable parameter `p` at global step `t`:

```cpp
float m_hat_correction = 1.0f / (1.0f - pow(beta1, t));
float v_hat_correction = 1.0f / (1.0f - pow(beta2, t));

for each element k:
    g = p->grad[k]
    m[k] = beta1 * m[k] + (1 - beta1) * g
    v[k] = beta2 * v[k] + (1 - beta2) * g * g
    m_hat = m[k] * m_hat_correction
    v_hat = v[k] * v_hat_correction
    p->data[k] -= lr * m_hat / (sqrt(v_hat) + eps)
```

The step counter `t` is incremented at the start of each `step()` call and is shared across all parameters in the same optimizer instance.

### `zero_grad()`

```cpp
void zero_grad();
```

Calls `tensor_clear` on every parameter's `grad` tensor. Must be called before each `backward()` pass.

---

## Full Training Loop Example

```cpp
auto* perm  = Arena::create(MiB(1024), MiB(64), true);
auto* graph = Arena::create(MiB(512),  MiB(32), true);

nn::Sequential seq;
// ... add layers ...

optim::Adam adam(perm, seq.parameters(), 0.0005f);
nn::CrossEntropyLoss criterion;

for (int epoch = 0; epoch < 40; epoch++) {
    loader->reset(true);
    while (loader->has_next()) {
        uint64_t pos = graph->get_pos();
        auto batch = loader->next(graph);

        auto* x    = autograd::create_leaf(graph, batch.features, false);
        auto* y    = autograd::create_leaf(graph, batch.labels,   false);
        auto* pred = seq.forward(graph, x);
        auto* loss = criterion.forward(graph, pred, y);

        adam.zero_grad();
        autograd::backward(graph, loss);
        adam.step();          // temp_arena not needed

        graph->pop_to(pos);
    }
}
```

---

## Hyperparameter Guide

### Learning rate (`lr`)

`0.001` is the canonical Adam default and works well for most tasks out of the box. Common deviations:

| Scenario | Suggested `lr` |
|---|---|
| General classification / regression | `0.001` |
| Large models or deep networks | `0.0001` to `0.0005` |
| Fine-tuning pretrained weights | `1e-5` to `1e-4` |
| Very small datasets | `0.001` with early stopping |

If loss oscillates wildly, halve the learning rate. If loss decreases very slowly, double it.

### `beta1` (first moment decay)

Controls the inertia of the gradient direction. The default `0.9` means the effective gradient is a 10-step exponential moving average. Lowering it (e.g. `0.8`) makes the optimizer more responsive to recent gradients; raising it (e.g. `0.95`) smooths more.

### `beta2` (second moment decay)

Controls how quickly the per-parameter effective learning rate adapts. The default `0.999` gives a 1000-step moving average of squared gradients. Lower values (e.g. `0.98`) make it adapt faster but less stably.

### `eps`

The stability constant. Rarely needs changing. In rare cases where parameters have extremely small gradients throughout training, increasing `eps` to `1e-7` or `1e-6` can prevent artificially large updates.

---

## When to Use Adam

Adam is the right first choice for most deep learning tasks:

- **Classification** (MNIST tutorial: Adam + CrossEntropyLoss, 97.77% in 40 epochs).
- **Regression** with well-conditioned features and targets.
- Any task where you want results quickly without hyperparameter tuning.

Consider switching to **AdamW** if you notice overfitting — Adam applies weight decay incorrectly (it interacts with the second moment estimate), whereas AdamW decouples it properly. Consider **SGD** if you have a very large model and the extra memory for `m` and `v` is a constraint.

:::warning Adam and weight decay
Standard Adam does not implement weight decay correctly. When you pass `weight_decay` to the gradient computation (as some frameworks do), it gets scaled by the adaptive learning rate, changing its effective magnitude throughout training. AdamW fixes this by applying weight decay directly to the weights, independently of the gradient. Use `optim::AdamW` when regularisation matters.
:::