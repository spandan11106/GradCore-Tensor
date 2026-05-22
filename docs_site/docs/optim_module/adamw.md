---
sidebar_position: 4
title: "AdamW"
---

# `optim::AdamW`

AdamW is Adam with **decoupled weight decay**. The key insight ([Loshchilov & Hutter, 2019](https://arxiv.org/pdf/1711.05101)) is that adding L2 regularisation to the loss and then running Adam is not the same as applying weight decay directly to the weights. Standard Adam scales the weight decay term by the adaptive learning rate, making its effective magnitude inconsistent across parameters and training steps. AdamW fixes this by applying weight decay as a direct multiplicative shrinkage of the weight, completely separate from the gradient update.

**Header**: `include/optim/adamw.hpp`  
**Namespace**: `gradientcore::optim`

:::info What it calls
`AdamW::step` applies weight decay with a direct scalar multiplication (`w -= lr * weight_decay * w`), then runs the standard Adam first/second moment update. All operations are in-place on the parameter and state tensors, with no temporary allocations needed.
:::

---

## Update Rule

```
# 1. Decoupled weight decay (applied first, directly to the weight)
w_t = w_{t-1} - lr * λ * w_{t-1}

# 2. Standard Adam gradient update
g_t = ∇L(w_{t-1})
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
w_t = w_t - lr * m̂_t / (√v̂_t + ε)
```

The critical difference from vanilla Adam + L2:

| Adam + L2 loss | AdamW |
|---|---|
| `g = ∇L + λ * w` | `g = ∇L` |
| `w -= lr * m̂(g) / (√v̂(g) + ε)` | `w -= lr * λ * w` then `w -= lr * m̂(∇L) / (√v̂(∇L) + ε)` |
| Weight decay scaled by adaptive LR | Weight decay at fixed effective rate |

In AdamW, `λ * w` never touches the moment estimators. This means the regularisation strength is consistent regardless of gradient magnitude — exactly what you want.

---

## Constructor

```cpp
optim::AdamW(Arena *perm_arena,
             const std::vector<autograd::Variable *> &params,
             float lr           = 0.001f,
             float beta1        = 0.9f,
             float beta2        = 0.999f,
             float eps          = 1e-8f,
             float weight_decay = 0.01f);
```

| Parameter | Default | Description |
|---|---|---|
| `perm_arena` | — | Permanent arena. Moment tensors live here. |
| `params` | — | Learnable parameters from `model->parameters()`. |
| `lr` | `0.001` | Learning rate. |
| `beta1` | `0.9` | First moment decay. |
| `beta2` | `0.999` | Second moment decay. |
| `eps` | `1e-8` | Denominator stability constant. |
| `weight_decay` | `0.01` | Decoupled weight decay coefficient (λ). |

```cpp
// Default weight decay
optim::AdamW adamw(perm_arena, params);

// Stronger regularisation
optim::AdamW adamw(perm_arena, params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.1f);

// No regularisation (same as Adam)
optim::AdamW adamw(perm_arena, params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
```

---

## Methods

### `step(Arena *temp_arena = nullptr)`

```cpp
void step(Arena *temp_arena = nullptr);
```

Applies one AdamW update. `temp_arena` is unused — no temporary allocations needed.

For each trainable parameter at step `t`:

```cpp
float m_hat_correction = 1.0f / (1.0f - pow(beta1, t));
float v_hat_correction = 1.0f / (1.0f - pow(beta2, t));

for each element k:
    g = p->grad[k]
    w = p->data[k]

    // Step 1: decoupled weight decay
    p->data[k] -= lr * weight_decay * w

    // Step 2: Adam update
    m[k] = beta1 * m[k] + (1 - beta1) * g
    v[k] = beta2 * v[k] + (1 - beta2) * g * g
    m_hat = m[k] * m_hat_correction
    v_hat = v[k] * v_hat_correction
    p->data[k] -= lr * m_hat / (sqrt(v_hat) + eps)
```

Note that `w` (the value before decay) is captured before the decay step — the gradient update uses the gradient of the loss, not of the regularised objective.

### `zero_grad()`

```cpp
void zero_grad();
```

Zeroes every parameter's `grad` tensor. Call before each `backward()`.

---

## Full Training Loop Example

The California Housing tutorial uses AdamW + HuberLoss:

```cpp
auto* perm  = Arena::create(MiB(1024), MiB(64), true);
auto* graph = Arena::create(MiB(512),  MiB(32), true);

nn::Sequential seq;
auto* l1   = perm->push<nn::Linear>();     new (l1)   nn::Linear(perm, 8, 128);
auto* bn1  = perm->push<nn::BatchNorm1d>(); new (bn1) nn::BatchNorm1d(perm, 128);
auto* r1   = perm->push<nn::ReLU>();       new (r1)   nn::ReLU();
auto* l2   = perm->push<nn::Linear>();     new (l2)   nn::Linear(perm, 128, 64);
auto* r2   = perm->push<nn::ReLU>();       new (r2)   nn::ReLU();
auto* l3   = perm->push<nn::Linear>();     new (l3)   nn::Linear(perm, 64, 1);
seq.add(l1); seq.add(bn1); seq.add(r1);
seq.add(l2); seq.add(r2); seq.add(l3);

optim::AdamW adamw(perm, seq.parameters(), 0.001f);
nn::HuberLoss criterion;

for (int epoch = 0; epoch < 200; epoch++) {
    loader->reset(true);
    while (loader->has_next()) {
        uint64_t pos = graph->get_pos();
        auto batch = loader->next(graph);

        auto* x    = autograd::create_leaf(graph, batch.features, false);
        auto* y    = autograd::create_leaf(graph, batch.labels,   false);
        auto* pred = seq.forward(graph, x);
        auto* loss = criterion.forward(graph, pred, y);

        adamw.zero_grad();
        autograd::backward(graph, loss);
        adamw.step();

        graph->pop_to(pos);
    }
}
```

---

## Weight Decay: How Much?

`weight_decay = 0.01` is the recommended default for most tasks. Intuition:

- **Too small** (`< 0.001`): Barely any regularisation — behaves like Adam.
- **Default** (`0.01`): Mild regularisation; good starting point for most experiments.
- **Moderate** (`0.1`): Noticeably shrinks weights; helpful for noisy data or large models.
- **Too large** (`> 1.0`): Weights collapse toward zero; training will fail.

Weight decay has a stronger relative effect at the start of training (when weights are large) and a weaker effect later (when weights have settled). This is intentional — it acts as a soft constraint that keeps the weight magnitudes from growing unbounded.

:::tip BatchNorm parameters and weight decay
It is generally considered incorrect practice to apply weight decay to `BatchNorm`'s `gamma` and `beta` parameters. AdamW in GradCore-Tensor applies weight decay to **all** registered parameters, including BatchNorm's. If you need to exclude specific parameters, construct the optimizer with two separate parameter lists (one with weight decay, one without) and use two optimizer instances — or filter the parameter list manually before construction.
:::

---

## AdamW vs Adam

Use **AdamW** when:
- You care about generalisation and want well-calibrated regularisation.
- Your model tends to overfit.
- You are following modern best practices (AdamW is the standard in most current research).

Use **Adam** when:
- You are experimenting quickly and don't want to tune weight decay.
- You know regularisation is not needed for your task.

In practice, for the same effective regularisation strength, AdamW with `weight_decay=λ` will outperform Adam with an L2 loss term of strength `λ`, because AdamW's decay is not distorted by the adaptive learning rate.