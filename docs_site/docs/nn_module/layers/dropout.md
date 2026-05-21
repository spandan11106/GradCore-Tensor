---
sidebar_position: 4
title: "Dropout"
---

# `nn::Dropout`

Dropout randomly zeroes out a fraction of activations during training. Each element is independently set to zero with probability `p`, and the surviving elements are scaled up by `1/(1-p)` to keep the expected sum the same. During evaluation, dropout is a complete no-op — every neuron is active.

**Header**: `include/nn/layers/dropout.hpp`  
**Inherits**: `nn::Module`

:::info What it calls
`Dropout::forward` uses `prng::randf()` (the thread-local PCG generator) to generate a uniform random value per element and applies the mask manually. It does **not** call an `autograd::dropout` op — the output is returned as an `autograd::create_leaf` with `requires_grad = true`, so backpropagation flows through the surviving (non-zeroed) elements correctly.
:::

---

## Constructor

```cpp
nn::Dropout(float dropout_prob = 0.5f);
```

| Parameter | Description |
|---|---|
| `dropout_prob` | Probability of zeroing each element. Must be in `[0, 1]`. Default `0.5`. |

```cpp
auto* drop = perm_arena->push<nn::Dropout>();
new (drop) nn::Dropout(0.3f);   // 30% of activations zeroed each forward pass
model.add_layer(drop);
```

A warning is printed if `p` is outside `[0, 1]`.

---

## Forward Pass

```cpp
autograd::Variable *forward(Arena *compute_arena,
                            autograd::Variable *x) override;
```

### Training mode (`is_training() == true`)

For each element:

```
mask_i = uniform(0, 1) < (1 - p)   →  1 (keep) or 0 (drop)
out_i  = x_i * mask_i / (1 - p)    →  scaled output or 0
```

The `1/(1-p)` scaling is **inverted dropout** — it ensures the expected value of each surviving output equals the input value, so you don't need to rescale at inference time.

```
Example (p = 0.5):
input:  [1.0, 2.0, 3.0, 4.0]
mask:   [1,   0,   1,   0  ]    (random)
scaled: [2.0, 0.0, 6.0, 0.0]   (survivors × 1/(1-0.5) = ×2)
```

### Eval mode (`is_training() == false`)

Returns `x` unchanged. No masking, no scaling, no randomness. This is the correct behaviour for inference.

### Edge cases

| Condition | Behaviour |
|---|---|
| `p == 0.0` | Returns `x` unchanged (nothing to drop, even in training) |
| `p == 1.0` | Zeros everything — mathematically valid but your network will learn nothing |
| Null input | Returns `nullptr` and prints an error |

---

## Inverted Dropout: Why Scale During Training?

The alternative approach (not used here) is to scale at inference time by multiplying all outputs by `(1 - p)`. Both approaches produce the same expected values, but inverted dropout is preferred because:

1. Inference is simpler — no scaling needed, the model "just works."
2. It's easier to swap between training and eval modes without forgetting to adjust scale.

---

## Placement in a Network

Dropout is typically placed **after** an activation and **before** the next linear layer:

```
Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → output
```

For small networks (like the MNIST MLP in the tutorials), dropout may be unnecessary. For larger networks or when overfitting is observed, `p = 0.2` to `p = 0.5` is a reasonable starting range.

```cpp
auto* l1   = perm->push<nn::Linear>();  new (l1)   nn::Linear(perm, 784, 512);
auto* r1   = perm->push<nn::ReLU>();    new (r1)   nn::ReLU();
auto* d1   = perm->push<nn::Dropout>(); new (d1)   nn::Dropout(0.3f);
auto* l2   = perm->push<nn::Linear>();  new (l2)   nn::Linear(perm, 512, 10);

model.add_layer(l1);
model.add_layer(r1);
model.add_layer(d1);
model.add_layer(l2);
```

---

## Effect on Gradient Flow

During training, elements that were zeroed by Dropout receive zero gradient from downstream — the zero in the forward pass propagates back as zero. This is handled implicitly because the zeroed output values produce zero contributions to the loss.

The surviving elements receive their upstream gradient scaled by `1/(1-p)`, matching the forward scaling. The net effect is that gradients are unbiased estimators of the full-network gradient.

---

## Reproducibility

Dropout uses the thread-local PRNG (`prng::randf()`). The mask changes every forward call. For reproducible results, seed the PRNG before training:

```cpp
prng::seed(42, 1);
model.train(X_train, Y_train);
```

See [PRNG](/docs/tensor-module/memory/prng) for details.

---

## Parameters

Dropout has **zero learnable parameters**. It does not appear in `model.num_parameters()` and contributes nothing to the save file.