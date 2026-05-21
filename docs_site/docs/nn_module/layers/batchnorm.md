---
sidebar_position: 3
title: "BatchNorm"
---

# `nn::BatchNorm1d` / `nn::BatchNorm2d`

Batch Normalisation normalises a layer's inputs across the batch dimension, then rescales and shifts with learned parameters. The result is faster training, reduced sensitivity to initialisation, and a mild regularisation effect — which is a very good return on investment for two extra tensors.

**Header**: `include/nn/layers/batchnorm.hpp`  
**Inherits**: `nn::Module`

:::info What it calls
`BatchNorm1d::forward` does not call a dedicated `autograd::batch_norm` — it implements the normalisation computation directly and returns an `autograd::Variable` leaf (created via `autograd::create_leaf`) with `requires_grad = true`. The gamma and beta parameters are registered via `register_parameter` and receive gradients through the standard backpropagation mechanism.
:::

---

## `nn::BatchNorm1d`

For 1D feature inputs — i.e. outputs of `nn::Linear`.

### Constructor

```cpp
nn::BatchNorm1d(Arena *perm_arena,
                uint32_t num_features,
                float momentum = 0.1f,
                float epsilon  = 1e-5f);
```

| Parameter | Description |
|---|---|
| `perm_arena` | Permanent arena — parameters and running stats live here. |
| `num_features` | Number of feature channels to normalise. Must match the second dimension of the input: `[batch, num_features]`. |
| `momentum` | Exponential moving average factor for updating running stats. `0.1` = 10% new, 90% old each batch. |
| `epsilon` | Small constant added to variance before taking the square root. Prevents division by zero. |

### What gets allocated

| Tensor | Shape | Initialisation | Trainable |
|---|---|---|---|
| `gamma` (scale) | `[1, num_features]` | Ones | ✓ |
| `beta` (shift) | `[1, num_features]` | Zeros | ✓ |
| `running_mean` | `[1, num_features]` | Zeros | ✗ |
| `running_var` | `[1, num_features]` | Ones | ✗ |

`running_mean` and `running_var` are not updated by the optimiser — they're updated by `forward()` itself during training.

### Forward pass

```cpp
autograd::Variable *forward(Arena *compute_arena,
                            autograd::Variable *x) override;
```

**Input**: `x` must be 2D: `[batch_size, num_features]`.

#### Training mode

Computes batch statistics and normalises:

```
μ_j   = mean of x[:, j] over the batch
σ²_j  = variance of x[:, j] over the batch
x̂_ij  = (x_ij - μ_j) / sqrt(σ²_j + ε)
out   = gamma * x̂ + beta        (broadcast over batch)
```

Then updates running statistics for inference:

```
running_mean = (1 - momentum) * running_mean + momentum * μ
running_var  = (1 - momentum) * running_var  + momentum * σ²
```

#### Eval mode

Uses stored running statistics (no batch-level computation):

```
x̂ = (x - running_mean) / sqrt(running_var + ε)
out = gamma * x̂ + beta
```

This ensures deterministic inference — the output depends only on the input and the learned parameters, not on what other samples happen to be in the batch.

### Construction example

```cpp
auto* bn = perm_arena->push<nn::BatchNorm1d>();
new (bn) nn::BatchNorm1d(perm_arena, 128);   // after a Linear(*, 128)
model.add_layer(bn);
```

### Placement in a network

```
Linear(8, 128) → BatchNorm1d(128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 1)
```

BatchNorm is placed **after** the linear layer and **before** the activation. This is the standard convention (though "after activation" is also sometimes used — both work in practice).

:::tip Skip the bias in Linear before BatchNorm
When `BatchNorm` follows `Linear`, the linear bias is redundant — `BatchNorm`'s `beta` parameter provides the same shift. Remove the bias to save parameters:

```cpp
new (l) nn::Linear(perm, 8, 128, /*use_bias=*/false);
new (bn) nn::BatchNorm1d(perm, 128);
```
:::

### Parameters

```
gamma:        num_features parameters
beta:         num_features parameters
total:        2 * num_features
running_mean: not counted (not trainable)
running_var:  not counted (not trainable)
```

---

## `nn::BatchNorm2d`

For 2D spatial inputs from convolutional layers — i.e. `[batch, channels, height, width]`.

### Constructor

```cpp
nn::BatchNorm2d(Arena *perm_arena,
                uint32_t num_features,
                float momentum = 0.1f,
                float epsilon  = 1e-5f);
```

`num_features` is the number of **channels** (dimension index 1).

### Forward pass

**Input**: `x` must be 4D: `[batch_size, channels, height, width]`.

Statistics are computed per-channel, averaging over `batch × height × width`:

```
μ_c   = mean over (batch, h, w) for channel c
σ²_c  = variance over (batch, h, w) for channel c
out_bchw = gamma_c * (x_bchw - μ_c) / sqrt(σ²_c + ε) + beta_c
```

### Construction example

```cpp
auto* bn2 = perm_arena->push<nn::BatchNorm2d>();
new (bn2) nn::BatchNorm2d(perm_arena, 64);   // after a Conv layer with 64 output channels
```

:::note No Conv layer yet
GradCore-Tensor does not currently include a `Conv2d` layer. `BatchNorm2d` is available for users who extend the library with custom convolutional operations.
:::

---

## Why BatchNorm Works

Without normalisation, activations can drift to very large or very small values as they pass through many layers — a phenomenon called **internal covariate shift**. This makes the loss landscape steep in some directions and flat in others, forcing the use of small learning rates and careful initialisation.

BatchNorm solves this by re-centering and re-scaling activations at every layer, every batch. The learned `gamma` and `beta` give the network the freedom to undo the normalisation if it turns out to be harmful — so the layer can only help, never hurt expressiveness.

---

## Common Mistakes

**Forgetting `eval()` before inference**: In eval mode, BatchNorm uses the running mean and variance accumulated during training. In training mode, it uses the current batch's statistics — which depend on whatever samples happened to be in that batch. Inference results will vary randomly from call to call. Always call `model.get_model()->eval()` before running inference.

**Using BatchNorm with batch size 1**: The variance of a single sample is 0. Division by `sqrt(0 + epsilon)` gives approximately `1/sqrt(1e-5)` ≈ 316, which is not what you want. BatchNorm requires a batch size of at least 2, and works best with larger batches (32+).