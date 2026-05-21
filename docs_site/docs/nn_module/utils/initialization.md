---
sidebar_position: 1
title: "Initialization"
---

# Weight Initialisation

Good weight initialisation is not optional — initialise too small and gradients vanish, too large and they explode. The functions in `nn::init` give you the standard schemes that the deep learning literature has converged on over the past decade.

**Header**: `include/nn/utils/initialization.hpp`  
**Namespace**: `gradientcore::nn::init`

:::info What they call
All initialisation functions write directly into a `Variable`'s underlying `Tensor` data buffer using `prng::randf()` or `prng::std_norm()` from the PRNG module. They modify the tensor in-place and return nothing. They do **not** build autograd graph nodes — initialisation happens before training, outside the computation graph.
:::

---

## General Signature

All initialisation functions take a single `autograd::Variable*`:

```cpp
void scheme_(autograd::Variable *weight);
```

The trailing underscore (`_`) follows the PyTorch convention indicating an in-place operation. The function modifies `weight->data` directly.

All functions print an error and return early if `weight` or `weight->data` is null.

---

## Kaiming (He) Initialisation

Designed for **ReLU-family activations**. Accounts for the fact that ReLU zeroes out roughly half of all activations, which would halve the variance at each layer if not corrected.

### `kaiming_normal_`

```cpp
void kaiming_normal_(autograd::Variable *weight);
```

Draws weights from a normal distribution:

```
std = sqrt(2 / fan_in)
W ~ N(0, std²)
```

The `sqrt(2)` factor is the He correction for ReLU. `fan_in` is the number of input units to the layer (i.e. `in_features` for a `Linear` layer).

**This is the default initialisation for `nn::Linear`.**

```cpp
init::kaiming_normal_(layer->weight);
```

### `kaiming_uniform_`

```cpp
void kaiming_uniform_(autograd::Variable *weight);
```

Draws weights uniformly:

```
limit = sqrt(6 / fan_in)
W ~ Uniform(-limit, limit)
```

Mathematically equivalent variance to `kaiming_normal_` but with bounded support. Useful when you want to guarantee no single initial weight is very large.

---

## Xavier (Glorot) Initialisation

Designed for **symmetric activations** (Tanh, Sigmoid) where the gradient is approximately linear near zero. Xavier initialisation keeps variance consistent both in the forward pass and the backward pass simultaneously.

### `xavier_normal_`

```cpp
void xavier_normal_(autograd::Variable *weight);
```

```
std = sqrt(2 / (fan_in + fan_out))
W ~ N(0, std²)
```

`fan_in` and `fan_out` are the number of input and output units respectively. For a `Linear(M, N)` layer: `fan_in = M`, `fan_out = N`.

```cpp
init::xavier_normal_(layer->weight);
```

### `xavier_uniform_`

```cpp
void xavier_uniform_(autograd::Variable *weight);
```

```
limit = sqrt(6 / (fan_in + fan_out))
W ~ Uniform(-limit, limit)
```

The original Glorot & Bengio (2010) formulation. Still widely used as a general-purpose default, especially with Tanh activations.

---

## Simple Distributions

### `uniform_`

```cpp
void uniform_(autograd::Variable *weight,
              float min_val = -1.0f,
              float max_val =  1.0f);
```

Fills with values drawn uniformly from `[min_val, max_val]`:

```
W ~ Uniform(min_val, max_val)
```

```cpp
init::uniform_(layer->weight, -0.1f, 0.1f);
```

### `normal_`

```cpp
void normal_(autograd::Variable *weight,
             float mean = 0.0f,
             float std  = 1.0f);
```

Fills with values drawn from a normal distribution:

```
W ~ N(mean, std²)
```

Uses the Box-Muller transform via `prng::std_norm()`, scaled and shifted.

```cpp
init::normal_(layer->weight, 0.0f, 0.02f);   // typical for small init
```

---

## Constant Initialisation

### `constant_`

```cpp
void constant_(autograd::Variable *weight, float value = 0.0f);
```

Sets every element to `value`.

```cpp
init::constant_(layer->bias, 0.1f);   // small positive bias for ReLU
```

### `zeros_`

```cpp
inline void zeros_(autograd::Variable *weight) {
    constant_(weight, 0.0f);
}
```

Sets all elements to zero. Used for bias initialisation (zero bias is the standard default).

```cpp
init::zeros_(layer->bias);
```

### `ones_`

```cpp
inline void ones_(autograd::Variable *weight) {
    constant_(weight, 1.0f);
}
```

Sets all elements to one. Used internally to initialise `BatchNorm`'s `gamma` (scale) parameter — the identity transform is the right starting point before the network has learned to rescale.

```cpp
init::ones_(bn->gamma);
init::zeros_(bn->beta);
```

---

## Fan Calculation

All Kaiming and Xavier functions use an internal `calculate_fans` helper:

```cpp
static void calculate_fans(const Tensor *tensor,
                            uint32_t &fan_in,
                            uint32_t &fan_out);
```

| Tensor shape | `fan_in` | `fan_out` |
|---|---|---|
| `[M, N]` (2D — Linear weights) | `M` | `N` |
| `[N]` (1D) | `N` | `N` |
| Higher-dimensional | Last dim | Product of all other dims |

For `nn::Linear(in=784, out=128)`, the weight shape is `[784, 128]`, so `fan_in = 784`, `fan_out = 128`.

---

## Choosing an Initialisation Scheme

| Activation | Recommended scheme |
|---|---|
| ReLU, ReLU6, LeakyReLU, ELU | `kaiming_normal_` or `kaiming_uniform_` |
| GELU, Swish | `kaiming_normal_` (works well in practice) |
| Tanh, Sigmoid | `xavier_normal_` or `xavier_uniform_` |
| Linear output (regression) | `xavier_uniform_` or `normal_(0, 0.01)` |
| Custom / experimental | `uniform_` or `normal_` with manual scale |
| Biases | `zeros_` (universal default) |
| BatchNorm gamma | `ones_` |
| BatchNorm beta | `zeros_` |

---

## Re-initialising a Layer After Construction

`nn::Linear` calls `reset_parameters()` in its constructor, which applies `kaiming_normal_` to weights and `zeros_` to bias. You can override this after construction:

```cpp
auto* l = perm->push<nn::Linear>();
new (l) nn::Linear(perm, 784, 128);   // kaiming_normal applied here

// Override with Xavier for a Tanh network:
init::xavier_uniform_(l->weight);
init::zeros_(l->bias);

model.add_layer(l);
```

---

## Implementation Notes

### Box-Muller Transform

`kaiming_normal_`, `xavier_normal_`, and `normal_` all use Box-Muller to generate normal random numbers from pairs of uniform samples:

```
u1, u2 ~ Uniform(0, 1)
z0 = sqrt(-2 * log(u1)) * cos(2π * u2)
z1 = sqrt(-2 * log(u1)) * sin(2π * u2)
```

Both `z0` and `z1` are standard normal. The initialisation loops stride by 2 and use both values to avoid wasting half the computation:

```cpp
for (uint64_t i = 0; i < size; i += 2) {
    float u1 = prng::randf();
    float u2 = prng::randf();
    if (u1 < 1e-7f) u1 = 1e-7f;   // guard against log(0)

    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    data[i] = z0 * std_dev;

    if (i + 1 < size) {
        float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * M_PI * u2);
        data[i + 1] = z1 * std_dev;
    }
}
```

If `size` is odd, the last element uses only `z0`.

### Reproducibility

Initialisation uses the global thread-local PRNG. Seed it before constructing your model for reproducible results:

```cpp
prng::seed(42, 1);   // fixed seed

nn::Model model(perm, graph);
// ... add layers — weights will be the same every run
```

See [PRNG](/docs/tensor-module/memory/prng) for details.