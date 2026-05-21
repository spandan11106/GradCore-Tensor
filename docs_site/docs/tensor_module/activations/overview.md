---
sidebar_position: 1
title: "Activation Functions"
---

# Activation Functions

Activation functions introduce non-linearity into your network. Without them, a 47-layer neural network would have the expressive power of a single matrix multiply — and we'd all be out of a hobby.

All activation functions have the same signature:

```cpp
bool tensor_<name>(Tensor *out, const Tensor *in, [optional params...]);
```

And their backward counterparts:

```cpp
bool tensor_<name>_grad(Tensor *out, const Tensor *in, const Tensor *grad, [optional params...]);
```

The `_grad` functions compute the **local gradient**, i.e. `d_activation/d_input * upstream_grad`. The autograd engine calls these during `backward()`.

All functions support contiguous tensors (fast path) and non-contiguous tensors (stride-aware path). All are parallelised with OpenMP when compiled with `-fopenmp`.

---

## ReLU

```
f(x) = max(0, x)
f'(x) = 1 if x > 0 else 0
```

```cpp
bool tensor_relu(Tensor *out, const Tensor *in);
bool tensor_relu_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

The workhorse. Fast, simple, and effective for most tasks. The dead neuron problem (neurons that always output 0) is real but manageable with careful initialisation (Kaiming Normal is designed for ReLU).

**nn layer**: `nn::ReLU`

---

## ReLU6

```
f(x) = min(max(0, x), 6)
f'(x) = 1 if 0 < x ≤ 6 else 0
```

```cpp
bool tensor_relu6(Tensor *out, const Tensor *in);
bool tensor_relu6_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

ReLU capped at 6. Designed for fixed-point quantisation (MobileNet uses this). The ceiling prevents very large activations from dominating representations.

**nn layer**: `nn::ReLU6`

---

## Leaky ReLU

```
f(x) = x       if x > 0
f(x) = α * x   otherwise
```

```cpp
bool tensor_leaky_relu(Tensor *out, const Tensor *in, float alpha);
bool tensor_leaky_relu_grad(Tensor *out, const Tensor *in, const Tensor *grad, float alpha);
```

Solves the dead neuron problem by giving negative inputs a small but non-zero gradient (`alpha`, typically 0.01). Neurons can still learn even when they're outputting negative values.

**nn layer**: `nn::LeakyReLU(float alpha = 0.01f)`

---

## ELU (Exponential Linear Unit)

```
f(x) = x                   if x > 0
f(x) = α * (exp(x) - 1)    otherwise
```

```cpp
bool tensor_elu(Tensor *out, const Tensor *in, float alpha);
bool tensor_elu_grad(Tensor *out, const Tensor *in, const Tensor *grad, float alpha);
```

Smooth for negative inputs, which can help with gradient flow compared to the hard kink of Leaky ReLU. Slightly more expensive due to `exp`.

**nn layer**: `nn::ELU(float alpha = 1.0f)`

---

## GELU (Gaussian Error Linear Unit)

```
f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

```cpp
bool tensor_gelu(Tensor *out, const Tensor *in);
bool tensor_gelu_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

The activation of choice in Transformers (BERT, GPT, etc.). Smooth everywhere, probabilistically gates inputs based on their magnitude. More expensive than ReLU but often trains better for NLP tasks.

The implementation uses the tanh approximation rather than the exact `erf` form — it's faster and the difference is negligible in practice.

**nn layer**: `nn::GELU`

---

## Swish

```
f(x) = x * σ(x) = x / (1 + exp(-x))
f'(x) = σ(x) * (1 + x * (1 - σ(x)))
```

```cpp
bool tensor_swish(Tensor *out, const Tensor *in);
bool tensor_swish_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

Self-gated: the input gates itself via sigmoid. Smooth, non-monotonic (it dips slightly below zero around x ≈ -1.3), and performs comparably to GELU on many benchmarks. Used in EfficientNet and MobileNetV3.

**nn layer**: `nn::Swish`

---

## Hard Sigmoid

```
f(x) = min(max(0, x + 3), 6) / 6
f'(x) = 1/6   if -3 < x < 3, else 0
```

```cpp
bool tensor_hard_sigmoid(Tensor *out, const Tensor *in);
bool tensor_hard_sigmoid_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

A piecewise-linear approximation to sigmoid. Much faster (no `exp`), suitable for mobile/embedded inference. Output is in [0, 1].

**nn layer**: `nn::HardSigmoid`

---

## Hard Swish

```
f(x) = x * HardSigmoid(x)
f'(x) = HardSwish'(x)  (piecewise, see source)
```

```cpp
bool tensor_hard_swish(Tensor *out, const Tensor *in);
bool tensor_hard_swish_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

Piecewise-linear approximation to Swish. Zero computation cost compared to the real thing on hardware without `exp` units.

```
derivative:
  x ≤ -3:  0
  x ≥  3:  1
  else:    (2x + 3) / 6
```

**nn layer**: `nn::HardSwish`

---

## Sigmoid

```
f(x) = 1 / (1 + exp(-x))
f'(x) = σ(x) * (1 - σ(x))
```

```cpp
bool tensor_sigmoid(Tensor *out, const Tensor *in);
bool tensor_sigmoid_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

Maps any real number to (0, 1). Used in binary classification output layers and gating mechanisms. Suffers from vanishing gradients for large `|x|` (gradient approaches 0), which is why ReLU-family activations are preferred in hidden layers.

**nn layer**: `nn::Sigmoid`

---

## Tanh

```
f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
f'(x) = 1 - tanh²(x)
```

```cpp
bool tensor_tanh(Tensor *out, const Tensor *in);
bool tensor_tanh_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

Maps to (-1, 1) — zero-centred unlike sigmoid. Still suffers from vanishing gradients at extremes. Common in RNNs, occasionally used in regression output layers.

**nn layer**: `nn::Tanh`

---

## Softmax

```
f(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

```cpp
bool tensor_softmax(Tensor *out, const Tensor *in, int32_t dim = -1);
bool tensor_softmax_grad(Tensor *out, const Tensor *softmax_out, const Tensor *grad, int32_t dim = -1);
```

Converts a vector of logits into a probability distribution (outputs sum to 1, all positive). The `max(x)` subtraction is the **log-sum-exp trick** — numerically critical for avoiding `exp` overflow:

```
Without trick: exp(1000) → inf
With trick:    exp(1000 - 1000) = exp(0) = 1  ✓
```

The `dim` parameter selects which axis to normalise over. `-1` means the last dimension (standard for classification logits).

:::note CrossEntropyLoss includes Softmax
`tensor_cross_entropy_loss` applies softmax internally. Do **not** add a `Softmax` layer before `CrossEntropyLoss` — you'll be applying it twice and your loss will be wrong (and confusingly finite).
:::

**Backward pass**: The Jacobian of softmax is:

```
∂f_i/∂x_j = f_i * (δ_ij - f_j)
```

The efficient form sums the upstream gradient first:

```
out_i = f_i * (grad_i - Σ_j(f_j * grad_j))
```

**nn layer**: `nn::Softmax(int32_t dim = -1)`

---

## SoftPlus

```
f(x) = log(1 + exp(x))
f'(x) = σ(x) = 1 / (1 + exp(-x))
```

```cpp
bool tensor_softplus(Tensor *out, const Tensor *in);
bool tensor_softplus_grad(Tensor *out, const Tensor *in, const Tensor *grad);
```

Smooth approximation to ReLU. Rarely used in hidden layers but appears in some probabilistic models. Uses `log1p(exp(x))` for numerical stability, with a linear approximation (`f(x) ≈ x`) for large values to avoid overflow.

**nn layer**: `nn::SoftPlus`

---

## Activation Comparison

| Activation | Range | Smooth? | Zero-centred? | Typical use |
|---|---|---|---|---|
| ReLU | [0, ∞) | No | No | Hidden layers (general) |
| ReLU6 | [0, 6] | No | No | Mobile/quantised models |
| LeakyReLU | (-∞, ∞) | No | Yes | When dead neurons are a concern |
| ELU | (-α, ∞) | Yes | Approx. | Faster convergence in some tasks |
| GELU | (-∞, ∞) | Yes | Approx. | Transformers, NLP |
| Swish | (-∞, ∞) | Yes | Approx. | EfficientNet, modern CNNs |
| HardSigmoid | [0, 1] | No | No | Fast mobile inference |
| HardSwish | (-∞, ∞) | No | Approx. | Fast mobile inference |
| Sigmoid | (0, 1) | Yes | No | Binary outputs, gates |
| Tanh | (-1, 1) | Yes | Yes | RNNs, some outputs |
| Softmax | (0, 1) | Yes | N/A | Multi-class output |
| SoftPlus | (0, ∞) | Yes | No | Probabilistic models |
