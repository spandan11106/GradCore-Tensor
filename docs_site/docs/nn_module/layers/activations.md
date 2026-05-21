---
sidebar_position: 2
title: "Activation Layers"
---

# Activation Layers

Activation layers are thin `nn::Module` wrappers around the `autograd::*` activation functions. They hold no parameters and no state — their only job is to sit in a `Sequential` and call the right autograd op when `forward()` is invoked.

**Header**: `include/nn/layers/activations.hpp`  
**Inherits**: `nn::Module` (all of them)

:::info What they call
Each activation layer's `forward` calls the corresponding `autograd::*` function (e.g. `autograd::relu`), which calls `tensor_relu` from the tensor module and wires up a backward node on the graph arena. No tensor logic lives in the `nn` layer itself.
:::

---

## Construction Pattern

All activation layers are constructed on the permanent arena:

```cpp
auto* relu = perm_arena->push<nn::ReLU>();
new (relu) nn::ReLU();
model.add_layer(relu);
```

None of them take constructor arguments except `Softmax` (dim), `LeakyReLU` (alpha), and `ELU` (alpha).

---

## `nn::Identity`

```cpp
class Identity : public Module;
```

Returns its input unchanged. Useful as a placeholder when you want to conditionally skip an activation:

```cpp
nn::Module* act = use_activation ? (nn::Module*) relu : (nn::Module*) identity;
```

No parameters, no state, no overhead. Just a pass-through.

---

## `nn::ReLU`

```cpp
class ReLU : public Module;
// Constructor: ReLU()
```

`f(x) = max(0, x)`

The default choice for hidden layers in MLPs and CNNs. Fast to compute, simple gradient (`1` where `x > 0`, `0` elsewhere). Kaiming Normal initialisation is designed specifically for ReLU.

```cpp
auto* r = perm->push<nn::ReLU>(); new (r) nn::ReLU();
```

---

## `nn::ReLU6`

```cpp
class ReLU6 : public Module;
// Constructor: ReLU6()
```

`f(x) = min(max(0, x), 6)`

ReLU with an output ceiling of 6. Used in quantisation-friendly architectures (MobileNet). Prevents very large activations from causing large fixed-point representation errors.

---

## `nn::LeakyReLU`

```cpp
class LeakyReLU : public Module;
// Constructor: LeakyReLU(float alpha = 0.01f)
```

`f(x) = x if x > 0 else alpha * x`

Gives negative inputs a small non-zero gradient, preventing the "dead neuron" problem where a ReLU unit outputs 0 forever. Typical `alpha` values: `0.01` (default), `0.1`, `0.2`.

```cpp
auto* lrelu = perm->push<nn::LeakyReLU>();
new (lrelu) nn::LeakyReLU(0.1f);
```

---

## `nn::ELU`

```cpp
class ELU : public Module;
// Constructor: ELU(float alpha = 1.0f)
```

`f(x) = x if x > 0 else alpha * (exp(x) - 1)`

Smooth for negative inputs. Can push mean activations closer to zero, which can speed up learning. More expensive than ReLU due to `exp`.

```cpp
auto* elu = perm->push<nn::ELU>();
new (elu) nn::ELU(1.0f);
```

---

## `nn::GELU`

```cpp
class GELU : public Module;
// Constructor: GELU()
```

`f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

The activation of choice in Transformers. Smooth everywhere, probabilistically gates the input. More expensive than ReLU but frequently achieves better results on NLP tasks.

```cpp
auto* gelu = perm->push<nn::GELU>(); new (gelu) nn::GELU();
```

---

## `nn::Swish`

```cpp
class Swish : public Module;
// Constructor: Swish()
```

`f(x) = x * σ(x) = x / (1 + exp(-x))`

Self-gated, smooth, non-monotonic. Used in EfficientNet and MobileNetV3. Performs comparably to GELU on many tasks.

---

## `nn::HardSigmoid`

```cpp
class HardSigmoid : public Module;
// Constructor: HardSigmoid()
```

`f(x) = clamp((x + 3) / 6, 0, 1)`

Piecewise-linear sigmoid approximation. No `exp` — suitable for inference on hardware without floating-point `exp` units. Output is in `[0, 1]`.

---

## `nn::HardSwish`

```cpp
class HardSwish : public Module;
// Constructor: HardSwish()
```

`f(x) = x * HardSigmoid(x)`

Piecewise-linear Swish approximation. Used in MobileNetV3 for inference efficiency.

---

## `nn::Sigmoid`

```cpp
class Sigmoid : public Module;
// Constructor: Sigmoid()
```

`f(x) = 1 / (1 + exp(-x))`

Maps to `(0, 1)`. Use as the output activation for **binary classification** (paired with `BCELoss`) or gating mechanisms. Avoid in hidden layers — vanishing gradients for large `|x|`.

---

## `nn::Tanh`

```cpp
class Tanh : public Module;
// Constructor: Tanh()
```

`f(x) = tanh(x)`

Maps to `(-1, 1)`. Zero-centred unlike Sigmoid. Still suffers vanishing gradients at extremes but less severely. Common in RNNs.

---

## `nn::Softmax`

```cpp
class Softmax : public Module;
// Constructor: Softmax(int32_t dim = -1)
```

`f(x)_i = exp(x_i) / Σ exp(x_j)`

Converts logits to a probability distribution. `dim` selects the axis to normalise over; `-1` means the last dimension.

```cpp
auto* sm = perm->push<nn::Softmax>();
new (sm) nn::Softmax(-1);   // normalise over last dim
```

:::warning Do not use with CrossEntropyLoss
`CrossEntropyLoss` applies softmax internally. Adding `nn::Softmax` before it means applying softmax twice — your loss will be wrong and you'll spend an evening wondering why accuracy plateaus at 10%.
:::

---

## `nn::SoftPlus`

```cpp
class SoftPlus : public Module;
// Constructor: SoftPlus()
```

`f(x) = log(1 + exp(x))`

Smooth approximation to ReLU. Output is always positive, unlike ReLU which clips at exactly 0. Occasionally used in probabilistic output layers.

---

## Quick Reference

| Class | Formula | Constructor args | Typical use |
|---|---|---|---|
| `Identity` | `x` | none | Placeholder |
| `ReLU` | `max(0, x)` | none | Hidden layers (default) |
| `ReLU6` | `clamp(x, 0, 6)` | none | Mobile / quantised |
| `LeakyReLU` | `x or α*x` | `alpha=0.01` | Avoiding dead neurons |
| `ELU` | `x or α*(eˣ-1)` | `alpha=1.0` | Smooth negative region |
| `GELU` | `~x*Φ(x)` | none | Transformers / NLP |
| `Swish` | `x*σ(x)` | none | EfficientNet |
| `HardSigmoid` | `clamp((x+3)/6)` | none | Fast inference |
| `HardSwish` | `x*HardSigmoid(x)` | none | Fast inference |
| `Sigmoid` | `1/(1+e⁻ˣ)` | none | Binary output, gates |
| `Tanh` | `tanh(x)` | none | RNNs |
| `Softmax` | `eˣⁱ/Σeˣʲ` | `dim=-1` | Multi-class output |
| `SoftPlus` | `log(1+eˣ)` | none | Probabilistic outputs |