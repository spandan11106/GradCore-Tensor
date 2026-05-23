---
sidebar_position: 4
title: "Activation Operations"
---

# Activation Operations

Every activation function in GradCore-Tensor has an autograd-layer wrapper. Each wrapper computes the activation in the forward pass, saves whatever tensors its backward function needs, and registers a `backward_fn` that calls the corresponding `tensor_*_grad` function.

**Header**: `include/autograd/autograd.hpp`  
**Source**: `src/autograd/ops/activations/`

:::info What is saved for backward
Most activations save the **pre-activation input** (the input tensor `in->data`), because the gradient formula needs the original input values — for example, `relu_grad` checks which elements were positive, and `sigmoid_grad` recomputes `σ(x)` from the input. The one notable exception is `softmax`, which saves the **output** (the softmax probabilities) because its Jacobian is expressed more cleanly in terms of the output values.
:::

---

## Common Structure

Every activation op follows this pattern:

```cpp
Variable *relu(Arena *arena, Variable *in) {
    // 1. Allocate output tensor and compute forward value
    Tensor *out_data = tensor_create_zeros(arena, in->data->ndims, in->data->shape);
    tensor_relu(out_data, in->data);

    // 2. Allocate output Variable
    Variable *out = arena->push<Variable>();
    out->data          = out_data;
    out->requires_grad = in->requires_grad;
    out->is_leaf       = false;

    if (out->requires_grad) {
        // 3. Allocate grad tensor, wire parent, save input for backward
        out->grad            = tensor_create_zeros(arena, out_data->ndims, out_data->shape);
        out->num_parents     = 1;
        out->parents         = arena->push_array<Edge>(1);
        out->parents[0]      = {in};
        out->num_saved       = 1;
        out->saved_tensors   = arena->push_array<Tensor *>(1);
        out->saved_tensors[0] = in->data;

        // 4. Register backward function
        out->backward_fn = [](Variable *self, Arena *temp_arena) {
            Variable *parent = self->parents[0].node;
            if (!parent->requires_grad) return;
            Tensor *local_grad = tensor_create_zeros(
                temp_arena, parent->grad->ndims, parent->grad->shape);
            tensor_relu_grad(local_grad, self->saved_tensors[0], self->grad);
            tensor_add(parent->grad, parent->grad, local_grad);
        };
    }
    return out;
}
```

The structure is identical for every activation — only the forward call, the saved tensor choice, and the backward call differ.

---

## `relu`

```cpp
Variable *relu(Arena *arena, Variable *in);
```

`f(x) = max(0, x)`. Saves `in->data` (pre-activation) for the backward pass.

**Backward**: `tensor_relu_grad` — passes through the upstream gradient where the input was positive, zeros it where the input was negative or zero.

**nn layer**: `nn::ReLU`

---

## `relu6`

```cpp
Variable *relu6(Arena *arena, Variable *in);
```

`f(x) = clamp(x, 0, 6)`. Saves `in->data` for the backward.

**Backward**: `tensor_relu6_grad` — passes the upstream gradient only where `0 < x ≤ 6`, zeros elsewhere.

**nn layer**: `nn::ReLU6`

---

## `leaky_relu`

```cpp
Variable *leaky_relu(Arena *arena, Variable *in, float alpha);
```

`f(x) = x if x > 0, else alpha * x`. The `alpha` value is stored in `out->metadata_float` and passed to the backward function. Saves `in->data`.

**Backward**: `tensor_leaky_relu_grad(out, saved_input, upstream_grad, self->metadata_float)`.

**nn layer**: `nn::LeakyReLU(float alpha = 0.01f)`

---

## `elu`

```cpp
Variable *elu(Arena *arena, Variable *in, float alpha);
```

`f(x) = x if x > 0, else alpha * (exp(x) - 1)`. `alpha` stored in `metadata_float`. Saves `in->data`.

**Backward**: `tensor_elu_grad` — derivative is `1` for positive inputs and `alpha * exp(x)` for negative inputs.

**nn layer**: `nn::ELU(float alpha = 1.0f)`

---

## `gelu`

```cpp
Variable *gelu(Arena *arena, Variable *in);
```

`f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`. Saves `in->data`.

**Backward**: `tensor_gelu_grad` — derivative is computed analytically from the tanh approximation:

```
f'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t²) * (√(2/π) + C * x²)
where t = tanh(√(2/π) * (x + 0.044715 * x³))
      C = 3 * 0.044715 * √(2/π) ≈ 0.1070322243
```

**nn layer**: `nn::GELU`

---

## `swish`

```cpp
Variable *swish(Arena *arena, Variable *in);
```

`f(x) = x * σ(x) = x / (1 + exp(-x))`. Saves `in->data`.

**Backward**: `tensor_swish_grad` — derivative is `σ(x) * (1 + x * (1 - σ(x)))`.

**nn layer**: `nn::Swish`

---

## `sigmoid`

```cpp
Variable *sigmoid(Arena *arena, Variable *in);
```

`f(x) = 1 / (1 + exp(-x))`. Saves `in->data`.

**Backward**: `tensor_sigmoid_grad` — derivative is `σ(x) * (1 - σ(x))`, recomputed from the saved input.

**nn layer**: `nn::Sigmoid`

---

## `tanh`

```cpp
Variable *tanh(Arena *arena, Variable *in);
```

`f(x) = tanh(x)`. Saves `in->data`.

**Backward**: `tensor_tanh_grad` — derivative is `1 - tanh²(x)`.

**nn layer**: `nn::Tanh`

---

## `hard_sigmoid`

```cpp
Variable *hard_sigmoid(Arena *arena, Variable *in);
```

`f(x) = clamp((x + 3) / 6, 0, 1)`. Saves `in->data`.

**Backward**: `tensor_hard_sigmoid_grad` — derivative is `1/6` where `-3 < x < 3`, zero elsewhere.

**nn layer**: `nn::HardSigmoid`

---

## `hard_swish`

```cpp
Variable *hard_swish(Arena *arena, Variable *in);
```

`f(x) = x * HardSigmoid(x)`. Saves `in->data`.

**Backward**: `tensor_hard_swish_grad` — piecewise derivative:

```
x ≤ -3 :  0
x ≥  3 :  1
else   :  (2x + 3) / 6
```

**nn layer**: `nn::HardSwish`

---

## `softplus`

```cpp
Variable *softplus(Arena *arena, Variable *in);
```

`f(x) = log(1 + exp(x))` (with linear approximation for large `x`). Saves `in->data`.

**Backward**: `tensor_softplus_grad` — derivative is `σ(x) = 1 / (1 + exp(-x))`, which is exactly the sigmoid of the input.

**nn layer**: `nn::SoftPlus`

---

## `softmax`

```cpp
Variable *softmax(Arena *arena, Variable *in, int32_t dim = -1);
```

`f(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))`, applied along `dim`. The `dim` value is stored in `out->metadata_float`.

**What is saved**: Unlike all other activations, `softmax` saves the **output** (`out_data`) rather than the input. The Jacobian of softmax is expressed in terms of its output values:

```
∂f_i/∂x_j = f_i * (δ_ij - f_j)
```

**Backward**: `tensor_softmax_grad` — the efficient formulation avoids constructing the full Jacobian:

```
out_i = f_i * (grad_i - Σ_j(f_j * grad_j))
```

This computes the dot product `Σ f_j * grad_j` once and uses it to adjust every element.

:::warning Do not use softmax before CrossEntropyLoss
`autograd::cross_entropy_loss` applies log-softmax internally. Adding `autograd::softmax` before it applies softmax twice, producing a wrong loss value and gradient. The `nn::CrossEntropyLoss` documentation has the same warning — it bears repeating here.
:::

**nn layer**: `nn::Softmax(int32_t dim = -1)`

---

## Quick Reference

| Function | Signature | Saves | nn layer |
|---|---|---|---|
| `relu` | `(arena, in)` | `in->data` | `nn::ReLU` |
| `relu6` | `(arena, in)` | `in->data` | `nn::ReLU6` |
| `leaky_relu` | `(arena, in, alpha)` | `in->data` | `nn::LeakyReLU` |
| `elu` | `(arena, in, alpha)` | `in->data` | `nn::ELU` |
| `gelu` | `(arena, in)` | `in->data` | `nn::GELU` |
| `swish` | `(arena, in)` | `in->data` | `nn::Swish` |
| `sigmoid` | `(arena, in)` | `in->data` | `nn::Sigmoid` |
| `tanh` | `(arena, in)` | `in->data` | `nn::Tanh` |
| `hard_sigmoid` | `(arena, in)` | `in->data` | `nn::HardSigmoid` |
| `hard_swish` | `(arena, in)` | `in->data` | `nn::HardSwish` |
| `softplus` | `(arena, in)` | `in->data` | `nn::SoftPlus` |
| `softmax` | `(arena, in, dim=-1)` | `out_data` (output!) | `nn::Softmax` |