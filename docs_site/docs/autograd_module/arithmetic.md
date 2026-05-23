---
sidebar_position: 3
title: "Arithmetic Operations"
---

# Arithmetic Operations

These are the differentiable arithmetic operations in the autograd layer. Each one calls the corresponding `tensor_*` function for the forward computation and registers a `backward_fn` that computes the local gradient and accumulates it into parent `grad` tensors.

**Header**: `include/autograd/autograd.hpp`  
**Source**: `src/autograd/ops/arithmetic/`

:::info Signature pattern
Every arithmetic op takes an `Arena *arena` as its first argument — this is always the **graph arena**, where the output `Variable`, its `grad` tensor, the `parents` array, and any `saved_tensors` are all allocated. The actual float data lives wherever the input tensors live (usually `perm_arena` for parameters, `graph_arena` for activations).
:::

---

## `add`

```cpp
Variable *add(Arena *arena, Variable *a, Variable *b);
```

Computes `out = a + b` element-wise with **broadcasting** support. The output shape is the broadcast result of `a` and `b`.

### Forward

Calls `tensor_add(out_data, a->data, b->data)`.

### Backward

```
∂L/∂a = ∂L/∂out   (summed over broadcast dims if a was broadcast)
∂L/∂b = ∂L/∂out   (summed over broadcast dims if b was broadcast)
```

The `backward_fn` calls `tensor_sum_to_shape` to reduce the upstream gradient back to the original shape of each parent before accumulating. This correctly handles the bias addition pattern in `nn::Linear`, where `bias` has shape `[1, out_features]` and is broadcast across the batch dimension.

### Common usage

```cpp
// Linear layer: out = x @ W + b
auto *z  = autograd::matmul(graph_arena, x, weight);
auto *out = autograd::add(graph_arena, z, bias);
// bias.shape = [1, out_features], z.shape = [batch, out_features]
// gradient w.r.t. bias is summed over the batch dimension automatically
```

---

## `sub`

```cpp
Variable *sub(Arena *arena, Variable *a, Variable *b);
```

Computes `out = a - b` element-wise with broadcasting.

### Forward

Calls `tensor_sub(out_data, a->data, b->data)`.

### Backward

```
∂L/∂a = +∂L/∂out   (summed to a's shape)
∂L/∂b = -∂L/∂out   (negated and summed to b's shape)
```

The `backward_fn` calls `tensor_sum_to_shape` for the `a` parent, then calls it again for `b` and applies `tensor_scale(reduced_grad, -1.0f)` before accumulating.

---

## `mul`

```cpp
Variable *mul(Arena *arena, Variable *a, Variable *b);
```

Computes `out = a * b` element-wise (Hadamard product) with broadcasting. This is **not** matrix multiplication — use `matmul` for that.

### Forward

Calls `tensor_mul(out_data, a->data, b->data)`. Both `a->data` and `b->data` are saved for the backward pass.

### Backward

```
∂L/∂a = ∂L/∂out * b
∂L/∂b = ∂L/∂out * a
```

Each saved tensor is multiplied element-wise with the upstream gradient (`tensor_mul`), then accumulated into the respective parent's `grad`.

### Common usage

```cpp
// Gating mechanism: out = gate * value
auto *out = autograd::mul(graph_arena, gate, value);

// Custom gradient masking
auto *masked = autograd::mul(graph_arena, activation, mask_variable);
```

---

## `matmul`

```cpp
Variable *matmul(Arena *arena, Variable *a, Variable *b);
```

Computes `out = a @ b` — standard 2D matrix multiplication. Both input Variables are saved for the backward pass.

:::note 2D only at the autograd level
The tensor-level `mat_mul` supports batched, multi-dimensional, and transposed matrix multiplication. The `autograd::matmul` wrapper assumes 2D inputs (`[M, K]` and `[K, N]`) and uses `transpose_a=false, transpose_b=false` for the forward pass. For batched matmul or transpose variants, call `mat_mul` at the tensor level directly.
:::

### Forward

```cpp
mat_mul(out_data, a->data, b->data, /*zero_out=*/true, false, false);
```

Output shape: `[a.shape[0], b.shape[1]]`.

### Backward

```
∂L/∂a = ∂L/∂out @ b^T       (shape: [M, K])
∂L/∂b = a^T @ ∂L/∂out       (shape: [K, N])
```

The `backward_fn` calls `mat_mul` twice, using the transpose flags to avoid actually transposing the saved tensors:

```cpp
// grad for a: local_grad_a = self->grad @ b^T
mat_mul(local_grad_a, self->grad, b_data, true, false, true);

// grad for b: local_grad_b = a^T @ self->grad
mat_mul(local_grad_b, a_data, self->grad, true, true, false);
```

### Common usage

```cpp
// nn::Linear forward pass
auto *out = autograd::matmul(graph_arena, x, weight);
// x.shape = [batch, in_features], weight.shape = [in_features, out_features]
// out.shape = [batch, out_features]
```

---

## `scale`

```cpp
Variable *scale(Arena *arena, Variable *in, float scale_factor);
```

Computes `out = in * scale_factor` — multiplies every element by a scalar constant. The `scale_factor` is stored in `out->metadata_float` for use in the backward pass.

### Forward

```cpp
tensor_copy(out_data, in->data);
tensor_scale(out_data, scale_factor);
```

### Backward

```
∂L/∂in = ∂L/∂out * scale_factor
```

The `backward_fn` copies the upstream gradient and multiplies by `self->metadata_float`:

```cpp
tensor_copy(local_grad, self->grad);
tensor_scale(local_grad, self->metadata_float);
tensor_add(parent->grad, parent->grad, local_grad);
```

### Common usage

```cpp
// Learning rate application (SGD does this at tensor level, but the pattern is the same)
auto *scaled = autograd::scale(graph_arena, param, 0.5f);

// Normalisation
auto *normalised = autograd::scale(graph_arena, logits, 1.0f / temperature);
```

---

## `sum`

```cpp
Variable *sum(Arena *arena, Variable *in);
```

Reduces all elements to a single scalar by summing. The output is a 1-element tensor.

### Forward

```cpp
float sum_val = tensor_sum(in->data);
out_data->storage->data[out_data->offset] = sum_val;
```

### Backward

```
∂L/∂in_i = ∂L/∂out   for all i
```

The upstream scalar gradient is broadcast uniformly to every element of `in`. The `backward_fn` fills a local gradient tensor with the upstream scalar value and accumulates:

```cpp
float grad_val = self->grad->storage->data[self->grad->offset];
tensor_fill(local_grad, grad_val);
tensor_add(parent->grad, parent->grad, local_grad);
```

### Common usage

```cpp
// Custom loss: sum of all elements
auto *total = autograd::sum(graph_arena, per_element_loss);
autograd::backward(graph_arena, total);

// Reducing a feature map before a scalar objective
auto *reduced = autograd::sum(graph_arena, feature_map);
```

---

## Relationship to Tensor-Level Arithmetic

Every autograd arithmetic op is a thin wrapper. The full mapping:

| Autograd op | Tensor forward | Tensor backward |
|---|---|---|
| `add(a, b)` | `tensor_add` | `tensor_sum_to_shape` |
| `sub(a, b)` | `tensor_sub` | `tensor_sum_to_shape`, `tensor_scale(-1)` |
| `mul(a, b)` | `tensor_mul` | `tensor_mul` |
| `matmul(a, b)` | `mat_mul` | `mat_mul` (with transpose flags) |
| `scale(in, s)` | `tensor_copy` + `tensor_scale` | `tensor_copy` + `tensor_scale` |
| `sum(in)` | `tensor_sum` | `tensor_fill` |

The autograd layer adds exactly: arena allocation of the output Variable, wiring of `parents`, saving of tensors needed by the backward, and the `backward_fn` lambda.