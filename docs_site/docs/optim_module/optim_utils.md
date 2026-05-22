---
sidebar_position: 8
title: "Optimizer Utilities"
---

# Optimizer Utilities (`optim_utils.hpp`)

`optim_utils.hpp` is a header-only collection of inline helper functions used internally by the L-BFGS optimizer to flatten parameter and gradient vectors into a single 1D tensor, and to restore parameters from that flat representation. They are also useful when implementing custom second-order optimizers.

**Header**: `include/optim/optim_utils.hpp`  
**Namespace**: `gradientcore::optim`

---

## Why Flatten?

L-BFGS (and second-order methods in general) reason about the entire parameter space as a single vector. The search direction `d`, the step `s_k = w_{k+1} - w_k`, and the gradient change `y_k = g_{k+1} - g_k` are all defined over this global vector. The utility functions here convert between the natural representation (a list of per-layer tensors) and the flat representation (a single contiguous 1D tensor) that second-order algorithms need.

First-order optimizers (Adam, SGD, etc.) never need to flatten — they process each parameter independently and never need to reason about the combined parameter space.

---

## `get_total_params_size`

```cpp
inline uint64_t
get_total_params_size(const std::vector<autograd::Variable *> &params);
```

Returns the total number of elements across all parameters that have `requires_grad = true`. Used by L-BFGS to allocate its flat working tensors.

```cpp
uint64_t n = get_total_params_size(seq->parameters());
uint32_t flat_shape[1] = {(uint32_t)n};
Tensor *flat = tensor_create_zeros(temp_arena, 1, flat_shape);
```

---

## `flatten_params`

```cpp
inline void flatten_params(const std::vector<autograd::Variable *> &params,
                           Tensor *flat_out);
```

Copies parameter values from all `requires_grad` parameters into a pre-allocated 1D tensor `flat_out`, in traversal order:

```
[param_0[0], param_0[1], ..., param_0[N₀-1],
 param_1[0], ..., param_1[N₁-1],
 ...]
```

`flat_out` must already be allocated with size equal to `get_total_params_size(params)`.

```cpp
Tensor *flat_w = tensor_create_zeros(temp_arena, 1, flat_shape);
flatten_params(params, flat_w);   // snapshot current weights
```

Used inside L-BFGS to save the current parameter values before tentatively applying a step during the line search.

---

## `unflatten_params`

```cpp
inline void unflatten_params(Tensor *flat_in,
                             const std::vector<autograd::Variable *> &params);
```

The inverse of `flatten_params`: copies values from the flat 1D tensor back into the per-parameter tensors. Writes to `param->data` directly.

```cpp
unflatten_params(new_flat_params, params);   // apply tentative step
float loss = closure();                       // evaluate at new position
if (loss > threshold) {
    unflatten_params(old_flat_params, params); // restore if rejected
}
```

---

## `flatten_grads`

```cpp
inline void flatten_grads(const std::vector<autograd::Variable *> &params,
                          Tensor *flat_out);
```

Copies gradient values (from `param->grad`) into `flat_out`. Only processes parameters with `requires_grad = true` and non-null `grad`.

```cpp
Tensor *flat_g = tensor_create_zeros(temp_arena, 1, flat_shape);
flatten_grads(params, flat_g);   // snapshot current gradients
```

Used inside L-BFGS to compute `y_k = g_{k+1} - g_k` across a step.

---

## `tensor_dot_1d`

```cpp
inline float tensor_dot_1d(Tensor *a, Tensor *b);
```

Computes the dot product of two 1D tensors:

```
result = Σ a[i] * b[i]
```

Both tensors must have the same size. This is a simple scalar loop — no SIMD, no OpenMP. Used by L-BFGS to compute quantities like `ρ_k = 1 / (y_k^T * s_k)` and the direction scalars `α_i`, `β_i` in the two-loop recursion.

```cpp
float ys = tensor_dot_1d(y_k, s_k);
float rho = 1.0f / ys;

float alpha = rho * tensor_dot_1d(s_k, direction);
```

---

## Using the Utilities Directly

These utilities are exposed for users implementing custom second-order optimizers. A pattern for a simple Newton-CG step:

```cpp
#include "optim/optim_utils.hpp"

uint64_t n = optim::get_total_params_size(params);
uint32_t shape[1] = {(uint32_t)n};

// Get current gradient as flat vector
Tensor *g = tensor_create_zeros(arena, 1, shape);
optim::flatten_grads(params, g);

// Compute some search direction d (e.g. conjugate gradient on the Hessian-vector product)
Tensor *d = tensor_create_zeros(arena, 1, shape);
// ... fill d ...

// Compute step size (e.g. from line search)
float t = 0.01f;

// Apply: w = w + t * d
Tensor *step = tensor_create_zeros(arena, 1, shape);
tensor_copy(step, d);
tensor_scale(step, t);

Tensor *flat_w = tensor_create_zeros(arena, 1, shape);
optim::flatten_params(params, flat_w);
tensor_add(flat_w, flat_w, step);
optim::unflatten_params(flat_w, params);
```