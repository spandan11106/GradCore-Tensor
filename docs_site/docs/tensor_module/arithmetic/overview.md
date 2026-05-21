---
sidebar_position: 1
title: "Arithmetic Operations"
---

# Arithmetic Operations

These are the low-level tensor math functions. Every layer in GradCore-Tensor ultimately calls one or more of these. They all follow the same conventions:

- Output tensor is always the **first argument**.
- The output must be **pre-allocated** by the caller (typically on an arena).
- Return `bool` — `true` on success, `false` on shape mismatch or null input.
- All support both contiguous (fast path) and non-contiguous (correct-but-slower) tensors.
- OpenMP parallelisation is enabled when compiled with `-fopenmp`.

## Element-wise Operations

### `tensor_add`

```cpp
bool tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
```

Computes `out = a + b` element-wise with **broadcasting**.

Broadcasting rules (NumPy-style): dimensions are compatible if they are equal or one of them is 1. The output shape is the element-wise maximum of the two input shapes, right-aligned.

```
a: shape [32, 128]       b: shape [1, 128]
─────────────────────────────────────────────
out: shape [32, 128]     (b is broadcast over the batch dimension)
```

This is how bias addition works in `nn::Linear`:

```cpp
// out = matmul(x, weight) + bias
// bias has shape [1, out_features], out has shape [batch, out_features]
out = autograd::add(compute_arena, out, bias);
```

### `tensor_sub`

```cpp
bool tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
```

`out = a - b`, with the same broadcasting rules as `tensor_add`. Used in gradient computations (e.g. `d(a-b)/db = -1`, so the gradient is negated after summation).

### `tensor_mul`

```cpp
bool tensor_mul(Tensor *out, const Tensor *a, const Tensor *b);
```

`out = a * b`, element-wise with broadcasting. This is Hadamard product, **not** matrix multiplication. Used in:

- Gradient of the `mul` autograd op: `grad_a = grad_output * b`
- Dropout mask application

### `tensor_scale`

```cpp
void tensor_scale(Tensor *t, float scale);
```

In-place scalar multiplication: `t *= scale`. Unlike the element-wise ops, this modifies `t` directly rather than writing to a separate output. Used by:

- Weight update: `tensor_scale(grad, learning_rate)`
- Gradient negation: `tensor_scale(reduced_grad, -1.0f)` in the `sub` backward pass
- LBFGS search direction scaling

## Matrix Multiplication

### `mat_mul`

```cpp
bool mat_mul(Tensor *out,
             const Tensor *a,
             const Tensor *b,
             bool zero_out,
             bool transpose_a,
             bool transpose_b);
```

Batched matrix multiplication supporting optional transposition of either operand. This is the performance-critical kernel.

| Parameter | Meaning |
|---|---|
| `out` | Pre-allocated output tensor |
| `a`, `b` | Input matrices (must be ≥ 2D) |
| `zero_out` | If `true`, zero `out` before accumulating |
| `transpose_a` | Treat `a` as transposed for the multiply |
| `transpose_b` | Treat `b` as transposed for the multiply |

The effective operation:

```
out = (transpose_a ? a^T : a)  @  (transpose_b ? b^T : b)
```

### Shapes

For 2D inputs with `transpose_a=false`, `transpose_b=false`:

```
a: [M, K]
b: [K, N]
out: [M, N]
```

For the common case of `Linear` forward pass:

```cpp
// x: [batch, in_features]   weight: [in_features, out_features]
// out: [batch, out_features]
mat_mul(out_data, x->data, weight->data, true, false, false);
```

And the backward pass:

```cpp
// grad_a = grad_output @ weight^T    (shape: [batch, in_features])
mat_mul(local_grad_a, self->grad, b_data, true, false, true);

// grad_b = x^T @ grad_output         (shape: [in_features, out_features])
mat_mul(local_grad_b, a_data, self->grad, true, true, false);
```

### Batched Operation

For inputs with `ndims > 2`, the leading dimensions are treated as a batch. Broadcasting applies to the batch dimensions:

```
a: [batch, M, K]
b: [1, K, N]        ← broadcast over batch
out: [batch, M, N]
```

### Implementation: Blocked GEMM

The kernel uses a classic blocked (tiled) GEMM strategy to maximise cache utilisation:

```
for m_block in [0, M, BLOCK_M=64]:
  for k_block in [0, K, BLOCK_K=256]:
    for n_block in [0, N, BLOCK_N=64]:
      micro_kernel(out[m..m+MR, n..n+NR],
                   a[m..m+MR, k..k+MK],
                   b[k..k+MK, n..n+NR])
```

The micro-kernel (`MR=6` × `NR=16`) uses AVX2 FMA instructions when available:

```
AVX2 path:    processes 6×16 = 96 output elements per k-iteration
              using eight 256-bit registers (_mm256_fmadd_ps)

Scalar path:  generic fallback for CPUs without AVX2
```

Compile with `-mavx2 -mfma` (see [Getting Started](/docs/Getting%20Started)) to enable the fast path. The `#if defined(USE_AVX2)` branch is selected at compile time.

## Reduction Operations

### `tensor_sum`

```cpp
float total = tensor_sum(Tensor *t);
```

Sum of all elements. Returns a scalar `float` (not a tensor). OpenMP reduction for the contiguous case.

### `tensor_sum_to_shape`

```cpp
bool tensor_sum_to_shape(Tensor *out, const Tensor *in);
```

Reduces `in` to the shape of `out` by summing over broadcast-expanded dimensions. This is the gradient accumulation operation for `tensor_add`:

```
in:  shape [32, 128]
out: shape [1, 128]   ← sum over dim 0 (batch)
```

Used in the `add` and `sub` backward passes to accumulate gradients back into bias tensors (which have shape `[1, features]`).

## Fast vs Slow Path

Every element-wise operation checks `tensor_is_contiguous` and takes one of two paths:

```cpp
if (tensor_is_contiguous(a) && tensor_is_contiguous(b) && tensor_is_contiguous(out)) {
    // Fast path: direct pointer arithmetic
    for (uint64_t i = 0; i < out->size; i++) {
        out->storage->data[out->offset + i] =
            a->storage->data[a->offset + i] + b->storage->data[b->offset + i];
    }
} else {
    // Correct path: stride-aware indexing
    uint32_t indices[MAX_TENSOR_DIMS] = {0};
    for (uint64_t i = 0; i < out->size; i++) {
        out->storage->data[tensor_get_flat_index(out, indices)] =
            a->storage->data[tensor_get_flat_index(a, indices)] +
            b->storage->data[tensor_get_flat_index(b, indices)];
        // advance indices ...
    }
}
```

In practice, tensors created by `tensor_create` and then used directly (without transposing) will always hit the fast path.
