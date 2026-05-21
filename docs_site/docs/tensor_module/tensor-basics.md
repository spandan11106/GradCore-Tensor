---
sidebar_position: 2
title: "Tensor Basics"
---

# Tensor Basics

The `Tensor` struct is the central data structure in GradCore-Tensor. It describes a multi-dimensional array of `float` values and knows how to index into them correctly, whether the data is contiguous in memory or is a strided view of something else.

## The Struct

```cpp
struct TensorStorage {
    float    *data;   // Raw float buffer (lives in an Arena)
    uint64_t  size;   // Number of elements in this buffer
};

struct Tensor {
    uint32_t      ndims;                    // Number of dimensions (≤ 10)
    uint32_t      shape[MAX_TENSOR_DIMS];   // Size along each axis
    uint32_t      strides[MAX_TENSOR_DIMS]; // Step size for each axis
    uint64_t      size;                     // Total element count
    uint64_t      offset;                   // Starting index into storage->data

    TensorStorage *storage;                 // Shared backing buffer
};
```

`MAX_TENSOR_DIMS` is 10, which is enough for any reasonable neural network tensor.

## Anatomy of a Tensor

```
Tensor (shape=[3,4], strides=[4,1], offset=0)
         │
         └── TensorStorage
               data ──► [a00, a01, a02, a03,
                          a10, a11, a12, a13,
                          a20, a21, a22, a23]
```

A 3×4 matrix laid out in row-major order. Element `[i, j]` is at:

```
index = offset + i * strides[0] + j * strides[1]
      = 0      + i * 4          + j * 1
```

This stride-based indexing is what makes views, reshapes, and transposes essentially free — you just change the metadata, not the data.

## Creating Tensors

### `tensor_create` — uninitialized

```cpp
uint32_t shape[2] = {batch_size, 784};
Tensor *t = tensor_create(arena, 2, shape);
```

Allocates the `Tensor` struct and the underlying `float` buffer on `arena`. The data is **zero-initialised** (the arena's `push_raw` zeroes by default). Returns `nullptr` if `ndims == 0`, any dimension is 0, or `ndims > MAX_TENSOR_DIMS`.

### `tensor_create_zeros` — explicitly zeroed

```cpp
Tensor *t = tensor_create_zeros(arena, 2, shape);
```

Identical to `tensor_create` but calls `tensor_fill(t, 0.0f)` for absolute clarity. Use this when correctness matters more than the marginal performance difference.

## Utility Functions

### Indexing

```cpp
uint32_t indices[2] = {1, 3};
uint64_t flat = tensor_get_flat_index(t, indices);
float val = t->storage->data[flat];
```

`tensor_get_flat_index` computes `offset + Σ(indices[i] * strides[i])`. Use this for non-contiguous tensors where you can't simply index with `data[offset + i]`.

### Contiguity check

```cpp
bool ok = tensor_is_contiguous(t);
```

A tensor is contiguous when iterating elements in memory order matches iterating in logical index order. Specifically:

```
strides[ndims-1] == 1
strides[i]       == strides[i+1] * shape[i+1]  (for i < ndims-1)
```

Dimensions of size 1 are excluded from the check (they can have any stride without affecting correctness). Contiguous tensors allow the fast path in arithmetic operations — direct pointer arithmetic instead of `tensor_get_flat_index` on every element.

### Fill

```cpp
tensor_fill(t, 3.14f);   // Set every element to 3.14
```

Works correctly for both contiguous and non-contiguous tensors.

### Clear (set to zero)

```cpp
tensor_clear(t);
```

Uses `memset` for contiguous tensors (fast), falls back to `tensor_fill(t, 0.0f)` otherwise.

### Copy

```cpp
bool ok = tensor_copy(dst, src);
```

Element-by-element copy from `src` to `dst`. Both must have identical shapes. Returns `false` on shape mismatch. Handles non-contiguous tensors correctly via `tensor_get_flat_index`.

### Shape match

```cpp
bool same = shape_match(a, b);
```

Returns `true` iff `a->ndims == b->ndims` and all `shape[i]` match. Used as a precondition check throughout the codebase.

### Broadcastability check

```cpp
uint32_t out_ndims;
uint32_t out_shape[MAX_TENSOR_DIMS];
bool ok = tensor_check_broadcastable(a, b, &out_ndims, out_shape);
```

Checks if `a` and `b` are broadcast-compatible (NumPy rules: dimensions are compatible if they're equal or one of them is 1). If compatible, fills `out_shape` with the result shape and `out_ndims` with its rank. Used by `tensor_add`, `tensor_sub`, `tensor_mul`.

### Sum

```cpp
float total = tensor_sum(t);
```

Returns the sum of all elements. Uses OpenMP reduction for contiguous tensors when `_OPENMP` is defined.

### Sum-to-shape (gradient reduction)

```cpp
tensor_sum_to_shape(out, in);
```

Reduces `in` to the shape of `out` by summing over broadcast dimensions. Used during backpropagation for `add` and `sub` to accumulate gradients into correctly-shaped parameter tensors.

### Scale (in-place)

```cpp
tensor_scale(t, 0.5f);   // Multiply every element by 0.5
```

In-place scalar multiplication. Fast path for contiguous tensors with OpenMP parallelisation.

## The `offset` Field — Why It Exists

Most tensors have `offset = 0`. The `offset` exists to support views that start partway through a shared buffer:

```
TensorStorage: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                                   ▲
Tensor (offset=5, size=3):       data[5..7]
```

When you call `tensor_view`, `tensor_transpose`, or `tensor_reshape`, the new `Tensor` shares the same `TensorStorage` but can have a different `offset`, `strides`, and `shape`. This is how strided views work at zero cost.

Always access data as `t->storage->data[t->offset + computed_index]`, never as `t->storage->data[computed_index]` directly.

## Constants

```cpp
constexpr uint32_t MAX_TENSOR_DIMS = 10;
```

Ten dimensions is plenty. If you need an 11-dimensional tensor, you may have a different problem.
