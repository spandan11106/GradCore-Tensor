---
sidebar_position: 3
title: "Tensor Views"
---

# Tensor Views

Views are tensors that share memory with another tensor. Changing the data through a view changes the original, and vice versa — because they're the same bytes, just described differently.

This is not a bug. It is, in fact, the entire point.

## `tensor_view` — shallow copy of metadata

```cpp
Tensor *view = tensor_view(arena, src);
```

Creates a new `Tensor` that is an exact copy of `src`'s metadata (`ndims`, `shape`, `strides`, `offset`, `size`) but **shares the same `TensorStorage`**. The new `Tensor` struct is allocated on `arena`; no float data is copied.

```
Original:                    View:
Tensor {                     Tensor {
  shape   = [3, 4]             shape   = [3, 4]
  strides = [4, 1]             strides = [4, 1]
  offset  = 0                  offset  = 0
  storage = ──────────────►  storage = ──────► (same buffer)
}                            }
```

`tensor_view` is the basis for both `tensor_reshape` and `tensor_transpose`.

## `tensor_reshape` — change shape, keep data

```cpp
uint32_t new_shape[1] = {12};
Tensor *flat = tensor_reshape(arena, src, 1, new_shape);
```

Returns a view of `src` with a new shape, provided:

1. The new total element count equals the old one (`product(new_shape) == src->size`).
2. The source tensor is **contiguous** — `tensor_is_contiguous(src)` must be true.

If either condition fails, `tensor_reshape` returns `nullptr`.

### Example: flatten then unflatten

```cpp
// Original: [batch=32, features=784]
uint32_t flat_shape[1] = {32 * 784};
Tensor *flat = tensor_reshape(graph_arena, input, 1, flat_shape);

// Back to 2D
uint32_t orig_shape[2] = {32, 784};
Tensor *back = tensor_reshape(graph_arena, flat, 2, orig_shape);
```

Both `flat` and `back` share `input`'s storage — zero bytes are copied.

### Why only contiguous tensors?

Reshape assumes the elements are laid out consecutively in memory so it can recompute strides from the new shape. A transposed tensor, for example, has rows interleaved in memory — there's no valid stride assignment for an arbitrary new shape. If you need to reshape a non-contiguous tensor, copy it first.

## `tensor_transpose` — swap two axes

```cpp
Tensor *t = tensor_transpose(arena, src, 0, 1);  // swap dim 0 and dim 1
```

Creates a view of `src` with `shape[dim0]` ↔ `shape[dim1]` and `strides[dim0]` ↔ `strides[dim1]` swapped. No data is moved.

### Transpose a matrix

```cpp
// src: shape=[M, K], strides=[K, 1]  (row-major)
Tensor *transposed = tensor_transpose(arena, src, 0, 1);
// transposed: shape=[K, M], strides=[1, K]  (column-major view of same data)
```

```
Original [M=3, K=4]:              Transposed [K=4, M=3]:
                                  
 0  1  2  3                        0  4  8
 4  5  6  7          ────►         1  5  9
 8  9 10 11                        2  6 10
                                   3  7 11

(same bytes, different strides)
```

Transpose is used extensively in the `matmul` backward pass:

```cpp
// grad_a = grad_output @ b^T
mat_mul(local_grad_a, self->grad, b_data, true, false, true);

// grad_b = a^T @ grad_output
mat_mul(local_grad_b, a_data, self->grad, true, true, false);
```

The `transpose_a` and `transpose_b` flags in `mat_mul` tell it to treat the inputs as if they were transposed without actually creating a new tensor.

## Non-Contiguity After Transpose

After a transpose, `tensor_is_contiguous` will typically return `false` (unless one of the swapped dimensions has size 1). Operations that require contiguous tensors — like `tensor_reshape` — will refuse to work on a transposed tensor.

Operations that work element-by-element (add, mul, activations) handle non-contiguous tensors via the index-walking path:

```cpp
// Fast path (contiguous)
for (uint64_t i = 0; i < out->size; i++) {
    out->storage->data[out->offset + i] =
        a->storage->data[a->offset + i] + b->storage->data[b->offset + i];
}

// Slow path (non-contiguous) — correct for any stride pattern
uint32_t indices[MAX_TENSOR_DIMS] = {0};
for (uint64_t i = 0; i < out->size; i++) {
    uint64_t a_idx   = tensor_get_flat_index(a, indices);
    uint64_t b_idx   = tensor_get_flat_index(b, indices);
    uint64_t out_idx = tensor_get_flat_index(out, indices);
    out->storage->data[out_idx] = a->storage->data[a_idx] + b->storage->data[b_idx];
    // advance indices ...
}
```

## Memory Implications

Because views share storage, they are valid only as long as the owning arena hasn't been rewound past the storage allocation. Typical usage:

```cpp
uint64_t pos = graph_arena->get_pos();

Tensor *t       = tensor_create(graph_arena, 2, shape);
Tensor *t_T     = tensor_transpose(graph_arena, t, 0, 1);   // view of t
// ... use t_T ...

graph_arena->pop_to(pos);   // Both t and t_T are gone — don't touch them after this
```
