---
sidebar_position: 1
title: "Tensor Module Overview"
---

# The Tensor Module

Welcome to the nerve centre of GradCore-Tensor. Everything — training, inference, backprop, your optimizer quietly panicking about learning rates — ultimately comes down to operations on `Tensor` objects. This section documents every corner of the tensor module so you know exactly what's happening under the hood.

## What Lives Here

The tensor module is split into four logical areas:

```
tensor/
├── tensor.hpp              ← The central header (include this)
├── memory_cpu/
│   ├── arena.hpp/cpp       ← Arena allocator (the memory wizard)
│   ├── platform.hpp/cpp    ← OS-level memory & entropy
│   └── platform_linux.cpp  ← Linux-specific implementation
├── prng.hpp/cpp            ← PCG-based random number generation
├── tensor_create.cpp       ← Tensor construction
├── tensor_utils.cpp        ← Shape checks, fills, copies, etc.
├── tensor_views.cpp        ← Views, reshape, transpose
├── arithmetic/             ← add, sub, mul, matmul, scale, sum
├── activations/            ← relu, sigmoid, softmax, gelu, …
├── activations_grad/       ← Backward passes for activations
├── loss_functions/         ← mse, cross_entropy, huber, …
└── loss_grad/              ← Backward passes for loss functions
```

## The Big Picture

Here's how tensors relate to everything else:

![Tensor module overview](/img/Intro.drawio.svg)


A `Tensor` does not own its memory. It holds a pointer to a `TensorStorage`, which in turn holds a pointer into an `Arena`'s slab. This design means allocating a tensor is just a pointer bump — no `malloc`, no heap fragmentation, no GC pausing at the worst possible moment.

## Quick Reference

| Topic                        | Page                                           |
| ---------------------------- | ---------------------------------------------- |
| Arena allocator              | [Memory & Arenas](./memory/arena.md)           |
| Platform layer               | [Platform](./memory/platform.md)               |
| Random number generation     | [PRNG](./memory/prng.md)                       |
| Tensor struct & construction | [Tensor Basics](./tensor-basics.md)            |
| Views, reshape, transpose    | [Tensor Views](./tensor-views.md)              |
| Arithmetic ops               | [Arithmetic](./arithmetic/overview.md)         |
| Activation functions         | [Activations](./activations/overview.md)       |
| Loss functions               | [Loss Functions](./loss_functions/overview.md) |

## Namespaces & Headers

All tensor code lives in the `gradientcore` namespace. The single include you need:

```cpp
#include "gradient.hpp"
using namespace gradientcore;
```

Or, if you only want the tensor layer:

```cpp
#include "tensor/tensor.hpp"
#include "tensor/memory_cpu/arena.hpp"
```

:::note
All functions that take an `Arena*` allocate their outputs on that arena. The caller decides the lifetime — there are no hidden global allocations.
:::
