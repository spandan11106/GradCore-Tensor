---
sidebar_position: 1
title: "Autograd Overview"
---

# The Autograd Engine

The autograd module is the machinery that makes training possible. Every call to `autograd::relu`, `autograd::matmul`, or `autograd::cross_entropy_loss` does two things simultaneously: it computes the output value (the forward pass), and it records a backward function that knows how to compute the gradient of that output with respect to its inputs. When you call `autograd::backward(arena, loss)`, GradCore-Tensor replays those recorded functions in reverse to propagate gradients all the way back to the leaf parameters.

**Header**: `include/autograd/autograd.hpp`  
**Namespace**: `gradientcore::autograd`

:::info The relationship to the tensor module
The autograd engine sits on top of the tensor module. Every autograd op calls the corresponding `tensor_*` function for the forward computation and `tensor_*_grad` function inside its backward closure. The autograd layer adds no mathematical logic of its own — it adds graph construction and gradient routing. See [Tensors in the Autograd Engine](../tensor_module/autograd-integration) for a deep dive into how the two layers connect.
:::

---

## What Lives in Autograd

```
autograd/
├── autograd.hpp         ← Single header — include this
│
├── engine.cpp           ← Variable, create_leaf, backward, topological sort
│
└── ops/
    ├── arithmetic/      ← add, sub, mul, matmul, scale, sum
    ├── activations/     ← relu, sigmoid, tanh, gelu, softmax, …
    └── loss/            ← mse_loss, cross_entropy_loss, bce_loss, …
```

---

## The Core Abstractions

### `Variable` — a tensor that remembers

```cpp
struct Variable {
    Tensor  *data;            // The actual values
    Tensor  *grad;            // Gradient accumulator (same shape)
    bool     requires_grad;   // Compute gradients for this?
    bool     is_leaf;         // Parameter (true) or intermediate (false)?

    Edge    *parents;         // Inputs that produced this Variable
    uint32_t num_parents;

    Tensor **saved_tensors;   // Data saved for the backward pass
    uint32_t num_saved;

    uint32_t     reduction;       // For loss ops (REDUCTION_NONE/MEAN/SUM)
    float        metadata_float;  // alpha, delta, scale factor, dim, etc.

    void (*backward_fn)(Variable *self, Arena *arena);
};
```

`Variable` structs live on the **graph arena**. They are freed automatically when `graph_arena->pop_to(pos)` is called at the end of each batch — no per-object bookkeeping, no reference counting, no garbage collection.

### `Edge` — a link in the graph

```cpp
struct Edge {
    Variable *node;
};
```

Each non-leaf `Variable` holds a `parents` array of `Edge` structs pointing to the `Variable` nodes that were its inputs. These links form the directed acyclic computation graph that `backward()` traverses.

---

## Lifecycle of a Forward Pass

```
1. Inputs and parameters wrapped as leaf Variables via create_leaf()
2. autograd::op() called → tensor_op() computes value, backward_fn recorded
3. Result is a new Variable on the graph arena with parents wired up
4. Repeat for each operation in the network
5. Loss Variable produced at the end of the chain
```

```
6. backward(arena, loss) called
7. Topological sort of the graph (DFS from loss node)
8. Each node's backward_fn called in reverse order
9. Gradients accumulated into leaf Variable::grad tensors
10. optimizer.step() reads those grad tensors and updates data tensors
11. graph_arena->pop_to(pos) frees the entire graph in O(1)
```

---

## Quick Reference

| What you want | Where to look |
|---|---|
| Wrapping tensors as Variables | [Variables & the Graph](./variable_and_graph.md) |
| Running backpropagation | [Variables & the Graph](./variable_and_graph.md) |
| Arithmetic ops (`add`, `matmul`, …) | [Arithmetic Operations](./arithmetic.md) |
| Activation ops (`relu`, `gelu`, …) | [Activation Operations](./activations.md) |
| Loss ops (`cross_entropy`, `mse`, …) | [Loss Operations](./losses.md) |
| How tensor ops plug in | [Tensors in the Autograd Engine](./autograd-integration.md) |

---

## Namespaces

All autograd code is in `gradientcore::autograd`. The `Reduction` enum (`REDUCTION_NONE`, `REDUCTION_MEAN`, `REDUCTION_SUM`) lives in `gradientcore` (defined in `tensor/tensor.hpp`) and is used directly by loss functions.

```cpp
#include "gradient.hpp"       // includes autograd.hpp automatically
using namespace gradientcore;

autograd::Variable *x = autograd::create_leaf(arena, tensor, true);
autograd::backward(arena, loss);
```