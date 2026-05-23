---
sidebar_position: 2
title: "Variables & the Graph"
---

# Variables & the Computation Graph

This page covers the two foundational functions of the autograd engine: `create_leaf`, which wraps a raw tensor into a graph-trackable `Variable`, and `backward`, which traverses the computation graph in reverse and accumulates gradients. Everything else in the autograd module builds on these two.

**Header**: `include/autograd/autograd.hpp`  
**Source**: `src/autograd/engine.cpp`

---

## `create_leaf` — Entering the Graph

```cpp
Variable *create_leaf(Arena *arena, Tensor *data, bool requires_grad);
```

`create_leaf` is the entry point into the autograd graph. It wraps a raw `Tensor` in a `Variable` and declares whether that tensor participates in gradient computation.

| Parameter | Description |
|---|---|
| `arena` | Arena on which the `Variable` struct and its `grad` tensor are allocated. |
| `data` | The tensor holding the actual float values. Not copied — the Variable holds a pointer to it. |
| `requires_grad` | If `true`, a zero-initialised gradient tensor is allocated alongside `data` and gradients will flow back through this node. |

```cpp
// A learnable parameter — gradients needed
Tensor *w_tensor = tensor_create(perm_arena, 2, shape);
autograd::Variable *W = autograd::create_leaf(perm_arena, w_tensor, true);

// An input batch — no gradient needed
Tensor *x_tensor = tensor_create(graph_arena, 2, batch_shape);
autograd::Variable *x = autograd::create_leaf(graph_arena, x_tensor, false);
```

### What `create_leaf` sets

```cpp
v->data          = data;
v->requires_grad = requires_grad;
v->is_leaf       = true;
v->backward_fn   = nullptr;   // leaves have no backward function
v->parents       = nullptr;   // leaves have no parents
v->num_parents   = 0;
v->grad          = requires_grad
                     ? tensor_create_zeros(arena, data->ndims, data->shape)
                     : nullptr;
```

Leaf `Variable` structs allocated on `perm_arena` persist for the program lifetime — that is how parameters outlive the graph. Leaf `Variable` structs for input data are typically allocated on `graph_arena` and freed with the rest of the graph after each batch.

### Two kinds of leaves

| Kind | `arena` | `requires_grad` | Purpose |
|---|---|---|---|
| **Parameter** | `perm_arena` | `true` | Model weights, biases, BatchNorm gamma/beta. Optimizer reads `grad` to update `data`. |
| **Input / target** | `graph_arena` | `false` | Batch features and labels. No gradient needed; `grad` is `nullptr`. |

---

## Non-leaf Variables

Every `autograd::op()` call (e.g. `autograd::relu`, `autograd::matmul`) produces a non-leaf `Variable`:

```cpp
v->is_leaf      = false;
v->backward_fn  = [](Variable *self, Arena *arena) { ... };
v->parents      = arena->push_array<Edge>(num_inputs);
v->num_parents  = num_inputs;
v->saved_tensors = ...;   // tensors the backward_fn needs
```

Non-leaf Variables are always allocated on the **graph arena**. They are the intermediate activations, the loss node, and everything in between. They exist only for the duration of a single forward–backward step.

---

## The Computation Graph

After a forward pass through a three-layer network:

```
x (leaf, no grad)
W1 (leaf, grad)  ──► matmul ──► z1 ──► relu ──► a1
b1 (leaf, grad)  ──► add ───────────────────────────► z2 ──► cross_entropy ──► loss
W2 (leaf, grad)  ──► matmul ──────────────────────────────►
b2 (leaf, grad)  ──► add ──────────────────────────────────►
```

Each interior node (`matmul`, `relu`, `add`, `cross_entropy`) is a `Variable` with:
- A `backward_fn` lambda that knows how to push gradients toward its parents.
- A `parents` array pointing to the `Variable`s that were its inputs.
- A `saved_tensors` array holding whatever tensor data the backward needs (e.g. `relu` saves the pre-activation input to know which elements were positive).

---

## `backward` — Running the Chain Rule

```cpp
void backward(Arena *arena, Variable *loss_node);
```

Runs the full backward pass from `loss_node` to all leaf parameters that have `requires_grad = true`.

| Parameter | Description |
|---|---|
| `arena` | The graph arena. Temporary tensors inside backward functions are allocated here and survive until `pop_to` is called externally. |
| `loss_node` | The scalar loss `Variable` to differentiate. Must have `requires_grad = true`. |

```cpp
autograd::backward(graph_arena, loss);
```

### What `backward` does internally

```cpp
// 1. Seed the loss gradient: dL/dL = 1
tensor_fill(loss_node->grad, 1.0f);

// 2. Build the topological order of the graph
std::vector<Variable *> topo;
std::unordered_set<Variable *> visited;
build_topo(loss_node, visited, topo);  // recursive DFS

// 3. Call each backward_fn in reverse topological order
for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward_fn)
        (*it)->backward_fn(*it, arena);
}
```

The topological sort (DFS from the loss node, appending each node after its children) guarantees that when `backward_fn` for node `v` is called, `v->grad` already contains the full accumulated upstream gradient — every downstream node has already contributed its piece via `tensor_add(parent->grad, parent->grad, local_grad)`.

### Gradient accumulation

Every `backward_fn` **adds** to `parent->grad` rather than assigning:

```cpp
tensor_add(parent->grad, parent->grad, local_grad);
```

This is correct because a parameter may appear in multiple places in the graph (e.g. a weight matrix used in several layers via `tensor_view`). Each use contributes a separate gradient contribution, and they all accumulate into the same `grad` tensor.

This is also why `zero_grad()` must be called before each `backward()` — without it, gradients from the previous step remain in the grad tensors and corrupt the current update.

---

## A Complete Step in Code

```cpp
// --- Permanent setup (once) ---
optim::Adam adam(perm_arena, seq->parameters(), 0.001f);
nn::CrossEntropyLoss criterion;

// --- Per-batch training loop ---
for (each batch) {
    uint64_t pos = graph_arena->get_pos();   // save graph arena position

    // 1. Wrap batch data as non-gradient leaves
    auto *x = autograd::create_leaf(graph_arena, batch_features, false);
    auto *y = autograd::create_leaf(graph_arena, batch_labels,   false);

    // 2. Forward pass — builds the computation graph on graph_arena
    auto *pred = seq->forward(graph_arena, x);
    auto *loss = criterion.forward(graph_arena, pred, y);

    // 3. Zero gradients — clear perm_arena grad tensors from last step
    adam.zero_grad();

    // 4. Backward pass — walks the graph, accumulates into perm grad tensors
    autograd::backward(graph_arena, loss);

    // 5. Optimizer step — reads grad tensors, updates data tensors
    adam.step();

    // 6. Free the graph — all non-leaf Variables and temporaries gone in O(1)
    graph_arena->pop_to(pos);
}
```

---

## `saved_tensors` vs `parents`

These two fields serve distinct purposes and are easy to confuse:

| Field | Type | Purpose |
|---|---|---|
| `parents` | `Edge*` (→ `Variable*`) | Graph topology. Used by `build_topo` for traversal and by `backward_fn` to reach the parent's `grad` tensor. |
| `saved_tensors` | `Tensor**` | Data needed by the backward computation. Contains raw tensor pointers, not Variables. |

For `relu`, the `backward_fn` needs to know which elements of the input were positive:

```cpp
// Forward:
out->saved_tensors[0] = in->data;      // the pre-activation input tensor

// Backward closure:
tensor_relu_grad(local_grad,
                 self->saved_tensors[0],  // pre-activation values
                 self->grad);             // upstream gradient
tensor_add(parent->grad, parent->grad, local_grad);
```

Only the tensor data is saved, not the entire `Variable` — because the backward function only needs the float values, not the graph structure.

---

## Checking `requires_grad`

Operations propagate `requires_grad` conservatively: if **any** input has `requires_grad = true`, the output also has `requires_grad = true`. If no input requires a gradient, the output doesn't need one either and no backward function is registered.

```cpp
Variable *out      = arena->push<Variable>();
out->requires_grad = a->requires_grad || b->requires_grad;

if (out->requires_grad) {
    out->grad         = tensor_create_zeros(arena, ...);
    out->backward_fn  = [...](Variable *self, Arena *arena) { ... };
    // ... wire parents, saved_tensors ...
}
```

This means operations involving only non-gradient inputs (e.g. two input batch tensors multiplied together for data augmentation) produce no graph nodes and incur no autograd overhead.

---

## Memory Layout

```
perm_arena:                        graph_arena (one batch):
┌─────────────────────┐            ┌─────────────────────────────────────┐ ← saved pos
│ W1.data  W1.grad    │            │ batch x, y (leaf Variable structs)  │
│ b1.data  b1.grad    │◄──────────►│ z1, a1, z2 (non-leaf Variables)     │
│ W2.data  W2.grad    │  grad acc  │ loss Variable + grad seed           │
│ b2.data  b2.grad    │            │ backward temporaries (local_grad …) │
└─────────────────────┘            └─────────────────────────────────────┘
                                                  ↑ pop_to(saved pos) → all gone
```

After `pop_to`, only `perm_arena` content survives: the updated parameter data tensors and their (now-zeroed, ready for next step) gradient tensors.