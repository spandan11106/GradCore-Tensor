---
sidebar_position: 7
title: "Tensors in the Autograd Engine"
---

# Tensors in the Autograd Engine

The tensor module provides the data. The autograd engine provides the *memory of what you did with that data* — so it can run the chain rule backwards. Here's how the two connect.

## `autograd::Variable`

Every tensor that participates in a differentiable computation is wrapped in a `Variable`:

```cpp
struct Variable {
    Tensor  *data;           // The actual tensor values
    Tensor  *grad;           // Gradient accumulator (same shape as data)
    bool     requires_grad;  // Should we compute gradients for this?
    bool     is_leaf;        // Is this a parameter (true) or intermediate (false)?

    Edge    *parents;        // Inputs to the op that created this Variable
    uint32_t num_parents;

    Tensor **saved_tensors;  // Tensors saved for the backward pass
    uint32_t num_saved;

    uint32_t     reduction;      // For loss ops
    float        metadata_float; // alpha, delta, scale, etc.

    void (*backward_fn)(Variable *self, Arena *arena);
};
```

`Variable` structs live on the **graph arena** — they're freed en masse when you call `graph_arena->pop_to(pos)` after each batch.

## create_leaf — wrapping a tensor

```cpp
Variable *x = autograd::create_leaf(graph_arena, t_x, false);
```

`create_leaf` is how you turn a raw `Tensor` into something the autograd graph can track:

```cpp
Variable *create_leaf(Arena *arena, Tensor *data, bool requires_grad) {
    Variable *v     = arena->push<Variable>();
    v->data         = data;
    v->requires_grad = requires_grad;
    v->is_leaf      = true;
    v->backward_fn  = nullptr;  // Leaves don't have a backward fn

    if (requires_grad) {
        v->grad = tensor_create_zeros(arena, data->ndims, data->shape);
    } else {
        v->grad = nullptr;
    }
    return v;
}
```

- `requires_grad = true`: This is a **parameter** — the optimizer will update it, and backward will accumulate into `v->grad`.
- `requires_grad = false`: This is **data** (input batch, targets) — no gradients needed, no grad tensor allocated.

## How Ops Build the Graph

Every differentiable operation (e.g. `autograd::relu`) does three things:

1. **Compute** the output using the corresponding `tensor_*` function.
2. **Allocate** a new `Variable` for the result on the graph arena.
3. **Wire up** the backward function.

```cpp
Variable *relu(Arena *arena, Variable *in) {
    // 1. Forward computation
    Tensor *out_data = tensor_create_zeros(arena, in->data->ndims, in->data->shape);
    tensor_relu(out_data, in->data);

    // 2. Allocate output Variable
    Variable *out = arena->push<Variable>();
    out->data         = out_data;
    out->requires_grad = in->requires_grad;
    out->is_leaf      = false;

    if (out->requires_grad) {
        // Allocate grad tensor
        out->grad = tensor_create_zeros(arena, out_data->ndims, out_data->shape);

        // Wire parents (for graph traversal)
        out->num_parents = 1;
        out->parents     = arena->push_array<Edge>(1);
        out->parents[0]  = {in};

        // Save tensors needed by backward
        out->num_saved       = 1;
        out->saved_tensors   = arena->push_array<Tensor *>(1);
        out->saved_tensors[0] = in->data;   // Need input to compute gradient mask

        // 3. Backward closure
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

The result is a DAG (directed acyclic graph) of `Variable` nodes:

```
input ─► relu ─► linear ─► cross_entropy ─► loss
  │        │        │             │
  └parent  └parent  └parent      └parent
            │        │
           [saved:  [saved:
            input]   input, weight]
```

## Backward Pass

`autograd::backward(arena, loss_node)` reverses the graph:

```cpp
void backward(Arena *arena, Variable *loss_node) {
    // Start gradient: d(loss)/d(loss) = 1
    tensor_fill(loss_node->grad, 1.0f);

    // Topological sort of the computation graph
    std::vector<Variable *> topo;
    build_topo(loss_node, visited, topo);

    // Reverse traversal: apply each backward_fn
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn)
            (*it)->backward_fn(*it, arena);
    }
}
```

Each `backward_fn`:
1. Computes the local gradient contribution using `tensor_*_grad`.
2. Adds it to the parent's `grad` tensor (`tensor_add(parent->grad, parent->grad, local_grad)`).

Gradients accumulate via `tensor_add` — this is the correct behaviour for parameters that appear in multiple places in the graph.

## Memory Layout During a Batch

```
graph_arena (before batch):
┌─────────────────────────────┐  ← saved_pos
│                             │
│                             │

graph_arena (during forward):
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  batch data  │  activations │   Variables  │ grad tensors │
│  (x, y)      │  (post-relu) │   (DAG)      │              │
└──────────────┴──────────────┴──────────────┴──────────────┘

After pop_to(saved_pos): entirely reclaimed → back to empty
```

Everything allocated after `saved_pos` — intermediate tensors, grad tensors, the graph nodes themselves — vanishes in a single pointer reset. The only things that survive are on `perm_arena`: the model parameters and their gradient accumulators.

## saved_tensors vs parents

These serve different purposes:

- **`parents`**: Links for graph traversal (the topology). Points to `Variable` nodes.
- **`saved_tensors`**: Data needed by the backward function. Points to `Tensor` data, because the input data is what you need to compute the local gradient — not the whole Variable.

For `relu`, the backward needs the pre-activation values (to know which elements were positive). For `cross_entropy_loss`, the backward needs both the logits and the targets. Only what's strictly necessary is saved.
