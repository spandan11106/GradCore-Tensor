---
sidebar_position: 2
title: "Sequential"
---

# `nn::Sequential`

`Sequential` is a `Module` that holds an ordered list of child modules and runs them one after another during `forward()`. The output of layer `N` is fed directly as input to layer `N+1`. That's it. Simple, composable, and the backbone of every model in GradCore-Tensor.

**Header**: `include/nn/core/sequential.hpp`  
**Inherits**: `nn::Module`

:::info Under the hood
`Sequential::forward` calls each child module's `forward()` in registration order, passing the output of one as the input of the next. Each call produces a new `autograd::Variable` on the graph arena. Parameter registration and gradient bookkeeping are inherited from `Module` — `Sequential` itself adds nothing beyond the ordered traversal.
:::

---

## Construction

`Sequential` has a default constructor. Layers are added with `add()`:

```cpp
nn::Sequential seq;
auto* l1   = perm->push<nn::Linear>();    new (l1)   nn::Linear(perm, 784, 128);
auto* relu = perm->push<nn::ReLU>();      new (relu) nn::ReLU();
auto* l2   = perm->push<nn::Linear>();    new (l2)   nn::Linear(perm, 128, 10);

seq.add(l1);
seq.add(relu);
seq.add(l2);
```

In practice you won't construct `Sequential` directly — `nn::Model` does this for you. You call `model.add_layer()` which delegates to `sequential->add()`. See [Model](../models/model).

---

## API Reference

### `add(Module *module)`

```cpp
void add(Module *module);
```

Appends a layer to the sequence. Internally calls `register_module(module)`, which makes the layer's parameters visible to `parameters()` and `save()`.

Passing `nullptr` prints a warning and does nothing (it does not crash). This is a defensive check — if your layer constructor failed silently, you'd rather get a warning than a segfault three batches later.

### `get(size_t index)`

```cpp
Module *layer = seq.get(0);   // First layer
Module *layer = seq.get(2);   // Third layer
```

Returns the layer at `index`. Returns `nullptr` and prints an error if `index >= num_modules()`. Useful for inspecting or modifying a specific layer after construction.

### `num_modules()`

```cpp
size_t n = seq.num_modules();  // e.g. 3 for Linear→ReLU→Linear
```

Number of layers currently in the sequence.

### `empty()`

```cpp
if (seq.empty()) {
    // Nothing added yet
}
```

Returns `true` if no layers have been added.

### `forward(Arena *compute_arena, autograd::Variable *x)`

```cpp
autograd::Variable *out = seq.forward(graph_arena, input);
```

Runs the forward pass through all layers in order:

```
x → layer[0] → h1 → layer[1] → h2 → … → layer[N] → output
```

Each intermediate result is a new `Variable` allocated on `compute_arena`.

Guards against:
- Null input: returns `nullptr` and prints an error.
- Empty sequence: returns `x` unchanged with a warning.
- Any layer returning `nullptr`: propagates `nullptr` and prints which layer failed.

### `summary()`

```cpp
seq.summary();
```

Prints each layer's summary in order:

```
Sequential(
  (0): Linear(in=784, out=128, bias=true) [100480 params]
  (1): ReLU
  (2): Linear(in=128, out=10, bias=true) [1290 params]
) [101770 params]
```

---

## Accessing the Sequential from Model

When using `nn::Model`, you can reach the underlying `Sequential` with:

```cpp
nn::Sequential *seq = model.get_model();

// Manually run inference (after loading weights)
seq->eval();
autograd::Variable *out = seq->forward(graph_arena, input_var);
```

This is the pattern used in the tutorials for single-sample inference after training. See [Tutorial 2](/docs/tutorials/tutorial-2-mnist) for a complete example.

---

## Data Flow Diagram

![nn Module Overview](/img/seq.drawio.svg)

---

## Common Mistakes

**Adding a layer twice**: don't. Its parameters will appear twice in `parameters()`, which means the optimizer applies two gradient updates to the same weights per step. Very exciting. Not in a good way.

**Not calling `seq->eval()` before inference**: `BatchNorm` will keep updating running stats from your test data, and `Dropout` will randomly zero out predictions. Always switch to eval mode.

**Expecting `Sequential` to own memory**: it holds pointers to modules allocated on the arena. If you pop the arena past the point where those modules were allocated, accessing them is undefined behaviour. Allocate layers on `perm_arena` so they outlive the training loop.