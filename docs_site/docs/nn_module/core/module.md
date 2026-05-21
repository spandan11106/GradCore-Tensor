---
sidebar_position: 1
title: "Module"
---

# `nn::Module`

`Module` is the abstract base class for every layer, activation, loss, and container in GradCore-Tensor. If you want to write a custom layer, you subclass `Module`. If you want to understand how a layer works internally, you understand `Module` first.

:::info Calls down to autograd
`Module::forward` is expected to call `autograd::*` operations, which in turn call `tensor_*` functions. `Module` itself does not touch raw tensors directly — it delegates to `autograd::Variable` ops so that the computation graph is built automatically for backpropagation.
:::

**Header**: `include/nn/core/module.hpp`

---

## Class Declaration (simplified)

```cpp
namespace gradientcore::nn {

class Module {
protected:
    std::vector<autograd::Variable *> _parameters;
    std::vector<Module *>             _modules;
    bool                              _training;

public:
    Module();
    virtual ~Module() = default;

    // Training / eval mode
    virtual void train(bool mode = true);
    void         eval();
    bool         is_training() const;

    // Registration
    void register_parameter(autograd::Variable *param);
    void register_module(Module *module);
    void register_forward_hook(ForwardHook hook);

    // Parameter access
    virtual std::vector<autograd::Variable *>              parameters();
    virtual std::map<std::string, autograd::Variable *>    named_parameters();
    virtual uint64_t  num_parameters();
    virtual uint64_t  num_trainable_parameters();

    // Persistence
    bool save(const std::string &path, const std::string &format = "binary") const;
    bool load(const std::string &path, Arena *arena);

    // Summary
    virtual void summary();

    // The one method every subclass must implement
    virtual autograd::Variable *forward(Arena *compute_arena,
                                        autograd::Variable *x) = 0;

    // Call operator — runs forward + hooks
    autograd::Variable *operator()(Arena *compute_arena, autograd::Variable *x);
};

} // namespace gradientcore::nn
```

---

## Training vs Eval Mode

Modules carry a `_training` flag that layers like `BatchNorm` and `Dropout` use to behave differently at training vs inference time.

### `train(bool mode = true)`

```cpp
model.train();        // Switch to training mode (default)
model.train(false);   // Switch to eval mode
```

Recursively sets `_training` on all registered sub-modules. You rarely call this directly — `Trainer::fit` calls `model->train(true)` at the start of training and `model->eval()` at the end.

### `eval()`

```cpp
model.eval();
// equivalent to model.train(false)
```

Switches the module (and all sub-modules) to evaluation mode. In eval mode:
- **`BatchNorm`** uses its stored running statistics instead of computing batch statistics.
- **`Dropout`** becomes a pass-through — no neurons are dropped.

:::warning Always call `eval()` before inference
Forgetting this is one of the most common bugs in deep learning code. A model left in training mode will give different (and wrong) results every time it runs due to Dropout randomness and BatchNorm's continued updating of running stats.
:::

### `is_training()`

```cpp
if (layer->is_training()) {
    // apply dropout, use batch stats, etc.
}
```

---

## Registration

Layers register their learnable parameters and child modules in their constructors. This is what makes `parameters()` and `save()` work automatically — the module hierarchy is a tree, and traversal collects everything.

### `register_parameter(autograd::Variable *param)`

```cpp
// Inside a custom layer's constructor:
weight = autograd::create_leaf(perm_arena, w_tensor, /*requires_grad=*/true);
register_parameter(weight);
```

Adds `param` to `_parameters`. Only call this for **learnable** variables (`requires_grad = true`). The parameter will be included in `parameters()`, counted by `num_parameters()`, and saved/loaded by `save()`/`load()`.

### `register_module(Module *module)`

```cpp
// Inside Sequential::add():
register_module(module);
```

Adds a child `Module` to `_modules`. Parameters of child modules are collected recursively by `parameters()`.

### `register_forward_hook(ForwardHook hook)`

```cpp
using ForwardHook = std::function<void(autograd::Variable *)>;

layer->register_forward_hook([](autograd::Variable *out) {
    std::cout << "Output shape: " << out->data->shape[0]
              << "x" << out->data->shape[1] << "\n";
});
```

Hooks are called with the output `Variable` after every `forward()` call. Useful for debugging — logging activation statistics, detecting NaNs, etc. — without modifying layer code. Multiple hooks can be registered; they fire in registration order.

---

## Parameter Access

### `parameters()`

```cpp
auto params = model.parameters();
// Returns std::vector<autograd::Variable *>
```

Returns a flat list of **all** learnable parameters in the module and all its children, in depth-first traversal order. This is what the optimizer receives.

Results are cached after the first call. The cache is invalidated when `register_parameter` or `register_module` is called.

### `named_parameters()`

```cpp
auto named = model.named_parameters();
for (auto& [name, param] : named) {
    std::cout << name << ": " << param->data->size << " elements\n";
}
```

Returns the same parameters as a `std::map<std::string, Variable*>` with auto-generated dot-notation names (e.g. `"0.0"`, `"0.1"`, `"1.0"`). Useful for debugging and selective freezing.

### `num_parameters()`

```cpp
std::cout << "Total params: " << model.num_parameters() << "\n";
// e.g. "Total params: 101770"  (MNIST MLP)
```

Sum of element counts across all parameters. Used for sanity-checking your architecture.

### `num_trainable_parameters()`

```cpp
std::cout << "Trainable: " << model.num_trainable_parameters() << "\n";
```

Same as `num_parameters()` but only counts parameters where `requires_grad == true`. Useful if you have frozen layers.

---

## Persistence

### `save(path, format)`

```cpp
bool ok = module.save("model.bin", "binary");
bool ok = module.save("model.json", "json");
bool ok = module.save("model.csv",  "csv");
```

Saves all parameters to disk. Three formats are supported:

| Format | Notes |
|---|---|
| `"binary"` | Compact, fast, not human-readable. **Recommended.** |
| `"json"` | Base64-encodes float data. Loading is simplified — use binary for production. |
| `"csv"` | One row per element: `param_index, element_index, value`. Inspectable but large. |

The binary format writes: a `uint32_t` parameter count, then for each parameter a `uint64_t` size followed by the raw `float` bytes. Simple and reliable.

Returns `true` on success, `false` on file-open or structure errors.

### `load(path, arena)`

```cpp
bool ok = module.load("model.bin", perm_arena);
```

Loads parameters from disk back into the module's existing parameter tensors. The module must already be constructed with the same architecture — `load` does not create layers, it fills them.

Checks:
- Parameter count must match exactly.
- Each parameter's element count must match exactly.

If either check fails, `load` returns `false` and prints an error. **Always check the return value.**

:::tip Use `model.load()` not `module.load()` directly
The `nn::Model` class wraps `load` with a nicer interface. See [Model](../models/model).
:::

---

## `forward()` — The One Method You Must Implement

```cpp
virtual autograd::Variable *forward(Arena *compute_arena,
                                    autograd::Variable *x) = 0;
```

Takes an input `Variable` and returns an output `Variable`. All intermediate tensors must be allocated on `compute_arena` (the graph arena), which will be rewound after each batch.

The `compute_arena` pointer is passed explicitly so each layer knows exactly where to allocate its outputs — there are no hidden global allocators.

### Calling convention

```cpp
autograd::Variable *out = layer->forward(graph_arena, x);
// or equivalently, using operator():
autograd::Variable *out = (*layer)(graph_arena, x);
```

`operator()` calls `forward` and then fires any registered hooks on the output.

### Implementing a custom layer

```cpp
class MyScaleLayer : public nn::Module {
    float scale_factor;
public:
    MyScaleLayer(float s) : scale_factor(s) {}

    autograd::Variable *forward(Arena *compute_arena,
                                autograd::Variable *x) override {
        // Delegate to an autograd op — this builds the graph automatically
        return autograd::scale(compute_arena, x, scale_factor);
    }
};
```

---

## `summary()`

```cpp
module.summary();
```

Prints a brief description to stdout:

```
Module Summary:
  Total Parameters: 101770
  Trainable Parameters: 101770
  Training Mode: true
```

Concrete subclasses (like `Linear` and `Sequential`) override this with more specific output.