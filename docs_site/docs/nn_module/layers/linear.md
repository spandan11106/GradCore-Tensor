---
sidebar_position: 1
title: "Linear"
---

# `nn::Linear`

The fully-connected layer. Takes an input of shape `[batch, in_features]`, multiplies by a weight matrix, optionally adds a bias, and produces output of shape `[batch, out_features]`.

**Header**: `include/nn/layers/linear.hpp`  
**Inherits**: `nn::Module`

:::info What it calls
`Linear::forward` calls `autograd::matmul` and, if bias is enabled, `autograd::add`. These in turn call `tensor_matmul` and `tensor_add` from the tensor module. Gradients are computed automatically during `autograd::backward`.
:::

---

## Constructor

```cpp
nn::Linear(Arena *perm_arena,
           uint32_t in_features,
           uint32_t out_features,
           bool use_bias = true);
```

| Parameter | Type | Description |
|---|---|---|
| `perm_arena` | `Arena*` | The **permanent** arena. Weight and bias tensors live here for the program's lifetime. |
| `in_features` | `uint32_t` | Number of input features per sample. |
| `out_features` | `uint32_t` | Number of output features per sample. |
| `use_bias` | `bool` | Whether to add a learnable bias term. Default `true`. |

The constructor:
1. Allocates a weight tensor of shape `[in_features, out_features]` on `perm_arena`.
2. Initialises weights with **Kaiming Normal** (appropriate for ReLU-family activations).
3. If `use_bias`, allocates a bias tensor of shape `[1, out_features]` and initialises it to zeros.
4. Calls `register_parameter` for both, so they appear in `parameters()` and get saved/loaded.

### Building a Linear layer

```cpp
auto* l1 = perm_arena->push<nn::Linear>();
new (l1) nn::Linear(perm_arena, 784, 128);
model.add_layer(l1);
```

The placement `new` pattern is required because `nn::Linear` is allocated on the arena (not the heap), so the constructor must be called manually.

---

## Public Members

```cpp
uint32_t in_features;
uint32_t out_features;
bool     has_bias;

autograd::Variable *weight;   // shape [in_features, out_features]
autograd::Variable *bias;     // shape [1, out_features], or nullptr
```

`weight` and `bias` are `public` for direct inspection. Don't reassign them — register new parameters through the constructor instead.

---

## Forward Pass

```cpp
autograd::Variable *forward(Arena *compute_arena,
                            autograd::Variable *x) override;
```

Computes:

```
out = x @ weight        (shape: [batch, out_features])
out = out + bias        (if use_bias, broadcast over batch dimension)
```

Where `@` is matrix multiplication via `autograd::matmul`.

### Input requirements

| Requirement | Detail |
|---|---|
| `x->data->ndims == 2` | Input must be 2-dimensional: `[batch_size, in_features]` |
| `x->data->shape[1] == in_features` | Feature dimension must match the layer's `in_features` |

If either check fails, `forward` returns `nullptr` and prints a diagnostic. A `nullptr` return from any layer causes `Sequential::forward` to abort and print which layer failed.

### Output

- Shape: `[batch_size, out_features]`
- Allocated on `compute_arena` (the graph arena) — freed when the batch is done.

### Graph built by forward

```
x ──► matmul ──► (x @ weight) ──► add ──► out
         │                           │
       weight                      bias
```

Both `weight` and `bias` are leaves with `requires_grad = true`, so gradients accumulate into `weight->grad` and `bias->grad` during backpropagation.

---

## Parameter Count

For a layer with `in_features = I`, `out_features = O`, and bias enabled:

```
weights:   I × O
bias:      O
total:     I × O + O  =  O × (I + 1)
```

Example — `Linear(784, 128)`:

```
784 × 128 + 128 = 100,480 parameters
```

---

## Weight Initialisation

Weights are initialised with **Kaiming Normal** by `reset_parameters()`:

```cpp
void reset_parameters() {
    init::kaiming_normal_(weight);   // std = sqrt(2 / fan_in)
    init::zeros_(bias);
}
```

Kaiming Normal draws from `N(0, sqrt(2/in_features))`. The `sqrt(2)` factor corrects for ReLU zeroing out roughly half of all activations, maintaining variance across layers.

You can re-initialise with a different scheme after construction:

```cpp
init::xavier_uniform_(l1->weight);
init::zeros_(l1->bias);
```

See [Initialization](../utils/initialization) for all available schemes.

---

## `summary()`

```cpp
layer->summary();
// Linear(in=784, out=128, bias=true) [100480 params]
```

---

## Usage Examples

### Single hidden layer

```cpp
auto* l = perm->push<nn::Linear>(); new (l) nn::Linear(perm, 8, 1);
model.add_layer(l);
// Input: [batch, 8]  →  Output: [batch, 1]
```

### No bias (useful when followed by BatchNorm)

```cpp
auto* l = perm->push<nn::Linear>(); new (l) nn::Linear(perm, 128, 64, false);
//   ↑ use_bias = false
```

`BatchNorm` learns its own shift parameter (`beta`), so the bias in `Linear` is redundant when the two are used together. Removing it saves parameters and avoids a redundant degree of freedom.

### Manual inference (bypassing Model)

```cpp
seq->eval();
uint32_t shape[2] = {1, 784};
Tensor *t = tensor_create(graph_arena, 2, shape);
std::memcpy(t->storage->data, sample.data(), 784 * sizeof(float));

auto* x   = autograd::create_leaf(graph_arena, t, false);
auto* out = seq->forward(graph_arena, x);

float prediction = out->data->storage->data[out->data->offset];
```