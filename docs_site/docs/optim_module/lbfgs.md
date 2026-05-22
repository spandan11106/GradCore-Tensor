---
sidebar_position: 7
title: "L-BFGS"
---

# `optim::LBFGS`

L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton optimizer. Rather than following the gradient directly, it uses a history of past gradient changes to build an approximation of the inverse Hessian — the curvature of the loss landscape — and uses that approximation to compute a better search direction. The result is much faster convergence per gradient evaluation than any first-order method, at the cost of requiring a closure (a function that re-evaluates the loss and its gradients on demand for the line search) and being fundamentally incompatible with mini-batch stochastic training.

**Header**: `include/optim/lbfgs.hpp`  
**Namespace**: `gradientcore::optim`

:::warning Not for mini-batch training
L-BFGS assumes a deterministic, consistent loss function. In stochastic mini-batch training, different batches produce different gradients, and L-BFGS's curvature estimate becomes meaningless and potentially harmful. Use L-BFGS only with the full dataset (or a fixed, large subset) evaluated in the closure.
:::

:::info What it calls
`LBFGS::step` calls several `tensor_*` utilities: `tensor_dot_1d`, `tensor_copy`, `tensor_scale`, `tensor_add`, `tensor_sub`. It also calls `flatten_params`, `flatten_grads`, and `unflatten_params` from `optim_utils.hpp` to work with the parameter space as a single flat vector. A double-loop L-BFGS two-loop recursion computes the search direction; Wolfe-condition backtracking line search determines the step size.
:::

---

## How L-BFGS Works

Standard Newton's method computes updates as `w -= H⁻¹ * g` where `H` is the Hessian. Computing and inverting the Hessian exactly is `O(n²)` in memory and `O(n³)` in time — completely infeasible for anything beyond tiny models.

L-BFGS approximates `H⁻¹ * g` using only the last `m` pairs of `(s_k, y_k)`:

```
s_k = w_{k+1} - w_k          # parameter change (step taken)
y_k = g_{k+1} - g_k          # gradient change
ρ_k = 1 / (y_k^T * s_k)
```

The **two-loop recursion** then computes the search direction `d = -H_k⁻¹ * g_k` in O(m * n) time and O(m * n) space, where `m` is the history size (typically 10–20) and `n` is the number of parameters. This makes L-BFGS practical for problems with up to millions of parameters, provided the loss can be evaluated cheaply.

---

## Update Rule (Two-Loop Recursion)

```
# Initialise search direction
d = -g    (negative gradient)

# Backward pass (newest to oldest):
for i = k-1, ..., k-m:
    α_i = ρ_i * (s_i^T * d)
    d   = d - α_i * y_i

# Scale by initial Hessian approximation:
γ = 1 / (ρ_{k-1} * (y_{k-1}^T * y_{k-1}))
d = γ * d

# Forward pass (oldest to newest):
for i = k-m, ..., k-1:
    β_i = ρ_i * (y_i^T * d)
    d   = d + (α_i - β_i) * s_i

# d is the L-BFGS search direction (approximates -H⁻¹ * g)
```

A backtracking Armijo line search (checking the sufficient decrease condition) determines the step size `t`, then updates:

```
w_{k+1} = w_k + t * d
```

---

## Constructor

```cpp
optim::LBFGS(Arena *perm_arena,
             const std::vector<autograd::Variable *> &params,
             float    lr           = 1.0f,
             uint32_t history_size = 20,
             float    tol_grad     = 1e-7f,
             float    tol_change   = 1e-9f);
```

| Parameter | Default | Description |
|---|---|---|
| `perm_arena` | — | Permanent arena. History tensors (`s`, `y`) and the search direction `d` live here. |
| `params` | — | Learnable parameters from `model->parameters()`. |
| `lr` | `1.0` | Initial step size for the line search. The line search may accept a smaller step. |
| `history_size` | `20` | Maximum number of `(s, y)` pairs retained. More history = better curvature estimate, more memory. |
| `tol_grad` | `1e-7` | Gradient norm convergence threshold. `step()` returns without updating if `‖g‖ < tol_grad`. |
| `tol_change` | `1e-9` | Reserved for parameter change convergence (currently checked via line search iterations). |

### Memory overhead

L-BFGS stores `2 * history_size` parameter-shaped tensors on `perm_arena` (one `s` and one `y` per history entry), plus one additional tensor `d` for the current search direction. For a model with `N` parameters and `m` history steps, this is `(2m + 1) * N * sizeof(float)` bytes — significant for large models.

---

## Methods

### `step(Arena *temp_arena, std::function<float()> closure)`

```cpp
float step(Arena *temp_arena, std::function<float()> closure);
```

This signature differs from all other optimizers. L-BFGS **requires a closure** — a callable that:

1. Computes a full forward pass and loss.
2. Calls `autograd::backward` to populate gradients.
3. Returns the scalar loss value as a `float`.

The line search inside `step()` calls the closure multiple times (up to 20 times) to find an acceptable step size.

Returns the loss value at the accepted step.

```cpp
// Define the closure
auto closure = [&]() -> float {
    uint64_t pos = graph->get_pos();

    // Full forward pass (full dataset, not a mini-batch)
    auto* x    = autograd::create_leaf(graph, X_full, false);
    auto* y    = autograd::create_leaf(graph, Y_full, false);
    auto* pred = seq.forward(graph, x);
    auto* loss = criterion.forward(graph, pred, y);

    float loss_val = loss->data->storage->data[loss->data->offset];

    lbfgs.zero_grad();
    autograd::backward(graph, loss);

    graph->pop_to(pos);
    return loss_val;
};

float final_loss = lbfgs.step(graph, closure);
```

:::warning Closure memory management
The closure is called repeatedly inside `step()`. Each call must leave the graph arena in the same state it was found — use `uint64_t pos = graph->get_pos()` and `graph->pop_to(pos)` at the start and end of each closure invocation. If the closure leaks graph arena memory, the allocator will grow unboundedly across line search iterations.
:::

### `zero_grad()`

```cpp
void zero_grad();
```

Zeroes every parameter's `grad` tensor. Must be called **inside the closure**, before `autograd::backward`, to prevent gradient accumulation across closure calls.

---

## Full Example

```cpp
auto* perm  = Arena::create(MiB(2048), MiB(128), true);
auto* graph = Arena::create(MiB(512),  MiB(32),  true);

// Small dataset, full-batch training
nn::Sequential seq;
auto* l1 = perm->push<nn::Linear>(); new (l1) nn::Linear(perm, 2, 16);
auto* r1 = perm->push<nn::Tanh>();   new (r1) nn::Tanh();
auto* l2 = perm->push<nn::Linear>(); new (l2) nn::Linear(perm, 16, 1);
seq.add(l1); seq.add(r1); seq.add(l2);

// Full dataset as tensors on perm
uint32_t shape_X[2] = {(uint32_t)X_train.size(), 2};
uint32_t shape_Y[2] = {(uint32_t)Y_train.size(), 1};
Tensor *t_X = tensor_create(perm, 2, shape_X);
Tensor *t_Y = tensor_create(perm, 2, shape_Y);
// ... copy X_train and Y_train into t_X and t_Y ...

optim::LBFGS lbfgs(perm, seq.parameters(), /*lr=*/1.0f, /*history=*/20);
nn::MSELoss criterion;

for (int iter = 0; iter < 100; iter++) {
    auto closure = [&]() -> float {
        uint64_t pos = graph->get_pos();

        auto* x    = autograd::create_leaf(graph, t_X, false);
        auto* y    = autograd::create_leaf(graph, t_Y, false);
        auto* pred = seq.forward(graph, x);
        auto* loss = criterion.forward(graph, pred, y);

        float loss_val = loss->data->storage->data[loss->data->offset];

        lbfgs.zero_grad();
        autograd::backward(graph, loss);

        graph->pop_to(pos);
        return loss_val;
    };

    float loss_val = lbfgs.step(graph, closure);

    if (iter % 10 == 0)
        std::cout << "Iter " << iter << " | Loss: " << loss_val << "\n";
}
```

---

## Line Search

`LBFGS::step` uses backtracking line search with the Armijo sufficient decrease condition:

```
L(w + t * d) ≤ L(w) + c₁ * t * (g^T * d)
```

where `c₁ = 1e-4` (standard Armijo constant). Starting from `t = lr`, the step size is halved up to 20 times until the condition is satisfied. If no acceptable step is found after 20 halvings, the last attempted step is accepted.

---

## History Size

| `history_size` | Memory | Approximation quality |
|---|---|---|
| `5` | Low | Minimal history — works for well-conditioned problems |
| `10` | Moderate | Good balance for most smooth objectives |
| `20` (default) | Moderate-high | Strong curvature approximation for most tasks |
| `50+` | High | Rarely beneficial; the gain from additional history diminishes quickly |

---

## When to Use L-BFGS

L-BFGS excels at:

- **Full-batch training on small datasets** — when you can afford to evaluate the full dataset in each closure call. The XOR network in `test/autograd/test_xor.cpp` is a canonical example.
- **Fine-tuning from a good initialisation** — when you want fast local convergence and the loss is approximately quadratic near the current point.
- **Scientific computing tasks** — physics-informed neural networks, PDE solving, any setting where the loss is smooth and deterministic.

L-BFGS is the wrong choice when:
- You have a large dataset where full-batch evaluation is too expensive.
- Your loss is non-smooth (e.g. hinge loss at the boundary, ReLU with many saturated units).
- You need mini-batch stochastic training for generalisation reasons (noise in SGD/Adam acts as implicit regularisation).

For the vast majority of deep learning tasks, **Adam** or **AdamW** will be your optimizer.