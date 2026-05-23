---
title: High-Level to Low-Level Flow
sidebar_position: 2
---

# High-Level to Low-Level Flow (`nn` → `autograd` → `tensor`)

GradCore-Tensor follows a clean **layered architecture**. This design makes the library easy to use at a high level while remaining fully transparent and educational at lower levels.

## Architecture Layers

| Level              | Module          | Responsibility                              | Key Types                     |
|--------------------|-----------------|---------------------------------------------|-------------------------------|
| **High-Level**     | `nn`            | User-friendly APIs, model building          | `nn::Model`, `nn::Module`, `nn::Linear` |
| **Mid-Level**      | `autograd`      | Differentiable computation graph            | `autograd::Variable`          |
| **Low-Level**      | `tensor`        | Raw data storage and mathematical operations| `Tensor`, Arena allocators    |

All memory is managed through **arenas** (`perm_arena` for long-lived parameters, `graph_arena` for temporary forward/backward tensors).

## Information Flow Overview

1. **Model Construction** (`nn`)
   - Layers create and register learnable parameters as `autograd::Variable`s.
   - These Variables wrap underlying `Tensor` data.

2. **Forward Pass**
   - High-level layers call tensor operations through the autograd API.
   - Each operation builds the computation graph by creating new `Variable`s.

3. **Loss Calculation**
   - Output Variables + targets → loss function (still returns an `autograd::Variable`).

4. **Backward Pass**
   - `loss->backward()` traverses the graph and populates `.grad` tensors.

5. **Optimization**
   - Optimizers read gradients and update parameter data.

6. **Memory Management**
   - All tensors are allocated via arenas for efficiency and cache locality.

---

## Concrete Example: `nn::Linear` Forward Pass

Here is how data flows through a **Linear layer** from high-level API down to raw tensor operations.

### 1. High-Level: `nn::Linear::forward()`

```cpp
// nn/layers/linear.hpp
autograd::Variable* Linear::forward(autograd::Variable* input) {
    // input is Variable from previous layer

    // High-level operation → routed through autograd
    auto output = autograd::matmul(input, weight);   // weight is a learnable Variable

    if (has_bias) {
        output = autograd::add(output, bias);        // supports broadcasting
    }

    return output;   // Returns new Variable with graph connection
}
```

### 2. Mid-Level: `autograd::matmul()` and `autograd::add()`

```cpp
// autograd/ops.hpp
Variable* matmul(Variable* a, Variable* b) {
    // Call low-level tensor operation
    Tensor* result_data = tensor_matmul(a->data, b->data);

    // Create new node in computation graph
    Variable* result = new Variable(result_data, true);  // usually requires_grad = true

    // Record graph edges for backward pass
    result->parents = {a, b};
    result->backward_fn = matmul_backward;           // function pointer
    result->saved_tensors = {a->data, b->data};      // needed for gradient computation

    return result;
}
```

### 3. Low-Level: `tensor_matmul()` (Core Tensor API)

```cpp
// tensor/ops/arithmetic.hpp
Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    // Shape validation and broadcasting logic
    Shape out_shape = {a->shape[0], b->shape[1]};
    
    // Allocate result using graph arena (temporary)
    Tensor* out = tensor_create_zeros(out_shape, graph_arena);

    // Actual matrix multiplication (OpenMP parallelized)
    #pragma omp parallel for
    for (int i = 0; i < a->shape[0]; ++i) {
        for (int j = 0; j < b->shape[1]; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a->shape[1]; ++k) {
                sum += a->at(i, k) * b->at(k, j);
            }
            out->at(i, j) = sum;
        }
    }
    return out;
}
```

### 4. Backward Flow (When `loss->backward()` is Called)

- Autograd engine walks the graph in reverse topological order.
- Calls matmul_backward(Variable* output) using saved tensors.
- Computes and accumulates gradients into weight->grad and input->grad.
- Optimizer then updates: weight->data = weight->data - lr * weight->grad.

## Full Training Loop Flow Summary
```cpp
// High-level usage
model.compile(OptimizerType::ADAMW, LossType::CROSS_ENTROPY, lr);
model.train(train_X, train_Y);   // internally does:

// Inside model.train():
for each batch {
    auto output = model.forward(batch);           // nn → autograd → tensor
    auto loss   = compute_loss(output, target);   // autograd loss
    loss->backward();                             // autograd backward pass
    optimizer.step();                             // optim uses .grad
    optimizer.zero_grad();                        // clear for next iteration
}
```

:::note
This layered design allows you to:

Use high-level nn::Model for quick experiments
Drop down to raw autograd::Variable + tensor_* functions for research/custom layers
Understand every step of the computation
:::

