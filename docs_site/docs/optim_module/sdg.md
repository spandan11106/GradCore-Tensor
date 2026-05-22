---
sidebar_position: 2
title: "SGD"
---

# `optim::SGD`

Stochastic Gradient Descent. The oldest optimizer, still competitive, and the easiest to reason about. Every weight moves directly opposite its gradient, scaled by the learning rate.

**Header**: `include/optim/sgd.hpp`  
**Namespace**: `gradientcore::optim`

:::info What it calls
`SGD::step` calls `tensor_copy`, `tensor_scale`, and `tensor_sub` from the tensor module. It allocates a temporary scaled-gradient tensor on the graph arena during each `step` call and frees it immediately via `pop_to`. There are no persistent state tensors — SGD holds no memory between steps beyond the parameters themselves.
:::

---

## Update Rule

```
w ← w - lr * ∇w
```

That's it. No momentum, no adaptive learning rates, no weight decay. The gradient is scaled by `lr` and subtracted from the weight.

---

## Constructor

```cpp
optim::SGD(const std::vector<autograd::Variable *> &params,
           float lr);
```

| Parameter | Type | Description |
|---|---|---|
| `params` | `vector<Variable*>` | All learnable parameters. Get via `model->parameters()`. |
| `lr` | `float` | Learning rate. A good starting range is `0.001`–`0.1`. |

SGD is the only optimizer that **does not** take a `perm_arena` — it allocates no persistent state, so it has nothing to put there.

```cpp
auto params = seq->parameters();
optim::SGD optimizer(params, 0.01f);
```

---

## Methods

### `step(Arena *temp_arena)`

```cpp
void step(Arena *temp_arena);
```

Applies one gradient update to every parameter that has `requires_grad = true` and a non-null, non-zero `grad` tensor.

For each parameter `p`:

```cpp
// Allocate temporary tensor on temp_arena
Tensor *scaled_grad = tensor_create_zeros(temp_arena, p->grad->ndims, p->grad->shape);
tensor_copy(scaled_grad, p->grad);
tensor_scale(scaled_grad, learning_rate);    // scaled_grad = lr * grad
tensor_sub(p->data, p->data, scaled_grad);   // w = w - lr * grad
temp_arena->pop_to(start_pos);               // free scaled_grad immediately
```

The temporary tensor is freed inside each parameter's update, not after the full loop — so memory usage is bounded by the size of a single parameter tensor.

### `zero_grad()`

```cpp
void zero_grad();
```

Calls `tensor_clear` on every parameter's `grad` tensor. Must be called before each `backward()` pass to prevent gradient accumulation across steps.

---

## Full Training Loop Example

```cpp
auto* perm  = Arena::create(MiB(512), MiB(32), true);
auto* graph = Arena::create(MiB(256), MiB(16), true);

// Build model (parameters live on perm)
nn::Sequential seq;
auto* l1 = perm->push<nn::Linear>(); new (l1) nn::Linear(perm, 2, 4);
auto* t1 = perm->push<nn::Tanh>();   new (t1) nn::Tanh();
auto* l2 = perm->push<nn::Linear>(); new (l2) nn::Linear(perm, 4, 1);
seq.add(l1); seq.add(t1); seq.add(l2);

// Construct optimizer — no perm_arena needed
optim::SGD optimizer(seq.parameters(), 0.05f);
nn::MSELoss criterion;

for (int epoch = 0; epoch < 2000; epoch++) {
    uint64_t pos = graph->get_pos();

    // ... create batch tensors x, y ...

    auto* pred = seq.forward(graph, x);
    auto* loss = criterion.forward(graph, pred, y);

    optimizer.zero_grad();
    autograd::backward(graph, loss);
    optimizer.step(graph);    // graph used for temporary scaled-grad

    graph->pop_to(pos);
}
```

---

## When to Use SGD

SGD is the right choice when:

- You want a **simple, reproducible baseline**. SGD's behaviour is deterministic given a fixed seed and has no adaptive momentum that can interact unexpectedly with your learning rate schedule.
- You are using **learning rate schedules** (warmup, cosine decay). SGD's update rule is straightforward enough that the effect of a learning rate change is exactly what you'd expect.
- You are training **large models** where Adam's per-parameter second moment tensors would consume significant memory. SGD stores nothing extra.

SGD without momentum tends to converge more slowly than Adam on most tasks. If training time matters more than memory, start with Adam.

:::tip SGD vs Adam on MNIST
The [MNIST tutorial](../tutorials/tutorial-2-mnist.md) uses Adam with `lr=0.0005` to reach 97.77% accuracy in 40 epochs. Running the same architecture with SGD at `lr=0.05` typically reaches similar accuracy in roughly 200 epochs. Both work; they differ in speed of convergence.
:::