---
sidebar_position: 1
title: "Optimizer Overview"
---

# Optimizers

Optimizers are responsible for updating model parameters after every backward pass. They read the gradient that `autograd::backward` accumulated into each `Variable::grad`, compute an update, and apply it to `Variable::data`. Everything else in the training loop — the forward pass, the loss, the backward pass — exists to produce those gradients. The optimizer is where gradients become learning.

**Header directory**: `include/optim/`  
**Namespace**: `gradientcore::optim`

:::info What optimizers actually do
Every optimizer in GradCore-Tensor follows the same contract. `step(Arena*)` iterates over every parameter that has `requires_grad = true` and a non-null `grad` tensor, reads the gradient values, updates any internal state tensors (moment estimates, accumulated squared gradients, etc.), and writes the new parameter values back. `zero_grad()` zeroes every gradient tensor so the next backward pass starts clean.
:::

---

## The Two-Arena Pattern

Optimizers interact with both arenas:

```
perm_arena:  model parameters (data + grad), optimizer state (m, v, sum_sq, …)
graph_arena: passed to step() for any temporaries (e.g. SGD's scaled gradient)
```

State tensors (`m`, `v`, etc.) are allocated **once** in the constructor on `perm_arena` and persist for the entire training run. They are never freed during training — they accumulate information across batches.

```cpp
optim::Adam adam(perm_arena, params, 0.001f);
//                ↑ state tensors allocated here, live until perm_arena->destroy()

for (int epoch = 0; epoch < 40; epoch++) {
    for (auto& batch : loader) {
        uint64_t pos = graph_arena->get_pos();
        // ... forward, loss, backward ...
        adam.zero_grad();
        adam.step(graph_arena);   // graph_arena used for temporaries only
        graph_arena->pop_to(pos);
    }
}
```

---

## Getting Parameters

Every optimizer constructor takes `const std::vector<autograd::Variable*>& params`. Obtain this from your model:

```cpp
// From nn::Sequential directly
auto params = seq->parameters();

// From nn::Model (via the wrapped Sequential)
auto params = model.get_model()->parameters();
```

`parameters()` returns a flat list of all `Variable*` with `requires_grad = true` across the full module hierarchy. See [Module](../nn_module/core/module.md#parameters) for details.

---

## Choosing an Optimizer

| Optimizer | Best for | Notes |
|---|---|---|
| `SGD` | Simple tasks, strong baselines | Needs careful LR tuning |
| `Adam` | General purpose | Great default; can over-fit on small datasets |
| `AdamW` | Regularised training | Like Adam but with correct weight decay |
| `RMSprop` | Non-stationary problems, RNNs | Classic adaptive method |
| `Adagrad` | Sparse gradients | LR decays to zero — use carefully |
| `LBFGS` | Small datasets, fine-tuning | Requires a closure; not mini-batch friendly |

For most new experiments, start with **Adam** (`lr=0.001`) or **AdamW** (`lr=0.001, weight_decay=0.01`). Switch to SGD with momentum if you need a reproducible baseline or are training very large models.

---

## Quick Reference

| Class | State per param | Key hyperparams |
|---|---|---|
| `SGD` | None | `lr` |
| `Adam` | `m`, `v` | `lr`, `beta1`, `beta2`, `eps` |
| `AdamW` | `m`, `v` | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` |
| `RMSprop` | `v` | `lr`, `alpha`, `eps`, `weight_decay` |
| `Adagrad` | `sum_sq` | `lr`, `eps`, `weight_decay` |
| `LBFGS` | history deque | `lr`, `history_size`, `tol_grad`, `tol_change` |

---

## Using Optimizers with `nn::Model`

The high-level `Model` API creates and manages the optimizer for you via `compile()`:

```cpp
model.compile(nn::OptimizerType::ADAMW,
              nn::LossType::HUBER,
              /*lr=*/0.001f, /*epochs=*/200, /*batch=*/128);
model.train(X_train, Y_train);
```

If you need a combination not in the `OptimizerType` enum, or you need access to optimizer internals (e.g. learning rate scheduling), construct the optimizer directly and use `nn::Trainer`:

```cpp
optim::RMSprop rms(perm_arena, seq->parameters(), 0.01f, 0.99f);
nn::HuberLoss  huber(2.0f);
nn::Trainer<optim::RMSprop, nn::HuberLoss> trainer(seq, &rms, &huber, graph);
trainer.fit_dataloader(loader, 100);
```

---

## The Contiguous Fast Path

Every optimizer checks `tensor_is_contiguous` on the gradient and parameter tensors. If both are contiguous — which is true for all parameters created by `nn::Linear`, `nn::BatchNorm`, etc. — the optimizer uses direct pointer arithmetic with optional OpenMP parallelisation. If either is non-contiguous, it falls back to stride-aware indexing via `tensor_get_flat_index`.

In practice, model parameters are always contiguous, so the fast path is always taken during training.