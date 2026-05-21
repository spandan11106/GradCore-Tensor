---
sidebar_position: 1
title: "Trainer"
---

# `nn::Trainer<OptimizerType, LossType>`

`Trainer` is the training loop engine. It owns no model weights — it receives a `Module*`, an optimizer, a loss function, and a graph arena, then runs the forward → loss → backward → step cycle for as many epochs as you ask for.

`nn::Model::train()` is a wrapper around `Trainer`. You only need to use `Trainer` directly when you want a combination of optimizer and loss that `Model` doesn't expose, or when you need full control over the training loop.

**Header**: `include/nn/training/trainer.hpp`  
**Namespace**: `gradientcore::nn`

:::info What it calls
On every batch, `Trainer` calls `module->forward(graph_arena, x)` → `criterion->forward(graph_arena, pred, y)` → `autograd::backward(graph_arena, loss)` → `optimizer.step(graph_arena)` → `graph_arena->pop_to(saved_pos)`. The graph arena is rewound after every batch, so all intermediate tensors and graph nodes are freed automatically.
:::

---

## Class Template

```cpp
template <typename OptimizerType, typename LossType>
class Trainer {
public:
    Trainer(Module       *model,
            OptimizerType *optimizer,
            LossType      *criterion,
            Arena         *graph_arena);

    void set_verbose(bool v);

    bool validate_data(const std::vector<std::vector<float>> &X,
                       const std::vector<std::vector<float>> &Y);

    // Train from raw vectors
    TrainingStats fit(const std::vector<std::vector<float>> &X_train,
                      const std::vector<std::vector<float>> &Y_train,
                      uint32_t epochs,
                      uint32_t batch_size   = 32,
                      uint32_t log_interval = 100);

    // Train from a DataLoader
    TrainingStats fit_dataloader(data::DataLoader *dataloader,
                                 uint32_t epochs,
                                 uint32_t log_interval = 100);

    // Evaluate from raw vectors
    float evaluate(const std::vector<std::vector<float>> &X_test,
                   const std::vector<std::vector<float>> &Y_test);

    // Evaluate from a DataLoader
    float evaluate_dataloader(data::DataLoader *dataloader);
};
```

`OptimizerType` must have `zero_grad()` and `step(Arena*)` methods (all optimizers in `gradientcore::optim` satisfy this). `LossType` must have a `forward(Arena*, Variable*, Variable*)` method (all `nn::LossFunction` subclasses satisfy this).

---

## Constructor

```cpp
Trainer<optim::Adam, nn::CrossEntropyLoss> trainer(
    model_seq,       // nn::Module* — the Sequential or any Module
    &adam_opt,       // OptimizerType* — already constructed with params
    &cross_entropy,  // LossType*
    graph_arena      // Arena* — the graph arena
);
```

The constructor prints errors to `stderr` if any pointer is null, but does not abort. A null pointer will cause a crash when `fit` is called — construct your objects before creating the `Trainer`.

---

## `set_verbose(bool)`

```cpp
trainer.set_verbose(true);   // print epoch loss (default: true)
trainer.set_verbose(false);  // silent training
```

When `verbose = true`, the trainer prints a configuration summary before training starts and logs loss at the epoch interval set by `log_interval`.

---

## `validate_data`

```cpp
bool ok = trainer.validate_data(X, Y);
```

Checks that:
- `X` and `Y` are non-empty.
- `X.size() == Y.size()` (same number of samples).
- All rows in `X` have the same number of columns.
- All rows in `Y` have the same number of columns.

Called automatically by `fit()`. You can call it manually to validate data before constructing the trainer.

---

## `fit` — Train from Raw Vectors

```cpp
TrainingStats fit(const std::vector<std::vector<float>> &X_train,
                  const std::vector<std::vector<float>> &Y_train,
                  uint32_t epochs,
                  uint32_t batch_size   = 32,
                  uint32_t log_interval = 100);
```

| Parameter | Description |
|---|---|
| `X_train` | Input features: `[num_samples][num_features]` |
| `Y_train` | Targets: `[num_samples][output_dim]` |
| `epochs` | Number of full passes over the data |
| `batch_size` | Samples per gradient update step |
| `log_interval` | Print loss every N epochs (and always on the last epoch) |

### What happens per batch

```
1. Save graph arena position:  pos = graph_arena->get_pos()
2. Copy batch into tensors:    t_x[batch_size × input_dim],
                                t_y[batch_size × output_dim]
3. Wrap as Variables:          x = create_leaf(graph_arena, t_x, false)
                                y = create_leaf(graph_arena, t_y, false)
4. Forward pass:               pred = model->forward(graph_arena, x)
5. Loss:                        loss = criterion->forward(graph_arena, pred, y)
6. Read scalar loss value
7. Zero gradients:             optimizer->zero_grad()
8. Backward:                   autograd::backward(graph_arena, loss)
9. Update weights:             optimizer->step(graph_arena)
10. Free batch graph:          graph_arena->pop_to(pos)
```

After all epochs complete, `model->eval()` is called automatically.

### Returns

A `TrainingStats` struct. If training aborts early (null forward/loss output), `epochs_trained` will reflect how many epochs actually completed. See [TrainingStats](./training_stats).

### Example

```cpp
// Construct optimizer and loss
auto params = seq->parameters();
optim::Adam adam(perm_arena, params, /*lr=*/0.001f);
nn::CrossEntropyLoss criterion;

// Construct trainer
Trainer<optim::Adam, nn::CrossEntropyLoss> trainer(
    seq, &adam, &criterion, graph_arena);

// Train
TrainingStats stats = trainer.fit(X_train, Y_train, /*epochs=*/40,
                                   /*batch_size=*/64, /*log_interval=*/5);
```

---

## `fit_dataloader` — Train from a DataLoader

```cpp
TrainingStats fit_dataloader(data::DataLoader *dataloader,
                              uint32_t epochs,
                              uint32_t log_interval = 100);
```

Identical to `fit` but accepts a pre-built `DataLoader` instead of raw vectors. The dataloader handles batching and shuffling. `reset(true)` is called on the dataloader at the start of each epoch.

This is the method `nn::Model::train()` uses internally.

```cpp
auto* features_ds = Dataset::create_2d(perm, X_train);
auto* labels_ds   = Dataset::create_2d(perm, Y_train);
auto* loader = DataLoader::create(features_ds, labels_ds, 64, /*shuffle=*/true);

TrainingStats stats = trainer.fit_dataloader(loader, /*epochs=*/40);
```

**Prefer `fit_dataloader` over `fit`** when your dataset is large — `DataLoader` only copies one batch at a time into the graph arena, whereas `fit` constructs its own batch tensors each step from the raw vectors.

---

## `evaluate` — Evaluate from Raw Vectors

```cpp
float evaluate(const std::vector<std::vector<float>> &X_test,
               const std::vector<std::vector<float>> &Y_test);
```

Runs the model in eval mode over the test data and returns the mean loss. Uses an internal batch size of `min(32, num_samples)`.

- Calls `model->eval()` at the start.
- Does **not** call `optimizer.step()` or `backward`.
- Frees each batch from the graph arena with `pop_to`.

Returns `-1.0f` on data validation failure.

```cpp
float test_loss = trainer.evaluate(X_test, Y_test);
std::cout << "Test loss: " << test_loss << "\n";
```

---

## `evaluate_dataloader` — Evaluate from a DataLoader

```cpp
float evaluate_dataloader(data::DataLoader *dataloader);
```

Same as `evaluate` but consumes batches from a `DataLoader`. Calls `dataloader->reset(false)` first (no reshuffling for evaluation — you want a deterministic order).

```cpp
float test_loss = trainer.evaluate_dataloader(test_loader);
```

Returns `-1.0f` if the dataloader is null or returns invalid batches.

---

## Verbose Output

When `verbose = true` (the default), `fit` prints:

```
=== Training Configuration ===
Epochs: 40
Batch Size: 64
Samples: 60000
Input Features: 784
Output Features: 10
Model Parameters: 101770
==============================

Epoch [1/40]  | Loss: 2.301842
Epoch [5/40]  | Loss: 0.483201
...
Epoch [40/40] | Loss: 0.082341
Training complete! Final Loss: 0.082341
```

`fit_dataloader` prints the same but also shows batches per epoch and the feature shape.

---

## Using Trainer vs Model

| | `nn::Model` | `nn::Trainer` |
|---|---|---|
| Interface | Keras-style | Explicit |
| Optimizer/loss combos | Fixed set of enums | Any template combination |
| Custom training logic | Not possible | Full control |
| Recommended for | Standard experiments | Research / custom loops |

For anything beyond the combinations in `LossType` and `OptimizerType` enums, use `Trainer` directly. For example, to use `optim::RMSprop` with `nn::HuberLoss(delta=2.0f)`:

```cpp
optim::RMSprop rms(perm, seq->parameters(), 0.01f);
nn::HuberLoss  huber(2.0f);
Trainer<optim::RMSprop, nn::HuberLoss> trainer(seq, &rms, &huber, graph);
TrainingStats stats = trainer.fit_dataloader(loader, 100);
```

`Model::train()` only supports the specific optimizer/loss pairings listed in its `compile()` documentation. `Trainer` supports any combination.