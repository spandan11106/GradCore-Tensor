---
sidebar_position: 1
title: "Model"
---

# `nn::Model`

`Model` is the top-level object you interact with when building and training networks in GradCore-Tensor. It owns a `Sequential` container, wires together an optimizer and loss function, and exposes a Keras-style interface: `compile` then `train` then `evaluate`.

**Header**: `include/nn/models/model.hpp`  
**Namespace**: `gradientcore::nn`

:::info What it wraps
`Model` is a convenience layer over `nn::Sequential` and `nn::Trainer`. Calling `model.train()` creates a `DataLoader` from your vectors, constructs a `Trainer<OptimizerType, LossType>`, and calls `trainer.fit_dataloader()`. All the real work happens in those lower layers — `Model` just removes the boilerplate.
:::

---

## Constructor

```cpp
nn::Model model(Arena *perm_arena, Arena *graph_arena);
```

| Parameter | Description |
|---|---|
| `perm_arena` | Permanent arena. The `Sequential`, optimizer, and loss objects are allocated here. |
| `graph_arena` | Graph arena. Passed to `Trainer`, which uses it for all per-batch allocations. |

The constructor allocates a `Sequential` on `perm_arena` using placement new. If either arena pointer is null, an error is printed and the model is unusable.

```cpp
auto* perm  = Arena::create(MiB(1024), MiB(64),  true);
auto* graph = Arena::create(MiB(512),  MiB(32),  true);

nn::Model model(perm, graph);
```

---

## Step 1 — Add Layers

### `add_layer(Module *layer)`

```cpp
void add_layer(Module *layer);
```

Appends a layer to the internal `Sequential`. Delegates to `sequential->add(layer)`.

Every layer must be constructed on `perm_arena` **before** being added:

```cpp
// Pattern: push to allocate, placement-new to construct, then add
auto* l1   = perm->push<nn::Linear>();    new (l1)   nn::Linear(perm, 784, 128);
auto* bn1  = perm->push<nn::BatchNorm1d>(); new (bn1) nn::BatchNorm1d(perm, 128);
auto* r1   = perm->push<nn::ReLU>();      new (r1)   nn::ReLU();
auto* drop = perm->push<nn::Dropout>();   new (drop) nn::Dropout(0.3f);
auto* l2   = perm->push<nn::Linear>();    new (l2)   nn::Linear(perm, 128, 10);

model.add_layer(l1);
model.add_layer(bn1);
model.add_layer(r1);
model.add_layer(drop);
model.add_layer(l2);
```

Passing `nullptr` prints a warning and does nothing. Layers are executed in the order they are added.

---

## Step 2 — Compile

### `compile(optimizer, loss, lr, epochs, batch_size)`

```cpp
void compile(OptimizerType optimizer,
             LossType      loss,
             float         lr,
             uint32_t      num_epochs,
             uint32_t      batch_sz);
```

Wires together the optimizer, loss function, and hyperparameters. Must be called before `train()` or `evaluate()`.

| Parameter | Type | Description |
|---|---|---|
| `optimizer` | `OptimizerType` enum | Which optimizer to use |
| `loss` | `LossType` enum | Which loss function to use |
| `lr` | `float` | Learning rate |
| `num_epochs` | `uint32_t` | Epochs to train for |
| `batch_sz` | `uint32_t` | Samples per gradient step |

### `OptimizerType` enum

```cpp
enum class OptimizerType { ADAM, SGD, ADAMW, RMSPROP, ADAGRAD };
```

| Value | Class | Default args |
|---|---|---|
| `ADAM` | `optim::Adam` | `lr, beta1=0.9, beta2=0.999, eps=1e-8` |
| `SGD` | `optim::SGD` | `lr` |
| `ADAMW` | `optim::AdamW` | `lr, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01` |
| `RMSPROP` | `optim::RMSprop` | `lr, alpha=0.99, eps=1e-8` |
| `ADAGRAD` | `optim::Adagrad` | `lr, eps=1e-10` |

### `LossType` enum

```cpp
enum class LossType {
    CROSS_ENTROPY,
    MSE,
    MAE,
    BCE,
    BCE_WITH_LOGITS,
    HUBER
};
```

| Value | Class | Notes |
|---|---|---|
| `CROSS_ENTROPY` | `CrossEntropyLoss` | Multi-class classification. Includes softmax — do not add a Softmax layer. |
| `MSE` | `MSELoss` | Regression. Sensitive to outliers. |
| `MAE` | `MAELoss` | Regression. Robust to outliers. |
| `BCE` | `BCELoss` | Binary classification with sigmoid output. |
| `BCE_WITH_LOGITS` | `BCEWithLogitsLoss` | Binary classification with raw logit output. Numerically stabler than BCE. |
| `HUBER` | `HuberLoss(delta=1.0)` | Regression. Robust to outliers, smoother than MAE. |

### Supported optimizer/loss combinations

`Model::train()` and `Model::evaluate()` support the following combinations. Attempting an unlisted pair prints an error and returns an empty `TrainingStats`:

| Optimizer | Loss |
|---|---|
| `ADAM` | `CROSS_ENTROPY` |
| `ADAM` | `MSE` |
| `SGD` | `CROSS_ENTROPY` |
| `ADAMW` | `HUBER` |

For any other combination, use `nn::Trainer` directly — it supports any `<OptimizerType, LossType>` template instantiation. See [Trainer](../training/trainer).

### Example

```cpp
model.compile(
    nn::OptimizerType::ADAM,
    nn::LossType::CROSS_ENTROPY,
    /*lr=*/    0.0005f,
    /*epochs=*/40,
    /*batch=*/ 64
);
```

After `compile()`, the optimizer and loss objects are allocated on `perm_arena` and `is_built()` returns `true`.

---

## Step 3 — Train

### `train(X_train, Y_train)`

```cpp
TrainingStats train(const std::vector<std::vector<float>> &X_train,
                    const std::vector<std::vector<float>> &Y_train);
```

Runs the full training loop for the number of epochs specified in `compile()`.

Internally:
1. Wraps `X_train` and `Y_train` in `Dataset::create_2d`.
2. Creates a shuffling `DataLoader` with `seed=42`.
3. Instantiates the appropriate `Trainer<Opt, Loss>` specialisation.
4. Calls `trainer.fit_dataloader(dataloader, epochs, log_interval=1)`.
5. Returns the `TrainingStats`.

```cpp
TrainingStats stats = model.train(X_train, Y_train);
std::cout << "Final loss: " << stats.final_loss << "\n";
```

Returns a default `TrainingStats` (with `epochs_trained=0`) if:
- `compile()` was not called first.
- Training data is empty or mismatched.
- The optimizer/loss combination is not supported.

The model is in **eval mode** after `train()` returns (the `Trainer` calls `model->eval()` at the end).

---

## Step 4 — Evaluate

### `evaluate(X_test, Y_test)`

```cpp
float evaluate(const std::vector<std::vector<float>> &X_test,
               const std::vector<std::vector<float>> &Y_test);
```

Runs the model over the test set in eval mode and returns the mean loss. Does not update weights.

```cpp
float test_loss = model.evaluate(X_test, Y_test);
std::cout << "Test Huber loss: " << test_loss << "\n";
```

Uses the same optimizer/loss combination as `train()`. Returns `-1.0f` on error.

---

## Step 5 — Save and Load

### `save(path, format)`

```cpp
bool save(const std::string &path,
          const std::string &format = "binary");
```

Saves all model parameters to disk. Delegates to `Sequential::save()` → `Module::save()`.

| Format | Extension | Notes |
|---|---|---|
| `"binary"` | `.bin` | Compact raw float bytes. **Recommended.** |
| `"json"` | `.json` | Base64-encoded. Useful for inspection; loading is limited. |
| `"csv"` | `.csv` | Human-readable `param_idx,elem_idx,value` format. Large files. |

```cpp
#include <filesystem>
std::filesystem::create_directories("bin");
model.save("bin/mnist_model.bin", "binary");
```

Returns `true` on success, `false` on file error or uninitialised model.

### `load(path)`

```cpp
bool load(const std::string &path);
```

Loads parameters from a previously saved file back into the model's existing parameter tensors. The model must already have the same architecture — `load` fills weights, it does not create layers.

```cpp
if (!model.load("bin/mnist_model.bin")) {
    std::cerr << "Failed to load model — did you run training first?\n";
    return 1;
}
model.get_model()->eval();   // switch to inference mode
```

Format is inferred from the file extension (`.json` → JSON, `.csv` → CSV, anything else → binary).

Returns `false` if the file cannot be opened, the parameter count doesn't match, or any parameter size mismatches.

---

## Accessing the Underlying Sequential

### `get_model()`

```cpp
Sequential *get_model();
```

Returns a raw pointer to the internal `Sequential`. Use this for:

- Manual inference (calling `forward` directly).
- Switching to eval mode explicitly: `model.get_model()->eval()`.
- Inspecting individual layers: `model.get_model()->get(0)`.
- Printing the full architecture: `model.get_model()->summary()`.

```cpp
// Manual single-sample inference
model.get_model()->eval();

uint32_t shape[2] = {1, 784};
Tensor *t = tensor_create(graph_arena, 2, shape);
std::memcpy(t->storage->data, sample.data(), 784 * sizeof(float));

auto* x   = autograd::create_leaf(graph_arena, t, false);
auto* out = model.get_model()->forward(graph_arena, x);

// Read argmax for classification
int predicted_class = 0;
float max_val = -1e9f;
for (int c = 0; c < 10; c++) {
    float v = out->data->storage->data[out->data->offset + c];
    if (v > max_val) { max_val = v; predicted_class = c; }
}
```

---

## Runtime Configuration

These methods can be called after `compile()` to adjust hyperparameters for a subsequent `train()` call (e.g. for learning-rate scheduling between runs):

```cpp
void set_learning_rate(float lr);
void set_epochs(uint32_t num_epochs);
void set_batch_size(uint32_t batch_sz);
```

Note: these do **not** rebuild the optimizer. The optimizer's internal learning rate is set once at construction. For mid-training learning rate changes, use `Trainer` directly and modify the optimizer's `learning_rate` field.

### `is_built()`

```cpp
bool is_built() const;
```

Returns `true` after `compile()` has been called successfully. Use as a guard before calling `train()` or `evaluate()` manually:

```cpp
if (!model.is_built()) {
    std::cerr << "Call compile() first.\n";
    return;
}
```

---

## `summary()`

```cpp
void summary() const;
```

Prints the total parameter count and compiled configuration:

```
=== Model Summary ===
Total Parameters: 101770

Compiled Configuration:
  Optimizer: Adam
  Learning Rate: 0.0005
  Epochs: 40
  Batch Size: 64
====================
```

If `compile()` has not been called, prints a message indicating the model is not yet compiled.

---

## Full Lifecycle Example

```cpp
// Arenas
auto* perm  = Arena::create(MiB(1024), MiB(64), true);
auto* graph = Arena::create(MiB(512),  MiB(32), true);

// Model
nn::Model model(perm, graph);

auto* l1  = perm->push<nn::Linear>();     new (l1)  nn::Linear(perm, 784, 128);
auto* r1  = perm->push<nn::ReLU>();       new (r1)  nn::ReLU();
auto* l2  = perm->push<nn::Linear>();     new (l2)  nn::Linear(perm, 128, 10);
model.add_layer(l1);
model.add_layer(r1);
model.add_layer(l2);

// Compile
model.compile(nn::OptimizerType::ADAM,
              nn::LossType::CROSS_ENTROPY,
              0.0005f, 40, 64);

model.summary();

// Train
TrainingStats stats = model.train(X_train, Y_train);
std::cout << "Final loss: " << stats.final_loss << "\n";

// Evaluate
float test_loss = model.evaluate(X_test, Y_test);
std::cout << "Test loss: " << test_loss << "\n";

// Save
model.save("model.bin", "binary");

// Later: load and infer
model.load("model.bin");
model.get_model()->eval();
// ... run forward pass manually ...
```