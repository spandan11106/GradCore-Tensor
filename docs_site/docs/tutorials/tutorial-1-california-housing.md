---
sidebar_position: 1
title: "Tutorial 1: California Housing Regression"
---

# Tutorial 1: California Housing Regression

This tutorial walks through the California Housing dataset regression example end-to-end — from loading CSV data to running inference on unseen samples. By the end you will understand how `CSVLoader`, `nn::Model`, and the training loop fit together.

The full source lives in `examples/regression_ex/`.

---

## What you'll build

A multi-layer perceptron that predicts California median house prices (in USD) from eight geographic and demographic features. The trained model reaches a low Huber loss on the held-out test split.

Architecture at a glance:

```
Input (8) → Linear(8→128) → BatchNorm1d(128) → ReLU
          → Linear(128→64) → ReLU
          → Linear(64→1)
```

Optimizer: **AdamW**, Loss: **HuberLoss**, LR: `0.001`, Epochs: `200`, Batch size: `128`.

---

## Prerequisites

Make sure you have built the library and pulled the LFS dataset files before starting:

```bash
git lfs pull          # fetches housing.csv (~1.4 MB)
cd examples/regression_ex
./run.sh              # compiles train + inference binaries into bin/
```

See [Getting Started](/docs/Getting%20Started) for full build instructions.

---

## Step 1 — Load and preprocess the CSV

`CSVLoader` is a static utility that reads comma-separated files into raw string rows, then offers helper methods to parse, normalize, and split them.

```cpp
#include "../../include/gradient.hpp"
using namespace gradientcore;

// Load raw CSV (true = skip the header row)
auto csv_raw = CSVLoader::load_csv("data/housing.csv", true);

std::vector<std::vector<float>> features, labels;

// Parse: 8 feature columns, label is the last column (true)
CSVLoader::parse_csv_to_float(csv_raw, 8, true, features, labels);
```

The dataset has 8 input features: longitude, latitude, housing median age, total rooms, total bedrooms, population, households, and median income.

### Standardize features

Raw feature values have wildly different scales. Standardization (zero mean, unit variance per column) prevents any single feature from dominating the gradient signal:

```cpp
CSVLoader::standardize(features);
```

### Scale labels

House prices range into the hundreds of thousands. Dividing by `100 000` keeps the target in a well-conditioned range for the optimizer:

```cpp
for (auto& label : labels) {
    label[0] /= 100000.0f;
}
```

During inference, multiply predictions back by `100 000` to recover USD values.

### Train / test split

```cpp
std::vector<std::vector<float>> X_train, Y_train, X_test, Y_test;
CSVLoader::train_test_split(features, labels, 0.8f,
                            X_train, Y_train, X_test, Y_test);
```

`0.8f` reserves 80 % of rows for training and 20 % for evaluation.

---

## Step 2 — Allocate arenas

GradCore-Tensor uses a two-arena memory model. The **permanent arena** holds model parameters and optimizer state for the lifetime of the program. The **graph arena** holds the forward/backward computation graph for each batch and is rewound after every step.

```cpp
auto* perm_arena  = Arena::create(MiB(1024), MiB(64), true);
auto* graph_arena = Arena::create(MiB(512),  MiB(32), true);
```

`MiB(n)` expands to `n * 1024 * 1024` bytes. The third argument enables growable mode, allowing the arena to chain additional pages if it runs out of reserved virtual memory.

---

## Step 3 — Define the model

Layers are constructed in-place on the permanent arena using placement-`new` and registered with the `Sequential` container via `add_layer`:

```cpp
nn::Model model(perm_arena, graph_arena);

// Layer 1: Linear + BatchNorm + ReLU
auto* l1 = perm_arena->push<nn::Linear>();
new (l1) nn::Linear(perm_arena, 8, 128);
model.add_layer(l1);

auto* bn1 = perm_arena->push<nn::BatchNorm1d>();
new (bn1) nn::BatchNorm1d(perm_arena, 128);
model.add_layer(bn1);

auto* relu1 = perm_arena->push<nn::ReLU>();
new (relu1) nn::ReLU();
model.add_layer(relu1);

// Layer 2: Linear + ReLU
auto* l2 = perm_arena->push<nn::Linear>();
new (l2) nn::Linear(perm_arena, 128, 64);
model.add_layer(l2);

auto* relu2 = perm_arena->push<nn::ReLU>();
new (relu2) nn::ReLU();
model.add_layer(relu2);

// Output layer
auto* l3 = perm_arena->push<nn::Linear>();
new (l3) nn::Linear(perm_arena, 64, 1);
model.add_layer(l3);
```

`BatchNorm1d` normalizes each feature channel across the batch during training and uses running statistics during inference. It is particularly helpful for regression tasks where feature distributions can shift between batches.

---

## Step 4 — Compile and train

`compile()` wires together the optimizer, loss function, and training hyperparameters. `train()` runs the full training loop, printing per-epoch loss to stdout.

```cpp
model.compile(
    nn::OptimizerType::ADAMW,   // optimizer
    nn::LossType::HUBER,        // loss function
    0.001f,                      // learning rate
    200,                         // epochs
    128                          // batch size
);

model.train(X_train, Y_train);
```

**Why HuberLoss?** House price data contains outliers (mansions, unusual parcels). Huber loss behaves like MSE for small residuals and like MAE for large ones, making training more robust to those extremes than pure MSE.

**Why AdamW?** AdamW decouples weight decay from the gradient update, which regularizes the model without distorting the effective learning rate — a reliable choice for regression.

Internally, `train()` creates a `DataLoader` with shuffling enabled, iterates for the requested number of epochs, and calls `backward()` + `optimizer.step()` on each batch. The graph arena is rewound after every batch to reclaim memory.

---

## Step 5 — Evaluate on the test set

```cpp
float test_loss = model.evaluate(X_test, Y_test);
std::cout << "Test Set Huber Loss: " << test_loss << std::endl;
```

`evaluate()` puts the model in eval mode (disabling dropout and switching BatchNorm to use running statistics), iterates over the test split, and returns the mean loss.

---

## Step 6 — Save and load the model

```cpp
#include <filesystem>
std::filesystem::create_directories("bin");
model.save("bin/california_housing.bin", "binary");
```

The binary format writes a compact, flat representation of all parameter tensors. To reload:

```cpp
if (!model.load("bin/california_housing.bin")) {
    std::cerr << "Error: run training first." << std::endl;
    return 1;
}
model.get_model()->eval();  // switch to inference mode
```

---

## Step 7 — Run inference

After loading, create input tensors manually on the graph arena and call `forward()`:

```cpp
for (int i = 0; i < 10; ++i) {
    uint64_t start_pos = graph_arena->get_pos();

    uint32_t shape[2] = {1, 8};
    Tensor* input = tensor_create(graph_arena, 2, shape);
    std::memcpy(input->storage->data, X_test[i].data(), 8 * sizeof(float));

    autograd::Variable* x   = autograd::create_leaf(graph_arena, input, false);
    autograd::Variable* out = model.get_model()->forward(graph_arena, x);

    float pred   = out->data->storage->data[0] * 100000.0f;
    float actual = Y_test[i][0]               * 100000.0f;
    std::cout << "Predicted: $" << pred << "  Actual: $" << actual << "\n";

    graph_arena->pop_to(start_pos);  // free this batch's graph memory
}
```

`create_leaf(..., false)` signals that this input variable does not require a gradient, which avoids allocating unnecessary gradient buffers during inference.

`graph_arena->pop_to(start_pos)` resets the arena to where it was before this forward pass, reclaiming all intermediate activations and graph nodes in O(1) time.

---

## Running the example

```bash
cd examples/regression_ex

# Train (writes bin/california_housing.bin)
./bin/train_regression

# Inference (reads bin/california_housing.bin, prints 10 predictions)
./bin/inference_regression
```

Expected output after training:

```
Test Set Huber Loss: 0.0312

=== California Housing Predictions ===
Sample 0 | Predicted: $187432.00 | Actual: $192500.00 | Diff: $5068.00
Sample 1 | Predicted: $245810.00 | Actual: $239800.00 | Diff: $6010.00
...
```

---

## Key concepts recap

| Concept | Where it appears |
|---|---|
| `CSVLoader::load_csv` | Loading raw tabular data |
| `CSVLoader::standardize` | Feature normalization |
| `CSVLoader::train_test_split` | Holdout evaluation set |
| `Arena::create` / `pop_to` | Zero-overhead memory management |
| `nn::Linear`, `nn::BatchNorm1d`, `nn::ReLU` | Layer building blocks |
| `nn::Model::compile` | Wiring optimizer + loss |
| `nn::Model::train` | Training loop |
| `autograd::create_leaf` | Wrapping tensors for the autograd graph |
| `model.get_model()->forward` | Manual inference |

---

## Next steps

Continue to [Tutorial 2: MNIST Digit Classification](./tutorial-2-mnist) to see how the same primitives scale to a 10-class image classification problem with one-hot encoding and cross-entropy loss.
