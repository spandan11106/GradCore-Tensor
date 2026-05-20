---
sidebar_position: 2
title: "Tutorial 2: MNIST Digit Classification"
---

# Tutorial 2: MNIST Digit Classification

This tutorial covers multi-class image classification using the MNIST handwritten digit dataset. You will learn how to set up batched loading with `DataLoader`, build a classifier, train with `CrossEntropyLoss`, evaluate test accuracy, and run per-sample inference with a visual terminal display.

The full source lives in `examples/mnist/`.

Dataset - [Github](https://github.com/spandan11106/GradCore-Tensor/tree/main/examples/mnist/data)

---

## What you'll build

A three-layer MLP that classifies 28×28 greyscale digit images (flattened to 784 floats) into one of 10 classes. This matches GradCore-Tensor's MNIST benchmark: **97.77 % test accuracy**.

Architecture at a glance:

```
Input (784) → Linear(784→128) → ReLU → Linear(128→10)
```

Optimizer: **Adam**, Loss: **CrossEntropyLoss**, LR: `0.0005`, Epochs: `40`, Batch size: `64`.

---

## Prerequisites

The MNIST CSVs are large files tracked by Git LFS (~110 MB train, ~18 MB test). Pull them before building:

```bash
git lfs pull
cd examples/mnist
./run.sh    # compiles train, inference, and autoencoder binaries
```

---

## Step 1 — Load and preprocess MNIST

MNIST is distributed as pixel rows in CSV format. The first column is the label (0–9); the remaining 784 columns are pixel intensities (0–255).

```cpp
#include "../../include/gradient.hpp"
using namespace gradientcore;

// Load training data (skip header row)
auto csv_raw = CSVLoader::load_csv("data/mnist_train.csv", true);

std::vector<std::vector<float>> features, labels_raw, labels_onehot;
CSVLoader::parse_mnist_csv(csv_raw, features, labels_raw);
```

### Normalize pixel values

`normalize_minmax` scales every value to `[0, 1]`, which stabilizes gradient magnitudes during training:

```cpp
CSVLoader::normalize_minmax(features);
```

### One-hot encode labels

`CrossEntropyLoss` expects a target distribution, not a scalar class index. `one_hot_encode` converts each integer label into a length-10 vector with a single `1.0` at the correct class position:

```cpp
CSVLoader::one_hot_encode(labels_raw, 10, labels_onehot);
// e.g. label "3" → [0,0,0,1,0,0,0,0,0,0]
```

---

## Step 2 — Allocate arenas

MNIST training loads 60 000 images. The permanent arena must be large enough to hold the model parameters plus Adam's two moment tensors per parameter:

```cpp
auto* perm_arena  = Arena::create(MiB(1024), MiB(64), true);
auto* graph_arena = Arena::create(MiB(512),  MiB(32), true);
```

The graph arena is rewound after each batch, so its size only needs to accommodate one batch at a time rather than the full dataset.

---

## Step 3 — Define the model

```cpp
nn::Model model(perm_arena, graph_arena);

auto* l1 = perm_arena->push<nn::Linear>();
new (l1) nn::Linear(perm_arena, 784, 128);
model.add_layer(l1);

auto* relu = perm_arena->push<nn::ReLU>();
new (relu) nn::ReLU();
model.add_layer(relu);

auto* l2 = perm_arena->push<nn::Linear>();
new (l2) nn::Linear(perm_arena, 128, 10);
model.add_layer(l2);
```

The final `Linear(128→10)` produces 10 raw logits — one per digit class. `CrossEntropyLoss` applies softmax internally, so **no explicit Softmax layer is needed here**.

### Verifying the parameter count

```cpp
std::cout << "Total parameters: "
          << model.get_model()->num_parameters() << std::endl;
// Expected: 784*128 + 128 + 128*10 + 10 = 101 770
if (model.get_model()->num_parameters() == 0) {
    std::cerr << "Error: 0 parameters — check add_layer." << std::endl;
    return 1;
}
```

---

## Step 4 — Compile and train

```cpp
model.compile(
    nn::OptimizerType::ADAM,
    nn::LossType::CROSS_ENTROPY,
    0.0005f,   // learning rate
    40,        // epochs
    64         // batch size
);

model.train(features, labels_onehot);
```

**Why CrossEntropyLoss?** It applies log-softmax to the logits internally and computes the negative log-likelihood against the one-hot target, penalizing confidently wrong predictions much more than uncertain ones.

**Why Adam with LR 0.0005?** A lower learning rate prevents the optimizer from overshooting the minimum in the later epochs, which is important when pushing past 97 % accuracy.

---

## Step 5 — Evaluate test accuracy

After training, load the test split and compute top-1 accuracy in batches of 100:

```cpp
auto test_csv = CSVLoader::load_csv("data/mnist_test.csv", true);
std::vector<std::vector<float>> test_features, test_labels_raw;
CSVLoader::parse_mnist_csv(test_csv, test_features, test_labels_raw);
CSVLoader::normalize_minmax(test_features);

model.get_model()->eval();   // disable dropout, use BatchNorm running stats

uint32_t correct = 0;
uint32_t batch_size = 100;

for (uint32_t i = 0; i < test_features.size(); i += batch_size) {
    uint32_t current_bs = std::min(batch_size, (uint32_t)test_features.size() - i);
    uint64_t start_pos  = graph_arena->get_pos();

    uint32_t shape_x[2] = {current_bs, 784};
    Tensor* t_x = tensor_create(graph_arena, 2, shape_x);

    for (uint32_t b = 0; b < current_bs; b++)
        for (uint32_t j = 0; j < 784; j++)
            t_x->storage->data[t_x->offset + b * 784 + j] =
                test_features[i + b][j];

    autograd::Variable* x   = autograd::create_leaf(graph_arena, t_x, false);
    autograd::Variable* out = model.get_model()->forward(graph_arena, x);

    for (uint32_t b = 0; b < current_bs; b++) {
        float max_v = -1e9f; int pred = 0;
        for (int c = 0; c < 10; c++) {
            float v = out->data->storage->data[out->data->offset + b * 10 + c];
            if (v > max_v) { max_v = v; pred = c; }
        }
        if (pred == (int)test_labels_raw[i + b][0]) correct++;
    }

    graph_arena->pop_to(start_pos);
}

float accuracy = 100.0f * correct / test_features.size();
std::cout << "Test Accuracy: " << correct << " / " << test_features.size()
          << " (" << accuracy << "%)" << std::endl;
// → Test Accuracy: 9777 / 10000 (97.77%)
```

:::note
`model.get_model()->eval()` is critical before evaluating. Without it, BatchNorm continues updating running statistics from test data, corrupting evaluation metrics.
:::

---

## Step 6 — Save the model

```cpp
#include <filesystem>
std::filesystem::create_directories("bin");
model.save("bin/mnist_model.bin", "binary");
```

---

## Step 7 — Inference with terminal visualization

The inference binary loads a saved model and lets you inspect any test sample interactively. It renders the digit using ANSI 256-color escape codes directly in your terminal.

```cpp
if (!model.load("bin/mnist_model.bin")) { return 1; }
model.get_model()->eval();

int n;
std::cout << "Enter sample index (0 to " << features.size() - 1 << "): ";
std::cin >> n;

draw_mnist_digit(features[n].data());   // ANSI terminal render

uint32_t shape[2] = {1, 784};
Tensor* input = tensor_create(graph_arena, 2, shape);
std::memcpy(input->storage->data, features[n].data(), 784 * sizeof(float));

autograd::Variable* x   = autograd::create_leaf(graph_arena, input, false);
autograd::Variable* out = model.get_model()->forward(graph_arena, x);

float max_v = -1e9f; int pred = 0;
for (int i = 0; i < 10; ++i)
    if (out->data->storage->data[i] > max_v) {
        max_v = out->data->storage->data[i]; pred = i;
    }

std::cout << "Actual: " << labels_raw[n][0]
          << "  Predicted: " << pred << std::endl;
```

---

## Running the example

```bash
cd examples/mnist

# Train (writes bin/mnist_model.bin)
./bin/train_mnist

# Interactive inference
./bin/inference_mnist
# → Enter sample index (0 to 9999): 42
# → [terminal digit visualization]
# → Actual: 7  Predicted: 7
```

---

## Bonus: Autoencoder

`examples/mnist/autoencoder.cpp` trains an encoder–decoder network to compress each 784-dimensional digit into a 32-dimensional latent space and reconstruct it:

```
Input(784) → Linear(784→128) → ReLU
           → Linear(128→32)  → ReLU   ← bottleneck
           → Linear(32→128)  → ReLU
           → Linear(128→784) → Sigmoid
```

Loss is `MSELoss` between input and reconstruction — no labels needed. After training it prints both the original and its reconstruction side-by-side in the terminal.

```bash
./bin/autoencoder
```

---

## Key concepts recap

| Concept | Where it appears |
|---|---|
| `CSVLoader::parse_mnist_csv` | Parsing MNIST CSV format |
| `CSVLoader::normalize_minmax` | Pixel value scaling |
| `CSVLoader::one_hot_encode` | Converting integer labels to distributions |
| `nn::Model::compile` / `train` | High-level training API |
| `model.get_model()->eval()` | Switching to inference mode |
| `model.save` / `model.load` | Checkpoint persistence |
| `autograd::create_leaf(..., false)` | Inference without gradient tracking |
| `graph_arena->pop_to` | O(1) graph memory reclamation |

## Next steps

You now understand the core training loop and inference patterns. From here you can explore:

- **Module deep dives** — how `autograd`, `tensor`, `nn`, and `optim` are implemented internally.

- **Adding new layers** — see the Contributing Guide for how to add a custom activation or layer type.

- **Extending the autoencoder** — try varying the bottleneck dimension or adding noise to inputs for denoising autoencoding.