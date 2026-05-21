---
sidebar_position: 1
title: "nn Module Overview"
---

# The `nn` Module

The `nn` module is the primary interface for building, training, and running neural networks in GradCore-Tensor. If the [Tensor Module](/docs/tensor-module) is the engine room, `nn` is the bridge — you issue commands here, and the lower layers handle all the mechanics.

:::info High-level API
Everything in `nn` is built on top of `autograd::Variable` operations and raw `Tensor` functions. When you call `model.train()`, it is ultimately calling `tensor_matmul`, `tensor_relu`, `autograd::backward`, and so on under the hood. The `nn` module exists so you don't have to think about any of that unless you want to.
:::

## The Layer Stack

![nn Module Overview](/img/NN.drawio.svg)

## What Lives in `nn`

```
nn/
├── nn.hpp                    ← Single include for the whole module
│
├── core/
│   ├── module.hpp            ← Base class for all layers
│   └── sequential.hpp        ← Ordered layer container
│
├── layers/
│   ├── linear.hpp            ← Fully-connected layer
│   ├── activations.hpp       ← ReLU, Sigmoid, Tanh, GELU, Swish, …
│   ├── batchnorm.hpp         ← BatchNorm1d, BatchNorm2d
│   └── dropout.hpp           ← Dropout
│
├── losses/
│   └── loss.hpp              ← All loss function classes
│
├── data/
│   ├── dataset.hpp           ← In-memory dataset abstraction
│   ├── dataloader.hpp        ← Batching, shuffling
│   └── csv_loader.hpp        ← CSV parsing, normalisation, splitting
│
├── training/
│   └── trainer.hpp           ← Trainer<Optimizer, Loss> template
│
├── models/
│   └── model.hpp             ← High-level Model class
│
└── utils/
    └── initialization.hpp    ← Kaiming, Xavier, uniform, normal, …
```

## The Single Include

```cpp
#include "gradient.hpp"   // includes everything, including nn
using namespace gradientcore;
```

Or just the nn module:

```cpp
#include "nn/nn.hpp"
using namespace gradientcore;
using namespace gradientcore::nn;
```

## Quick Reference

| What you want | Where to look |
|---|---|
| The `Module` base class | [Module](./core/module.md) |
| `Sequential` container | [Sequential](./core/sequential.md) |
| `Linear` layer | [Linear](./layers/linear.md) |
| Activation layers | [Activations](./layers/activations.md) |
| `BatchNorm1d` / `BatchNorm2d` | [BatchNorm](./layers/batchnorm.md) |
| `Dropout` | [Dropout](./layers/dropout.md) |
| All loss classes | [Loss Functions](./losses/overview.md) |
| `Dataset` | [Dataset](./data/dataset.md) |
| `DataLoader` | [DataLoader](./data/dataloader.md) |
| `Trainer` | [Trainer](./training/trainer.md) |
| `Model` (high-level API) | [Model](./models/model.md) |
| Weight initialisation | [Initialization](./utils/initialization.md) |

## Namespace

All `nn` code lives in `gradientcore::nn`. Loss functions are in `gradientcore::nn`, data utilities in `gradientcore::nn::data`, and optimizers in `gradientcore::optim`.

## A Minimal Training Example

To orient yourself, here is the complete lifecycle from arena to trained model:

```cpp
#include "gradient.hpp"
using namespace gradientcore;

// 1. Arenas — two are all you need
auto* perm  = Arena::create(MiB(512), MiB(32), true);
auto* graph = Arena::create(MiB(256), MiB(16), true);

// 2. Build the model
nn::Model model(perm, graph);
auto* l1 = perm->push<nn::Linear>(); new (l1) nn::Linear(perm, 784, 128);
auto* r1 = perm->push<nn::ReLU>();   new (r1) nn::ReLU();
auto* l2 = perm->push<nn::Linear>(); new (l2) nn::Linear(perm, 128, 10);
model.add_layer(l1);
model.add_layer(r1);
model.add_layer(l2);

// 3. Compile
model.compile(nn::OptimizerType::ADAM,
              nn::LossType::CROSS_ENTROPY,
              /*lr=*/0.0005f, /*epochs=*/40, /*batch=*/64);

// 4. Train
model.train(X_train, Y_train);

// 5. Evaluate
float loss = model.evaluate(X_test, Y_test);

// 6. Save
model.save("mnist.bin", "binary");
```