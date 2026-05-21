---
sidebar_position: 2
title: "TrainingStats"
---

# `TrainingStats`

`TrainingStats` is a plain data struct returned by `Trainer::fit` and `Trainer::fit_dataloader` at the end of a training run. It gives you a record of how the loss evolved over every epoch, plus a few headline numbers.

**Header**: `include/nn/training/trainer.hpp`  
**Namespace**: `gradientcore::nn`

---

## Struct Definition

```cpp
struct TrainingStats {
    std::vector<float> train_losses;   // Mean loss for each epoch, in order
    float              final_loss;     // Loss of the last epoch
    uint32_t           epochs_trained; // Number of epochs completed
    uint32_t           total_batches;  // Total batches processed across all epochs
};
```

---

## Fields

### `train_losses`

```cpp
std::vector<float> train_losses;
```

One entry per epoch: the mean loss across all batches in that epoch. The vector grows as training progresses — `train_losses[0]` is epoch 1, `train_losses[N-1]` is the final epoch.

Use this to plot a learning curve or detect divergence:

```cpp
TrainingStats stats = model.train(X_train, Y_train);

for (size_t i = 0; i < stats.train_losses.size(); i++) {
    std::cout << "Epoch " << (i + 1)
              << "  loss: " << stats.train_losses[i] << "\n";
}
```

### `final_loss`

```cpp
float final_loss;
```

The mean loss of the last completed epoch. Equivalent to `train_losses.back()`. Convenient for a quick check without indexing.

```cpp
std::cout << "Final loss: " << stats.final_loss << "\n";
```

### `epochs_trained`

```cpp
uint32_t epochs_trained;
```

Number of epochs actually completed. Under normal circumstances this equals the `epochs` argument passed to `compile()` or `fit()`. It will be less if training aborted early due to an error (e.g. a layer returning `nullptr`).

### `total_batches`

```cpp
uint32_t total_batches;
```

Total number of gradient update steps taken. Equals `epochs_trained × batches_per_epoch` for complete runs.

```cpp
std::cout << "Steps taken: " << stats.total_batches << "\n";
```

---

## Default State

A default-constructed `TrainingStats` has `final_loss = 0.0f`, `epochs_trained = 0`, `total_batches = 0`, and an empty `train_losses`. This is returned by `Model::train()` and `Trainer::fit()` on early-exit error conditions — always check `epochs_trained > 0` if you need to detect failures.

```cpp
TrainingStats stats = model.train(X_train, Y_train);

if (stats.epochs_trained == 0) {
    std::cerr << "Training failed — check your model and data.\n";
    return 1;
}
```

---

## Example: Detecting Divergence

```cpp
TrainingStats stats = model.train(X_train, Y_train);

for (float loss : stats.train_losses) {
    if (std::isnan(loss) || std::isinf(loss)) {
        std::cerr << "Loss diverged! Try a lower learning rate.\n";
        break;
    }
}
```

---

## Example: Saving a Loss Curve to CSV

```cpp
std::ofstream curve("loss_curve.csv");
curve << "epoch,loss\n";
for (size_t i = 0; i < stats.train_losses.size(); i++) {
    curve << (i + 1) << "," << stats.train_losses[i] << "\n";
}
```