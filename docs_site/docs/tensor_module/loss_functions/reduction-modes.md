---
sidebar_position: 2
title: "Reduction Modes"
---

# Reduction Modes

Reduction modes control whether a loss function returns a per-element tensor or collapses to a scalar. Every loss function in GradCore-Tensor accepts a `Reduction` argument.

## The Enum

```cpp
enum Reduction {
    REDUCTION_NONE,   // Return a tensor with the same shape as the input
    REDUCTION_MEAN,   // Return a scalar: average of all losses
    REDUCTION_SUM,    // Return a scalar: sum of all losses
};
```

## Behaviour by Mode

### `REDUCTION_NONE`

The output tensor has the same shape as the input predictions. Each element gets its own loss value — useful for custom per-sample weighting or analysis.

```cpp
// pred:   shape [4, 1]
// target: shape [4, 1]
// out:    shape [4, 1]  ← one loss value per sample
Tensor *out = tensor_create(arena, 2, pred->shape);
tensor_mse_loss(out, pred, target, REDUCTION_NONE);
```

### `REDUCTION_MEAN`

The output is a scalar (`size = 1`). The formula is:

```
L = (1/N) * Σ L_i
```

where N is the total number of elements in the prediction tensor. This normalises by the batch size and any other dimensions — the loss magnitude is independent of batch size.

**Use this for training.** It makes hyperparameters (learning rate, etc.) transferable across different batch sizes.

### `REDUCTION_SUM`

The output is a scalar. No normalisation:

```
L = Σ L_i
```

The loss grows with batch size. Occasionally useful for specific mathematical formulations but rarely the right choice for training loops.

## Effect on Gradients

The reduction mode scales the backward pass gradients:

```cpp
float scale = 1.0f;
if (reduction == REDUCTION_MEAN) {
    scale = 1.0f / static_cast<float>(pred->size);
}
// gradient = scale * local_gradient
```

For `REDUCTION_NONE`, each gradient flows back individually.

For `REDUCTION_MEAN`, every gradient is divided by N — which keeps gradient magnitudes consistent regardless of how many elements contribute to the loss.

## In the `nn` API

The high-level `Model` API always uses `REDUCTION_MEAN`. The enum is exposed at the tensor level for users who call the loss functions directly (e.g. in a custom training loop that needs per-sample losses for sample weighting or curriculum learning).

## Cross-Entropy: What is N?

For `tensor_cross_entropy_loss`, N is `num_batches` (number of samples), not `num_batches * num_classes`. The loss for each sample is already summed over classes before the batch reduction:

```
L_batch = -(1/N) * Σ_batch [ Σ_class target * log(softmax(logits)) ]
```

This matches PyTorch's `nn.CrossEntropyLoss` behaviour.
