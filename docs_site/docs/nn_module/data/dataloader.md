---
sidebar_position: 2
title: "DataLoader"
---

# `nn::data::DataLoader`

`DataLoader` slices a `Dataset` into mini-batches, optionally shuffles the sample order at the start of each epoch, and hands them to the training loop one batch at a time.

**Header**: `include/nn/data/dataloader.hpp`  
**Namespace**: `gradientcore::nn::data`

:::info What it calls
`DataLoader::next` calls `tensor_create` on the **graph arena** to produce a batch tensor, then `memcpy`s the selected samples into it. The batch tensor is a fresh allocation every call — it is freed when `graph_arena->pop_to(saved_pos)` is called at the end of the training step.
:::

---

## The `Batch` Struct

```cpp
struct Batch {
    Tensor   *features;     // shape [batch_size, *sample_shape]
    Tensor   *labels;       // shape [batch_size, *label_shape]
    uint32_t  batch_size;   // actual number of samples (may be < requested for last batch)
    uint32_t  start_idx;    // index of first sample in this batch
    uint32_t  shape[MAX_TENSOR_DIMS];
    uint32_t  ndims;
};
```

`features` and `labels` are both allocated on the graph arena. They share no memory with the `Dataset` — they are copies of the selected rows. After `graph_arena->pop_to(saved_pos)`, both are invalid.

---

## Factory Method

```cpp
static DataLoader *create(Dataset *features_dataset,
                           Dataset *labels_dataset,
                           uint32_t batch_size,
                           bool shuffle = false,
                           uint32_t seed = 0);
```

| Parameter | Description |
|---|---|
| `features_dataset` | Input features. Must not be null. |
| `labels_dataset` | Labels / targets. May be null if you're doing unsupervised training. |
| `batch_size` | Number of samples per batch. |
| `shuffle` | If `true`, randomises sample order at the start of each epoch. |
| `seed` | Seed for the shuffle RNG. Ignored if `shuffle = false`. |

Returns `nullptr` on failure (null dataset, batch size 0, or mismatched sample counts between features and labels datasets).

```cpp
auto* loader = nn::data::DataLoader::create(
    features_ds, labels_ds,
    /*batch_size=*/64,
    /*shuffle=*/true,
    /*seed=*/42
);
```

The `DataLoader` struct itself is allocated on the features dataset's arena (`perm_arena` in normal usage), so it lives as long as the arena does.

---

## Iterating Batches

### `has_next()`

```cpp
bool has_next() const;
```

Returns `true` if there are more batches to yield in the current epoch. Becomes `false` after the last batch.

### `next(Arena *graph_arena)`

```cpp
Batch next(Arena *graph_arena);
```

Returns the next batch and advances the internal cursor. The batch tensors are allocated on `graph_arena`.

The last batch of an epoch may have fewer samples than `batch_size` if `num_samples` is not divisible by `batch_size`. Check `batch.batch_size` rather than assuming it equals the requested size.

### `reset(bool reshuffle = true)`

```cpp
void reset(bool reshuffle = true);
```

Resets the batch cursor to the beginning. If `reshuffle = true` and shuffling is enabled, the sample order is re-randomised. Call this at the start of each epoch.

### Manual iteration

```cpp
for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    loader->reset(true);   // re-shuffle each epoch

    while (loader->has_next()) {
        uint64_t pos = graph_arena->get_pos();

        Batch batch = loader->next(graph_arena);

        auto* x    = autograd::create_leaf(graph_arena, batch.features, false);
        auto* y    = autograd::create_leaf(graph_arena, batch.labels,   false);
        auto* pred = model_seq->forward(graph_arena, x);
        auto* loss = criterion.forward(graph_arena, pred, y);

        float loss_val = loss->data->storage->data[loss->data->offset];

        optimizer.zero_grad();
        autograd::backward(graph_arena, loss);
        optimizer.step(graph_arena);

        graph_arena->pop_to(pos);   // free batch + graph
    }
}
```

---

## Random-access Batches

```cpp
Batch get_batch(uint32_t batch_idx, Arena *graph_arena);
```

Fetches a specific batch by index without advancing the cursor. Useful for evaluation when you want to iterate in a fixed order while the training dataloader is mid-epoch.

---

## Query Methods

```cpp
uint32_t get_batch_size()    const;
uint32_t get_num_batches()   const;
uint32_t get_current_batch() const;
uint32_t get_dataset_size()  const;
uint32_t get_feature_ndims() const;
const uint32_t *get_feature_shape() const;
uint32_t get_label_ndims()   const;
const uint32_t *get_label_shape()   const;
uint64_t get_feature_sample_size()  const;
```

```cpp
std::cout << "Batches per epoch: " << loader->get_num_batches() << "\n";
// e.g. ceil(60000 / 64) = 938
```

---

## Shuffling Internals

Shuffling works on an **index array**, not the data itself. The `Dataset` backing buffer is never moved or reordered. Instead, `DataLoader` maintains a `std::vector<uint32_t> indices` where `indices[i]` is the dataset row to use for position `i` in the current epoch order.

`next()` uses this index array when copying samples into the batch tensor:

```cpp
for (uint32_t i = 0; i < num_samples; i++) {
    uint32_t sample_idx = indices[start_idx + i];
    memcpy(dst + i * sample_size,
           src + sample_idx * sample_size,
           sample_size * sizeof(float));
}
```

Shuffling is Fisher-Yates using `std::mt19937`. Each epoch gets a fresh shuffle when `reset(true)` is called.

---

## Memory Diagram

```
perm_arena:
┌──────────────────────────────────────────────────┐
│  Dataset (features) │ Dataset (labels) │ Loader  │
│  [60000 × 784 floats]│ [60000×10 floats]│ struct │
└──────────────────────────────────────────────────┘

graph_arena (per batch step):
┌─────────────────────────────────────────────────┐  ← saved pos
│  Batch features   │  Batch labels               │
│  [64 × 784 floats]│  [64 × 10 floats]           │
│                   │                             │
│  + all autograd Variables, intermediate tensors │
└─────────────────────────────────────────────────┘
                                                 ← pop_to(saved pos)
```