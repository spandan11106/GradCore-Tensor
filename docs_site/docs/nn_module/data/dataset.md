---
sidebar_position: 1
title: "Dataset"
---

# `nn::data::Dataset`

`Dataset` is a thin wrapper that takes a block of float data (e.g. your training features or labels), stores it on the permanent arena, and exposes a shape-aware interface for `DataLoader` to batch from.

**Header**: `include/nn/data/dataset.hpp`  
**Namespace**: `gradientcore::nn::data`

:::info What it calls
`Dataset` calls `tensor_create` to allocate a single contiguous `Tensor` holding all samples on the permanent arena. `DataLoader::next` reads from this tensor directly — no data is copied until a batch view is created on the graph arena.
:::

In normal usage you do **not** construct `Dataset` yourself — `nn::Model::train()` calls `Dataset::create_2d` internally. `Dataset` is documented here for users writing custom training loops with `Trainer` directly.

---

## Factory Methods

`Dataset` has a private constructor. Use one of three static factories:

### `create_2d` — from a 2D vector (most common)

```cpp
static Dataset *create_2d(Arena *perm_arena,
                           const std::vector<std::vector<float>> &data);
```

Creates a dataset from a 2D `vector<vector<float>>`. All inner vectors must have the same length (consistent number of features).

```cpp
// X_train: shape [60000, 784]
auto* features = nn::data::Dataset::create_2d(perm_arena, X_train);
// Y_train: shape [60000, 10]
auto* labels   = nn::data::Dataset::create_2d(perm_arena, Y_train);
```

Internally flattens to a contiguous buffer and stores as a `[num_samples, num_features]` tensor on `perm_arena`.

### `create` — from a raw float buffer

```cpp
static Dataset *create(Arena *perm_arena,
                        const float *data,
                        const uint32_t *shape,
                        uint32_t ndims);
```

Creates a dataset from a raw `float*` pointer. The data is **copied** into the arena. `shape` describes the full tensor layout including the sample count as `shape[0]`.

```cpp
float raw[4] = {1.f, 2.f, 3.f, 4.f};
uint32_t shape[2] = {2, 2};   // 2 samples, 2 features each
auto* ds = Dataset::create(perm_arena, raw, shape, 2);
```

### `create_from_samples` — from per-sample vectors

```cpp
static Dataset *create_from_samples(Arena *perm_arena,
                                     const std::vector<std::vector<float>> &samples,
                                     const uint32_t *sample_shape,
                                     uint32_t sample_ndims);
```

Useful for non-1D samples (e.g. images stored as `[H, W, C]`). `sample_shape` describes the shape of a single sample; the dataset automatically prepends the sample count to form the full shape.

---

## Query Methods

```cpp
Tensor   *get_data()        const;   // The underlying tensor
uint32_t  get_num_samples() const;   // shape[0]
uint32_t  get_ndims()       const;   // Number of dimensions
const uint32_t *get_shape() const;   // Full shape array
uint64_t  get_sample_size() const;   // Product of shape[1..ndims-1]
Arena    *get_arena()        const;
void      get_sample_shape(uint32_t *out_shape,
                            uint32_t &out_ndims) const;
```

Example:

```cpp
auto* ds = Dataset::create_2d(perm, X_train);
std::cout << ds->get_num_samples() << "\n";  // 60000
std::cout << ds->get_sample_size()  << "\n"; // 784
```

---

## Memory Model

All data is stored on `perm_arena`. The `Dataset` struct itself is also on `perm_arena`. The underlying tensor is one contiguous allocation of `num_samples * sample_size * sizeof(float)` bytes.

`DataLoader` does **not** copy the full dataset into the graph arena. It copies only the samples for each batch, on demand, as `graph_arena` allocations. This keeps peak memory usage proportional to the batch size, not the dataset size.

---

## Returns `nullptr` On Failure

All three factory methods return `nullptr` and print an error if:
- The arena is null.
- Any dimension is 0.
- Inner vectors have inconsistent sizes (`create_2d`, `create_from_samples`).
- The tensor allocation fails.

Always check the return value before passing to `DataLoader::create`.