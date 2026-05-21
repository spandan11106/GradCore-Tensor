---
sidebar_position: 3
title: "CSVLoader"
---

# `CSVLoader`

`CSVLoader` is a static utility class for loading, parsing, and preprocessing tabular data from CSV files. It handles the entire pipeline from raw file to training-ready vectors — no external libraries, no Python dependency, no nonsense.

**Header**: `include/nn/data/csv_loader.hpp`  
**Namespace**: `gradientcore`

All methods are `static` — you never instantiate `CSVLoader`. Just call the methods directly.

---

## Step 1 — Load Raw CSV

### `load_csv`

```cpp
static std::vector<std::vector<std::string>>
load_csv(const std::string &filepath, bool skip_header = false);
```

| Parameter | Description |
|---|---|
| `filepath` | Path to the CSV file. |
| `skip_header` | If `true`, the first row is discarded. Default `false`. |

Reads the file line by line, splits each line on commas, and trims whitespace from each cell. Returns a 2D vector of strings — one `vector<string>` per row.

Returns an empty vector and prints an error if the file cannot be opened.

```cpp
auto raw = CSVLoader::load_csv("data/housing.csv", /*skip_header=*/true);
// raw[0][0] = "-122.23"  (first feature of first sample, as a string)
```

The raw format is intentionally untyped — you decide how to parse the strings in the next step.

---

## Step 2 — Parse to Floats

### `parse_csv_to_float`

For general tabular CSVs with a fixed number of feature columns:

```cpp
static void parse_csv_to_float(
    const std::vector<std::vector<std::string>> &csv_data,
    uint32_t feature_cols,
    bool has_label,
    std::vector<std::vector<float>> &features,   // output
    std::vector<std::vector<float>> &labels);    // output
```

| Parameter | Description |
|---|---|
| `csv_data` | Raw string rows from `load_csv`. |
| `feature_cols` | Number of columns to treat as features (read from the left). |
| `has_label` | If `true`, the column after the feature columns is parsed as the label. |
| `features` | Output: one `vector<float>` per sample. |
| `labels` | Output: one `vector<float>` per sample (single element). |

Rows with fewer columns than `feature_cols` are silently skipped. Cells that fail `std::stof` are replaced with `0.0f`.

```cpp
std::vector<std::vector<float>> features, labels;
CSVLoader::parse_csv_to_float(raw, /*feature_cols=*/8, /*has_label=*/true,
                               features, labels);
// features[i] has 8 floats, labels[i] has 1 float
std::cout << "Loaded " << features.size() << " samples\n";
```

### `parse_mnist_csv`

Specialised parser for the MNIST CSV format (label in column 0, 784 pixels in columns 1–784):

```cpp
static void parse_mnist_csv(
    const std::vector<std::vector<std::string>> &csv_data,
    std::vector<std::vector<float>> &features,   // output: 784-element vectors
    std::vector<std::vector<float>> &labels);    // output: 1-element vectors [0..9]
```

```cpp
auto raw = CSVLoader::load_csv("data/mnist_train.csv", true);
std::vector<std::vector<float>> features, labels;
CSVLoader::parse_mnist_csv(raw, features, labels);
// features[i]: 784 floats (pixel values 0–255)
// labels[i]:   1 float    (digit class 0–9)
```

Rows with fewer than 785 columns are silently skipped.

---

## Step 3 — Preprocess

### `normalize_minmax`

```cpp
static void normalize_minmax(std::vector<std::vector<float>> &features);
```

Scales each feature column independently to `[0, 1]`:

```
x_norm = (x - min) / (max - min)
```

Operates in-place. If a column is constant (`min == max`), it is left unchanged. Used for MNIST pixel normalisation (pixel values 0–255 → 0–1).

```cpp
CSVLoader::normalize_minmax(features);
```

### `standardize`

```cpp
static void standardize(std::vector<std::vector<float>> &features);
```

Scales each feature column to zero mean and unit variance:

```
x_std = (x - mean) / std_dev
```

Operates in-place. Columns with zero standard deviation are left unchanged. Used for the California Housing dataset where features have wildly different natural scales (longitude vs total rooms vs median income).

```cpp
CSVLoader::standardize(features);
```

**`normalize_minmax` vs `standardize`**: use min-max for pixel data or bounded inputs, standardize for continuous features with unknown range and possible outliers.

---

## Step 4 — Encode Labels

### `one_hot_encode`

```cpp
static void one_hot_encode(
    const std::vector<std::vector<float>> &labels,
    uint32_t num_classes,
    std::vector<std::vector<float>> &encoded);  // output
```

Converts integer class labels to one-hot vectors:

```
label = 3,  num_classes = 10
→ [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

| Parameter | Description |
|---|---|
| `labels` | Input: each element is a 1-element vector containing an integer class index. |
| `num_classes` | Total number of classes. Determines output vector length. |
| `encoded` | Output: one-hot vectors of length `num_classes`. |

Required for `CrossEntropyLoss`, which expects a probability distribution (one-hot is the hard-label special case of a distribution).

```cpp
std::vector<std::vector<float>> labels_onehot;
CSVLoader::one_hot_encode(labels, /*num_classes=*/10, labels_onehot);
```

Class indices outside `[0, num_classes)` produce all-zero vectors (not an error, but worth knowing).

---

## Step 5 — Split

### `train_test_split`

```cpp
static void train_test_split(
    const std::vector<std::vector<float>> &features,
    const std::vector<std::vector<float>> &labels,
    float train_ratio,
    std::vector<std::vector<float>> &X_train,  // output
    std::vector<std::vector<float>> &Y_train,  // output
    std::vector<std::vector<float>> &X_test,   // output
    std::vector<std::vector<float>> &Y_test);  // output
```

Splits data into training and test sets by taking the first `train_ratio * N` samples for training and the rest for testing. The split is **deterministic** and **not shuffled** — the order of rows in the CSV determines which samples end up in each split.

| Parameter | Description |
|---|---|
| `train_ratio` | Fraction of data for training. `0.8` = 80% train, 20% test. |

```cpp
std::vector<std::vector<float>> X_train, Y_train, X_test, Y_test;
CSVLoader::train_test_split(features, labels, 0.8f,
                             X_train, Y_train,
                             X_test,  Y_test);
std::cout << X_train.size() << " train, " << X_test.size() << " test\n";
```

:::tip Shuffle before splitting
`train_test_split` does not shuffle. If your CSV is sorted by class or time, a sequential split will produce biased train/test sets. Shuffle the data externally (or use the `DataLoader`'s built-in shuffle during training) to mitigate this.
:::

---

## Complete Pipeline Examples

### California Housing (regression)

```cpp
// 1. Load
auto raw = CSVLoader::load_csv("data/housing.csv", true);

// 2. Parse
std::vector<std::vector<float>> features, labels;
CSVLoader::parse_csv_to_float(raw, 8, true, features, labels);

// 3. Preprocess
CSVLoader::standardize(features);
for (auto& label : labels) label[0] /= 100000.0f;   // scale to ~[0, 5]

// 4. Split
std::vector<std::vector<float>> X_train, Y_train, X_test, Y_test;
CSVLoader::train_test_split(features, labels, 0.8f,
                             X_train, Y_train, X_test, Y_test);
```

### MNIST (classification)

```cpp
// 1. Load
auto raw = CSVLoader::load_csv("data/mnist_train.csv", true);

// 2. Parse (MNIST-specific)
std::vector<std::vector<float>> features, labels_raw;
CSVLoader::parse_mnist_csv(raw, features, labels_raw);

// 3. Normalise pixels
CSVLoader::normalize_minmax(features);

// 4. One-hot encode labels
std::vector<std::vector<float>> labels_onehot;
CSVLoader::one_hot_encode(labels_raw, 10, labels_onehot);

// 5. No split needed — test set is a separate file
auto test_raw = CSVLoader::load_csv("data/mnist_test.csv", true);
std::vector<std::vector<float>> test_features, test_labels;
CSVLoader::parse_mnist_csv(test_raw, test_features, test_labels);
CSVLoader::normalize_minmax(test_features);
```