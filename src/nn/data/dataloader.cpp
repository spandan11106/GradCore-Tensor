#include "nn/data/dataloader.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace gradientcore {
namespace nn {
namespace data {

DataLoader* DataLoader::create(Dataset *features_dataset,
                               Dataset *labels_dataset,
                               uint32_t batch_size,
                               bool shuffle,
                               uint32_t seed) {
  if (features_dataset == nullptr) {
    std::cerr << "Error: features_dataset is nullptr" << std::endl;
    return nullptr;
  }

  if (batch_size == 0) {
    std::cerr << "Error: Batch size must be > 0" << std::endl;
    return nullptr;
  }

  uint32_t num_samples = features_dataset->get_num_samples();
  
  if (labels_dataset != nullptr &&
      labels_dataset->get_num_samples() != num_samples) {
    std::cerr << "Error: features_dataset and labels_dataset have different "
              << "number of samples" << std::endl;
    std::cerr << "  Features: " << num_samples
              << " Labels: " << labels_dataset->get_num_samples() << std::endl;
    return nullptr;
  }

  if (batch_size > num_samples) {
    std::cerr << "Warning: Batch size (" << batch_size
              << ") > dataset size (" << num_samples << ")" << std::endl;
  }

  Arena *arena = features_dataset->get_arena();
  DataLoader *loader = arena->push<DataLoader>();
  new (loader) DataLoader(features_dataset, labels_dataset, batch_size,
                          shuffle, arena);

  loader->num_batches = (num_samples + batch_size - 1) / batch_size;

  loader->indices.resize(num_samples);
  for (uint32_t i = 0; i < num_samples; i++) {
    loader->indices[i] = i;
  }

  if (shuffle) {
    loader->shuffle_indices(seed);
  }

  return loader;
}

void DataLoader::reset(bool reshuffle) {
  current_batch = 0;
  if (reshuffle && shuffle) {
    shuffle_indices(0);
  }
}

void DataLoader::shuffle_indices(uint32_t seed) {
  if (!shuffle || indices.empty()) {
    return;
  }

  std::mt19937 rng(seed);
  uint32_t n = indices.size();

  for (uint32_t i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<uint32_t> dist(0, i);
    uint32_t j = dist(rng);
    std::swap(indices[i], indices[j]);
  }
}

Tensor* DataLoader::create_batch_view(const Dataset *dataset,
                                      uint32_t start_idx,
                                      uint32_t num_samples,
                                      Arena *graph_arena,
                                      uint32_t *out_shape) {
  if (!dataset || start_idx >= dataset->get_num_samples()) {
    return nullptr;
  }

  Tensor *view = graph_arena->push<Tensor>();
  const Tensor *original = dataset->get_data();
  *view = *original;

  uint32_t original_ndims = dataset->get_ndims();
  const uint32_t *original_shape = dataset->get_shape();
  uint64_t sample_size = dataset->get_sample_size();

  view->ndims = original_ndims;
  view->shape[0] = num_samples;
  view->size = (uint64_t)num_samples * sample_size;

  for (uint32_t i = 1; i < original_ndims; i++) {
    view->shape[i] = original_shape[i];
  }

  for (uint32_t i = 0; i < original_ndims; i++) {
    out_shape[i] = view->shape[i];
  }

  uint32_t permuted_idx = indices[start_idx];
  view->offset = original->offset + (uint64_t)permuted_idx * sample_size;

  view->strides[original_ndims - 1] = 1;
  for (int32_t i = (int32_t)original_ndims - 2; i >= 0; i--) {
    view->strides[i] = view->strides[i + 1] * view->shape[i + 1];
  }

  return view;
}

Batch DataLoader::next(Arena *graph_arena) {
  if (!has_next()) {
    return Batch();
  }

  return get_batch(current_batch++, graph_arena);
}

Batch DataLoader::get_batch(uint32_t batch_idx, Arena *graph_arena) {
  Batch batch;

  if (batch_idx >= num_batches) {
    return batch;
  }

  uint32_t num_samples = features_dataset->get_num_samples();
  uint32_t start_idx = batch_idx * batch_size;
  uint32_t actual_batch_size = std::min(batch_size, num_samples - start_idx);

  batch.batch_size = actual_batch_size;
  batch.start_idx = start_idx;
  batch.ndims = features_dataset->get_ndims();

  batch.features = create_batch_view(features_dataset, start_idx,
                                     actual_batch_size, graph_arena,
                                     batch.shape);

  if (labels_dataset != nullptr) {
    batch.labels = create_batch_view(labels_dataset, start_idx,
                                     actual_batch_size, graph_arena,
                                     batch.shape);  
  }

  return batch;
}

} // namespace data
} // namespace nn
} // namespace gradientcore

