#pragma once

#include "dataset.hpp"
#include "../../tensor/tensor.hpp"
#include "../../tensor/memory_cpu/arena.hpp"
#include <vector>
#include <cstdint>
#include <random>

namespace gradientcore {
namespace nn {
namespace data {

struct Batch {
  Tensor *features;      
  Tensor *labels;       
  uint32_t batch_size;   
  uint32_t start_idx;    
  
  uint32_t shape[MAX_TENSOR_DIMS];  
  uint32_t ndims;                   

  Batch() : features(nullptr), labels(nullptr), batch_size(0), 
             start_idx(0), ndims(0) {}

  uint64_t get_memory_size() const {
    uint64_t size = 1;
    for (uint32_t i = 0; i < ndims; i++) {
      size *= shape[i];
    }
    return size * sizeof(float);
  }
};

class DataLoader {
public:

  static DataLoader* create(Dataset *features_dataset,
                           Dataset *labels_dataset,
                           uint32_t batch_size,
                           bool shuffle = false,
                           uint32_t seed = 0);

  void reset(bool reshuffle = true);

  bool has_next() const { return current_batch < num_batches; }

  Batch next(Arena *graph_arena);

  Batch get_batch(uint32_t batch_idx, Arena *graph_arena);

  uint32_t get_batch_size() const { return batch_size; }
  uint32_t get_num_batches() const { return num_batches; }
  uint32_t get_current_batch() const { return current_batch; }
  uint32_t get_dataset_size() const { 
    return features_dataset ? features_dataset->get_num_samples() : 0;
  }
  uint32_t get_feature_ndims() const {
    return features_dataset ? features_dataset->get_ndims() : 0;
  }
  const uint32_t* get_feature_shape() const {
    return features_dataset ? features_dataset->get_shape() : nullptr;
  }
  uint32_t get_label_ndims() const {
    return labels_dataset ? labels_dataset->get_ndims() : 0;
  }
  const uint32_t* get_label_shape() const {
    return labels_dataset ? labels_dataset->get_shape() : nullptr;
  }

  uint64_t get_feature_sample_size() const {
    return features_dataset ? features_dataset->get_sample_size() : 0;
  }

private:
  Dataset *features_dataset;
  Dataset *labels_dataset;
  uint32_t batch_size;
  bool shuffle;
  uint32_t num_batches;
  uint32_t current_batch;
  
  std::vector<uint32_t> indices;  
  Arena *perm_arena;

  DataLoader(Dataset *feat, Dataset *lab, uint32_t batch_sz, 
             bool shuf, Arena *arena)
      : features_dataset(feat), labels_dataset(lab), batch_size(batch_sz),
        shuffle(shuf), current_batch(0), perm_arena(arena) {}

  void shuffle_indices(uint32_t seed);

  Tensor* create_batch_view(const Dataset *dataset,
                           uint32_t start_idx,
                           uint32_t num_samples,
                           Arena *graph_arena,
                           uint32_t *out_shape);
};

} // namespace data
} // namespace nn
} // namespace gradientcore
