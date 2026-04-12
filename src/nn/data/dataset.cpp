#include "nn/data/dataset.hpp"
#include <iostream>
#include <cstring>

namespace gradientcore {
namespace nn {
namespace data {

Dataset* Dataset::create(Arena *perm_arena, const float *data,
                        const uint32_t *shape, uint32_t ndims) {
 
  if (!perm_arena || !data || !shape || ndims == 0 || ndims > MAX_TENSOR_DIMS) {
    std::cerr << "Error: Invalid parameters for Dataset::create" << std::endl;
    return nullptr;
  }

  if (shape[0] == 0) {
    std::cerr << "Error: Number of samples (shape[0]) must be > 0" << std::endl;
    return nullptr;
  }

  uint64_t total_size = 1;
  for (uint32_t i = 0; i < ndims; i++) {
    if (shape[i] == 0) {
      std::cerr << "Error: Shape dimension " << i << " is 0" << std::endl;
      return nullptr;
    }
    total_size *= shape[i];
  }

  uint32_t flat_shape[1] = {(uint32_t)total_size};
  Tensor *tensor = tensor_create(perm_arena, 1, flat_shape);

  if (tensor == nullptr) {
    std::cerr << "Error: Failed to allocate tensor for dataset" << std::endl;
    return nullptr;
  }

  std::memcpy(tensor->storage->data + tensor->offset, data,
              total_size * sizeof(float));

  Dataset *dataset = perm_arena->push<Dataset>();
  new (dataset) Dataset(perm_arena, tensor, shape, ndims);

  return dataset;
}

Dataset* Dataset::create_from_samples(
    Arena *perm_arena,
    const std::vector<std::vector<float>> &samples,
    const uint32_t *sample_shape,
    uint32_t sample_ndims) {
  
  if (samples.empty() || !sample_shape || sample_ndims == 0) {
    std::cerr << "Error: Empty samples or invalid sample shape" << std::endl;
    return nullptr;
  }

  uint32_t num_samples = samples.size();

  uint64_t sample_size = 1;
  for (uint32_t i = 0; i < sample_ndims; i++) {
    sample_size *= sample_shape[i];
  }

  for (size_t i = 0; i < samples.size(); i++) {
    if (samples[i].size() != sample_size) {
      std::cerr << "Error: Sample " << i << " has size " << samples[i].size()
                << ", expected " << sample_size << std::endl;
      return nullptr;
    }
  }

  float *flattened = perm_arena->push_array<float>(
      (size_t)num_samples * sample_size);

  if (flattened == nullptr) {
    std::cerr << "Error: Failed to allocate flattened data buffer" << std::endl;
    return nullptr;
  }

  for (uint32_t i = 0; i < num_samples; i++) {
    std::memcpy(flattened + (uint64_t)i * sample_size,
                samples[i].data(),
                sample_size * sizeof(float));
  }

  uint32_t full_shape[MAX_TENSOR_DIMS];
  full_shape[0] = num_samples;
  for (uint32_t i = 0; i < sample_ndims; i++) {
    full_shape[i + 1] = sample_shape[i];
  }

  return create(perm_arena, flattened, full_shape, sample_ndims + 1);
}

Dataset* Dataset::create_2d(Arena *perm_arena,
                           const std::vector<std::vector<float>> &data) {
  if (data.empty()) {
    std::cerr << "Error: Empty 2D data" << std::endl;
    return nullptr;
  }

  uint32_t num_samples = data.size();
  uint32_t num_features = data[0].size();

  for (size_t i = 0; i < data.size(); i++) {
    if (data[i].size() != num_features) {
      std::cerr << "Error: Sample " << i << " has " << data[i].size()
                << " features, expected " << num_features << std::endl;
      return nullptr;
    }
  }

  float *flattened = perm_arena->push_array<float>(
      (size_t)num_samples * num_features);

  if (flattened == nullptr) {
    std::cerr << "Error: Failed to allocate 2D data buffer" << std::endl;
    return nullptr;
  }

  for (uint32_t i = 0; i < num_samples; i++) {
    std::memcpy(flattened + (uint64_t)i * num_features,
                data[i].data(),
                num_features * sizeof(float));
  }

  uint32_t shape[2] = {num_samples, num_features};
  return create(perm_arena, flattened, shape, 2);
}

void Dataset::get_sample_shape(uint32_t *out_shape, uint32_t &out_ndims) const {
  out_ndims = (ndims > 1) ? (ndims - 1) : 0;
  for (uint32_t i = 0; i < out_ndims; i++) {
    out_shape[i] = shape[i + 1];
  }
}

uint64_t Dataset::get_sample_size() const {
  uint64_t size = 1;
  for (uint32_t i = 1; i < ndims; i++) {
    size *= shape[i];
  }
  return size;
}

} // namespace data
} // namespace nn
} // namespace gradientcore
