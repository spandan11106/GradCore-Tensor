#pragma once

#include "../../tensor/tensor.hpp"
#include "../../tensor/memory_cpu/arena.hpp"
#include <vector>
#include <cstdint>

namespace gradientcore {
namespace nn {
namespace data {

class Dataset {
public:
  static Dataset* create(Arena *perm_arena, const float *data,
                        const uint32_t *shape, uint32_t ndims);

  static Dataset* create_from_samples(Arena *perm_arena,
                                     const std::vector<std::vector<float>> &samples,
                                     const uint32_t *sample_shape,
                                     uint32_t sample_ndims);

  static Dataset* create_2d(Arena *perm_arena,
                           const std::vector<std::vector<float>> &data);

  Tensor* get_data() const { return data_tensor; }
  uint32_t get_num_samples() const { return shape[0]; }
  uint32_t get_ndims() const { return ndims; }
  const uint32_t* get_shape() const { return shape; }
  uint64_t get_sample_size() const;
  void get_sample_shape(uint32_t *out_shape, uint32_t &out_ndims) const;
  Arena* get_arena() const { return perm_arena; }

private:
  Tensor *data_tensor;
  Arena *perm_arena;
  uint32_t shape[MAX_TENSOR_DIMS];
  uint32_t ndims;

  Dataset(Arena *arena, Tensor *tensor, const uint32_t *s, uint32_t nd)
      : data_tensor(tensor), perm_arena(arena), ndims(nd) {
    for (uint32_t i = 0; i < nd; i++) {
      shape[i] = s[i];
    }
  }
};

} // namespace data
} // namespace nn
} // namespace gradientcore
