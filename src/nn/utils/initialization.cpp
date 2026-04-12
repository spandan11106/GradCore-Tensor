#include "../../../include/nn/utils/initialization.hpp"
#include "../../../include/tensor/prng.hpp"
#include <cmath>
#include <iostream>
#include <cstring>

namespace gradientcore {
namespace nn {
namespace init {

static void calculate_fans(const Tensor *tensor, uint32_t &fan_in, uint32_t &fan_out) {
  if (!tensor || tensor->ndims < 1) {
    fan_in = 1;
    fan_out = 1;
    return;
  }
  
  if (tensor->ndims == 2) {
    fan_in = tensor->shape[0];
    fan_out = tensor->shape[1];
  } else if (tensor->ndims == 1) {
    fan_in = tensor->shape[0];
    fan_out = tensor->shape[0];
  } else {
    fan_out = 1;
    for (uint32_t i = 0; i < tensor->ndims - 1; i++) {
      fan_out *= tensor->shape[i];
    }
    fan_in = tensor->shape[tensor->ndims - 1];
  }
}

void xavier_uniform_(autograd::Variable *weight) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  Tensor *tensor = weight->data;
  uint32_t fan_in, fan_out;
  calculate_fans(tensor, fan_in, fan_out);
  
  float limit = std::sqrt(6.0f / (fan_in + fan_out));
  
  float *data = tensor->storage->data + tensor->offset;
  uint64_t size = tensor->size;
  
  for (uint64_t i = 0; i < size; i++) {
    // Random uniform in [-limit, limit]
    data[i] = -limit + 2.0f * limit * prng::randf();
  }
}

void xavier_normal_(autograd::Variable *weight) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  Tensor *tensor = weight->data;
  uint32_t fan_in, fan_out;
  calculate_fans(tensor, fan_in, fan_out);
  
  float std = std::sqrt(2.0f / (fan_in + fan_out));
  
  float *data = tensor->storage->data + tensor->offset;
  uint64_t size = tensor->size;
  
  // Box-Muller transform for normal distribution
  for (uint64_t i = 0; i < size; i += 2) {
    float u1 = prng::randf();
    float u2 = prng::randf();
    
    // Avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;
    
    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    data[i] = z0 * std;
    
    if (i + 1 < size) {
      float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * 3.14159265f * u2);
      data[i + 1] = z1 * std;
    }
  }
}

void kaiming_uniform_(autograd::Variable *weight) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  Tensor *tensor = weight->data;
  uint32_t fan_in, fan_out;
  calculate_fans(tensor, fan_in, fan_out);
  
  float limit = std::sqrt(6.0f / fan_in);
  
  float *data = tensor->storage->data + tensor->offset;
  uint64_t size = tensor->size;
  
  for (uint64_t i = 0; i < size; i++) {
    // Random uniform in [-limit, limit]
    data[i] = -limit + 2.0f * limit * prng::randf();
  }
}

void kaiming_normal_(autograd::Variable *weight) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  Tensor *tensor = weight->data;
  uint32_t fan_in, fan_out;
  calculate_fans(tensor, fan_in, fan_out);
  
  float std = std::sqrt(2.0f / fan_in);
  
  float *data = tensor->storage->data + tensor->offset;
  uint64_t size = tensor->size;
  
  // Box-Muller transform for normal distribution
  for (uint64_t i = 0; i < size; i += 2) {
    float u1 = prng::randf();
    float u2 = prng::randf();
    
    // Avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;
    
    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    data[i] = z0 * std;
    
    if (i + 1 < size) {
      float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * 3.14159265f * u2);
      data[i + 1] = z1 * std;
    }
  }
}

void uniform_(autograd::Variable *weight, float min_val, float max_val) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  float *data = weight->data->storage->data + weight->data->offset;
  uint64_t size = weight->data->size;
  
  float range = max_val - min_val;
  for (uint64_t i = 0; i < size; i++) {
    data[i] = min_val + range * prng::randf();
  }
}

void normal_(autograd::Variable *weight, float mean, float std) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  float *data = weight->data->storage->data + weight->data->offset;
  uint64_t size = weight->data->size;
  
  // Box-Muller transform for normal distribution
  for (uint64_t i = 0; i < size; i += 2) {
    float u1 = prng::randf();
    float u2 = prng::randf();
    
    // Avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;
    
    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    data[i] = mean + z0 * std;
    
    if (i + 1 < size) {
      float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * 3.14159265f * u2);
      data[i + 1] = mean + z1 * std;
    }
  }
}

void constant_(autograd::Variable *weight, float value) {
  if (!weight || !weight->data) {
    std::cerr << "Error: Invalid weight variable for initialization" << std::endl;
    return;
  }
  
  float *data = weight->data->storage->data + weight->data->offset;
  uint64_t size = weight->data->size;
  
  for (uint64_t i = 0; i < size; i++) {
    data[i] = value;
  }
}

} // namespace init
} // namespace nn
} // namespace gradientcore
