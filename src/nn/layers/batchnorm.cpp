#include "../../../include/nn/layers/batchnorm.hpp"
#include "../../../include/nn/utils/initialization.hpp"
#include <cstring>

namespace gradientcore {
namespace nn {

BatchNorm1d::BatchNorm1d(Arena *perm_arena, uint32_t num_features,
                         float momentum, float epsilon)
    : num_features(num_features), momentum(momentum), epsilon(epsilon),
      num_batches_tracked(0) {
  
  if (num_features == 0) {
    std::cerr << "Error: BatchNorm1d num_features must be > 0" << std::endl;
    gamma = nullptr;
    beta = nullptr;
    running_mean = nullptr;
    running_var = nullptr;
    return;
  }

  uint32_t param_shape[2] = {1, num_features};
  Tensor *gamma_tensor = tensor_create(perm_arena, 2, param_shape);
  if (gamma_tensor == nullptr) {
    std::cerr << "Error: Failed to create gamma tensor" << std::endl;
    gamma = nullptr;
    beta = nullptr;
    running_mean = nullptr;
    running_var = nullptr;
    return;
  }
  gamma = autograd::create_leaf(perm_arena, gamma_tensor, true);
  register_parameter(gamma);
  
  init::ones_(gamma);

  Tensor *beta_tensor = tensor_create(perm_arena, 2, param_shape);
  if (beta_tensor == nullptr) {
    std::cerr << "Error: Failed to create beta tensor" << std::endl;
    beta = nullptr;
    running_mean = nullptr;
    running_var = nullptr;
    return;
  }
  beta = autograd::create_leaf(perm_arena, beta_tensor, true);
  register_parameter(beta);
  
  init::zeros_(beta);

  running_mean = tensor_create(perm_arena, 2, param_shape);
  if (running_mean == nullptr) {
    std::cerr << "Error: Failed to create running_mean tensor" << std::endl;
    running_var = nullptr;
    return;
  }
  init::zeros_(autograd::create_leaf(perm_arena, running_mean, false));

  running_var = tensor_create(perm_arena, 2, param_shape);
  if (running_var == nullptr) {
    std::cerr << "Error: Failed to create running_var tensor" << std::endl;
    return;
  }
  init::ones_(autograd::create_leaf(perm_arena, running_var, false));
}

autograd::Variable *BatchNorm1d::forward(Arena *compute_arena,
                                        autograd::Variable *x) {
  if (!x || !x->data || !gamma || !beta) {
    std::cerr << "Error: Invalid input to BatchNorm1d or layer not initialized" << std::endl;
    return nullptr;
  }

  if (x->data->ndims != 2) {
    std::cerr << "Error: BatchNorm1d expects 2D input [batch_size, features]" << std::endl;
    std::cerr << "       Got " << x->data->ndims << "D tensor" << std::endl;
    return nullptr;
  }

  if (x->data->shape[1] != num_features) {
    std::cerr << "Error: Input features mismatch in BatchNorm1d" << std::endl;
    std::cerr << "       Expected: " << num_features << " Got: " << x->data->shape[1] << std::endl;
    return nullptr;
  }

  uint32_t batch_size = x->data->shape[0];
  uint32_t num_feats = x->data->shape[1];

  if (_training) {
    
    uint32_t out_shape[2] = {batch_size, num_feats};
    Tensor *out_tensor = tensor_create(compute_arena, 2, out_shape);
    if (!out_tensor) {
      std::cerr << "Error: Failed to allocate output tensor" << std::endl;
      return nullptr;
    }

    autograd::Variable *out = autograd::create_leaf(compute_arena, out_tensor, true);
    
    float *in_data = x->data->storage->data + x->data->offset;
    float *out_data = out->data->storage->data + out->data->offset;
    float *gamma_data = gamma->data->storage->data + gamma->data->offset;
    float *beta_data = beta->data->storage->data + beta->data->offset;
    float *mean_data = running_mean->storage->data + running_mean->offset;
    float *var_data = running_var->storage->data + running_var->offset;

    for (uint32_t j = 0; j < num_feats; j++) {
      float sum = 0.0f;
      for (uint32_t i = 0; i < batch_size; i++) {
        sum += in_data[(uint64_t)i * num_feats + j];
      }
      float batch_mean = sum / static_cast<float>(batch_size);

      float var_sum = 0.0f;
      for (uint32_t i = 0; i < batch_size; i++) {
        float diff = in_data[(uint64_t)i * num_feats + j] - batch_mean;
        var_sum += diff * diff;
      }
      float batch_var = var_sum / static_cast<float>(batch_size);

      float inv_std = 1.0f / std::sqrt(batch_var + epsilon);
      for (uint32_t i = 0; i < batch_size; i++) {
        uint64_t idx = (uint64_t)i * num_feats + j;
        float normalized = (in_data[idx] - batch_mean) * inv_std;
        out_data[idx] = normalized * gamma_data[j] + beta_data[j];
      }

      mean_data[j] = (1.0f - momentum) * mean_data[j] + momentum * batch_mean;
      var_data[j] = (1.0f - momentum) * var_data[j] + momentum * batch_var;
    }

    num_batches_tracked++;
    return out;
  } else {
    uint32_t out_shape[2] = {batch_size, num_feats};
    Tensor *out_tensor = tensor_create(compute_arena, 2, out_shape);
    if (!out_tensor) {
      std::cerr << "Error: Failed to allocate output tensor" << std::endl;
      return nullptr;
    }

    autograd::Variable *out = autograd::create_leaf(compute_arena, out_tensor, true);
    
    float *in_data = x->data->storage->data + x->data->offset;
    float *out_data = out->data->storage->data + out->data->offset;
    float *gamma_data = gamma->data->storage->data + gamma->data->offset;
    float *beta_data = beta->data->storage->data + beta->data->offset;
    float *mean_data = running_mean->storage->data + running_mean->offset;
    float *var_data = running_var->storage->data + running_var->offset;

    for (uint32_t i = 0; i < batch_size; i++) {
      for (uint32_t j = 0; j < num_feats; j++) {
        uint64_t idx = (uint64_t)i * num_feats + j;
        float normalized = (in_data[idx] - mean_data[j]) / std::sqrt(var_data[j] + epsilon);
        out_data[idx] = normalized * gamma_data[j] + beta_data[j];
      }
    }

    return out;
  }
}

BatchNorm2d::BatchNorm2d(Arena *perm_arena, uint32_t num_features,
                         float momentum, float epsilon)
    : num_features(num_features), momentum(momentum), epsilon(epsilon),
      num_batches_tracked(0) {
  
  if (num_features == 0) {
    std::cerr << "Error: BatchNorm2d num_features must be > 0" << std::endl;
    gamma = nullptr;
    beta = nullptr;
    running_mean = nullptr;
    running_var = nullptr;
    return;
  }

  uint32_t param_shape[2] = {1, num_features};
  Tensor *gamma_tensor = tensor_create(perm_arena, 2, param_shape);
  if (gamma_tensor == nullptr) {
    std::cerr << "Error: Failed to create gamma tensor" << std::endl;
    gamma = nullptr;
    beta = nullptr;
    running_mean = nullptr;
    running_var = nullptr;
    return;
  }
  gamma = autograd::create_leaf(perm_arena, gamma_tensor, true);
  register_parameter(gamma);
  init::ones_(gamma);

  Tensor *beta_tensor = tensor_create(perm_arena, 2, param_shape);
  if (beta_tensor == nullptr) {
    std::cerr << "Error: Failed to create beta tensor" << std::endl;
    beta = nullptr;
    running_mean = nullptr;
    running_var = nullptr;
    return;
  }
  beta = autograd::create_leaf(perm_arena, beta_tensor, true);
  register_parameter(beta);
  init::zeros_(beta);

  running_mean = tensor_create(perm_arena, 2, param_shape);
  if (running_mean == nullptr) {
    std::cerr << "Error: Failed to create running_mean tensor" << std::endl;
    running_var = nullptr;
    return;
  }
  init::zeros_(autograd::create_leaf(perm_arena, running_mean, false));

  running_var = tensor_create(perm_arena, 2, param_shape);
  if (running_var == nullptr) {
    std::cerr << "Error: Failed to create running_var tensor" << std::endl;
    return;
  }
  init::ones_(autograd::create_leaf(perm_arena, running_var, false));
}

autograd::Variable *BatchNorm2d::forward(Arena *compute_arena,
                                        autograd::Variable *x) {
  if (!x || !x->data || !gamma || !beta) {
    std::cerr << "Error: Invalid input to BatchNorm2d or layer not initialized" << std::endl;
    return nullptr;
  }

  if (x->data->ndims != 4) {
    std::cerr << "Error: BatchNorm2d expects 4D input [batch, channels, height, width]" << std::endl;
    std::cerr << "       Got " << x->data->ndims << "D tensor" << std::endl;
    return nullptr;
  }

  if (x->data->shape[1] != num_features) {
    std::cerr << "Error: Input channels mismatch in BatchNorm2d" << std::endl;
    std::cerr << "       Expected: " << num_features << " Got: " << x->data->shape[1] << std::endl;
    return nullptr;
  }

  uint32_t batch_size = x->data->shape[0];
  uint32_t channels = x->data->shape[1];
  uint32_t height = x->data->shape[2];
  uint32_t width = x->data->shape[3];

  uint32_t out_shape[4] = {batch_size, channels, height, width};
  Tensor *out_tensor = tensor_create(compute_arena, 4, out_shape);
  if (!out_tensor) {
    std::cerr << "Error: Failed to allocate output tensor" << std::endl;
    return nullptr;
  }

  autograd::Variable *out = autograd::create_leaf(compute_arena, out_tensor, true);
  
  float *in_data = x->data->storage->data + x->data->offset;
  float *out_data = out->data->storage->data + out->data->offset;
  float *gamma_data = gamma->data->storage->data + gamma->data->offset;
  float *beta_data = beta->data->storage->data + beta->data->offset;

  if (_training) {
    float *mean_data = running_mean->storage->data + running_mean->offset;
    float *var_data = running_var->storage->data + running_var->offset;

    uint64_t spatial_size = (uint64_t)height * width;
    uint64_t count = (uint64_t)batch_size * spatial_size;

    for (uint32_t c = 0; c < channels; c++) {
      float sum = 0.0f;
      for (uint32_t b = 0; b < batch_size; b++) {
        for (uint64_t s = 0; s < spatial_size; s++) {
          uint64_t idx = (uint64_t)b * channels * spatial_size + c * spatial_size + s;
          sum += in_data[idx];
        }
      }
      float batch_mean = sum / static_cast<float>(count);

      float var_sum = 0.0f;
      for (uint32_t b = 0; b < batch_size; b++) {
        for (uint64_t s = 0; s < spatial_size; s++) {
          uint64_t idx = (uint64_t)b * channels * spatial_size + c * spatial_size + s;
          float diff = in_data[idx] - batch_mean;
          var_sum += diff * diff;
        }
      }
      float batch_var = var_sum / static_cast<float>(count);

      float inv_std = 1.0f / std::sqrt(batch_var + epsilon);
      for (uint32_t b = 0; b < batch_size; b++) {
        for (uint64_t s = 0; s < spatial_size; s++) {
          uint64_t idx = (uint64_t)b * channels * spatial_size + c * spatial_size + s;
          float normalized = (in_data[idx] - batch_mean) * inv_std;
          out_data[idx] = normalized * gamma_data[c] + beta_data[c];
        }
      }

      mean_data[c] = (1.0f - momentum) * mean_data[c] + momentum * batch_mean;
      var_data[c] = (1.0f - momentum) * var_data[c] + momentum * batch_var;
    }
  } else {
    float *mean_data = running_mean->storage->data + running_mean->offset;
    float *var_data = running_var->storage->data + running_var->offset;

    uint64_t spatial_size = (uint64_t)height * width;
    for (uint32_t b = 0; b < batch_size; b++) {
      for (uint32_t c = 0; c < channels; c++) {
        for (uint64_t s = 0; s < spatial_size; s++) {
          uint64_t idx = (uint64_t)b * channels * spatial_size + c * spatial_size + s;
          float normalized = (in_data[idx] - mean_data[c]) / std::sqrt(var_data[c] + epsilon);
          out_data[idx] = normalized * gamma_data[c] + beta_data[c];
        }
      }
    }
  }

  num_batches_tracked++;
  return out;
}

} // namespace nn
} // namespace gradientcore
