#pragma once

#include "../../autograd/autograd.hpp"
#include "../../tensor/tensor.hpp"
#include <cmath>

namespace gradientcore {
namespace nn {
namespace init {

void xavier_uniform_(autograd::Variable *weight);

void xavier_normal_(autograd::Variable *weight);

void kaiming_uniform_(autograd::Variable *weight);

void kaiming_normal_(autograd::Variable *weight);

void uniform_(autograd::Variable *weight, float min_val = -1.0f, float max_val = 1.0f);

void normal_(autograd::Variable *weight, float mean = 0.0f, float std = 1.0f);

void constant_(autograd::Variable *weight, float value = 0.0f);

inline void zeros_(autograd::Variable *weight) {
  constant_(weight, 0.0f);
}

inline void ones_(autograd::Variable *weight) {
  constant_(weight, 1.0f);
}

} // namespace init
} // namespace nn
} // namespace gradientcore
