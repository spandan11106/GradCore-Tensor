#include "optim/adagrad.hpp"
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {
namespace optim {

Adagrad::Adagrad(Arena *perm_arena,
                 const std::vector<autograd::Variable *> &params, float lr,
                 float eps, float wd)
    : parameters(params), learning_rate(lr), epsilon(eps), weight_decay(wd) {
  states.reserve(parameters.size());
  for (auto *p : parameters) {
    AdagradState state;
    if (p->requires_grad) {
      state.sum_sq =
          tensor_create_zeros(perm_arena, p->data->ndims, p->data->shape);
    } else {
      state.sum_sq = nullptr;
    }
    states.push_back(state);
  }
}

void Adagrad::step(Arena *temp_arena) {
  for (size_t i = 0; i < parameters.size(); i++) {
    auto *p = parameters[i];
    if (!p->requires_grad || !p->grad)
      continue;

    Tensor *grad = p->grad;
    Tensor *sum_sq = states[i].sum_sq;
    Tensor *data = p->data;

    if (tensor_is_contiguous(grad) && tensor_is_contiguous(data)) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t k = 0; k < grad->size; k++) {
        float g = grad->storage->data[grad->offset + k];

        if (weight_decay != 0.0f) {
          g += weight_decay * data->storage->data[data->offset + k];
        }

        sum_sq->storage->data[sum_sq->offset + k] += g * g;

        float update =
            g /
            (std::sqrt(sum_sq->storage->data[sum_sq->offset + k]) + epsilon);
        data->storage->data[data->offset + k] -= learning_rate * update;
      }
    } else {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};

      for (uint64_t k = 0; k < grad->size; k++) {
        uint64_t g_idx = tensor_get_flat_index(grad, indices);
        uint64_t s_idx = tensor_get_flat_index(sum_sq, indices);
        uint64_t d_idx = tensor_get_flat_index(data, indices);

        float g = grad->storage->data[g_idx];

        if (weight_decay != 0.0f) {
          g += weight_decay * data->storage->data[d_idx];
        }

        sum_sq->storage->data[s_idx] += g * g;

        float update =
            g /
            (std::sqrt(sum_sq->storage->data[s_idx]) + epsilon);
        data->storage->data[d_idx] -= learning_rate * update;

        for (int32_t d = grad->ndims - 1; d >= 0; d--) {
          indices[d]++;
          if (indices[d] < grad->shape[d])
            break;
          indices[d] = 0;
        }
      }
    }
  }
}

void Adagrad::zero_grad() {
  for (auto *p : parameters) {
    if (p->requires_grad && p->grad != nullptr)
      tensor_clear(p->grad);
  }
}

} // namespace optim
} // namespace gradientcore
