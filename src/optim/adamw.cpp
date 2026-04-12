#include "optim/adamw.hpp"
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gradientcore {
namespace optim {

AdamW::AdamW(Arena *perm_arena, const std::vector<autograd::Variable *> &params,
             float lr, float b1, float b2, float eps, float wd)
    : parameters(params), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps),
      weight_decay(wd), t(0) {
  states.reserve(parameters.size());
  for (auto *p : parameters) {
    AdamWState state;
    if (p->requires_grad) {
      state.m = tensor_create_zeros(perm_arena, p->data->ndims, p->data->shape);
      state.v = tensor_create_zeros(perm_arena, p->data->ndims, p->data->shape);
    } else {
      state.m = nullptr;
      state.v = nullptr;
    }
    states.push_back(state);
  }
}

void AdamW::step(Arena *temp_arena) {
  t++;
  float m_hat_correction = 1.0f / (1.0f - std::pow(beta1, t));
  float v_hat_correction = 1.0f / (1.0f - std::pow(beta2, t));

  for (size_t i = 0; i < parameters.size(); i++) {
    auto *p = parameters[i];
    if (!p->requires_grad || !p->grad)
      continue;

    Tensor *grad = p->grad;
    Tensor *m = states[i].m;
    Tensor *v = states[i].v;
    Tensor *data = p->data;

    if (tensor_is_contiguous(grad) && tensor_is_contiguous(data)) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (uint64_t k = 0; k < grad->size; k++) {
        float g = grad->storage->data[grad->offset + k];
        float w = data->storage->data[data->offset + k];

        // AdamW Decoupled Weight Decay: Apply directly to the weight
        data->storage->data[data->offset + k] -=
            learning_rate * weight_decay * w;

        // Standard Adam updates
        m->storage->data[m->offset + k] =
            beta1 * m->storage->data[m->offset + k] + (1.0f - beta1) * g;
        v->storage->data[v->offset + k] =
            beta2 * v->storage->data[v->offset + k] + (1.0f - beta2) * g * g;

        float m_hat = m->storage->data[m->offset + k] * m_hat_correction;
        float v_hat = v->storage->data[v->offset + k] * v_hat_correction;

        data->storage->data[data->offset + k] -=
            learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
      }
    } else {
      uint32_t indices[MAX_TENSOR_DIMS] = {0};

      for (uint64_t k = 0; k < grad->size; k++) {
        uint64_t g_idx = tensor_get_flat_index(grad, indices);
        uint64_t m_idx = tensor_get_flat_index(m, indices);
        uint64_t v_idx = tensor_get_flat_index(v, indices);
        uint64_t d_idx = tensor_get_flat_index(data, indices);

        float g = grad->storage->data[g_idx];
        float w = data->storage->data[d_idx];

        // AdamW Decoupled Weight Decay: Apply directly to the weight
        data->storage->data[d_idx] -=
            learning_rate * weight_decay * w;

        // Standard Adam updates
        m->storage->data[m_idx] =
            beta1 * m->storage->data[m_idx] + (1.0f - beta1) * g;
        v->storage->data[v_idx] =
            beta2 * v->storage->data[v_idx] + (1.0f - beta2) * g * g;

        float m_hat = m->storage->data[m_idx] * m_hat_correction;
        float v_hat = v->storage->data[v_idx] * v_hat_correction;

        data->storage->data[d_idx] -=
            learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

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

void AdamW::zero_grad() {
  for (auto *p : parameters) {
    if (p->requires_grad && p->grad != nullptr)
      tensor_clear(p->grad);
  }
}

} // namespace optim
} // namespace gradientcore
