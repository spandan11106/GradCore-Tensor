#include "../../include/optim/lbfgs.hpp"
#include "../../include/optim/optim_utils.hpp"
#include <cmath>

namespace gradientcore {
namespace optim {

LBFGS::LBFGS(Arena *p_arena, const std::vector<autograd::Variable *> &params,
             float lr, uint32_t h_size, float tol_grad, float tol_change)
    : parameters(params), perm_arena(p_arena), history_size(h_size),
      learning_rate(lr), tolerance_grad(tol_grad), tolerance_change(tol_change),
      n_iter(0) {

  num_params = get_total_params_size(parameters);

  uint32_t shape[1] = {(uint32_t)num_params};
  d = tensor_create_zeros(perm_arena, 1, shape);
}

void LBFGS::zero_grad() {
  for (auto *p : parameters) {
    if (p->requires_grad && p->grad)
      tensor_clear(p->grad);
  }
}

float LBFGS::step(Arena *temp_arena, std::function<float()> closure) {
  uint32_t shape[1] = {(uint32_t)num_params};

  float orig_loss = closure();

  Tensor *flat_grad = tensor_create_zeros(temp_arena, 1, shape);
  flatten_grads(parameters, flat_grad);

  float grad_norm = std::sqrt(tensor_dot_1d(flat_grad, flat_grad));
  if (grad_norm < tolerance_grad)
    return orig_loss;

  tensor_copy(d, flat_grad);
  tensor_scale(d, -1.0f);

  if (!history.empty()) {
    uint32_t k = history.size();
    std::vector<float> alphas(k);

    for (int i = k - 1; i >= 0; i--) {
      alphas[i] = history[i].rho * tensor_dot_1d(history[i].s, d);

      Tensor *scaled_y = tensor_create_zeros(temp_arena, 1, shape);
      tensor_copy(scaled_y, history[i].y);
      tensor_scale(scaled_y, alphas[i]);
      tensor_sub(d, d, scaled_y);
    }

    float y_dot_y = tensor_dot_1d(history.back().y, history.back().y);
    float gamma = 1.0f / (history.back().rho * y_dot_y);
    tensor_scale(d, gamma);

    for (int i = 0; i < k; i++) {
      float beta = history[i].rho * tensor_dot_1d(history[i].y, d);

      Tensor *scaled_s = tensor_create_zeros(temp_arena, 1, shape);
      tensor_copy(scaled_s, history[i].s);
      tensor_scale(scaled_s, alphas[i] - beta);
      tensor_add(d, d, scaled_s);
    }
  }

  float t = learning_rate;
  float c1 = 1e-4f;
  float dir_dot_g = tensor_dot_1d(d, flat_grad);

  Tensor *flat_params = tensor_create_zeros(temp_arena, 1, shape);
  flatten_params(parameters, flat_params);

  float new_loss = 0.0f;
  Tensor *step_tensor = tensor_create_zeros(temp_arena, 1, shape);

  int ls_iter = 0;
  while (ls_iter < 20) {
    tensor_copy(step_tensor, d);
    tensor_scale(step_tensor, t);

    Tensor *new_params = tensor_create_zeros(temp_arena, 1, shape);
    tensor_copy(new_params, flat_params);
    tensor_add(new_params, new_params, step_tensor);

    unflatten_params(new_params, parameters);

    new_loss = closure();

    if (new_loss <= orig_loss + c1 * t * dir_dot_g) {
      break;
    }

    t *= 0.5f;
    ls_iter++;
  }

  Tensor *new_flat_grad = tensor_create_zeros(temp_arena, 1, shape);
  flatten_grads(parameters, new_flat_grad);

  Tensor *s_k = tensor_create_zeros(perm_arena, 1, shape);
  tensor_copy(s_k, step_tensor);

  Tensor *y_k = tensor_create_zeros(perm_arena, 1, shape);
  tensor_copy(y_k, new_flat_grad);
  tensor_sub(y_k, y_k, flat_grad);

  float y_dot_s = tensor_dot_1d(y_k, s_k);
  if (y_dot_s > 1e-10f) {
    LBFGS_StepData step_data;
    step_data.s = s_k;
    step_data.y = y_k;
    step_data.rho = 1.0f / y_dot_s;

    if (history.size() == history_size) {
      history.pop_front();
    }
    history.push_back(step_data);
  }

  n_iter++;
  return new_loss;
}

} // namespace optim
} // namespace gradientcore
