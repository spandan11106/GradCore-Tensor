#include "../../include/optim/sgd.hpp"

namespace gradientcore {
namespace optim {

SGD::SGD(const std::vector<autograd::Variable *> &params, float lr)
    : parameters(params), learning_rate(lr) {}

void SGD::step(Arena *temp_arena) {
  for (auto *p : parameters) {
    if (p->requires_grad && p->grad != nullptr) {
      uint64_t start_pos = temp_arena->get_pos();

      Tensor *scaled_grad =
          tensor_create_zeros(temp_arena, p->grad->ndims, p->grad->shape);
      tensor_copy(scaled_grad, p->grad);
      tensor_scale(scaled_grad, learning_rate);

      tensor_sub(p->data, p->data, scaled_grad);

      temp_arena->pop_to(start_pos);
    }
  }
}

void SGD::zero_grad() {
  for (auto *p : parameters) {
    if (p->requires_grad && p->grad != nullptr) {
      tensor_clear(p->grad);
    }
  }
}

} // namespace optim
} // namespace gradientcore
