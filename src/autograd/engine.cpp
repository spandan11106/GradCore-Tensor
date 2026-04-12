#include "autograd/autograd.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace gradientcore {
namespace autograd {

Variable *create_leaf(Arena *arena, Tensor *data, bool requires_grad) {
  Variable *v = arena->push<Variable>();
  v->data = data;
  v->requires_grad = requires_grad;
  v->is_leaf = true;
  v->parents = nullptr;
  v->num_parents = 0;
  v->saved_tensors = nullptr;
  v->num_saved = 0;
  v->backward_fn = nullptr;

  if (requires_grad) {
    v->grad = tensor_create_zeros(arena, data->ndims, data->shape);
  } else {
    v->grad = nullptr;
  }
  return v;
}

static void build_topo(Variable *v, std::unordered_set<Variable *> &visited,
                       std::vector<Variable *> &topo) {
  if (!v || visited.count(v))
    return;
  visited.insert(v);

  for (uint32_t i = 0; i < v->num_parents; i++) {
    build_topo(v->parents[i].node, visited, topo);
  }
  topo.push_back(v);
}

void backward(Arena *arena, Variable *loss_node) {
  if (!loss_node || !loss_node->requires_grad)
    return;

  tensor_fill(loss_node->grad, 1.0f);

  std::vector<Variable *> topo;
  std::unordered_set<Variable *> visited;
  build_topo(loss_node, visited, topo);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Variable *v = *it;
    if (v->backward_fn) {
      v->backward_fn(v, arena);
    }
  }
}

} // namespace autograd
} // namespace gradientcore
