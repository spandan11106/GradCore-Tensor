#pragma once
#include "module.hpp"

namespace gradientcore {
namespace nn {

class Sequential : public Module {
public:
  Sequential() = default;

  void add(Module *module) { register_module(module); }

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    autograd::Variable *out = x;
    for (auto *m : _modules) {
      out = m->forward(compute_arena, out);
    }
    return out;
  }
};

} // namespace nn
} // namespace gradientcore
