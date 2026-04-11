#pragma once
#include "../autograd/autograd.hpp"
#include <vector>

namespace gradientcore {
namespace nn {

class Module {
protected:
  std::vector<autograd::Variable *> _parameters;
  std::vector<Module *> _modules;

public:
  virtual ~Module() = default;

  void register_parameter(autograd::Variable *param) {
    _parameters.push_back(param);
  }

  void register_module(Module *module) { _modules.push_back(module); }

  virtual std::vector<autograd::Variable *> parameters() {
    std::vector<autograd::Variable *> all_params = _parameters;
    for (auto *m : _modules) {
      auto sub_params = m->parameters();
      all_params.insert(all_params.end(), sub_params.begin(), sub_params.end());
    }
    return all_params;
  }

  virtual autograd::Variable *forward(Arena *compute_arena,
                                      autograd::Variable *x) = 0;

  autograd::Variable *operator()(Arena *compute_arena, autograd::Variable *x) {
    return forward(compute_arena, x);
  }
};

} // namespace nn
} // namespace gradientcore
