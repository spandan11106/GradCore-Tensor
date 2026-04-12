#pragma once
#include "../../autograd/autograd.hpp"
#include <vector>
#include <iostream>

namespace gradientcore {
namespace nn {

class Module {
protected:
  std::vector<autograd::Variable *> _parameters;
  std::vector<Module *> _modules;
  bool _training;  

public:
  Module() : _training(true) {}
  
  virtual ~Module() = default;

  virtual void train(bool mode = true) {
    _training = mode;
    for (auto *m : _modules) {
      m->train(mode);
    }
  }

  void eval() { train(false); }

  bool is_training() const { return _training; }

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

  virtual uint64_t num_parameters() {
    uint64_t count = 0;
    for (auto *p : parameters()) {
      count += p->data->size;
    }
    return count;
  }

  virtual uint64_t num_trainable_parameters() {
    uint64_t count = 0;
    for (auto *p : parameters()) {
      if (p->requires_grad) {
        count += p->data->size;
      }
    }
    return count;
  }

  virtual void summary() {
    std::cout << "Module Summary:" << std::endl;
    std::cout << "  Total Parameters: " << num_parameters() << std::endl;
    std::cout << "  Trainable Parameters: " << num_trainable_parameters() << std::endl;
    std::cout << "  Training Mode: " << (_training ? "true" : "false") << std::endl;
  }

  virtual autograd::Variable *forward(Arena *compute_arena,
                                      autograd::Variable *x) = 0;

  autograd::Variable *operator()(Arena *compute_arena, autograd::Variable *x) {
    return forward(compute_arena, x);
  }
};

} // namespace nn
} // namespace gradientcore
