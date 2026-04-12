#pragma once
#include "module.hpp"
#include <iostream>

namespace gradientcore {
namespace nn {

class Sequential : public Module {
public:
  Sequential() = default;

  void add(Module *module) {
    if (module == nullptr) {
      std::cerr << "Warning: Attempting to add nullptr module to Sequential" << std::endl;
      return;
    }
    register_module(module);
  }

  Module *get(size_t index) {
    if (index >= _modules.size()) {
      std::cerr << "Error: Module index " << index << " out of bounds (size=" 
                << _modules.size() << ")" << std::endl;
      return nullptr;
    }
    return _modules[index];
  }

  size_t num_modules() const { return _modules.size(); }

  bool empty() const { return _modules.empty(); }

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    if (x == nullptr) {
      std::cerr << "Error: Invalid input to Sequential" << std::endl;
      return nullptr;
    }

    if (_modules.empty()) {
      std::cerr << "Warning: Forward pass on empty Sequential, returning input" << std::endl;
      return x;  
    }

    autograd::Variable *out = x;
    for (size_t i = 0; i < _modules.size(); i++) {
      if (_modules[i] == nullptr) {
        std::cerr << "Error: Module " << i << " is nullptr in Sequential" << std::endl;
        return nullptr;
      }
      autograd::Variable *new_out = _modules[i]->forward(compute_arena, out);
      if (new_out == nullptr) {
        std::cerr << "Error: Module " << i << " returned nullptr in Sequential" << std::endl;
        return nullptr;
      }
      out = new_out;
    }
    return out;
  }

  void summary() override {
    std::cout << "Sequential(" << std::endl;
    for (size_t i = 0; i < _modules.size(); i++) {
      std::cout << "  (" << i << "): ";
      _modules[i]->summary();
    }
    std::cout << ")";
    std::cout << " [" << num_parameters() << " params]" << std::endl;
  }
};

} // namespace nn
} // namespace gradientcore
