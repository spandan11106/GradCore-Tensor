#pragma once
#include "../../autograd/autograd.hpp"
#include <vector>
#include <iostream>
#include <functional>
#include <fstream>
#include <sstream>
#include <cstring>
#include <map>

namespace gradientcore {
namespace nn {

using ForwardHook = std::function<void(autograd::Variable *)>;

class Module {
protected:
  std::vector<autograd::Variable *> _parameters;
  std::vector<Module *> _modules;
  bool _training;
  std::vector<ForwardHook> _forward_hooks;
  mutable std::vector<autograd::Variable *> _cached_parameters;
  mutable std::map<std::string, autograd::Variable *> _named_parameters;
  mutable bool _parameters_cached;  

public:
  Module() : _training(true), _parameters_cached(false) {}
  
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
    _parameters_cached = false;
  }

  void register_module(Module *module) {
    _modules.push_back(module);
    _parameters_cached = false;
  }

  void register_forward_hook(ForwardHook hook) {
    _forward_hooks.push_back(hook);
  }

  void _build_parameters_recursive(const std::string &prefix = "") {
    for (size_t i = 0; i < _parameters.size(); i++) {
      std::string param_name = prefix.empty() ? std::to_string(i) : prefix + "." + std::to_string(i);
      _cached_parameters.push_back(_parameters[i]);
      _named_parameters[param_name] = _parameters[i];
    }
    
    for (size_t i = 0; i < _modules.size(); i++) {
      auto *m = _modules[i];
      std::string module_prefix = prefix.empty() ? std::to_string(i) : prefix + "." + std::to_string(i);
      m->_build_parameters_recursive(module_prefix);
    }
  }

  void _build_named_parameters_recursive(const std::string &prefix = "") {
    for (size_t i = 0; i < _parameters.size(); i++) {
      std::string param_name = prefix.empty() ? std::to_string(i) : prefix + "." + std::to_string(i);
      _named_parameters[param_name] = _parameters[i];
    }
    
    for (size_t i = 0; i < _modules.size(); i++) {
      auto *m = _modules[i];
      std::string module_prefix = prefix.empty() ? std::to_string(i) : prefix + "." + std::to_string(i);
      m->_build_named_parameters_recursive(module_prefix);
    }
  }

  bool save(const std::string &path, const std::string &format = "binary") const;

  bool load(const std::string &path, Arena *arena);

  virtual std::vector<autograd::Variable *> parameters() {
    if (!_parameters_cached) {
      _cached_parameters.clear();
      _build_parameters_recursive();
      _parameters_cached = true;
    }
    return _cached_parameters;
  }

  virtual std::map<std::string, autograd::Variable *> named_parameters() {
    if (!_parameters_cached) {
      _cached_parameters.clear();
      _named_parameters.clear();
      _build_named_parameters_recursive();
      _parameters_cached = true;
    }
    return _named_parameters;
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
    autograd::Variable *output = forward(compute_arena, x);
    
    for (auto &hook : _forward_hooks) {
      hook(output);
    }
    
    return output;
  }
};

} // namespace nn
} // namespace gradientcore
