#pragma once
#include "module.hpp"

namespace gradientcore {
namespace nn {

class Identity : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return x;
  }
};

class ReLU : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::relu(compute_arena, x);
  }
};

class Tanh : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::tanh(compute_arena, x);
  }
};

class Sigmoid : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::sigmoid(compute_arena, x);
  }
};

class Softmax : public Module {
private:
  int32_t dim;

public:
  Softmax(int32_t dim = -1) : dim(dim) {}

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::softmax(compute_arena, x, dim);
  }
};

class LeakyReLU : public Module {
private:
  float alpha;

public:
  LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::leaky_relu(compute_arena, x, alpha);
  }
};

class ELU : public Module {
private:
  float alpha;

public:
  ELU(float alpha = 1.0f) : alpha(alpha) {}

  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::elu(compute_arena, x, alpha);
  }
};

class Swish : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::swish(compute_arena, x);
  }
};

class GELU : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::gelu(compute_arena, x);
  }
};

class ReLU6 : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::relu6(compute_arena, x);
  }
};

class HardSigmoid : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::hard_sigmoid(compute_arena, x);
  }
};

class HardSwish : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::hard_swish(compute_arena, x);
  }
};

class SoftPlus : public Module {
public:
  autograd::Variable *forward(Arena *compute_arena,
                              autograd::Variable *x) override {
    return autograd::softplus(compute_arena, x);
  }
};

} // namespace nn
} // namespace gradientcore
