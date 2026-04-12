#pragma once
#include "../../autograd/autograd.hpp"
#include "../core/module.hpp"
#include <iostream>

namespace gradientcore {
namespace nn {

class LossFunction {
public:
  Reduction reduction;

  virtual ~LossFunction() = default;

  virtual autograd::Variable *forward(Arena *arena, autograd::Variable *pred,
                                      autograd::Variable *target) = 0;

  autograd::Variable *operator()(Arena *arena, autograd::Variable *pred,
                                 autograd::Variable *target) {
    return forward(arena, pred, target);
  }
};

class MSELoss : public LossFunction {
public:
  MSELoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to MSELoss" << std::endl;
      return nullptr;
    }
    return autograd::mse_loss(compute_arena, pred, target, reduction);
  }
};

class L1Loss : public LossFunction {
public:
  L1Loss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to L1Loss" << std::endl;
      return nullptr;
    }
    return autograd::l1_loss(compute_arena, pred, target, reduction);
  }
};

class L2Loss : public LossFunction {
public:
  L2Loss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr) {
      std::cerr << "Error: Invalid input to L2Loss" << std::endl;
      return nullptr;
    }
    return autograd::l2_loss(compute_arena, pred, reduction);
  }
};

class MAELoss : public LossFunction {
public:
  MAELoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to MAELoss" << std::endl;
      return nullptr;
    }
    return autograd::l1_loss(compute_arena, pred, target, reduction);
  }
};

class BCELoss : public LossFunction {
public:
  BCELoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to BCELoss" << std::endl;
      return nullptr;
    }
    return autograd::bce_loss(compute_arena, pred, target, reduction);
  }
};

class BCEWithLogitsLoss : public LossFunction {
public:
  BCEWithLogitsLoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *logits,
                              autograd::Variable *target) override {
    if (logits == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to BCEWithLogitsLoss" << std::endl;
      return nullptr;
    }
    return autograd::bce_with_logits_loss(compute_arena, logits, target, reduction);
  }
};

class CrossEntropyLoss : public LossFunction {
public:
  CrossEntropyLoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *logits,
                              autograd::Variable *target) override {
    if (logits == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to CrossEntropyLoss" << std::endl;
      return nullptr;
    }
    return autograd::cross_entropy_loss(compute_arena, logits, target, reduction);
  }
};

class NLLLoss : public LossFunction {
public:
  NLLLoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *log_probs,
                              autograd::Variable *target) override {
    if (log_probs == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to NLLLoss" << std::endl;
      return nullptr;
    }
    return autograd::nll_loss(compute_arena, log_probs, target, reduction);
  }
};

class KLDivLoss : public LossFunction {
public:
  KLDivLoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to KLDivLoss" << std::endl;
      return nullptr;
    }
    return autograd::kl_div_loss(compute_arena, pred, target, reduction);
  }
};

class HingeLoss : public LossFunction {
public:
  HingeLoss(Reduction red = REDUCTION_MEAN) { reduction = red; }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to HingeLoss" << std::endl;
      return nullptr;
    }
    return autograd::hinge_loss(compute_arena, pred, target, reduction);
  }
};

class HuberLoss : public LossFunction {
private:
  float delta;

public:
  HuberLoss(float d = 1.0f, Reduction red = REDUCTION_MEAN)
      : delta(d) {
    reduction = red;
  }

  autograd::Variable *forward(Arena *compute_arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    if (pred == nullptr || target == nullptr) {
      std::cerr << "Error: Invalid input to HuberLoss" << std::endl;
      return nullptr;
    }
    return autograd::huber_loss(compute_arena, pred, target, delta, reduction);
  }
};

class CosineEmbeddingLoss : public LossFunction {
private:
  float margin;

public:
  CosineEmbeddingLoss(float m = 0.0f, Reduction red = REDUCTION_MEAN)
      : margin(m) {
    reduction = red;
  }

  // Specialized forward for 3-input case (x1, x2, target)
  autograd::Variable *forward_triplet(Arena *compute_arena, autograd::Variable *x1,
                                      autograd::Variable *x2,
                                      autograd::Variable *target = nullptr) {
    if (x1 == nullptr || x2 == nullptr) {
      std::cerr << "Error: Invalid input to CosineEmbeddingLoss" << std::endl;
      return nullptr;
    }
    return autograd::cosine_embedding_loss(compute_arena, x1, x2, margin, target, reduction);
  }

  // Base class override (not typically used for this loss)
  autograd::Variable *forward(Arena *arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    std::cerr << "Warning: CosineEmbeddingLoss requires 3 inputs. Use forward_triplet() instead." << std::endl;
    return nullptr;
  }
};

class TripletLoss : public LossFunction {
private:
  float margin;

public:
  TripletLoss(float m = 1.0f, Reduction red = REDUCTION_MEAN)
      : margin(m) {
    reduction = red;
  }

  // Specialized forward for triplet case (anchor, positive, negative)
  autograd::Variable *forward_triplet(Arena *compute_arena, autograd::Variable *anchor,
                                      autograd::Variable *positive,
                                      autograd::Variable *negative) {
    if (anchor == nullptr || positive == nullptr || negative == nullptr) {
      std::cerr << "Error: Invalid input to TripletLoss" << std::endl;
      return nullptr;
    }
    return autograd::triplet_loss(compute_arena, anchor, positive, negative, margin, reduction);
  }

  // Base class override (not typically used for this loss)
  autograd::Variable *forward(Arena *arena, autograd::Variable *pred,
                              autograd::Variable *target) override {
    std::cerr << "Warning: TripletLoss requires 3 inputs. Use forward_triplet() instead." << std::endl;
    return nullptr;
  }
};

} // namespace nn
} // namespace gradientcore
