#pragma once

#include "../core/sequential.hpp"
#include "../losses/loss.hpp"
#include "../training/trainer.hpp"
#include "../data/dataloader.hpp"
#include <string>
#include <iostream>
#include <vector>
#include <memory>

namespace gradientcore {
namespace nn {

enum class OptimizerType {
  ADAM,
  SGD,
  ADAMW,
  RMSPROP,
  ADAGRAD
};

enum class LossType {
  CROSS_ENTROPY,
  MSE,
  MAE,
  BCE,
  BCE_WITH_LOGITS
};

class Model {
private:
  Sequential *sequential;
  Arena *perm_arena;
  Arena *graph_arena;
  
  OptimizerType optimizer_type;
  LossType loss_type;
  float learning_rate;
  uint32_t epochs;
  uint32_t batch_size;
  bool is_compiled;
  
  void* optimizer_ptr;   
  void* loss_ptr;        
  
  void* create_optimizer();
  
  void* create_loss();
  
public:
  Model(Arena *perm_arena, Arena *graph_arena);
  
  void add_layer(Module *layer);
  
  void compile(OptimizerType optimizer,
               LossType loss,
               float lr,
               uint32_t num_epochs,
               uint32_t batch_sz);
  
  TrainingStats train(const std::vector<std::vector<float>> &X_train,
                      const std::vector<std::vector<float>> &Y_train);
  
  float evaluate(const std::vector<std::vector<float>> &X_test,
                 const std::vector<std::vector<float>> &Y_test);
  
  Sequential* get_model() { return sequential; }
  
  void summary() const;
  
  bool is_built() const { return is_compiled; }
  
  void set_learning_rate(float lr) { learning_rate = lr; }
  
  void set_epochs(uint32_t num_epochs) { epochs = num_epochs; }
  
  void set_batch_size(uint32_t batch_sz) { batch_size = batch_sz; }
  
  bool save(const std::string &path, const std::string &format = "binary") {
    if (!sequential) {
      std::cerr << "Error: Model not initialized" << std::endl;
      return false;
    }
    return sequential->save(path, format);
  }
  
  bool load(const std::string &path) {
    if (!sequential) {
      std::cerr << "Error: Model not initialized" << std::endl;
      return false;
    }
    return sequential->load(path, perm_arena);
  }
};

} // namespace nn
} // namespace gradientcore
