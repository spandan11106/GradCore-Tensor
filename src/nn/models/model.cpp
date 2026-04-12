#include "../../../include/nn/models/model.hpp"
#include "../../../include/optim/adam.hpp"
#include "../../../include/optim/sgd.hpp"
#include "../../../include/optim/adamw.hpp"
#include "../../../include/optim/rmsprop.hpp"
#include "../../../include/optim/adagrad.hpp"
#include <iostream>
#include <cstring>

namespace gradientcore {
namespace nn {

Model::Model(Arena *perm_arena, Arena *graph_arena)
    : perm_arena(perm_arena), graph_arena(graph_arena),
      learning_rate(0.001f), epochs(10), batch_size(32),
      is_compiled(false), optimizer_ptr(nullptr), loss_ptr(nullptr) {
  
  if (perm_arena == nullptr || graph_arena == nullptr) {
    std::cerr << "Error: Invalid arenas provided to Model" << std::endl;
    return;
  }
  
  sequential = perm_arena->push<Sequential>();
  new (sequential) Sequential();
}

void Model::add_layer(Module *layer) {
  if (!sequential) {
    std::cerr << "Error: Sequential model not initialized" << std::endl;
    return;
  }
  
  if (layer == nullptr) {
    std::cerr << "Error: Cannot add nullptr layer" << std::endl;
    return;
  }
  
  sequential->add(layer);
}

void* Model::create_optimizer() {
  if (!sequential) {
    std::cerr << "Error: Model not initialized" << std::endl;
    return nullptr;
  }
  
  auto params = sequential->parameters();
  
  switch (optimizer_type) {
    case OptimizerType::ADAM: {
      optim::Adam *opt = perm_arena->push<optim::Adam>();
      new (opt) optim::Adam(params, learning_rate);
      return opt;
    }
    case OptimizerType::SGD: {
      optim::SGD *opt = perm_arena->push<optim::SGD>();
      new (opt) optim::SGD(params, learning_rate);
      return opt;
    }
    case OptimizerType::ADAMW: {
      optim::AdamW *opt = perm_arena->push<optim::AdamW>();
      new (opt) optim::AdamW(params, learning_rate);
      return opt;
    }
    case OptimizerType::RMSPROP: {
      optim::RMSProp *opt = perm_arena->push<optim::RMSProp>();
      new (opt) optim::RMSProp(params, learning_rate);
      return opt;
    }
    case OptimizerType::ADAGRAD: {
      optim::Adagrad *opt = perm_arena->push<optim::Adagrad>();
      new (opt) optim::Adagrad(params, learning_rate);
      return opt;
    }
    default:
      std::cerr << "Error: Unknown optimizer type" << std::endl;
      return nullptr;
  }
}

void* Model::create_loss() {
  switch (loss_type) {
    case LossType::CROSS_ENTROPY: {
      CrossEntropyLoss *loss = perm_arena->push<CrossEntropyLoss>();
      new (loss) CrossEntropyLoss();
      return loss;
    }
    case LossType::MSE: {
      MSELoss *loss = perm_arena->push<MSELoss>();
      new (loss) MSELoss();
      return loss;
    }
    case LossType::MAE: {
      MAELoss *loss = perm_arena->push<MAELoss>();
      new (loss) MAELoss();
      return loss;
    }
    case LossType::BCE: {
      BCELoss *loss = perm_arena->push<BCELoss>();
      new (loss) BCELoss();
      return loss;
    }
    case LossType::BCE_WITH_LOGITS: {
      BCEWithLogitsLoss *loss = perm_arena->push<BCEWithLogitsLoss>();
      new (loss) BCEWithLogitsLoss();
      return loss;
    }
    default:
      std::cerr << "Error: Unknown loss type" << std::endl;
      return nullptr;
  }
}

void Model::compile(OptimizerType optimizer,
                    LossType loss,
                    float lr,
                    uint32_t num_epochs,
                    uint32_t batch_sz) {
  
  if (!sequential) {
    std::cerr << "Error: Model not initialized" << std::endl;
    return;
  }
  
  optimizer_type = optimizer;
  loss_type = loss;
  learning_rate = lr;
  epochs = num_epochs;
  batch_size = batch_sz;
  
  // Create optimizer and loss
  optimizer_ptr = create_optimizer();
  loss_ptr = create_loss();
  
  if (optimizer_ptr == nullptr || loss_ptr == nullptr) {
    std::cerr << "Error: Failed to create optimizer or loss" << std::endl;
    return;
  }
  
  is_compiled = true;
  
  std::cout << "Model compiled successfully!" << std::endl;
  std::cout << "  Optimizer: ";
  switch (optimizer_type) {
    case OptimizerType::ADAM: std::cout << "Adam"; break;
    case OptimizerType::SGD: std::cout << "SGD"; break;
    case OptimizerType::ADAMW: std::cout << "AdamW"; break;
    case OptimizerType::RMSPROP: std::cout << "RMSProp"; break;
    case OptimizerType::ADAGRAD: std::cout << "Adagrad"; break;
  }
  std::cout << std::endl;
  std::cout << "  Loss: ";
  switch (loss_type) {
    case LossType::CROSS_ENTROPY: std::cout << "CrossEntropyLoss"; break;
    case LossType::MSE: std::cout << "MSELoss"; break;
    case LossType::MAE: std::cout << "MAELoss"; break;
    case LossType::BCE: std::cout << "BCELoss"; break;
    case LossType::BCE_WITH_LOGITS: std::cout << "BCEWithLogitsLoss"; break;
  }
  std::cout << std::endl;
  std::cout << "  Learning Rate: " << learning_rate << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
}

TrainingStats Model::train(const std::vector<std::vector<float>> &X_train,
                           const std::vector<std::vector<float>> &Y_train) {
  
  TrainingStats empty_stats;
  
  if (!is_compiled) {
    std::cerr << "Error: Model must be compiled before training. Call compile() first." << std::endl;
    return empty_stats;
  }
  
  if (X_train.empty() || Y_train.empty()) {
    std::cerr << "Error: Training data is empty" << std::endl;
    return empty_stats;
  }
  
  if (X_train.size() != Y_train.size()) {
    std::cerr << "Error: X_train and Y_train have different sizes" << std::endl;
    return empty_stats;
  }
  
  Dataset *features_dataset = Dataset::create_2d(perm_arena, X_train);
  Dataset *labels_dataset = Dataset::create_2d(perm_arena, Y_train);
  
  if (!features_dataset || !labels_dataset) {
    std::cerr << "Error: Failed to create datasets" << std::endl;
    return empty_stats;
  }
  
  data::DataLoader *dataloader = data::DataLoader::create(
    features_dataset,
    labels_dataset,
    batch_size,
    true,  /
    42     
  );
  
  if (!dataloader) {
    std::cerr << "Error: Failed to create dataloader" << std::endl;
    return empty_stats;
  }
  
  #define TRAIN_WITH_OPTIMIZER(OPT_TYPE) \
    do { \
      Trainer<OPT_TYPE, CrossEntropyLoss> trainer( \
        sequential, \
        (OPT_TYPE*)optimizer_ptr, \
        (CrossEntropyLoss*)loss_ptr, \
        graph_arena \
      ); \
      return trainer.fit_dataloader(dataloader, epochs, 1); \
    } while(0)
  
  if (optimizer_type == OptimizerType::ADAM && loss_type == LossType::CROSS_ENTROPY) {
    Trainer<optim::Adam, CrossEntropyLoss> trainer(
      sequential,
      (optim::Adam*)optimizer_ptr,
      (CrossEntropyLoss*)loss_ptr,
      graph_arena
    );
    return trainer.fit_dataloader(dataloader, epochs, 1);
  }
  else if (optimizer_type == OptimizerType::SGD && loss_type == LossType::CROSS_ENTROPY) {
    Trainer<optim::SGD, CrossEntropyLoss> trainer(
      sequential,
      (optim::SGD*)optimizer_ptr,
      (CrossEntropyLoss*)loss_ptr,
      graph_arena
    );
    return trainer.fit_dataloader(dataloader, epochs, 1);
  }
  else if (optimizer_type == OptimizerType::ADAM && loss_type == LossType::MSE) {
    Trainer<optim::Adam, MSELoss> trainer(
      sequential,
      (optim::Adam*)optimizer_ptr,
      (MSELoss*)loss_ptr,
      graph_arena
    );
    return trainer.fit_dataloader(dataloader, epochs, 1);
  }
  else {
    std::cerr << "Error: Optimizer/Loss combination not yet supported" << std::endl;
    return empty_stats;
  }
}

float Model::evaluate(const std::vector<std::vector<float>> &X_test,
                     const std::vector<std::vector<float>> &Y_test) {
  
  if (!is_compiled) {
    std::cerr << "Error: Model must be compiled before evaluation" << std::endl;
    return -1.0f;
  }
  
  if (X_test.empty() || Y_test.empty()) {
    std::cerr << "Error: Test data is empty" << std::endl;
    return -1.0f;
  }
  
  Dataset *features_dataset = Dataset::create_2d(perm_arena, X_test);
  Dataset *labels_dataset = Dataset::create_2d(perm_arena, Y_test);
  
  if (!features_dataset || !labels_dataset) {
    std::cerr << "Error: Failed to create test datasets" << std::endl;
    return -1.0f;
  }
  
  data::DataLoader *dataloader = data::DataLoader::create(
    features_dataset,
    labels_dataset,
    batch_size,
    false  
  );
  
  if (!dataloader) {
    std::cerr << "Error: Failed to create test dataloader" << std::endl;
    return -1.0f;
  }
  
  if (optimizer_type == OptimizerType::ADAM && loss_type == LossType::CROSS_ENTROPY) {
    Trainer<optim::Adam, CrossEntropyLoss> trainer(
      sequential,
      (optim::Adam*)optimizer_ptr,
      (CrossEntropyLoss*)loss_ptr,
      graph_arena
    );
    return trainer.evaluate_dataloader(dataloader);
  }
  else if (optimizer_type == OptimizerType::SGD && loss_type == LossType::CROSS_ENTROPY) {
    Trainer<optim::SGD, CrossEntropyLoss> trainer(
      sequential,
      (optim::SGD*)optimizer_ptr,
      (CrossEntropyLoss*)loss_ptr,
      graph_arena
    );
    return trainer.evaluate_dataloader(dataloader);
  }
  else if (optimizer_type == OptimizerType::ADAM && loss_type == LossType::MSE) {
    Trainer<optim::Adam, MSELoss> trainer(
      sequential,
      (optim::Adam*)optimizer_ptr,
      (MSELoss*)loss_ptr,
      graph_arena
    );
    return trainer.evaluate_dataloader(dataloader);
  }
  else {
    std::cerr << "Error: Optimizer/Loss combination not yet supported" << std::endl;
    return -1.0f;
  }
}

void Model::summary() const {
  if (!sequential) {
    std::cout << "Model not initialized" << std::endl;
    return;
  }
  
  std::cout << "\n=== Model Summary ===" << std::endl;
  std::cout << "Total Parameters: " << sequential->num_parameters() << std::endl;
  
  if (is_compiled) {
    std::cout << "\nCompiled Configuration:" << std::endl;
    std::cout << "  Optimizer: ";
    switch (optimizer_type) {
      case OptimizerType::ADAM: std::cout << "Adam"; break;
      case OptimizerType::SGD: std::cout << "SGD"; break;
      case OptimizerType::ADAMW: std::cout << "AdamW"; break;
      case OptimizerType::RMSPROP: std::cout << "RMSProp"; break;
      case OptimizerType::ADAGRAD: std::cout << "Adagrad"; break;
    }
    std::cout << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch Size: " << batch_size << std::endl;
  } else {
    std::cout << "\nStatus: Not compiled. Call compile() to set hyperparameters." << std::endl;
  }
  std::cout << "===================" << std::endl;
}

} // namespace nn
} // namespace gradientcore
