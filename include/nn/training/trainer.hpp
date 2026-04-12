#pragma once
#include "../../tensor/memory_cpu/arena.hpp"
#include "../core/module.hpp"
#include "../losses/loss.hpp"
#include "../data/dataloader.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

namespace gradientcore {
namespace nn {

struct TrainingStats {
  std::vector<float> train_losses;  
  float final_loss;
  uint32_t epochs_trained;
  uint32_t total_batches;
  
  TrainingStats() 
      : final_loss(0.0f), epochs_trained(0), total_batches(0) {}
};

template <typename OptimizerType, typename LossType>
class Trainer {
private:
  Module *model;
  OptimizerType *optimizer;
  LossType *criterion;
  Arena *graph_arena;
  bool verbose;

public:
  Trainer(Module *mod, OptimizerType *opt, LossType *crit, Arena *arena)
      : model(mod), optimizer(opt), criterion(crit), graph_arena(arena), verbose(true) {
    
    if (model == nullptr) {
      std::cerr << "Error: Model is nullptr" << std::endl;
    }
    if (optimizer == nullptr) {
      std::cerr << "Error: Optimizer is nullptr" << std::endl;
    }
    if (criterion == nullptr) {
      std::cerr << "Error: Criterion is nullptr" << std::endl;
    }
    if (graph_arena == nullptr) {
      std::cerr << "Error: Arena is nullptr" << std::endl;
    }
  }

  void set_verbose(bool v) { verbose = v; }

  bool validate_data(const std::vector<std::vector<float>> &X,
                     const std::vector<std::vector<float>> &Y) {
    if (X.empty() || Y.empty()) {
      std::cerr << "Error: Empty training data" << std::endl;
      return false;
    }

    if (X.size() != Y.size()) {
      std::cerr << "Error: X and Y have different number of samples" << std::endl;
      std::cerr << "  X size: " << X.size() << " Y size: " << Y.size() << std::endl;
      return false;
    }

    size_t expected_X_dim = X[0].size();
    size_t expected_Y_dim = Y[0].size();

    for (size_t i = 0; i < X.size(); i++) {
      if (X[i].size() != expected_X_dim) {
        std::cerr << "Error: Inconsistent X dimensions at sample " << i << std::endl;
        std::cerr << "  Expected: " << expected_X_dim 
                  << " Got: " << X[i].size() << std::endl;
        return false;
      }
      if (Y[i].size() != expected_Y_dim) {
        std::cerr << "Error: Inconsistent Y dimensions at sample " << i << std::endl;
        std::cerr << "  Expected: " << expected_Y_dim 
                  << " Got: " << Y[i].size() << std::endl;
        return false;
      }
    }

    return true;
  }

  TrainingStats fit(const std::vector<std::vector<float>> &X_train,
                    const std::vector<std::vector<float>> &Y_train,
                    uint32_t epochs, uint32_t batch_size = 32,
                    uint32_t log_interval = 100) {
    
    TrainingStats stats;

    if (!validate_data(X_train, Y_train)) {
      return stats;
    }

    if (model == nullptr || optimizer == nullptr || criterion == nullptr) {
      std::cerr << "Error: Trainer not properly initialized" << std::endl;
      return stats;
    }

    if (graph_arena == nullptr) {
      std::cerr << "Error: Arena not valid" << std::endl;
      return stats;
    }

    if (batch_size == 0) {
      std::cerr << "Error: Batch size must be > 0" << std::endl;
      return stats;
    }

    // Set model to training mode
    model->train(true);

    uint32_t num_samples = X_train.size();
    uint32_t input_dim = X_train[0].size();
    uint32_t output_dim = Y_train[0].size();

    if (verbose) {
      std::cout << "=== Training Configuration ===" << std::endl;
      std::cout << "Epochs: " << epochs << std::endl;
      std::cout << "Batch Size: " << batch_size << std::endl;
      std::cout << "Samples: " << num_samples << std::endl;
      std::cout << "Input Features: " << input_dim << std::endl;
      std::cout << "Output Features: " << output_dim << std::endl;
      std::cout << "Model Parameters: " << model->num_parameters() << std::endl;
      std::cout << "=============================" << std::endl << std::endl;
    }

    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
      float epoch_loss = 0.0f;
      uint32_t num_batches = 0;

      for (uint32_t i = 0; i < num_samples; i += batch_size) {
        uint32_t current_batch_size = std::min(batch_size, num_samples - i);

        uint64_t start_pos = graph_arena->get_pos();

        uint32_t shape_X[2] = {current_batch_size, input_dim};
        uint32_t shape_Y[2] = {current_batch_size, output_dim};
        
        Tensor *t_x = tensor_create(graph_arena, 2, shape_X);
        Tensor *t_y = tensor_create(graph_arena, 2, shape_Y);

        if (t_x == nullptr || t_y == nullptr) {
          std::cerr << "Error: Failed to create batch tensors" << std::endl;
          return stats;
        }

        for (uint32_t b = 0; b < current_batch_size; b++) {
          for (uint32_t j = 0; j < input_dim; j++) {
            t_x->storage->data[t_x->offset + b * input_dim + j] =
                X_train[i + b][j];
          }
          for (uint32_t j = 0; j < output_dim; j++) {
            t_y->storage->data[t_y->offset + b * output_dim + j] =
                Y_train[i + b][j];
          }
        }

        autograd::Variable *x = autograd::create_leaf(graph_arena, t_x, false);
        autograd::Variable *y = autograd::create_leaf(graph_arena, t_y, false);

        if (x == nullptr || y == nullptr) {
          std::cerr << "Error: Failed to create variables" << std::endl;
          return stats;
        }

        autograd::Variable *pred = model->forward(graph_arena, x);
        if (pred == nullptr) {
          std::cerr << "Error: Model forward pass returned nullptr at epoch " 
                    << epoch << " batch " << num_batches << std::endl;
          return stats;
        }

        autograd::Variable *loss = criterion->forward(graph_arena, pred, y);
        if (loss == nullptr) {
          std::cerr << "Error: Loss computation failed" << std::endl;
          return stats;
        }

        float batch_loss = loss->data->storage->data[loss->data->offset];
        epoch_loss += batch_loss;
        num_batches++;
        stats.total_batches++;

        optimizer->zero_grad();
        autograd::backward(graph_arena, loss);
        optimizer->step(graph_arena);

        graph_arena->pop_to(start_pos);
      }

      float avg_epoch_loss = epoch_loss / num_batches;
      stats.train_losses.push_back(avg_epoch_loss);
      stats.final_loss = avg_epoch_loss;
      stats.epochs_trained = epoch + 1;

      if (verbose && (epoch % log_interval == 0 || epoch == epochs - 1)) {
        std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "] | "
                  << "Loss: " << avg_epoch_loss << std::endl;
      }
    }

    model->eval();

    if (verbose) {
      std::cout << "Training complete! Final Loss: " << stats.final_loss << std::endl;
    }

    return stats;
  }

  float evaluate(const std::vector<std::vector<float>> &X_test,
                 const std::vector<std::vector<float>> &Y_test) {
    
    if (!validate_data(X_test, Y_test)) {
      return -1.0f;
    }

    model->eval();  

    uint32_t num_samples = X_test.size();
    uint32_t input_dim = X_test[0].size();
    uint32_t output_dim = Y_test[0].size();

    float total_loss = 0.0f;
    uint32_t num_batches = 0;

    uint32_t batch_size = std::min(32u, (uint32_t)num_samples);

    for (uint32_t i = 0; i < num_samples; i += batch_size) {
      uint32_t current_batch_size = std::min(batch_size, num_samples - i);

      uint64_t start_pos = graph_arena->get_pos();

      uint32_t shape_X[2] = {current_batch_size, input_dim};
      uint32_t shape_Y[2] = {current_batch_size, output_dim};
      Tensor *t_x = tensor_create(graph_arena, 2, shape_X);
      Tensor *t_y = tensor_create(graph_arena, 2, shape_Y);

      for (uint32_t b = 0; b < current_batch_size; b++) {
        for (uint32_t j = 0; j < input_dim; j++) {
          t_x->storage->data[t_x->offset + b * input_dim + j] = X_test[i + b][j];
        }
        for (uint32_t j = 0; j < output_dim; j++) {
          t_y->storage->data[t_y->offset + b * output_dim + j] = Y_test[i + b][j];
        }
      }

      autograd::Variable *x = autograd::create_leaf(graph_arena, t_x, false);
      autograd::Variable *y = autograd::create_leaf(graph_arena, t_y, false);

      autograd::Variable *pred = model->forward(graph_arena, x);
      autograd::Variable *loss = criterion->forward(graph_arena, pred, y);

      total_loss += loss->data->storage->data[loss->data->offset];
      num_batches++;

      graph_arena->pop_to(start_pos);
    }

    return total_loss / num_batches;
  }

  TrainingStats fit_dataloader(data::DataLoader *dataloader, uint32_t epochs,
                               uint32_t log_interval = 100) {
    TrainingStats stats;

    if (dataloader == nullptr) {
      std::cerr << "Error: DataLoader is nullptr" << std::endl;
      return stats;
    }

    if (model == nullptr || optimizer == nullptr || criterion == nullptr) {
      std::cerr << "Error: Trainer not properly initialized" << std::endl;
      return stats;
    }

    if (graph_arena == nullptr) {
      std::cerr << "Error: Arena not valid" << std::endl;
      return stats;
    }

    model->train(true);

    uint32_t num_batches = dataloader->get_num_batches();
    uint32_t batch_size = dataloader->get_batch_size();
    uint32_t dataset_size = dataloader->get_dataset_size();

    if (verbose) {
      std::cout << "=== Training Configuration (DataLoader) ===" << std::endl;
      std::cout << "Epochs: " << epochs << std::endl;
      std::cout << "Batch Size: " << batch_size << std::endl;
      std::cout << "Samples: " << dataset_size << std::endl;
      std::cout << "Batches per Epoch: " << num_batches << std::endl;
      std::cout << "Model Parameters: " << model->num_parameters() << std::endl;
      
      // Print feature shape
      uint32_t feat_ndims = dataloader->get_feature_ndims();
      const uint32_t* feat_shape = dataloader->get_feature_shape();
      std::cout << "Feature shape: [";
      for (uint32_t i = 0; i < feat_ndims; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << feat_shape[i];
      }
      std::cout << "]" << std::endl;
      std::cout << "==========================================" << std::endl << std::endl;
    }

    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
      float epoch_loss = 0.0f;
      uint32_t batch_count = 0;

      dataloader->reset(true);

      while (dataloader->has_next()) {
        uint64_t start_pos = graph_arena->get_pos();

        data::Batch batch = dataloader->next(graph_arena);

        if (batch.features == nullptr || batch.labels == nullptr) {
          std::cerr << "Error: Failed to get batch from DataLoader" << std::endl;
          return stats;
        }

        autograd::Variable *x = autograd::create_leaf(graph_arena, 
                                                       batch.features, false);
        autograd::Variable *y = autograd::create_leaf(graph_arena, 
                                                       batch.labels, false);

        if (x == nullptr || y == nullptr) {
          std::cerr << "Error: Failed to create variables from batch" << std::endl;
          return stats;
        }

        autograd::Variable *pred = model->forward(graph_arena, x);
        if (pred == nullptr) {
          std::cerr << "Error: Model forward pass returned nullptr at epoch " 
                    << epoch << " batch " << batch_count << std::endl;
          return stats;
        }

        autograd::Variable *loss = criterion->forward(graph_arena, pred, y);
        if (loss == nullptr) {
          std::cerr << "Error: Loss computation failed" << std::endl;
          return stats;
        }

        float batch_loss = loss->data->storage->data[loss->data->offset];
        epoch_loss += batch_loss;
        batch_count++;
        stats.total_batches++;

        optimizer->zero_grad();
        autograd::backward(graph_arena, loss);
        optimizer->step(graph_arena);

        graph_arena->pop_to(start_pos);
      }

      float avg_epoch_loss = epoch_loss / batch_count;
      stats.train_losses.push_back(avg_epoch_loss);
      stats.final_loss = avg_epoch_loss;
      stats.epochs_trained = epoch + 1;

      if (verbose && (epoch % log_interval == 0 || epoch == epochs - 1)) {
        std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "] | "
                  << "Loss: " << avg_epoch_loss << std::endl;
      }
    }

    model->eval();

    if (verbose) {
      std::cout << "Training complete! Final Loss: " << stats.final_loss << std::endl;
    }

    return stats;
  }

  float evaluate_dataloader(data::DataLoader *dataloader) {
    if (dataloader == nullptr) {
      std::cerr << "Error: DataLoader is nullptr" << std::endl;
      return -1.0f;
    }

    model->eval();

    float total_loss = 0.0f;
    uint32_t num_batches = 0;

    dataloader->reset(false);  

    while (dataloader->has_next()) {
      uint64_t start_pos = graph_arena->get_pos();

      data::Batch batch = dataloader->next(graph_arena);

      if (batch.features == nullptr || batch.labels == nullptr) {
        std::cerr << "Error: Failed to get batch from DataLoader" << std::endl;
        return -1.0f;
      }

      autograd::Variable *x = autograd::create_leaf(graph_arena, 
                                                     batch.features, false);
      autograd::Variable *y = autograd::create_leaf(graph_arena, 
                                                     batch.labels, false);

      autograd::Variable *pred = model->forward(graph_arena, x);
      autograd::Variable *loss = criterion->forward(graph_arena, pred, y);

      total_loss += loss->data->storage->data[loss->data->offset];
      num_batches++;

      graph_arena->pop_to(start_pos);
    }

    return num_batches > 0 ? (total_loss / num_batches) : -1.0f;
  }
};

} // namespace nn
} // namespace gradientcore
