#pragma once

/**
 * @file nn.hpp
 * @brief Main header for the neural network module
 * 
 * NN Module Structure:
 * ====================
 * 
 * core/                   - Core abstractions
 *   ├── module.hpp        Base class for all layers
 *   └── sequential.hpp    Sequential container
 * 
 * layers/                 - Layer implementations
 *   ├── linear.hpp        Fully connected layers
 *   └── activations.hpp   Activation functions (ReLU, Sigmoid, Tanh, etc.)
 * 
 * losses/                 - Loss functions
 *   └── loss.hpp          CrossEntropyLoss, MSELoss, BCELoss, etc.
 * 
 * data/                   - Data loading & batching
 *   ├── dataset.hpp       Dataset abstraction (supports any shape)
 *   └── dataloader.hpp    DataLoader for efficient batching
 * 
 * training/               - Training utilities
 *   └── trainer.hpp       Trainer class for training loops
 * 
 * models/                 - High-level model APIs
 *   └── model.hpp         User-friendly Model class (Keras-like)
 * 
 * utils/                  - Utility functions
 *   ├── initialization.hpp Weight initialization (Xavier, He, etc.)
 *   └── metrics.hpp       Evaluation metrics (Accuracy, Precision, etc.)
 */

// Core abstractions
#include "core/module.hpp"
#include "core/sequential.hpp"

// Layer implementations
#include "layers/activations.hpp"
#include "layers/linear.hpp"

// Loss functions
#include "losses/loss.hpp"

// Data loading
#include "data/dataset.hpp"
#include "data/dataloader.hpp"

// Training utilities
#include "training/trainer.hpp"

// High-level API
#include "models/model.hpp"

namespace gradientcore {
namespace nn {

// Convenience namespace for direct access

} // namespace nn
} // namespace gradientcore
