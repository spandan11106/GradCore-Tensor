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
 *   └── initialization.hpp Weight initialization (Xavier, Kaiming, etc.)
 */

// Core abstractions
#include "core/module.hpp"
#include "core/sequential.hpp"

// Layer implementations
#include "layers/activations.hpp"
#include "layers/linear.hpp"
#include "layers/batchnorm.hpp"
#include "layers/dropout.hpp"

// Loss functions
#include "losses/loss.hpp"

// Data loading
#include "data/dataset.hpp"
#include "data/dataloader.hpp"

// Training utilities
#include "training/trainer.hpp"

// High-level API
#include "models/model.hpp"

// Utility functions
#include "utils/initialization.hpp"

namespace gradientcore {
namespace nn {

// Convenience namespace for direct access

} // namespace nn
} // namespace gradientcore
