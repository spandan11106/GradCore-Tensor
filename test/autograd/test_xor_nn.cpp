#include "../../include/autograd/autograd.hpp"
#include "../../include/tensor/memory_cpu/arena.hpp"
#include "../../include/tensor/prng.hpp"

#include "../../include/nn/nn.hpp"
#include "../../include/optim/sgd.hpp"

#include <iostream>
#include <vector>

using namespace gradientcore;

int main() {
  prng::seed_from_entropy();

  Arena *perm_arena = Arena::create(MiB(16), MiB(1), false);
  Arena *graph_arena = Arena::create(MiB(64), MiB(1), true);

  // 1. Define Architecture
  nn::Sequential model;
  nn::Linear fc1(perm_arena, 2, 4);
  nn::Tanh tanh1;
  nn::Linear fc2(perm_arena, 4, 1);

  model.add(&fc1);
  model.add(&tanh1);
  model.add(&fc2);

  // 2. Setup Loss and Optimizer
  nn::MSELoss criterion(REDUCTION_MEAN);
  float learning_rate = 0.05f;
  optim::SGD optimizer(model.parameters(), learning_rate);

  // 3. Prepare Data
  std::vector<std::vector<float>> X_train = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  std::vector<std::vector<float>> Y_train = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

  // 4. Initialize Trainer and Fit
  nn::Trainer<optim::SGD, nn::MSELoss> trainer(&model, &optimizer, &criterion,
                                               graph_arena);

  trainer.fit(X_train, Y_train, /*epochs=*/2000, /*batch_size=*/4,
              /*log_interval=*/250);

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
