#include "../../include/autograd/autograd.hpp"
#include "../../include/nn/activations.hpp"
#include "../../include/nn/linear.hpp"
#include "../../include/nn/sequential.hpp"
#include "../../include/optim/sgd.hpp"
#include "../../include/tensor/memory_cpu/arena.hpp"
#include "../../include/tensor/prng.hpp"
#include <iostream>

using namespace gradientcore;

int main() {
  prng::seed_from_entropy();

  Arena *perm_arena = Arena::create(MiB(16), MiB(1), false);
  Arena *graph_arena = Arena::create(MiB(64), MiB(1), true);

  // 1. Define Architecture using nn::Module
  nn::Sequential model;

  nn::Linear fc1(perm_arena, 2, 4);
  nn::Tanh tanh1;
  nn::Linear fc2(perm_arena, 4, 1);

  model.add(&fc1);
  model.add(&tanh1);
  model.add(&fc2);

  // 2. Setup Optimizer using the dynamically gathered parameters
  optim::SGD optimizer(model.parameters(), 0.05f);

  // XOR Dataset
  float X_data[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  float Y_data[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

  std::cout << "Starting XOR Training..." << std::endl;

  for (int epoch = 0; epoch < 2000; epoch++) {
    float epoch_loss = 0.0f;

    for (int i = 0; i < 4; i++) {
      uint64_t start_pos = graph_arena->get_pos();

      uint32_t shape_X[2] = {1, 2};
      uint32_t shape_Y[2] = {1, 1};
      Tensor *t_x = tensor_create(graph_arena, 2, shape_X);
      Tensor *t_y = tensor_create(graph_arena, 2, shape_Y);
      t_x->storage->data[t_x->offset + 0] = X_data[i][0];
      t_x->storage->data[t_x->offset + 1] = X_data[i][1];
      t_y->storage->data[t_y->offset + 0] = Y_data[i][0];

      autograd::Variable *x = autograd::create_leaf(graph_arena, t_x, false);
      autograd::Variable *y = autograd::create_leaf(graph_arena, t_y, false);

      // 3. Clean Forward Pass
      autograd::Variable *pred = model(graph_arena, x);

      auto *loss = autograd::mse_loss(graph_arena, pred, y, REDUCTION_MEAN);
      epoch_loss += loss->data->storage->data[loss->data->offset];

      // 4. Clean Backward & Update
      optimizer.zero_grad();
      autograd::backward(graph_arena, loss);
      optimizer.step(graph_arena);

      graph_arena->pop_to(start_pos);
    }

    if (epoch % 250 == 0) {
      std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / 4.0f
                << std::endl;
    }
  }

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
