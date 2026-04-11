#include "../../include/autograd/autograd.hpp"
#include "../../include/tensor/memory_cpu/arena.hpp"
#include <cstdlib>
#include <iostream>

using namespace gradientcore;

// Quick helper to randomly initialize weight tensors between -1.0 and 1.0
void random_init(Tensor *t) {
  for (uint64_t i = 0; i < t->size; i++) {
    t->storage->data[t->offset + i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
}

int main() {
  // Permanent arena for weights/biases. Transient arena for the dynamic graph.
  Arena *perm_arena = Arena::create(MiB(16), MiB(1), false);
  Arena *graph_arena = Arena::create(MiB(64), MiB(1), true);

  // 1. Define Architecture & Initialize Weights (2 -> 4 -> 1)
  uint32_t shape_W1[2] = {2, 4};
  uint32_t shape_b1[2] = {1, 4};
  uint32_t shape_W2[2] = {4, 1};
  uint32_t shape_b2[2] = {1, 1};

  Tensor *t_W1 = tensor_create(perm_arena, 2, shape_W1);
  Tensor *t_b1 = tensor_create_zeros(perm_arena, 2, shape_b1);
  Tensor *t_W2 = tensor_create(perm_arena, 2, shape_W2);
  Tensor *t_b2 = tensor_create_zeros(perm_arena, 2, shape_b2);

  random_init(t_W1);
  random_init(t_W2);

  autograd::Variable *W1 = autograd::create_leaf(perm_arena, t_W1, true);
  autograd::Variable *b1 = autograd::create_leaf(perm_arena, t_b1, true);
  autograd::Variable *W2 = autograd::create_leaf(perm_arena, t_W2, true);
  autograd::Variable *b2 = autograd::create_leaf(perm_arena, t_b2, true);

  // XOR Dataset
  float X_data[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  float Y_data[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

  float learning_rate = 0.05f;

  // --- Manual Optimizer Lambdas ---
  auto zero_grad = [](autograd::Variable *p) {
    if (p->requires_grad && p->grad) {
      tensor_clear(p->grad);
    }
  };

  auto step = [&](autograd::Variable *p) {
    if (p->requires_grad && p->grad) {
      uint64_t start_pos = graph_arena->get_pos();

      // Temporary tensor to hold scaled gradient: (grad * lr)
      Tensor *scaled_grad =
          tensor_create_zeros(graph_arena, p->grad->ndims, p->grad->shape);
      tensor_copy(scaled_grad, p->grad);
      tensor_scale(scaled_grad, learning_rate);

      // In-place update: data = data - scaled_grad
      tensor_sub(p->data, p->data, scaled_grad);

      // Free the temporary tensor
      graph_arena->pop_to(start_pos);
    }
  };

  std::cout << "Starting XOR Training..." << std::endl;

  for (int epoch = 0; epoch < 2000; epoch++) {
    float epoch_loss = 0.0f;

    for (int i = 0; i < 4; i++) {
      // Memory Checkpoint
      uint64_t start_pos = graph_arena->get_pos();

      // 2. Setup Input & Target for this step
      uint32_t shape_X[2] = {1, 2};
      uint32_t shape_Y[2] = {1, 1};
      Tensor *t_x = tensor_create(graph_arena, 2, shape_X);
      Tensor *t_y = tensor_create(graph_arena, 2, shape_Y);
      t_x->storage->data[t_x->offset + 0] = X_data[i][0];
      t_x->storage->data[t_x->offset + 1] = X_data[i][1];
      t_y->storage->data[t_y->offset + 0] = Y_data[i][0];

      autograd::Variable *x = autograd::create_leaf(graph_arena, t_x, false);
      autograd::Variable *y = autograd::create_leaf(graph_arena, t_y, false);

      // 3. Dynamic Forward Pass
      auto *z1 =
          autograd::add(graph_arena, autograd::matmul(graph_arena, x, W1), b1);
      auto *a1 = autograd::tanh(graph_arena, z1);
      auto *z2 =
          autograd::add(graph_arena, autograd::matmul(graph_arena, a1, W2), b2);

      auto *loss = autograd::mse_loss(graph_arena, z2, y, REDUCTION_MEAN);
      epoch_loss += loss->data->storage->data[loss->data->offset];

      // 4. Manual Zero Grad
      zero_grad(W1);
      zero_grad(b1);
      zero_grad(W2);
      zero_grad(b2);

      // 5. Backward Pass
      autograd::backward(graph_arena, loss);

      // 6. Manual Step (Update Weights)
      step(W1);
      step(b1);
      step(W2);
      step(b2);

      // 7. Memory Wipe: Destroy the dynamic graph!
      graph_arena->pop_to(start_pos);
    }

    if (epoch % 250 == 0) {
      std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / 4.0f
                << std::endl;
    }
  }

  // Test Phase
  std::cout << "\nFinal XOR Predictions:" << std::endl;
  for (int i = 0; i < 4; i++) {
    uint64_t start_pos = graph_arena->get_pos();

    uint32_t shape_X[2] = {1, 2};
    Tensor *t_x = tensor_create(graph_arena, 2, shape_X);
    t_x->storage->data[t_x->offset + 0] = X_data[i][0];
    t_x->storage->data[t_x->offset + 1] = X_data[i][1];

    autograd::Variable *x = autograd::create_leaf(graph_arena, t_x, false);

    auto *z1 =
        autograd::add(graph_arena, autograd::matmul(graph_arena, x, W1), b1);
    auto *a1 = autograd::tanh(graph_arena, z1);
    auto *z2 =
        autograd::add(graph_arena, autograd::matmul(graph_arena, a1, W2), b2);

    float pred = z2->data->storage->data[z2->data->offset];
    std::cout << X_data[i][0] << " XOR " << X_data[i][1] << " = " << pred
              << std::endl;

    graph_arena->pop_to(start_pos);
  }

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
