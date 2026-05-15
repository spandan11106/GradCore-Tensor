#include "../../include/gradient.hpp"
#include <iostream>

using namespace gradientcore;

void draw_mnist_digit(float *data) {
  for (uint32_t y = 0; y < 28; y++) {
    for (uint32_t x = 0; x < 28; x++) {
      float num = data[x + y * 28];
      uint32_t col = 232 + (uint32_t)(num * 23);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m\n");
}

int main() {
  auto *perm_arena = Arena::create(MiB(1024), MiB(64), true);
  auto *graph_arena = Arena::create(MiB(512), MiB(32), true);

  std::cout << "Loading MNIST data for Autoencoder..." << std::endl;
  auto csv_raw = CSVLoader::load_csv("data/mnist_test.csv", true);

  std::vector<std::vector<float>> features, labels_raw;
  CSVLoader::parse_mnist_csv(csv_raw, features, labels_raw);
  CSVLoader::normalize_minmax(features);

  nn::Model autoencoder(perm_arena, graph_arena);

  // --- ENCODER ---
  auto *enc1 = perm_arena->push<nn::Linear>();
  new (enc1) nn::Linear(perm_arena, 784, 128);
  autoencoder.add_layer(enc1);

  auto *relu1 = perm_arena->push<nn::ReLU>();
  new (relu1) nn::ReLU();
  autoencoder.add_layer(relu1);

  auto *enc2 = perm_arena->push<nn::Linear>();
  new (enc2) nn::Linear(perm_arena, 128, 32); // Compress to 32 dimensions
  autoencoder.add_layer(enc2);

  auto *relu2 = perm_arena->push<nn::ReLU>();
  new (relu2) nn::ReLU();
  autoencoder.add_layer(relu2);

  // --- DECODER ---
  auto *dec1 = perm_arena->push<nn::Linear>();
  new (dec1) nn::Linear(perm_arena, 32, 128);
  autoencoder.add_layer(dec1);

  auto *relu3 = perm_arena->push<nn::ReLU>();
  new (relu3) nn::ReLU();
  autoencoder.add_layer(relu3);

  auto *dec2 = perm_arena->push<nn::Linear>();
  new (dec2) nn::Linear(perm_arena, 128, 784); // Decompress back to 784
  autoencoder.add_layer(dec2);

  auto *sig = perm_arena->push<nn::Sigmoid>();
  new (sig) nn::Sigmoid();
  autoencoder.add_layer(sig);

  autoencoder.compile(nn::OptimizerType::ADAM, nn::LossType::MSE, 0.001f, 15,
                      64);

  std::cout << "\nStarting Autoencoder training..." << std::endl;
  autoencoder.train(features, features);

  std::cout << "\n=== Visualizing Reconstruction ===" << std::endl;
  autoencoder.get_model()->eval();

  int test_idx = 7;

  std::cout << "\nOriginal Image:" << std::endl;
  draw_mnist_digit(features[test_idx].data());

  uint64_t start_pos = graph_arena->get_pos();
  uint32_t shape[2] = {1, 784};
  Tensor *input = tensor_create(graph_arena, 2, shape);
  std::memcpy(input->storage->data, features[test_idx].data(),
              784 * sizeof(float));

  autograd::Variable *x = autograd::create_leaf(graph_arena, input, false);
  autograd::Variable *out = autoencoder.get_model()->forward(graph_arena, x);

  std::cout << "\nReconstructed Image (from 32 compressed features):"
            << std::endl;
  draw_mnist_digit(out->data->storage->data + out->data->offset);

  graph_arena->pop_to(start_pos);

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
