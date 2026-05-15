#include "../../include/gradient.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

using namespace gradientcore;

int main() {
  auto *perm_arena = Arena::create(MiB(1024), MiB(64), true);
  auto *graph_arena = Arena::create(MiB(512), MiB(32), true);

  std::cout << "Loading California Housing dataset..." << std::endl;

  auto csv_raw = CSVLoader::load_csv("data/housing.csv", true);

  std::vector<std::vector<float>> features, labels;
  // 8 features: longitude, latitude, age, rooms, bed_rooms, pop, households,
  // income
  CSVLoader::parse_csv_to_float(csv_raw, 8, true, features, labels);

  CSVLoader::standardize(features);

  for (auto &label : labels) {
    label[0] /= 100000.0f;
  }

  std::vector<std::vector<float>> X_train, Y_train, X_test, Y_test;
  CSVLoader::train_test_split(features, labels, 0.8f, X_train, Y_train, X_test,
                              Y_test);

  nn::Model model(perm_arena, graph_arena);

  auto *l1 = perm_arena->push<nn::Linear>();
  new (l1) nn::Linear(perm_arena, 8, 128);
  model.add_layer(l1);

  auto *bn1 = perm_arena->push<nn::BatchNorm1d>();
  new (bn1) nn::BatchNorm1d(perm_arena, 128);
  model.add_layer(bn1);

  auto *relu1 = perm_arena->push<nn::ReLU>();
  new (relu1) nn::ReLU();
  model.add_layer(relu1);

  auto *l2 = perm_arena->push<nn::Linear>();
  new (l2) nn::Linear(perm_arena, 128, 64);
  model.add_layer(l2);

  auto *relu2 = perm_arena->push<nn::ReLU>();
  new (relu2) nn::ReLU();
  model.add_layer(relu2);

  auto *l3 = perm_arena->push<nn::Linear>();
  new (l3) nn::Linear(perm_arena, 64, 1);
  model.add_layer(l3);

  model.compile(nn::OptimizerType::ADAMW, nn::LossType::HUBER, 0.001f, 200,
                128);

  std::cout << "\nStarting regression training..." << std::endl;
  model.train(X_train, Y_train);

  std::filesystem::create_directories("bin");
  model.save("bin/california_housing.bin", "binary");

  float test_loss = model.evaluate(X_test, Y_test);
  std::cout << "\nTest Set MSE Loss: " << test_loss << std::endl;

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
