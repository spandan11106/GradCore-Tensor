#include "../../include/gradient.hpp"
#include <iomanip>
#include <iostream>

using namespace gradientcore;

int main() {
  auto *perm_arena = Arena::create(MiB(128), MiB(16), true);
  auto *graph_arena = Arena::create(MiB(64), MiB(8), true);

  auto csv_raw = CSVLoader::load_csv("data/housing.csv", true);
  std::vector<std::vector<float>> features, labels;
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

  if (!model.load("bin/california_housing.bin")) {
    std::cerr << "Error: Run training first." << std::endl;
    return 1;
  }

  model.get_model()->eval();

  std::cout << "\n=== California Housing Predictions ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2);

  for (int i = 0; i < 10; ++i) {
    uint64_t start_pos = graph_arena->get_pos();
    uint32_t shape[2] = {1, 8};
    Tensor *input = tensor_create(graph_arena, 2, shape);
    std::memcpy(input->storage->data, X_test[i].data(), 8 * sizeof(float));

    autograd::Variable *x = autograd::create_leaf(graph_arena, input, false);
    autograd::Variable *out = model.get_model()->forward(graph_arena, x);

    float pred = out->data->storage->data[0];
    float actual = Y_test[i][0];

    std::cout << "Sample " << i << " | Predicted: $" << (pred * 100000.0f)
              << " | Actual: $" << (actual * 100000.0f) << " | Diff: $"
              << std::abs(pred - actual) * 100000.0f << std::endl;

    graph_arena->pop_to(start_pos);
  }

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
