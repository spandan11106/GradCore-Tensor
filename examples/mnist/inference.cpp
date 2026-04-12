#include "../../include/gradient.hpp"
#include <iostream>

using namespace gradientcore;

int main() {
    auto* perm_arena = Arena::create(MiB(128), MiB(16), true);
    auto* graph_arena = Arena::create(MiB(64), MiB(8), true);

    auto csv_raw = CSVLoader::load_csv("data/mnist_test.csv", true);
    std::vector<std::vector<float>> features, labels_raw;
    CSVLoader::parse_mnist_csv(csv_raw, features, labels_raw);
    CSVLoader::normalize_minmax(features);

    nn::Model model(perm_arena, graph_arena);
    auto* l1 = perm_arena->push<nn::Linear>();
    new (l1) nn::Linear(perm_arena, 784, 128);
    model.add_layer(l1);

    auto* relu = perm_arena->push<nn::ReLU>();
    new (relu) nn::ReLU();
    model.add_layer(relu);

    auto* l2 = perm_arena->push<nn::Linear>();
    new (l2) nn::Linear(perm_arena, 128, 10);
    model.add_layer(l2);

    if (!model.load("bin/mnist_model.bin")) {
        std::cerr << "Error: Run training first." << std::endl;
        return 1;
    }
    model.get_model()->eval();

    int n;
    std::cout << "Enter sample index (0 to " << features.size() - 1 << "): ";
    std::cin >> n;

    if (n < 0 || n >= (int)features.size()) {
        std::cout << "Index out of range." << std::endl;
        return 1;
    }

    uint32_t shape[2] = {1, 784};
    Tensor* input = tensor_create(graph_arena, 2, shape);
    std::memcpy(input->storage->data, features[n].data(), 784 * sizeof(float));

    autograd::Variable* x = autograd::create_leaf(graph_arena, input, false);
    autograd::Variable* out = model.get_model()->forward(graph_arena, x);

    float max_v = -1e9; int pred = 0;
    for(int i=0; i<10; ++i) {
        if(out->data->storage->data[i] > max_v) {
            max_v = out->data->storage->data[i];
            pred = i;
        }
    }
    std::cout << "Actual Label: " << labels_raw[n][0] << " | Predicted: " << pred << std::endl;

    perm_arena->destroy();
    graph_arena->destroy();
    return 0;
}