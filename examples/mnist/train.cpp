#include "../../include/gradient.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <vector>

using namespace gradientcore;

int main() {
    auto* perm_arena = Arena::create(MiB(1024), MiB(64), true);
    auto* graph_arena = Arena::create(MiB(512), MiB(32), true);

    std::cout << "Loading MNIST training data..." << std::endl;
    auto csv_raw = CSVLoader::load_csv("data/mnist_train.csv", true);
    
    std::vector<std::vector<float>> features, labels_raw, labels_onehot;
    CSVLoader::parse_mnist_csv(csv_raw, features, labels_raw);
    CSVLoader::normalize_minmax(features);
    CSVLoader::one_hot_encode(labels_raw, 10, labels_onehot);

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

    model.compile(nn::OptimizerType::ADAM, nn::LossType::CROSS_ENTROPY, 0.0005f, 40, 64);
    
    std::cout << "Total parameters registered: " << model.get_model()->num_parameters() << std::endl;
    if (model.get_model()->num_parameters() == 0) {
        std::cerr << "Error: Model still has 0 parameters. Check your add_layer implementation." << std::endl;
        return 1;
    }

    std::cout << "Starting training..." << std::endl;
    model.train(features, labels_onehot); 

    std::filesystem::create_directories("bin");
    model.save("bin/mnist_model.bin", "binary");

    std::cout << "\n=== Evaluating on test set ===" << std::endl;

    auto test_csv = CSVLoader::load_csv("data/mnist_test.csv", true);
    std::vector<std::vector<float>> test_features, test_labels_raw;
    CSVLoader::parse_mnist_csv(test_csv, test_features, test_labels_raw);
    CSVLoader::normalize_minmax(test_features);

    model.get_model()->eval();

    uint32_t num_test = test_features.size();
    uint32_t batch_size = 100;
    uint32_t correct = 0;

    for (uint32_t i = 0; i < num_test; i += batch_size) {
        uint32_t current_bs = std::min(batch_size, num_test - i);
        uint64_t start_pos = graph_arena->get_pos();

        uint32_t shape_x[2] = {current_bs, 784};
        Tensor* t_x = tensor_create(graph_arena, 2, shape_x);

        for (uint32_t b = 0; b < current_bs; b++) {
            for (uint32_t j = 0; j < 784; j++) {
                t_x->storage->data[t_x->offset + b * 784 + j] = test_features[i + b][j];
            }
        }

        autograd::Variable* x = autograd::create_leaf(graph_arena, t_x, false);
        autograd::Variable* out = model.get_model()->forward(graph_arena, x);

        for (uint32_t b = 0; b < current_bs; b++) {
            float max_v = -1e9f;
            int pred = 0;
            for (int c = 0; c < 10; c++) {
                float v = out->data->storage->data[out->data->offset + b * 10 + c];
                if (v > max_v) {
                    max_v = v;
                    pred = c;
                }
            }
            int true_label = static_cast<int>(test_labels_raw[i + b][0]);
            if (pred == true_label) correct++;
        }

        graph_arena->pop_to(start_pos);
    }

    float accuracy = 100.0f * correct / num_test;
    std::cout << "Test Accuracy: " << correct << " / " << num_test
              << " (" << std::fixed << std::setprecision(2) << accuracy << "%)" << std::endl;

    perm_arena->destroy();
    graph_arena->destroy();
    return 0;
}