#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

namespace gradientcore {

class CSVLoader {
public:

  static std::vector<std::vector<std::string>> load_csv(
      const std::string &filepath, bool skip_header = false) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file " << filepath << std::endl;
      return data;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
      if (first_line && skip_header) {
        first_line = false;
        continue;
      }
      first_line = false;

      std::vector<std::string> row;
      std::stringstream ss(line);
      std::string cell;

      while (std::getline(ss, cell, ',')) {
        // Trim whitespace
        cell.erase(0, cell.find_first_not_of(" \t\r\n"));
        cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
        row.push_back(cell);
      }

      if (!row.empty()) {
        data.push_back(row);
      }
    }

    file.close();
    return data;
  }

  static void parse_csv_to_float(
      const std::vector<std::vector<std::string>> &csv_data,
      uint32_t feature_cols,
      bool has_label,
      std::vector<std::vector<float>> &features,
      std::vector<std::vector<float>> &labels) {
    features.clear();
    labels.clear();

    for (const auto &row : csv_data) {
      if (row.size() < feature_cols) {
        continue; 
      }

      std::vector<float> feature_vec;

      for (uint32_t i = 0; i < feature_cols; i++) {
        try {
          float val = std::stof(row[i]);
          feature_vec.push_back(val);
        } catch (...) {
          feature_vec.push_back(0.0f);
        }
      }

      features.push_back(feature_vec);

      if (has_label && row.size() > feature_cols) {
        try {
          float label = std::stof(row[feature_cols]);
          labels.push_back({label});
        } catch (...) {
          labels.push_back({0.0f});
        }
      }
    }

    std::cout << "Loaded " << features.size() << " samples" << std::endl;
  }

  // MNIST CSV format: label in column 0, 784 pixel values in columns 1-784
  static void parse_mnist_csv(
      const std::vector<std::vector<std::string>> &csv_data,
      std::vector<std::vector<float>> &features,
      std::vector<std::vector<float>> &labels) {
    features.clear();
    labels.clear();

    for (const auto &row : csv_data) {
      if (row.size() < 785) {  // 1 label + 784 pixels
        continue;
      }

      // Extract label from column 0
      std::vector<float> label_vec;
      try {
        float label = std::stof(row[0]);
        label_vec.push_back(label);
      } catch (...) {
        label_vec.push_back(0.0f);
      }
      labels.push_back(label_vec);

      // Extract 784 pixels from columns 1-784
      std::vector<float> feature_vec;
      for (uint32_t i = 1; i <= 784; i++) {
        try {
          float val = std::stof(row[i]);
          feature_vec.push_back(val);
        } catch (...) {
          feature_vec.push_back(0.0f);
        }
      }
      features.push_back(feature_vec);
    }

    std::cout << "Loaded " << features.size() << " samples" << std::endl;
  }

  static void normalize_minmax(
      std::vector<std::vector<float>> &features) {
    if (features.empty() || features[0].empty()) return;

    uint32_t num_features = features[0].size();

    for (uint32_t j = 0; j < num_features; j++) {
      float min_val = features[0][j];
      float max_val = features[0][j];

      for (const auto &sample : features) {
        if (sample[j] < min_val) min_val = sample[j];
        if (sample[j] > max_val) max_val = sample[j];
      }

      if (min_val == max_val) continue; 

      for (auto &sample : features) {
        sample[j] = (sample[j] - min_val) / (max_val - min_val);
      }
    }
  }

  static void standardize(
      std::vector<std::vector<float>> &features) {
    if (features.empty() || features[0].empty()) return;

    uint32_t num_features = features[0].size();
    uint32_t num_samples = features.size();

    for (uint32_t j = 0; j < num_features; j++) {
      float mean = 0.0f;
      for (const auto &sample : features) {
        mean += sample[j];
      }
      mean /= num_samples;

      float variance = 0.0f;
      for (const auto &sample : features) {
        float diff = sample[j] - mean;
        variance += diff * diff;
      }
      variance /= num_samples;
      float std_dev = std::sqrt(variance);

      if (std_dev == 0) continue;

      for (auto &sample : features) {
        sample[j] = (sample[j] - mean) / std_dev;
      }
    }
  }

  static void one_hot_encode(
      const std::vector<std::vector<float>> &labels,
      uint32_t num_classes,
      std::vector<std::vector<float>> &encoded) {
    encoded.clear();
    for (const auto &label_vec : labels) {
      int class_idx = static_cast<int>(label_vec[0]);
      std::vector<float> one_hot(num_classes, 0.0f);
      if (class_idx >= 0 && class_idx < (int)num_classes) {
        one_hot[class_idx] = 1.0f;
      }
      encoded.push_back(one_hot);
    }
  }

  static void train_test_split(
      const std::vector<std::vector<float>> &features,
      const std::vector<std::vector<float>> &labels,
      float train_ratio,
      std::vector<std::vector<float>> &X_train,
      std::vector<std::vector<float>> &Y_train,
      std::vector<std::vector<float>> &X_test,
      std::vector<std::vector<float>> &Y_test) {
    uint32_t num_samples = features.size();
    uint32_t train_size = static_cast<uint32_t>(num_samples * train_ratio);

    X_train.assign(features.begin(), features.begin() + train_size);
    Y_train.assign(labels.begin(), labels.begin() + train_size);
    
    X_test.assign(features.begin() + train_size, features.end());
    Y_test.assign(labels.begin() + train_size, labels.end());
  }
};

} // namespace gradientcore
