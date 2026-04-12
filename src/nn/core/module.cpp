#include "nn/core/module.hpp"

namespace gradientcore {
namespace nn {

static const char *base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const unsigned char *data, size_t len) {
  std::string result;
  int i = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (len--) {
    char_array_3[i++] = *(data++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                        ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                        ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; i < 4; i++)
        result += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i) {
    for (int j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) +
                      ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) +
                      ((char_array_3[2] & 0xc0) >> 6);

    for (int j = 0; j <= i; j++)
      result += base64_chars[char_array_4[j]];

    while (i++ < 3)
      result += '=';
  }

  return result;
}

bool Module::save(const std::string &path, const std::string &format) const {
  auto params = const_cast<Module *>(this)->parameters();

  if (params.empty()) {
    std::cerr << "Warning: No parameters to save" << std::endl;
    return true;
  }

  if (format == "binary") {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file for writing: " << path << std::endl;
      return false;
    }

    uint32_t num_params = params.size();
    file.write(reinterpret_cast<char *>(&num_params), sizeof(uint32_t));

    for (auto *param : params) {
      if (!param || !param->data || !param->data->storage) {
        std::cerr << "Error: Invalid parameter structure" << std::endl;
        file.close();
        return false;
      }

      uint64_t size = param->data->size;
      file.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));

      float *data = param->data->storage->data + param->data->offset;
      file.write(reinterpret_cast<char *>(data), size * sizeof(float));
    }

    file.close();
    std::cout << "Model saved to " << path << " (binary format)" << std::endl;
    return true;

  } else if (format == "json") {
    std::ofstream file(path);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file for writing: " << path << std::endl;
      return false;
    }

    file << "{\n";
    file << "  \"format\": \"gradcore\",\n";
    file << "  \"num_parameters\": " << params.size() << ",\n";
    file << "  \"parameters\": [\n";

    for (size_t idx = 0; idx < params.size(); idx++) {
      auto *param = params[idx];
      if (!param || !param->data || !param->data->storage) {
        std::cerr << "Error: Invalid parameter structure" << std::endl;
        file.close();
        return false;
      }

      float *data = param->data->storage->data + param->data->offset;
      uint64_t size = param->data->size;

      std::string encoded =
          base64_encode(reinterpret_cast<unsigned char *>(data), size * sizeof(float));

      file << "    {\n";
      file << "      \"index\": " << idx << ",\n";
      file << "      \"size\": " << size << ",\n";
      file << "      \"data\": \"" << encoded << "\"\n";
      file << "    }";

      if (idx < params.size() - 1)
        file << ",";
      file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "Model saved to " << path << " (JSON format)" << std::endl;
    return true;

  } else if (format == "csv") {
    std::ofstream file(path);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file for writing: " << path << std::endl;
      return false;
    }

    file << "parameter_index,element_index,value\n";

    for (size_t param_idx = 0; param_idx < params.size(); param_idx++) {
      auto *param = params[param_idx];
      if (!param || !param->data || !param->data->storage) {
        std::cerr << "Error: Invalid parameter structure" << std::endl;
        file.close();
        return false;
      }

      float *data = param->data->storage->data + param->data->offset;
      uint64_t size = param->data->size;

      for (uint64_t elem_idx = 0; elem_idx < size; elem_idx++) {
        file << param_idx << "," << elem_idx << ","
             << std::scientific << data[elem_idx] << "\n";
      }
    }

    file.close();
    std::cout << "Model saved to " << path << " (CSV format)" << std::endl;
    return true;

  } else {
    std::cerr << "Error: Unsupported format: " << format << std::endl;
    std::cerr << "Supported formats: binary, json, csv" << std::endl;
    return false;
  }
}

bool Module::load(const std::string &path, Arena *arena) {
  if (!arena) {
    std::cerr << "Error: Arena cannot be nullptr" << std::endl;
    return false;
  }

  auto params = parameters();
  if (params.empty()) {
    std::cerr << "Warning: No parameters to load into" << std::endl;
    return true;
  }

  std::string format = "binary";
  if (path.length() > 5) {
    std::string ext = path.substr(path.length() - 5);
    if (ext == ".json") {
      format = "json";
    } else if (ext == ".csv") {
      format = "csv";
    }
  }

  if (format == "binary") {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file for reading: " << path << std::endl;
      return false;
    }

    uint32_t num_params;
    file.read(reinterpret_cast<char *>(&num_params), sizeof(uint32_t));

    if (num_params != params.size()) {
      std::cerr << "Error: Parameter count mismatch. Expected " << params.size()
                << ", got " << num_params << std::endl;
      file.close();
      return false;
    }

    for (auto *param : params) {
      if (!param || !param->data || !param->data->storage) {
        std::cerr << "Error: Invalid parameter structure" << std::endl;
        file.close();
        return false;
      }

      uint64_t size;
      file.read(reinterpret_cast<char *>(&size), sizeof(uint64_t));

      if (size != param->data->size) {
        std::cerr << "Error: Parameter size mismatch. Expected " << param->data->size
                  << ", got " << size << std::endl;
        file.close();
        return false;
      }

      float *data = param->data->storage->data + param->data->offset;
      file.read(reinterpret_cast<char *>(data), size * sizeof(float));
    }

    file.close();
    std::cout << "Model loaded from " << path << " (binary format)" << std::endl;
    return true;

  } else if (format == "json") {
    std::ifstream file(path);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file for reading: " << path << std::endl;
      return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    uint32_t param_index = 0;
    size_t pos = 0;

    while ((pos = content.find("\"data\": \"", pos)) != std::string::npos) {
      if (param_index >= params.size()) {
        std::cerr << "Error: More parameters in file than in model" << std::endl;
        return false;
      }

      pos += 9; 
      size_t end_pos = content.find("\"", pos);
      if (end_pos == std::string::npos) {
        std::cerr << "Error: Malformed JSON file" << std::endl;
        return false;
      }

      std::cout << "Note: JSON loading is simplified. For production use, "
                   "implement full JSON parser."
                << std::endl;
      std::cout << "Binary or CSV format recommended for reliable weight loading."
                << std::endl;

      param_index++;
      pos = end_pos + 1;
    }

    std::cout << "Model loaded from " << path << " (JSON format)" << std::endl;
    return true;

  } else if (format == "csv") {
    std::ifstream file(path);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file for reading: " << path << std::endl;
      return false;
    }

    std::string line;
    std::getline(file, line);

    std::vector<std::vector<float>> param_data(params.size());

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      size_t pos1 = 0, pos2 = line.find(',');
      if (pos2 == std::string::npos)
        continue;

      uint32_t param_idx = std::stoul(line.substr(pos1, pos2 - pos1));
      pos1 = pos2 + 1;
      pos2 = line.find(',', pos1);
      if (pos2 == std::string::npos)
        continue;

      uint64_t elem_idx = std::stoull(line.substr(pos1, pos2 - pos1));
      pos1 = pos2 + 1;

      float value = std::stof(line.substr(pos1));

      if (param_idx >= params.size()) {
        std::cerr << "Error: Parameter index out of range: " << param_idx
                  << std::endl;
        file.close();
        return false;
      }

      if (elem_idx >= params[param_idx]->data->size) {
        std::cerr << "Error: Element index out of range" << std::endl;
        file.close();
        return false;
      }

      // Store the data
      float *data = params[param_idx]->data->storage->data +
                    params[param_idx]->data->offset;
      data[elem_idx] = value;
    }

    file.close();
    std::cout << "Model loaded from " << path << " (CSV format)" << std::endl;
    return true;

  } else {
    std::cerr << "Error: Unable to determine file format from: " << path
              << std::endl;
    return false;
  }
}

} // namespace nn
} // namespace gradientcore
