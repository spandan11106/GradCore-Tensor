#pragma once

#include <cstdint>

namespace gradientcore {
// Based on https://www.pcg-random.org
class PRNG {
private:
  uint64_t state;
  uint64_t increment;

public:
  PRNG();
  PRNG(uint64_t init_state, uint64_t init_seq);

  void seed(uint64_t init_state, uint64_t init_seq);

  uint32_t rand();

  // Generates a uniform random number in [0, 1)
  float randf();

  // Generates a standard normal distribution (mean=0, std=1)
  float std_norm();
};

namespace prng {
void seed(uint64_t init_state, uint64_t init_seq);
void seed_from_entropy(); // Automatically seeds using OS entropy

uint32_t rand();
float randf();
float std_norm();
} // namespace prng

} // namespace gradientcore
