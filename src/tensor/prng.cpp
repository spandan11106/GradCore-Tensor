#include "tensor/prng.hpp"
#include "tensor/memory_cpu/platform.hpp"
#include <cmath>

namespace gradientcore {

PRNG::PRNG() : state(0x853c49e6748fea9bULL), increment(0xda3e39cb94b95bdbULL) {}

PRNG::PRNG(uint64_t init_state, uint64_t init_seq) {
  seed(init_state, init_seq);
}

void PRNG::seed(uint64_t init_state, uint64_t init_seq) {
  this->state = 0;
  this->increment = (init_seq << 1) | 1;

  this->rand();
  this->state += init_state;
  this->rand();
}

uint32_t PRNG::rand() {
  uint64_t old_state = this->state;

  this->state = old_state * 6364136223846793005ULL + this->increment;

  uint32_t xorshifted =
      static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
  uint32_t rot = static_cast<uint32_t>(old_state >> 59u);

  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float PRNG::randf() {
  return std::ldexp(static_cast<float>(this->rand()), -32);
}

float PRNG::std_norm() {
  static const float epsilon = 1e-7f;

  float u1, u2;

  do {
    u1 = this->randf();
  } while (u1 <= epsilon);

  u2 = this->randf();

  float mag = std::sqrt(-2.0f * std::log(u1));
  float z0 = mag * std::cos(2.0f * 3.141592653f * u2);

  return z0;
}

namespace prng {

static thread_local PRNG s_rng;

void seed(uint64_t init_state, uint64_t init_seq) {
  s_rng.seed(init_state, init_seq);
}

void seed_from_entropy() {
  uint64_t entropy[2];
  platform::get_entropy(entropy, sizeof(entropy));
  s_rng.seed(entropy[0], entropy[1]);
}

uint32_t rand() { return s_rng.rand(); }

float randf() { return s_rng.randf(); }

float std_norm() { return s_rng.std_norm(); }

} // namespace prng
} // namespace gradientcore
