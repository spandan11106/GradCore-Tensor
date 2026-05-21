---
sidebar_position: 3
title: "PRNG"
---

# PRNG — Pseudo-Random Number Generation

GradCore-Tensor uses a **PCG (Permuted Congruential Generator)** for all random number generation. PCG is fast, has excellent statistical properties, and produces 32 bits of randomness per step — which is exactly what you need for weight initialisation and data shuffling, and absolutely overkill for anything that doesn't need cryptographic security (which is everything in a neural network training loop).

> *"Any sufficiently advanced random number generator is indistinguishable from magic."*  
> — Nobody, but they should have.

## Class: `PRNG`

```cpp
class PRNG {
public:
    PRNG();                                          // Default seed
    PRNG(uint64_t init_state, uint64_t init_seq);   // Explicit seed

    void     seed(uint64_t init_state, uint64_t init_seq);

    uint32_t rand();    // Uniform integer in [0, 2^32)
    float    randf();   // Uniform float  in [0, 1)
    float    std_norm(); // Standard normal N(0, 1)
};
```

### `rand()`

Returns a uniformly distributed `uint32_t`. This is the raw PCG output:

```
state = state * 6364136223846793005 + increment
output = rotate_right(xorshift(state), rot_bits)
```

The default seed (`0x853c49e6748fea9b` / `0xda3e39cb94b95bdb`) is the canonical PCG example seed — change it if you need reproducibility.

### `randf()`

```cpp
float r = prng.randf();  // e.g. 0.3742...
```

Converts the 32-bit integer output to a float in [0, 1) using `std::ldexp`:

```cpp
return std::ldexp(static_cast<float>(this->rand()), -32);
```

`ldexp(x, -32)` = `x * 2^-32`, which neatly maps `[0, 2^32)` to `[0, 1)` without any division.

### `std_norm()`

```cpp
float z = prng.std_norm();  // From N(0, 1)
```

Samples from a standard normal distribution using the **Box-Muller transform**:

```
z = sqrt(-2 * ln(u1)) * cos(2π * u2)
```

where `u1`, `u2` ~ Uniform(0, 1). Used by Kaiming Normal and Xavier Normal weight initialisation.

## Global Thread-local Interface

For convenience, a thread-local `PRNG` instance is available via free functions in `gradientcore::prng`:

```cpp
namespace prng {
    void     seed(uint64_t init_state, uint64_t init_seq);
    void     seed_from_entropy();   // Seeds from OS randomness
    uint32_t rand();
    float    randf();
    float    std_norm();
}
```

This is what the initialisation utilities and Dropout use. Each thread gets its own independent generator — no locking, no contention.

### Seeding from entropy

```cpp
prng::seed_from_entropy();
```

Calls `platform::get_entropy` to get two 64-bit values from the OS and seeds the thread-local PRNG with them. Call this once at program startup for non-deterministic training runs.

For reproducible results:

```cpp
prng::seed(42, 1);   // Fixed seed — same results every time
```

## Usage Examples

### Weight initialisation (Kaiming Normal)

```cpp
float std = std::sqrt(2.0f / fan_in);

for (uint64_t i = 0; i < size; i += 2) {
    float u1 = prng::randf();
    float u2 = prng::randf();
    if (u1 < 1e-7f) u1 = 1e-7f;  // Avoid log(0)

    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    data[i] = z0 * std;

    if (i + 1 < size) {
        float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * M_PI * u2);
        data[i + 1] = z1 * std;
    }
}
```

Box-Muller generates two values per call (z0 and z1), so the loop strides by 2 to use both and avoid wasting half the computation.

### DataLoader shuffling

```cpp
std::mt19937 rng(seed);  // DataLoader uses std::mt19937 for shuffle
for (uint32_t i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<uint32_t> dist(0, i);
    std::swap(indices[i], indices[dist(rng)]);
}
```

Note: the `DataLoader` uses `std::mt19937` (not the custom `PRNG`) for index shuffling. This is intentional — the Fisher-Yates shuffle needs integers in a bounded range, and `std::uniform_int_distribution` handles the modulo bias correctly.

## PCG vs Other PRNGs

| Generator | Period | Quality | Speed |
|---|---|---|---|
| `rand()` from libc | 2^32 | Poor | Fast |
| Mersenne Twister | 2^19937 | Good | Moderate |
| PCG32 (this library) | 2^64 | Excellent | Very fast |
| xoshiro256++ | 2^256 | Excellent | Very fast |

PCG was chosen for its simplicity, small state (128 bits), and the fact that its output passes all PractRand and TestU01 statistical tests. For neural network training, this is more than sufficient.
