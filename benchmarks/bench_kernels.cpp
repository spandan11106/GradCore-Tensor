// Times core tensor kernels (mat_mul, relu, sigmoid, add). Built once per
// compute variant (scalar / AVX2 SIMD / SIMD + OpenMP); run_compute_bench.sh
// compares them.
//
// Usage: bench_kernels_<variant> [--reps N]
// Lines starting with "RESULT" are parsed by run_compute_bench.sh.

#include "../include/gradient.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef GC_VARIANT
#define GC_VARIANT "unknown"
#endif

using namespace gradientcore;

namespace {
uint32_t g_reps = 7;

// Median seconds per call. Each rep runs the kernel enough times to last
// >= 50 ms, so short kernels are not dominated by timer resolution.
double time_kernel(const std::function<void()> &fn) {
  using clock = std::chrono::steady_clock;
  fn(); // warmup

  uint32_t inner = 1;
  for (;;) {
    auto t0 = clock::now();
    for (uint32_t i = 0; i < inner; i++) fn();
    double s = std::chrono::duration<double>(clock::now() - t0).count();
    if (s >= 0.05) break;
    inner *= 2;
  }

  std::vector<double> samples;
  for (uint32_t r = 0; r < g_reps; r++) {
    auto t0 = clock::now();
    for (uint32_t i = 0; i < inner; i++) fn();
    samples.push_back(
        std::chrono::duration<double>(clock::now() - t0).count() / inner);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

Tensor *random_tensor(Arena *arena, uint32_t rows, uint32_t cols) {
  uint32_t shape[2] = {rows, cols};
  Tensor *t = tensor_create(arena, 2, shape);
  for (uint64_t i = 0; i < t->size; i++)
    t->storage->data[t->offset + i] = prng::randf() * 2.0f - 1.0f;
  return t;
}

// Max relative error of out = a @ b against a double-precision reference.
double matmul_error(const Tensor *out, const Tensor *a, const Tensor *b,
                    uint32_t M, uint32_t K, uint32_t N) {
  const float *A = a->storage->data + a->offset;
  const float *B = b->storage->data + b->offset;
  const float *C = out->storage->data + out->offset;
  std::vector<double> row(N);
  double max_err = 0.0;
  for (uint32_t m = 0; m < M; m++) {
    std::fill(row.begin(), row.end(), 0.0);
    for (uint32_t k = 0; k < K; k++)
      for (uint32_t n = 0; n < N; n++) row[n] += double(A[m * K + k]) * B[k * N + n];
    for (uint32_t n = 0; n < N; n++)
      max_err = std::max(max_err, std::abs(C[m * N + n] - row[n]) /
                                      std::max(1.0, std::abs(row[n])));
  }
  return max_err;
}

void bench_matmul(Arena *arena, const char *name, uint32_t M, uint32_t K,
                  uint32_t N) {
  uint64_t pos = arena->get_pos();
  Tensor *a = random_tensor(arena, M, K);
  Tensor *b = random_tensor(arena, K, N);
  uint32_t shape[2] = {M, N};
  Tensor *out = tensor_create(arena, 2, shape);

  double s = time_kernel([&] { mat_mul(out, a, b, true, false, false); });
  double gflops = 2.0 * M * K * N / s / 1e9;
  double err = matmul_error(out, a, b, M, K, N);

  std::printf("  matmul %-22s %10.3f ms %8.2f GFLOP/s  max rel err %.1e\n",
              name, s * 1e3, gflops, err);
  std::printf("RESULT matmul_%s_ms %.6f\n", name, s * 1e3);
  std::printf("RESULT matmul_%s_gflops %.4f\n", name, gflops);
  arena->pop_to(pos);
}

void bench_elementwise(Arena *arena, uint64_t n) {
  uint64_t pos = arena->get_pos();
  uint32_t rows = static_cast<uint32_t>(n / 1024);
  Tensor *a = random_tensor(arena, rows, 1024);
  Tensor *b = random_tensor(arena, rows, 1024);
  uint32_t shape[2] = {rows, 1024};
  Tensor *out = tensor_create(arena, 2, shape);

  const char *label = n >= (1u << 22) ? "4M" : "64K";
  struct Op {
    const char *name;
    std::function<void()> fn;
  } ops[] = {
      {"relu", [&] { tensor_relu(out, a); }},
      {"sigmoid", [&] { tensor_sigmoid(out, a); }},
      {"add", [&] { tensor_add(out, a, b); }},
  };
  for (auto &op : ops) {
    double s = time_kernel(op.fn);
    std::printf("  %-7s %-21s %10.4f ms %8.2f Gelem/s\n", op.name, label,
                s * 1e3, n / s / 1e9);
    std::printf("RESULT %s_%s_ms %.6f\n", op.name, label, s * 1e3);
  }
  arena->pop_to(pos);
}
} // namespace

int main(int argc, char **argv) {
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--reps")) {
      g_reps = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
    } else {
      std::fprintf(stderr, "Unknown option %s\n", argv[i]);
      return 1;
    }
  }
  prng::seed(42, 54);

  int threads = 1;
#if defined(_OPENMP)
  threads = omp_get_max_threads();
#endif
  std::printf("variant=%s threads=%d reps=%u\n", GC_VARIANT, threads, g_reps);
  std::printf("RESULT threads %d\n", threads);

  Arena *arena = Arena::create(MiB(1024), MiB(64), true);

  // 64x784x128 is the first Linear layer of the MNIST example at batch 64.
  bench_matmul(arena, "mnist_64x784x128", 64, 784, 128);
  bench_matmul(arena, "256", 256, 256, 256);
  bench_matmul(arena, "512", 512, 512, 512);
  bench_matmul(arena, "1024", 1024, 1024, 1024);

  bench_elementwise(arena, 1u << 16);
  bench_elementwise(arena, 1u << 22);

  arena->destroy();
  return 0;
}
