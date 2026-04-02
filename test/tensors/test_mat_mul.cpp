#include "../../include/tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

using namespace gradientcore;

namespace gradientcore {
bool mat_mul(Tensor *out, const Tensor *a, const Tensor *b, bool zero_out,
             bool transpose_a, bool transpose_b);
}

static int g_passed = 0;
static int g_failed = 0;

static void report(const std::string &name, bool ok) {
  if (ok) {
    std::cout << "  [PASS] " << name << "\n";
    g_passed++;
  } else {
    std::cerr << "  [FAIL] " << name << "\n";
    g_failed++;
  }
}

static void tensor_fill_seq(Tensor *t, float start = 0.f, float step = 1.f) {
  for (uint64_t i = 0; i < t->size; i++)
    t->storage->data[t->offset + i] = start + step * (float)i;
}

static void ref_matmul_2d(const float *a, const float *b, float *out,
                          uint32_t M, uint32_t K, uint32_t N, uint64_t a_sm,
                          uint64_t a_sk, uint64_t b_sk, uint64_t b_sn,
                          uint64_t o_sm, uint64_t o_sn) {
  for (uint32_t m = 0; m < M; m++)
    for (uint32_t n = 0; n < N; n++) {
      float acc = 0.f;
      for (uint32_t k = 0; k < K; k++)
        acc += a[m * a_sm + k * a_sk] * b[k * b_sk + n * b_sn];
      out[m * o_sm + n * o_sn] = acc;
    }
}

static bool close(const Tensor *got, const float *ref, uint64_t n,
                  float rtol = 1e-4f, float atol = 1e-5f) {
  for (uint64_t i = 0; i < n; i++) {
    float g = got->storage->data[got->offset + i];
    float r = ref[i];
    float diff = std::fabs(g - r);
    if (diff > atol + rtol * std::fabs(r)) {
      std::cerr << "    mismatch at [" << i << "]: got=" << g << " ref=" << r
                << " diff=" << diff << "\n";
      return false;
    }
  }
  return true;
}

static bool test_2d_basic(Arena *a) {
  uint32_t sa[2] = {64, 128}, sb[2] = {128, 64}, so[2] = {64, 64};
  auto *ta = tensor_create(a, 2, sa), *tb = tensor_create(a, 2, sb),
       *to = tensor_create_zeros(a, 2, so);
  tensor_fill(ta, 1.f);
  tensor_fill(tb, 2.f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  float exp = 128 * 2.f;
  for (uint64_t i = 0; i < to->size; i++)
    if (to->storage->data[to->offset + i] != exp)
      return false;
  return true;
}

static bool test_2d_vs_ref(Arena *a) {
  constexpr uint32_t M = 32, K = 48, N = 40;
  uint32_t sa[2] = {M, K}, sb[2] = {K, N}, so[2] = {M, N};
  auto *ta = tensor_create(a, 2, sa), *tb = tensor_create(a, 2, sb),
       *to = tensor_create_zeros(a, 2, so);
  tensor_fill_seq(ta, 0.f, 0.01f);
  tensor_fill_seq(tb, 0.f, 0.01f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(M * N, true);
  ref_matmul_2d(ta->storage->data + ta->offset, tb->storage->data + tb->offset,
                ref, M, K, N, ta->strides[0], ta->strides[1], tb->strides[0],
                tb->strides[1], N, 1); // ref output is flat row-major
  return close(to, ref, M * N);
}

static bool test_transpose_a(Arena *a) {
  constexpr uint32_t M = 30, K = 50, N = 40;
  uint32_t sa[2] = {K, M}, sb[2] = {K, N}, so[2] = {M, N};
  auto *ta = tensor_create(a, 2, sa), *tb = tensor_create(a, 2, sb),
       *to = tensor_create_zeros(a, 2, so);
  tensor_fill_seq(ta, 0.f, 0.01f);
  tensor_fill_seq(tb, 0.f, 0.01f);
  if (!mat_mul(to, ta, tb, true, true, false))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(M * N, true);
  ref_matmul_2d(ta->storage->data + ta->offset, tb->storage->data + tb->offset,
                ref, M, K, N, ta->strides[1], ta->strides[0], // transposed
                tb->strides[0], tb->strides[1], N, 1);
  return close(to, ref, M * N);
}

static bool test_transpose_b(Arena *a) {
  constexpr uint32_t M = 30, K = 50, N = 40;
  uint32_t sa[2] = {M, K}, sb[2] = {N, K}, so[2] = {M, N};
  auto *ta = tensor_create(a, 2, sa), *tb = tensor_create(a, 2, sb),
       *to = tensor_create_zeros(a, 2, so);
  tensor_fill_seq(ta, 0.f, 0.01f);
  tensor_fill_seq(tb, 0.f, 0.01f);
  if (!mat_mul(to, ta, tb, true, false, true))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(M * N, true);
  ref_matmul_2d(ta->storage->data + ta->offset, tb->storage->data + tb->offset,
                ref, M, K, N, ta->strides[0], ta->strides[1], tb->strides[1],
                tb->strides[0], // transposed
                N, 1);
  return close(to, ref, M * N);
}

static bool test_transpose_both(Arena *a) {
  constexpr uint32_t M = 28, K = 36, N = 44;
  uint32_t sa[2] = {K, M}, sb[2] = {N, K}, so[2] = {M, N};
  auto *ta = tensor_create(a, 2, sa), *tb = tensor_create(a, 2, sb),
       *to = tensor_create_zeros(a, 2, so);
  tensor_fill_seq(ta, 0.f, 0.01f);
  tensor_fill_seq(tb, 0.f, 0.01f);
  if (!mat_mul(to, ta, tb, true, true, true))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(M * N, true);
  ref_matmul_2d(ta->storage->data + ta->offset, tb->storage->data + tb->offset,
                ref, M, K, N, ta->strides[1], ta->strides[0], tb->strides[1],
                tb->strides[0], N, 1);
  return close(to, ref, M * N);
}

static bool test_3d_batched(Arena *a) {
  constexpr uint32_t B = 8, M = 32, K = 48, N = 40;
  uint32_t sa[3] = {B, M, K}, sb[3] = {B, K, N}, so[3] = {B, M, N};
  auto *ta = tensor_create(a, 3, sa), *tb = tensor_create(a, 3, sb),
       *to = tensor_create_zeros(a, 3, so);
  tensor_fill_seq(ta, 0.f, 0.001f);
  tensor_fill_seq(tb, 0.f, 0.001f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(B * M * N, true);
  for (uint32_t b = 0; b < B; b++)
    ref_matmul_2d(ta->storage->data + ta->offset + b * ta->strides[0],
                  tb->storage->data + tb->offset + b * tb->strides[0],
                  ref + b * M * N, M, K, N, ta->strides[1], ta->strides[2],
                  tb->strides[1], tb->strides[2], N, 1);
  return close(to, ref, B * M * N);
}

static bool test_4d_batched(Arena *a) {
  constexpr uint32_t B1 = 3, B2 = 4, M = 24, K = 32, N = 20;
  uint32_t sa[4] = {B1, B2, M, K}, sb[4] = {B1, B2, K, N},
           so[4] = {B1, B2, M, N};
  auto *ta = tensor_create(a, 4, sa), *tb = tensor_create(a, 4, sb),
       *to = tensor_create_zeros(a, 4, so);
  tensor_fill_seq(ta, 0.f, 0.0005f);
  tensor_fill_seq(tb, 0.f, 0.0005f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(B1 * B2 * M * N, true);
  for (uint32_t b1 = 0; b1 < B1; b1++)
    for (uint32_t b2 = 0; b2 < B2; b2++)
      ref_matmul_2d(ta->storage->data + ta->offset + b1 * ta->strides[0] +
                        b2 * ta->strides[1],
                    tb->storage->data + tb->offset + b1 * tb->strides[0] +
                        b2 * tb->strides[1],
                    ref + (b1 * B2 + b2) * M * N, M, K, N, ta->strides[2],
                    ta->strides[3], tb->strides[2], tb->strides[3], N, 1);
  return close(to, ref, B1 * B2 * M * N);
}

static bool test_broadcast_batch(Arena *a) {
  constexpr uint32_t B = 6, M = 24, K = 32, N = 28;
  uint32_t sa[3] = {1, M, K}, sb[3] = {B, K, N}, so[3] = {B, M, N};
  auto *ta = tensor_create(a, 3, sa), *tb = tensor_create(a, 3, sb),
       *to = tensor_create_zeros(a, 3, so);
  tensor_fill_seq(ta, 0.f, 0.001f);
  tensor_fill_seq(tb, 0.f, 0.001f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  ArenaTemp tmp(a);
  float *ref = a->push_array<float>(B * M * N, true);
  for (uint32_t b = 0; b < B; b++)
    ref_matmul_2d(ta->storage->data + ta->offset, // broadcast: always slice 0
                  tb->storage->data + tb->offset + b * tb->strides[0],
                  ref + b * M * N, M, K, N, ta->strides[1], ta->strides[2],
                  tb->strides[1], tb->strides[2], N, 1);
  return close(to, ref, B * M * N);
}

static bool test_accumulate(Arena *a) {
  uint32_t s[2] = {16, 16};
  auto *ta = tensor_create(a, 2, s), *tb = tensor_create(a, 2, s),
       *to = tensor_create_zeros(a, 2, s);
  tensor_fill(ta, 1.f);
  tensor_fill(tb, 1.f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  if (!mat_mul(to, ta, tb, false, false, false))
    return false;
  float exp = 2.f * 16.f;
  for (uint64_t i = 0; i < to->size; i++)
    if (to->storage->data[to->offset + i] != exp)
      return false;
  return true;
}

static bool test_scalar(Arena *a) {
  uint32_t s[2] = {1, 1};
  auto *ta = tensor_create(a, 2, s), *tb = tensor_create(a, 2, s),
       *to = tensor_create_zeros(a, 2, s);
  tensor_fill(ta, 3.f);
  tensor_fill(tb, 7.f);
  if (!mat_mul(to, ta, tb, true, false, false))
    return false;
  return to->storage->data[to->offset] == 21.f;
}

static bool test_null_guards(Arena *a) {
  uint32_t s[2] = {4, 4};
  auto *ta = tensor_create(a, 2, s), *tb = tensor_create(a, 2, s),
       *to = tensor_create_zeros(a, 2, s);
  if (mat_mul(nullptr, ta, tb, true, false, false))
    return false;
  if (mat_mul(to, nullptr, tb, true, false, false))
    return false;
  if (mat_mul(to, ta, nullptr, true, false, false))
    return false;
  uint32_t sb2[2] = {5, 4};
  auto *tb2 = tensor_create(a, 2, sb2);
  if (mat_mul(to, ta, tb2, true, false, false))
    return false;
  return true;
}

static bool test_large_2d(Arena *a) {
  constexpr uint32_t M = 4096, K = 4096, N = 4096;
  uint32_t sa[2] = {M, K}, sb[2] = {K, N}, so[2] = {M, N};
  auto *ta = tensor_create(a, 2, sa), *tb = tensor_create(a, 2, sb),
       *to = tensor_create_zeros(a, 2, so);
  tensor_fill(ta, 1.f);
  tensor_fill(tb, 2.f);
  auto t0 = std::chrono::high_resolution_clock::now();
  bool ok = mat_mul(to, ta, tb, true, false, false);
  auto t1 = std::chrono::high_resolution_clock::now();
  double dt = std::chrono::duration<double>(t1 - t0).count();
  double gflops = 2.0 * M * K * N / dt / 1e9;
  std::cout << "    large_2d: " << dt << "s  (" << gflops << " GFLOP/s)\n";
  if (!ok)
    return false;
  float exp = (float)K * 2.f;
  for (uint64_t i = 0; i < to->size; i++)
    if (to->storage->data[to->offset + i] != exp)
      return false;
  return true;
}

int main() {
  Arena *arena = Arena::create(MiB(2048), MiB(64), true);
  std::cout << "=== mat_mul test suite ===\n\n";

  std::cout << "[Group 1] Basic 2-D\n";
  report("2d_basic", test_2d_basic(arena));
  report("2d_vs_reference", test_2d_vs_ref(arena));

  std::cout << "\n[Group 2] Transpose variants\n";
  report("transpose_a", test_transpose_a(arena));
  report("transpose_b", test_transpose_b(arena));
  report("transpose_both", test_transpose_both(arena));

  std::cout << "\n[Group 3] Batched (many dims)\n";
  report("3d_batched", test_3d_batched(arena));
  report("4d_batched", test_4d_batched(arena));
  report("broadcast_batch", test_broadcast_batch(arena));

  std::cout << "\n[Group 4] Edge cases\n";
  report("accumulate", test_accumulate(arena));
  report("scalar_1x1", test_scalar(arena));
  report("null_guards", test_null_guards(arena));

  std::cout << "\n[Group 5] Large / performance\n";
  report("large_2d_2048", test_large_2d(arena));

  std::cout << "\n=== Results: " << g_passed << " passed, " << g_failed
            << " failed ===\n";
  arena->destroy();
  return g_failed == 0 ? 0 : 1;
}

// g++ -O3 -mavx2 -mfma -fopenmp -I ./include \
//   -o test_matmul \
//   test/tensors/test_mat_mul.cpp \
//   src/tensor/memory_cpu/arena.cpp \
//   src/tensor/memory_cpu/platform_linux.cpp \
//   src/tensor/tensor_create.cpp \
//   src/tensor/tensor_utils.cpp \
//   src/tensor/tensor_views.cpp \
//   src/tensor/arithmetic/mat_mul.cpp \
// && ./test_matmul
