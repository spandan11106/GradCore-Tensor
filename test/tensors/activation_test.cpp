#include "../../include/tensor/tensor.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

using namespace gradientcore;

// ---------------------------------------------------------------------------
// Minimal test harness
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fill tensor with explicit values; count must equal t->size
static void fill_values(Tensor *t, const float *vals) {
  for (uint64_t i = 0; i < t->size; i++)
    t->storage->data[t->offset + i] = vals[i];
}

// Compare every element of a contiguous tensor against expected[], with
// tolerance
static bool approx_eq(const Tensor *t, const float *expected, uint64_t n,
                      float rtol = 1e-5f, float atol = 1e-6f) {
  for (uint64_t i = 0; i < n; i++) {
    float got = t->storage->data[t->offset + i];
    float exp = expected[i];
    float diff = std::fabs(got - exp);
    if (diff > atol + rtol * std::fabs(exp)) {
      std::cerr << "    mismatch at [" << i << "]: got=" << got
                << " expected=" << exp << " diff=" << diff << "\n";
      return false;
    }
  }
  return true;
}

// Confirm every output is in [lo, hi]
static bool all_in_range(const Tensor *t, float lo, float hi) {
  for (uint64_t i = 0; i < t->size; i++) {
    float v = t->storage->data[t->offset + i];
    if (v < lo || v > hi) {
      std::cerr << "    value " << v << " out of range [" << lo << ", " << hi
                << "]\n";
      return false;
    }
  }
  return true;
}

// Confirm no NaN / Inf in output
static bool no_nan_inf(const Tensor *t) {
  for (uint64_t i = 0; i < t->size; i++) {
    float v = t->storage->data[t->offset + i];
    if (std::isnan(v) || std::isinf(v)) {
      std::cerr << "    NaN/Inf at [" << i << "]\n";
      return false;
    }
  }
  return true;
}

// Make a 1-D tensor of size n
static Tensor *make1d(Arena *a, uint32_t n) {
  uint32_t s[1] = {n};
  return tensor_create(a, 1, s);
}

// Make a 2-D tensor
static Tensor *make2d(Arena *a, uint32_t r, uint32_t c) {
  uint32_t s[2] = {r, c};
  return tensor_create(a, 2, s);
}

// Make a 3-D tensor
static Tensor *make3d(Arena *a, uint32_t d0, uint32_t d1, uint32_t d2) {
  uint32_t s[3] = {d0, d1, d2};
  return tensor_create(a, 3, s);
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------
static bool test_relu_basic(Arena *a) {
  // relu(x) = max(0, x)
  float in_vals[] = {-3.f, -1.f, 0.f, 1.f, 3.f};
  float expected[] = {0.f, 0.f, 0.f, 1.f, 3.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_relu(out, in))
    return false;
  return approx_eq(out, expected, 5);
}

static bool test_relu_all_negative(Arena *a) {
  float in_vals[] = {-5.f, -2.f, -0.001f};
  float expected[] = {0.f, 0.f, 0.f};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_relu(out, in))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_relu_all_positive(Arena *a) {
  float in_vals[] = {1.f, 2.f, 3.f};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_relu(out, in))
    return false;
  return approx_eq(out, in_vals, 3); // output == input
}

static bool test_relu_null_guard(Arena *a) {
  auto *t = make1d(a, 4);
  return !tensor_relu(nullptr, t) && !tensor_relu(t, nullptr);
}

static bool test_relu_shape_mismatch(Arena *a) {
  auto *in = make1d(a, 4), *out = make1d(a, 5);
  return !tensor_relu(out, in);
}

// ---------------------------------------------------------------------------
// Leaky ReLU
// ---------------------------------------------------------------------------
static bool test_leaky_relu_basic(Arena *a) {
  float in_vals[] = {-2.f, -1.f, 0.f, 1.f, 2.f};
  float alpha = 0.1f;
  float expected[] = {-0.2f, -0.1f, 0.f, 1.f, 2.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_leaky_relu(out, in, alpha))
    return false;
  return approx_eq(out, expected, 5);
}

static bool test_leaky_relu_zero_alpha(Arena *a) {
  // alpha=0 => same as ReLU
  float in_vals[] = {-3.f, 0.f, 3.f};
  float expected[] = {0.f, 0.f, 3.f};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_leaky_relu(out, in, 0.0f))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_leaky_relu_negative_alpha(Arena *a) {
  // Unusual but should still compute: alpha=-0.1
  float in_vals[] = {-2.f, 2.f};
  float alpha = -0.1f;
  float expected[] = {0.2f, 2.f};
  auto *in = make1d(a, 2), *out = make1d(a, 2);
  fill_values(in, in_vals);
  if (!tensor_leaky_relu(out, in, alpha))
    return false;
  return approx_eq(out, expected, 2);
}

// ---------------------------------------------------------------------------
// ELU
// ---------------------------------------------------------------------------
static bool test_elu_basic(Arena *a) {
  // elu(x) = x if x>0, alpha*(exp(x)-1) otherwise
  float in_vals[] = {-1.f, 0.f, 1.f, 2.f};
  float alpha = 1.0f;
  float expected[] = {alpha * (std::exp(-1.f) - 1.f), 0.f, 1.f, 2.f};
  auto *in = make1d(a, 4), *out = make1d(a, 4);
  fill_values(in, in_vals);
  if (!tensor_elu(out, in, alpha))
    return false;
  return approx_eq(out, expected, 4);
}

static bool test_elu_alpha_2(Arena *a) {
  float in_vals[] = {-1.f, 1.f};
  float alpha = 2.0f;
  float expected[] = {2.0f * (std::exp(-1.f) - 1.f), 1.f};
  auto *in = make1d(a, 2), *out = make1d(a, 2);
  fill_values(in, in_vals);
  if (!tensor_elu(out, in, alpha))
    return false;
  return approx_eq(out, expected, 2);
}

static bool test_elu_positive_range(Arena *a) {
  // For positive inputs, ELU is identity — output must equal input
  float in_vals[] = {0.5f, 1.0f, 5.0f, 10.0f};
  auto *in = make1d(a, 4), *out = make1d(a, 4);
  fill_values(in, in_vals);
  if (!tensor_elu(out, in, 1.0f))
    return false;
  return approx_eq(out, in_vals, 4);
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------
static bool test_sigmoid_basic(Arena *a) {
  float in_vals[] = {0.f, 1.f, -1.f};
  float expected[] = {0.5f, 1.f / (1.f + std::exp(-1.f)),
                      1.f / (1.f + std::exp(1.f))};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_sigmoid(out, in))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_sigmoid_range(Arena *a) {
  // sigmoid output must always be in (0, 1)
  float in_vals[] = {-100.f, -10.f, 0.f, 10.f, 100.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_sigmoid(out, in))
    return false;
  return all_in_range(out, 0.f, 1.f) && no_nan_inf(out);
}

static bool test_sigmoid_symmetry(Arena *a) {
  // sigmoid(x) + sigmoid(-x) == 1
  float in_pos[] = {0.5f, 1.f, 2.f, 3.f};
  float in_neg[] = {-0.5f, -1.f, -2.f, -3.f};
  auto *tp = make1d(a, 4), *tn = make1d(a, 4);
  auto *op = make1d(a, 4), *on = make1d(a, 4);
  fill_values(tp, in_pos);
  fill_values(tn, in_neg);
  if (!tensor_sigmoid(op, tp))
    return false;
  if (!tensor_sigmoid(on, tn))
    return false;
  for (int i = 0; i < 4; i++) {
    float sum =
        op->storage->data[op->offset + i] + on->storage->data[on->offset + i];
    if (std::fabs(sum - 1.0f) > 1e-5f)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------
static bool test_tanh_basic(Arena *a) {
  float in_vals[] = {0.f, 1.f, -1.f};
  float expected[] = {0.f, std::tanh(1.f), std::tanh(-1.f)};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_tanh(out, in))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_tanh_range(Arena *a) {
  // tanh output must be in (-1, 1)
  float in_vals[] = {-100.f, -10.f, 0.f, 10.f, 100.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_tanh(out, in))
    return false;
  return all_in_range(out, -1.f, 1.f) && no_nan_inf(out);
}

static bool test_tanh_odd_function(Arena *a) {
  // tanh(-x) == -tanh(x)
  float in_pos[] = {0.5f, 1.f, 2.f};
  float in_neg[] = {-0.5f, -1.f, -2.f};
  auto *tp = make1d(a, 3), *tn = make1d(a, 3);
  auto *op = make1d(a, 3), *on = make1d(a, 3);
  fill_values(tp, in_pos);
  fill_values(tn, in_neg);
  if (!tensor_tanh(op, tp))
    return false;
  if (!tensor_tanh(on, tn))
    return false;
  for (int i = 0; i < 3; i++) {
    float pos = op->storage->data[op->offset + i];
    float neg = on->storage->data[on->offset + i];
    if (std::fabs(pos + neg) > 1e-5f)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
static bool test_softmax_sums_to_one(Arena *a) {
  float in_vals[] = {1.f, 2.f, 3.f, 4.f};
  auto *in = make1d(a, 4), *out = make1d(a, 4);
  fill_values(in, in_vals);
  if (!tensor_softmax(out, in, 0))
    return false;
  float sum = 0.f;
  for (int i = 0; i < 4; i++)
    sum += out->storage->data[out->offset + i];
  return std::fabs(sum - 1.0f) < 1e-5f && all_in_range(out, 0.f, 1.f);
}

static bool test_softmax_uniform(Arena *a) {
  // All same inputs => uniform distribution
  float in_vals[] = {2.f, 2.f, 2.f, 2.f};
  float expected[] = {0.25f, 0.25f, 0.25f, 0.25f};
  auto *in = make1d(a, 4), *out = make1d(a, 4);
  fill_values(in, in_vals);
  if (!tensor_softmax(out, in, 0))
    return false;
  return approx_eq(out, expected, 4);
}

static bool test_softmax_2d_last_dim(Arena *a) {
  // 2x4 tensor, softmax along last dim (dim=1)
  // Each row must sum to 1
  auto *in = make2d(a, 2, 4), *out = make2d(a, 2, 4);
  float in_vals[] = {1.f, 2.f, 3.f, 4.f, 4.f, 3.f, 2.f, 1.f};
  fill_values(in, in_vals);
  if (!tensor_softmax(out, in, 1))
    return false;
  for (int row = 0; row < 2; row++) {
    float sum = 0.f;
    for (int col = 0; col < 4; col++)
      sum += out->storage->data[out->offset + row * 4 + col];
    if (std::fabs(sum - 1.f) > 1e-5f)
      return false;
  }
  return true;
}

static bool test_softmax_2d_first_dim(Arena *a) {
  // 4x2, softmax along dim=0; each column must sum to 1
  auto *in = make2d(a, 4, 2), *out = make2d(a, 4, 2);
  float in_vals[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  fill_values(in, in_vals);
  if (!tensor_softmax(out, in, 0))
    return false;
  for (int col = 0; col < 2; col++) {
    float sum = 0.f;
    for (int row = 0; row < 4; row++)
      sum += out->storage->data[out->offset + row * 2 + col];
    if (std::fabs(sum - 1.f) > 1e-5f)
      return false;
  }
  return true;
}

static bool test_softmax_3d(Arena *a) {
  // 2x3x4, softmax along dim=2; each of the 6 rows of 4 must sum to 1
  auto *in = make3d(a, 2, 3, 4), *out = make3d(a, 2, 3, 4);
  tensor_fill(in, 1.f);
  if (!tensor_softmax(out, in, 2))
    return false;
  for (int i = 0; i < 6; i++) {
    float sum = 0.f;
    for (int j = 0; j < 4; j++)
      sum += out->storage->data[out->offset + i * 4 + j];
    if (std::fabs(sum - 1.f) > 1e-5f)
      return false;
  }
  return true;
}

static bool test_softmax_negative_dim(Arena *a) {
  // dim=-1 should be same as last dim
  auto *in = make2d(a, 3, 5), *out1 = make2d(a, 3, 5), *out2 = make2d(a, 3, 5);
  float vals[15];
  for (int i = 0; i < 15; i++)
    vals[i] = (float)(i % 5);
  fill_values(in, vals);
  if (!tensor_softmax(out1, in, -1))
    return false;
  if (!tensor_softmax(out2, in, 1))
    return false;
  return approx_eq(out1, &out2->storage->data[out2->offset], 15);
}

static bool test_softmax_large_inputs_no_nan(Arena *a) {
  // Very large values; numerical stability via max subtraction
  float in_vals[] = {1000.f, 1001.f, 1002.f};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_softmax(out, in, 0))
    return false;
  return no_nan_inf(out) && all_in_range(out, 0.f, 1.f);
}

static bool test_softmax_invalid_dim(Arena *a) {
  auto *in = make1d(a, 4), *out = make1d(a, 4);
  return !tensor_softmax(out, in, 5) && !tensor_softmax(out, in, -5);
}

// ---------------------------------------------------------------------------
// ReLU6
// ---------------------------------------------------------------------------
static bool test_relu6_basic(Arena *a) {
  float in_vals[] = {-1.f, 0.f, 3.f, 6.f, 7.f, 10.f};
  float expected[] = {0.f, 0.f, 3.f, 6.f, 6.f, 6.f};
  auto *in = make1d(a, 6), *out = make1d(a, 6);
  fill_values(in, in_vals);
  if (!tensor_relu6(out, in))
    return false;
  return approx_eq(out, expected, 6);
}

static bool test_relu6_range(Arena *a) {
  float in_vals[] = {-100.f, -1.f, 0.f, 3.f, 6.f, 100.f};
  auto *in = make1d(a, 6), *out = make1d(a, 6);
  fill_values(in, in_vals);
  if (!tensor_relu6(out, in))
    return false;
  return all_in_range(out, 0.f, 6.f);
}

// ---------------------------------------------------------------------------
// Hard Sigmoid
// ---------------------------------------------------------------------------
static bool test_hard_sigmoid_basic(Arena *a) {
  // hard_sigmoid(x) = clip((x+3)/6, 0, 1)
  float in_vals[] = {-3.f, 0.f, 3.f, -10.f, 10.f};
  float expected[] = {0.f, 0.5f, 1.f, 0.f, 1.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_hard_sigmoid(out, in))
    return false;
  return approx_eq(out, expected, 5);
}

static bool test_hard_sigmoid_range(Arena *a) {
  float in_vals[] = {-100.f, -5.f, -3.f, 0.f, 3.f, 5.f, 100.f};
  auto *in = make1d(a, 7), *out = make1d(a, 7);
  fill_values(in, in_vals);
  if (!tensor_hard_sigmoid(out, in))
    return false;
  return all_in_range(out, 0.f, 1.f);
}

// ---------------------------------------------------------------------------
// Hard Swish
// ---------------------------------------------------------------------------
static bool test_hard_swish_basic(Arena *a) {
  // hard_swish(x) = x * hard_sigmoid(x)
  float in_vals[] = {-3.f, 0.f, 3.f};
  float expected[] = {0.f, 0.f, 3.f}; // -3*(0/6)=0, 0*0.5=0, 3*1=3
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_hard_swish(out, in))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_hard_swish_consistency(Arena *a) {
  // hard_swish(x) == x * hard_sigmoid(x) for all x
  float in_vals[] = {-4.f, -2.f, 0.f, 1.f, 4.f};
  auto *in = make1d(a, 5), *hs_out = make1d(a, 5), *sig_out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_hard_swish(hs_out, in))
    return false;
  if (!tensor_hard_sigmoid(sig_out, in))
    return false;
  for (int i = 0; i < 5; i++) {
    float expected = in_vals[i] * sig_out->storage->data[sig_out->offset + i];
    float got = hs_out->storage->data[hs_out->offset + i];
    if (std::fabs(got - expected) > 1e-5f)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Softplus
// ---------------------------------------------------------------------------
static bool test_softplus_basic(Arena *a) {
  // softplus(x) = log(1 + exp(x))
  float in_vals[] = {0.f, 1.f, -1.f};
  float expected[] = {std::log(2.f), std::log(1.f + std::exp(1.f)),
                      std::log(1.f + std::exp(-1.f))};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_softplus(out, in))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_softplus_threshold(Arena *a) {
  // For large x, softplus(x) ≈ x (linear region kicks in at BETA_THRESHOLD=20)
  float in_vals[] = {25.f, 50.f, 100.f};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_softplus(out, in))
    return false;
  return approx_eq(out, in_vals, 3, 1e-3f, 1e-3f);
}

static bool test_softplus_always_positive(Arena *a) {
  // softplus(x) > 0 for all x
  float in_vals[] = {-100.f, -10.f, 0.f, 10.f, 100.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_softplus(out, in))
    return false;
  for (uint64_t i = 0; i < out->size; i++)
    if (out->storage->data[out->offset + i] <= 0.f)
      return false;
  return no_nan_inf(out);
}

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
static bool test_gelu_basic(Arena *a) {
  // gelu(0) == 0; gelu(x) > 0 for x > 0; gelu(x) slightly negative for x < 0
  float in_vals[] = {-1.f, 0.f, 1.f};
  // Reference values from the tanh approximation
  auto gelu_ref = [](float x) -> float {
    constexpr float S2P = 0.7978845608f, C = 0.044715f;
    return 0.5f * x * (1.f + std::tanh(S2P * (x + C * x * x * x)));
  };
  float expected[] = {gelu_ref(-1.f), gelu_ref(0.f), gelu_ref(1.f)};
  auto *in = make1d(a, 3), *out = make1d(a, 3);
  fill_values(in, in_vals);
  if (!tensor_gelu(out, in))
    return false;
  return approx_eq(out, expected, 3);
}

static bool test_gelu_no_nan(Arena *a) {
  float in_vals[] = {-10.f, -1.f, 0.f, 1.f, 10.f};
  auto *in = make1d(a, 5), *out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_gelu(out, in))
    return false;
  return no_nan_inf(out);
}

// ---------------------------------------------------------------------------
// Swish
// ---------------------------------------------------------------------------
static bool test_swish_basic(Arena *a) {
  // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
  float in_vals[] = {-1.f, 0.f, 1.f, 2.f};
  auto swish_ref = [](float x) { return x / (1.f + std::exp(-x)); };
  float expected[] = {swish_ref(-1.f), swish_ref(0.f), swish_ref(1.f),
                      swish_ref(2.f)};
  auto *in = make1d(a, 4), *out = make1d(a, 4);
  fill_values(in, in_vals);
  if (!tensor_swish(out, in))
    return false;
  return approx_eq(out, expected, 4);
}

static bool test_swish_zero(Arena *a) {
  float in_vals[] = {0.f};
  float expected[] = {0.f};
  auto *in = make1d(a, 1), *out = make1d(a, 1);
  fill_values(in, in_vals);
  if (!tensor_swish(out, in))
    return false;
  return approx_eq(out, expected, 1);
}

static bool test_swish_consistency_with_sigmoid(Arena *a) {
  // swish(x) == x * sigmoid(x)
  float in_vals[] = {-2.f, -1.f, 0.f, 1.f, 2.f};
  auto *in = make1d(a, 5), *sw_out = make1d(a, 5), *sig_out = make1d(a, 5);
  fill_values(in, in_vals);
  if (!tensor_swish(sw_out, in))
    return false;
  if (!tensor_sigmoid(sig_out, in))
    return false;
  for (int i = 0; i < 5; i++) {
    float expected = in_vals[i] * sig_out->storage->data[sig_out->offset + i];
    float got = sw_out->storage->data[sw_out->offset + i];
    if (std::fabs(got - expected) > 1e-5f)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// 2-D (non-contiguous) path — transpose makes a tensor non-contiguous
// so we exercise the index-walking code in each activation
// ---------------------------------------------------------------------------
static bool test_activation_non_contiguous(Arena *a) {
  // Build a 2x3 tensor, transpose it to 3x2 (non-contiguous), apply relu
  uint32_t s[2] = {2, 3};
  auto *base = tensor_create(a, 2, s);
  float vals[] = {-1.f, 2.f, -3.f, 4.f, -5.f, 6.f};
  fill_values(base, vals);

  auto *tr = tensor_transpose(a, base, 0, 1); // shape 3x2, non-contiguous
  uint32_t out_shape[2] = {3, 2};
  auto *out = tensor_create(a, 2, out_shape);

  if (!tensor_relu(out, tr))
    return false;

  // Manually compute expected: relu applied to transposed values
  // tr[i][j] = base[j][i]
  // base layout: [-1, 2, -3, 4, -5, 6]
  // tr[0][0]=base[0][0]=-1→0, tr[0][1]=base[1][0]=4→4
  // tr[1][0]=base[0][1]=2→2,  tr[1][1]=base[1][1]=-5→0
  // tr[2][0]=base[0][2]=-3→0, tr[2][1]=base[1][2]=6→6
  float expected[] = {0.f, 4.f, 2.f, 0.f, 0.f, 6.f};
  return approx_eq(out, expected, 6);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  Arena *arena = Arena::create(MiB(256), MiB(4), true);
  std::cout << "=== Activation function test suite ===\n\n";

  std::cout << "[ReLU]\n";
  report("relu_basic", test_relu_basic(arena));
  report("relu_all_negative", test_relu_all_negative(arena));
  report("relu_all_positive", test_relu_all_positive(arena));
  report("relu_null_guard", test_relu_null_guard(arena));
  report("relu_shape_mismatch", test_relu_shape_mismatch(arena));

  std::cout << "\n[Leaky ReLU]\n";
  report("leaky_relu_basic", test_leaky_relu_basic(arena));
  report("leaky_relu_zero_alpha", test_leaky_relu_zero_alpha(arena));
  report("leaky_relu_negative_alpha", test_leaky_relu_negative_alpha(arena));

  std::cout << "\n[ELU]\n";
  report("elu_basic", test_elu_basic(arena));
  report("elu_alpha_2", test_elu_alpha_2(arena));
  report("elu_positive_range", test_elu_positive_range(arena));

  std::cout << "\n[Sigmoid]\n";
  report("sigmoid_basic", test_sigmoid_basic(arena));
  report("sigmoid_range", test_sigmoid_range(arena));
  report("sigmoid_symmetry", test_sigmoid_symmetry(arena));

  std::cout << "\n[Tanh]\n";
  report("tanh_basic", test_tanh_basic(arena));
  report("tanh_range", test_tanh_range(arena));
  report("tanh_odd_function", test_tanh_odd_function(arena));

  std::cout << "\n[Softmax]\n";
  report("softmax_sums_to_one", test_softmax_sums_to_one(arena));
  report("softmax_uniform", test_softmax_uniform(arena));
  report("softmax_2d_last_dim", test_softmax_2d_last_dim(arena));
  report("softmax_2d_first_dim", test_softmax_2d_first_dim(arena));
  report("softmax_3d", test_softmax_3d(arena));
  report("softmax_negative_dim", test_softmax_negative_dim(arena));
  report("softmax_large_inputs_no_nan",
         test_softmax_large_inputs_no_nan(arena));
  report("softmax_invalid_dim", test_softmax_invalid_dim(arena));

  std::cout << "\n[ReLU6]\n";
  report("relu6_basic", test_relu6_basic(arena));
  report("relu6_range", test_relu6_range(arena));

  std::cout << "\n[Hard Sigmoid]\n";
  report("hard_sigmoid_basic", test_hard_sigmoid_basic(arena));
  report("hard_sigmoid_range", test_hard_sigmoid_range(arena));

  std::cout << "\n[Hard Swish]\n";
  report("hard_swish_basic", test_hard_swish_basic(arena));
  report("hard_swish_consistency", test_hard_swish_consistency(arena));

  std::cout << "\n[Softplus]\n";
  report("softplus_basic", test_softplus_basic(arena));
  report("softplus_threshold", test_softplus_threshold(arena));
  report("softplus_always_positive", test_softplus_always_positive(arena));

  std::cout << "\n[GELU]\n";
  report("gelu_basic", test_gelu_basic(arena));
  report("gelu_no_nan", test_gelu_no_nan(arena));

  std::cout << "\n[Swish]\n";
  report("swish_basic", test_swish_basic(arena));
  report("swish_zero", test_swish_zero(arena));
  report("swish_consistency_with_sigmoid",
         test_swish_consistency_with_sigmoid(arena));

  std::cout << "\n[Non-contiguous path]\n";
  report("relu_non_contiguous", test_activation_non_contiguous(arena));

  std::cout << "\n=== Results: " << g_passed << " passed, " << g_failed
            << " failed ===\n";
  arena->destroy();
  return g_failed == 0 ? 0 : 1;
}

// Build command:
// g++ -O2 -fopenmp -I ./include \
//   -o test_activations \
//   test/activations/test_activations.cpp \
//   src/tensor/memory_cpu/arena.cpp \
//   src/tensor/memory_cpu/platform_linux.cpp \
//   src/tensor/tensor_create.cpp \
//   src/tensor/tensor_utils.cpp \
//   src/tensor/tensor_views.cpp \
//   src/tensor/activations/tensor_relu.cpp \
//   src/tensor/activations/tensor_leaky_relu.cpp \
//   src/tensor/activations/tensor_elu.cpp \
//   src/tensor/activations/tensor_sigmoid.cpp \
//   src/tensor/activations/tensor_tanh.cpp \
//   src/tensor/activations/tensor_softmax.cpp \
//   src/tensor/activations/tensor_relu6.cpp \
//   src/tensor/activations/tensor_hard_sigmoid.cpp \
//   src/tensor/activations/tensor_hard_swish.cpp \
//   src/tensor/activations/tensor_softplus.cpp \
//   src/tensor/activations/tensor_gelu.cpp \
//   src/tensor/activations/tensor_swish.cpp \
// && ./test_activations
