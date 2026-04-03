#include "../../../include/tensor/tensor.hpp"
#include <algorithm>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#elif defined(__SSE2__)
#include <emmintrin.h>
#define USE_SSE2
#endif

namespace gradientcore {

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_K = 256;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t MR = 6;
constexpr uint32_t NR = 16;

static void micro_kernel(float *__restrict__ out_tile,
                         const float *__restrict__ a_panel,
                         const float *__restrict__ b_panel, uint32_t k_len,
                         uint32_t mr, uint32_t nr, uint64_t a_stride_m,
                         uint64_t a_stride_k, uint64_t b_stride_k,
                         uint64_t b_stride_n, uint64_t out_stride_m,
                         uint64_t out_stride_n) {
#if defined(USE_AVX2)
  if (mr == MR && nr == NR && out_stride_n == 1 && b_stride_n == 1) {
    __m256 c[MR][2];
    for (uint32_t m = 0; m < MR; m++) {
      c[m][0] = _mm256_loadu_ps(out_tile + m * out_stride_m);
      c[m][1] = _mm256_loadu_ps(out_tile + m * out_stride_m + 8);
    }
    for (uint32_t k = 0; k < k_len; k++) {
      __m256 b0 = _mm256_loadu_ps(b_panel + k * b_stride_k);
      __m256 b1 = _mm256_loadu_ps(b_panel + k * b_stride_k + 8);
      for (uint32_t m = 0; m < MR; m++) {
        __m256 av = _mm256_set1_ps(a_panel[m * a_stride_m + k * a_stride_k]);
        c[m][0] = _mm256_fmadd_ps(av, b0, c[m][0]);
        c[m][1] = _mm256_fmadd_ps(av, b1, c[m][1]);
      }
    }
    for (uint32_t m = 0; m < MR; m++) {
      _mm256_storeu_ps(out_tile + m * out_stride_m, c[m][0]);
      _mm256_storeu_ps(out_tile + m * out_stride_m + 8, c[m][1]);
    }
    return;
  }
#endif

  float acc[MR][NR] = {};
  for (uint32_t m = 0; m < mr; m++)
    for (uint32_t n = 0; n < nr; n++)
      acc[m][n] = out_tile[m * out_stride_m + n * out_stride_n];

  for (uint32_t k = 0; k < k_len; k++) {
    for (uint32_t m = 0; m < mr; m++) {
      float a_val = a_panel[m * a_stride_m + k * a_stride_k];
      for (uint32_t n = 0; n < nr; n++) {
        acc[m][n] += a_val * b_panel[k * b_stride_k + n * b_stride_n];
      }
    }
  }

  for (uint32_t m = 0; m < mr; m++)
    for (uint32_t n = 0; n < nr; n++)
      out_tile[m * out_stride_m + n * out_stride_n] = acc[m][n];
}

static void matmul_2d(float *__restrict__ out, const float *__restrict__ a,
                      const float *__restrict__ b, uint32_t M, uint32_t K,
                      uint32_t N, uint64_t a_stride_m, uint64_t a_stride_k,
                      uint64_t b_stride_k, uint64_t b_stride_n,
                      uint64_t out_stride_m, uint64_t out_stride_n) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (uint32_t m_block = 0; m_block < M; m_block += BLOCK_M) {
    uint32_t m_end = std::min(M, m_block + BLOCK_M);

    for (uint32_t k_block = 0; k_block < K; k_block += BLOCK_K) {
      uint32_t k_end = std::min(K, k_block + BLOCK_K);
      uint32_t k_len = k_end - k_block;

      for (uint32_t n_block = 0; n_block < N; n_block += BLOCK_N) {
        uint32_t n_end = std::min(N, n_block + BLOCK_N);

        for (uint32_t m = m_block; m < m_end; m += MR) {
          uint32_t mr = std::min(MR, m_end - m);

          for (uint32_t n = n_block; n < n_end; n += NR) {
            uint32_t nr = std::min(NR, n_end - n);

            const float *a_panel = a + m * a_stride_m + k_block * a_stride_k;
            const float *b_panel = b + k_block * b_stride_k + n * b_stride_n;
            float *out_tile = out + m * out_stride_m + n * out_stride_n;

            micro_kernel(out_tile, a_panel, b_panel, k_len, mr, nr, a_stride_m,
                         a_stride_k, b_stride_k, b_stride_n, out_stride_m,
                         out_stride_n);
          }
        }
      }
    }
  }
}

bool mat_mul(Tensor *out, const Tensor *a, const Tensor *b, bool zero_out,
             bool transpose_a, bool transpose_b) {
  if (out == nullptr || a == nullptr || b == nullptr)
    return false;
  if (out->ndims < 2 || a->ndims < 2 || b->ndims < 2)
    return false;
  if (a->ndims != b->ndims || out->ndims != a->ndims)
    return false;

  uint32_t ndims = a->ndims;
  uint32_t a_m_idx = transpose_a ? ndims - 1 : ndims - 2;
  uint32_t a_k_idx = transpose_a ? ndims - 2 : ndims - 1;
  uint32_t b_k_idx = transpose_b ? ndims - 1 : ndims - 2;
  uint32_t b_n_idx = transpose_b ? ndims - 2 : ndims - 1;

  uint32_t M = a->shape[a_m_idx];
  uint32_t K = a->shape[a_k_idx];
  uint32_t N = b->shape[b_n_idx];

  if (K != b->shape[b_k_idx] || out->shape[ndims - 2] != M ||
      out->shape[ndims - 1] != N)
    return false;

  for (uint32_t i = 0; i < ndims - 2; i++) {
    uint32_t dim_a = a->shape[i];
    uint32_t dim_b = b->shape[i];
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
      return false;
    if (out->shape[i] != std::max(dim_a, dim_b))
      return false;
  }

  if (zero_out)
    tensor_clear(out);

  uint64_t num_batches = 1;
  for (uint32_t i = 0; i < ndims - 2; i++)
    num_batches *= out->shape[i];

  uint64_t a_stride_m = a->strides[a_m_idx];
  uint64_t a_stride_k = a->strides[a_k_idx];
  uint64_t b_stride_k = b->strides[b_k_idx];
  uint64_t b_stride_n = b->strides[b_n_idx];
  uint64_t out_stride_m = out->strides[ndims - 2];
  uint64_t out_stride_n = out->strides[ndims - 1];

  for (uint64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    uint32_t batch_indices[MAX_TENSOR_DIMS] = {0};
    uint64_t temp_idx = batch_idx;
    for (int32_t d = (int32_t)ndims - 3; d >= 0; d--) {
      batch_indices[d] = temp_idx % out->shape[d];
      temp_idx /= out->shape[d];
    }

    uint64_t a_batch_offset = a->offset;
    uint64_t b_batch_offset = b->offset;
    uint64_t out_batch_offset = out->offset;

    for (uint32_t i = 0; i < ndims - 2; i++) {
      uint32_t a_dim_idx = (a->shape[i] == 1) ? 0 : batch_indices[i];
      uint32_t b_dim_idx = (b->shape[i] == 1) ? 0 : batch_indices[i];
      a_batch_offset += a_dim_idx * a->strides[i];
      b_batch_offset += b_dim_idx * b->strides[i];
      out_batch_offset += batch_indices[i] * out->strides[i];
    }

    matmul_2d(out->storage->data + out_batch_offset,
              a->storage->data + a_batch_offset,
              b->storage->data + b_batch_offset, M, K, N, a_stride_m,
              a_stride_k, b_stride_k, b_stride_n, out_stride_m, out_stride_n);
  }

  return true;
}

} // namespace gradientcore
