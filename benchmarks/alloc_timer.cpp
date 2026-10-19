// Times every Arena allocation and free via the linker's --wrap, so both the
// real arena and the malloc baseline are measured without editing either.

#include "alloc_timer.hpp"
#include <atomic>
#include <chrono>

namespace gradientcore {
struct Arena;
}
using gradientcore::Arena;

namespace {
std::atomic<uint64_t> g_push_calls{0}, g_push_bytes{0}, g_pop_calls{0};
std::atomic<uint64_t> g_alloc_ns{0}, g_free_ns{0};

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
} // namespace

extern "C" {
void *__real__ZN12gradientcore5Arena8push_rawEmb(Arena *, uint64_t, bool);
void __real__ZN12gradientcore5Arena6pop_toEm(Arena *, uint64_t);
void __real__ZN12gradientcore5Arena3popEm(Arena *, uint64_t);

void *__wrap__ZN12gradientcore5Arena8push_rawEmb(Arena *a, uint64_t size,
                                                 bool non_zero) {
  uint64_t t0 = now_ns();
  void *p = __real__ZN12gradientcore5Arena8push_rawEmb(a, size, non_zero);
  g_alloc_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
  g_push_calls.fetch_add(1, std::memory_order_relaxed);
  g_push_bytes.fetch_add(size, std::memory_order_relaxed);
  return p;
}

void __wrap__ZN12gradientcore5Arena6pop_toEm(Arena *a, uint64_t pos) {
  uint64_t t0 = now_ns();
  __real__ZN12gradientcore5Arena6pop_toEm(a, pos);
  g_free_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
  g_pop_calls.fetch_add(1, std::memory_order_relaxed);
}

void __wrap__ZN12gradientcore5Arena3popEm(Arena *a, uint64_t size) {
  uint64_t t0 = now_ns();
  __real__ZN12gradientcore5Arena3popEm(a, size);
  g_free_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
  g_pop_calls.fetch_add(1, std::memory_order_relaxed);
}
}

namespace bench {
AllocStats alloc_stats() {
  return {g_push_calls.load(), g_push_bytes.load(), g_pop_calls.load(),
          g_alloc_ns.load(), g_free_ns.load()};
}

void alloc_stats_reset() {
  g_push_calls = g_push_bytes = g_pop_calls = 0;
  g_alloc_ns = g_free_ns = 0;
}

double timer_overhead_ns() {
  constexpr int N = 1000000;
  uint64_t start = now_ns(), sink = 0;
  for (int i = 0; i < N; i++) {
    uint64_t t0 = now_ns();
    sink += now_ns() - t0;
  }
  // Each wrapped call reads the clock twice, like one loop iteration.
  volatile uint64_t keep = sink;
  (void)keep;
  return double(now_ns() - start) / N / 2.0;
}
} // namespace bench
