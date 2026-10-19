#pragma once
#include <cstdint>

namespace bench {
struct AllocStats {
  uint64_t push_calls;
  uint64_t push_bytes;
  uint64_t pop_calls;
  uint64_t alloc_ns;
  uint64_t free_ns;
};

AllocStats alloc_stats();
void alloc_stats_reset();
// Approximate cost of one clock read pair, subtracted from each timed call.
double timer_overhead_ns();
} // namespace bench
