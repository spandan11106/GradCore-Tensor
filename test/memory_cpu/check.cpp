#include "../../include/tensor/memory_cpu/arena.hpp"

using namespace gradientcore;

int main() {
  Arena *perm_arena = Arena::create(MiB(1024), MiB(1), false);

  perm_arena->destroy();
  return 0;
}

// g++ -I ./include -o check_test test/memory_cpu/check.cpp
// src/tensor/memory_cpu/arena.cpp src/tensor/memory_cpu/platform_linux.cpp &&
// ./check_test
