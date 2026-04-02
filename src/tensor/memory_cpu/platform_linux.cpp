#include "../../../include/tensor/memory_cpu/platform.hpp"

#include <cstdint>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/random.h>
#include <unistd.h>

namespace gradientcore {
namespace platform {

void exit(int32_t code) { std::exit(code); }

uint32_t page_size() { return static_cast<uint32_t>(sysconf(_SC_PAGESIZE)); }

void *mem_reserve(uint64_t size) {
  void *ptr =
      mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return nullptr;
  }
  return ptr;
}

bool mem_commit(void *ptr, uint64_t size) {
  return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
}

bool mem_decommit(void *ptr, uint64_t size) {
  return madvise(ptr, size, MADV_DONTNEED) == 0;
}

bool mem_release(void *ptr, uint64_t size) { return munmap(ptr, size) == 0; }

void get_entropy(void *data, uint64_t size) { getrandom(data, size, 0); }

} // namespace platform
} // namespace gradientcore
