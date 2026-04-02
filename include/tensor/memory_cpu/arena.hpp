#pragma once

#include <cstdint>
#include <cstdlib> // To avoid malloc
#include <cstring> // For memset
#include <sys/types.h>

namespace gradientcore {

constexpr size_t ARENA_ALIGN = sizeof(void *);
constexpr size_t ARENA_NUM_SCRATCH = 2;

constexpr uint64_t MiB(uint64_t n) { return n << 20; }
constexpr uint64_t KiB(uint64_t n) { return n << 10; }

constexpr uint64_t ARENA_SCRATCH_RESERVE = MiB(64);
constexpr uint64_t ARENA_SCRATCH_COMMIT = KiB(64);

struct Arena {
  Arena *current;
  Arena *prev;

  uint64_t reserve_size;
  uint64_t commit_size;
  bool growable;

  uint64_t base_pos;
  uint64_t pos;
  uint64_t commit_pos;

  static Arena *create(uint64_t reserve_size, uint64_t commit_size,
                       bool growable);
  void destroy();

  uint64_t get_pos() const;
  void *push_raw(uint64_t size, bool non_zero = false);
  void pop(uint64_t size);
  void pop_to(uint64_t pos);

  template <typename T> T *push(bool non_zero = false) {
    return static_cast<T *>(push_raw(sizeof(T), non_zero));
  }

  template <typename T> T *push_array(size_t count, bool non_zero = false) {
    return static_cast<T *>(push_raw(sizeof(T) * count, non_zero));
  }
};

struct ArenaTemp {
  Arena *arena;
  uint64_t start_pos;

  explicit ArenaTemp(Arena *arena)
      : arena(arena), start_pos(arena ? arena->get_pos() : 0) {}

  ~ArenaTemp() {
    if (arena)
      arena->pop_to(start_pos);
  }

  ArenaTemp(const ArenaTemp &) = delete;
  ArenaTemp &operator=(const ArenaTemp &) = delete;

  ArenaTemp(ArenaTemp &&other) noexcept
      : arena(other.arena), start_pos(other.start_pos) {
    other.arena = nullptr;
  }
  ArenaTemp &operator=(ArenaTemp &&other) noexcept {
    if (this != &other) {
      arena = other.arena;
      start_pos = other.start_pos;
      other.arena = nullptr;
    }
    return *this;
  }
};

ArenaTemp scratch_get(Arena **conflicts, uint64_t num_conflicts);
} // namespace gradientcore
