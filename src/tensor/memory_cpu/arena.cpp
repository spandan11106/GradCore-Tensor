#include "../../../include/tensor/memory_cpu/arena.hpp"
#include "../../../include/tensor/memory_cpu/platform.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/types.h>

namespace gradientcore {
constexpr uint64_t align_up_pow2(uint64_t val, uint64_t align) {
  return (val + align - 1) & ~(align - 1);
}

Arena *Arena::create(uint64_t reserve_size, uint64_t commit_size,
                     bool growable) {
  uint32_t page_size = platform::page_size();

  reserve_size = align_up_pow2(reserve_size, page_size);
  commit_size = align_up_pow2(commit_size, page_size);

  Arena *arena = static_cast<Arena *>(platform::mem_reserve(reserve_size));

  if (!platform::mem_commit(arena, commit_size)) {
    arena = nullptr;
  }

  if (arena == nullptr) {
    std::fprintf(stderr, "Fatal error unable to commit memory for arena\n");
    platform::exit(1);
  }

  arena->current = arena;
  arena->prev = nullptr;

  arena->reserve_size = reserve_size;
  arena->commit_size = commit_size;
  arena->growable = growable;

  arena->base_pos = 0;
  arena->pos = sizeof(Arena);
  arena->commit_pos = commit_size;

  return arena;
}

void Arena::destroy() {
  Arena *curr = this->current;

  while (curr != nullptr) {
    Arena *previous = curr->prev;
    platform::mem_release(curr, curr->reserve_size);
    curr = previous;
  }
}

uint64_t Arena::get_pos() const {
  return this->current->base_pos + this->current->pos;
}

void *Arena::push_raw(uint64_t size, bool non_zero) {
  void *out = nullptr;
  Arena *curr = this->current;

  uint64_t pos_aligned = align_up_pow2(curr->pos, ARENA_ALIGN);
  out = static_cast<uint8_t *>(static_cast<void *>(curr)) + pos_aligned;
  uint64_t new_pos = pos_aligned + size;

  if (new_pos > curr->reserve_size) {
    out = nullptr;

    if (this->growable) {
      uint64_t res_size = this->reserve_size;
      uint64_t com_size = this->commit_size;

      if (size + sizeof(Arena) > res_size) {
        res_size = align_up_pow2(size + sizeof(Arena), ARENA_ALIGN);
      }

      Arena *new_arena = Arena::create(res_size, com_size, true);
      new_arena->base_pos = curr->base_pos + curr->reserve_size;

      Arena *prev_curr = curr;
      curr = new_arena;
      curr->prev = prev_curr;
      this->current = curr;

      pos_aligned = align_up_pow2(curr->pos, ARENA_ALIGN);
      out = static_cast<uint8_t *>(static_cast<void *>(curr)) + pos_aligned;
      new_pos = pos_aligned + size;
    }
  }

  if (new_pos > curr->commit_pos) {
    uint64_t new_commit_pos = new_pos;
    new_commit_pos += curr->commit_size - 1;
    new_commit_pos -= new_commit_pos % curr->commit_size;
    new_commit_pos = std::min(new_commit_pos, curr->reserve_size);

    uint64_t commit_amt = new_commit_pos - curr->commit_pos;
    uint8_t *commint_pointer =
        static_cast<uint8_t *>(static_cast<void *>(curr)) + curr->commit_pos;

    if (!platform::mem_commit(commint_pointer, commit_amt)) {
      out = nullptr;
    } else {
      curr->commit_pos = new_commit_pos;
    }
  }

  if (out == nullptr) {
    std::fprintf(stderr, "Fatal error: failed to allocate memory on arena\n");
    platform::exit(1);
  }

  curr->pos = new_pos;

  if (!non_zero) {
    std::memset(out, 0, size);
  }

  return out;
}

void Arena::pop(uint64_t size) {
  size = std::min(size, this->get_pos());

  Arena *curr = this->current;

  while (curr != nullptr && size > curr->pos) {
    Arena *previous = curr->prev;

    size -= curr->pos;
    platform::mem_release(curr, curr->reserve_size);

    curr = previous;
  }

  this->current = curr;
  size = std::min(curr->pos - sizeof(Arena), size);
  curr->pos -= size;
}

void Arena::pop_to(uint64_t pos) {
  uint64_t cur_pos = this->get_pos();
  pos = std::min(pos, cur_pos);
  this->pop(cur_pos - pos);
}

// thread_local replaces TS_THREAD_LOCAL
static thread_local Arena *scratch_arenas[ARENA_NUM_SCRATCH] = {nullptr};

ArenaTemp scratch_get(Arena **conflicts, uint64_t num_conflicts) {
  int32_t scratch_index = -1;

  for (int32_t i = 0; i < ARENA_NUM_SCRATCH; i++) {
    bool conflict_found = false;

    for (uint32_t j = 0; j < num_conflicts; j++) {
      if (scratch_arenas[i] == conflicts[j]) {
        conflict_found = true;
        break;
      }
    }

    if (!conflict_found) {
      scratch_index = i;
      break;
    }
  }

  if (scratch_index == -1) {
    return ArenaTemp(nullptr);
  }

  if (scratch_arenas[scratch_index] == nullptr) {
    scratch_arenas[scratch_index] =
        Arena::create(ARENA_SCRATCH_RESERVE, ARENA_SCRATCH_COMMIT, false);
  }
  return ArenaTemp(scratch_arenas[scratch_index]);
}
} // namespace gradientcore
