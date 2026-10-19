// Baseline allocator for benchmarking: implements the Arena interface with one
// calloc/free per allocation, i.e. what a framework without an arena does.
// Linked in place of src/tensor/memory_cpu/arena.cpp; the library is untouched.
//
// "Positions" are logical byte counters, so get_pos()/pop_to() keep the same
// scoped-free semantics: pop_to(p) frees every block pushed at or after p.

#include "../include/tensor/memory_cpu/arena.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace gradientcore {

namespace {
struct MallocArena : Arena {
  std::vector<std::pair<uint64_t, void *>> allocs; // (pos before push, block)
};

MallocArena *as_malloc(Arena *a) { return static_cast<MallocArena *>(a); }
} // namespace

Arena *Arena::create(uint64_t reserve_size, uint64_t commit_size,
                     bool growable) {
  MallocArena *arena = new MallocArena();
  arena->current = arena;
  arena->prev = nullptr;
  arena->reserve_size = reserve_size;
  arena->commit_size = commit_size;
  arena->growable = growable;
  arena->base_pos = 0;
  arena->pos = 0;
  arena->commit_pos = 0;
  return arena;
}

void Arena::destroy() {
  MallocArena *self = as_malloc(this);
  for (auto &a : self->allocs)
    std::free(a.second);
  delete self;
}

uint64_t Arena::get_pos() const { return this->pos; }

void *Arena::push_raw(uint64_t size, bool non_zero) {
  uint64_t n = size ? size : 1;
  void *out = non_zero ? std::malloc(n) : std::calloc(1, n);
  if (out == nullptr) {
    std::fprintf(stderr, "Fatal error: malloc failed\n");
    std::exit(1);
  }
  as_malloc(this)->allocs.emplace_back(this->pos, out);
  this->pos += (size + ARENA_ALIGN - 1) & ~(ARENA_ALIGN - 1);
  return out;
}

void Arena::pop(uint64_t size) {
  this->pop_to(this->pos - std::min(size, this->pos));
}

void Arena::pop_to(uint64_t target) {
  auto &allocs = as_malloc(this)->allocs;
  while (!allocs.empty() && allocs.back().first >= target) {
    std::free(allocs.back().second);
    allocs.pop_back();
  }
  this->pos = std::min(this->pos, target);
}

static thread_local Arena *scratch_arenas[ARENA_NUM_SCRATCH] = {nullptr};

ArenaTemp scratch_get(Arena **conflicts, uint64_t num_conflicts) {
  for (uint32_t i = 0; i < ARENA_NUM_SCRATCH; i++) {
    bool conflict_found = false;
    for (uint32_t j = 0; j < num_conflicts; j++) {
      if (scratch_arenas[i] == conflicts[j]) {
        conflict_found = true;
        break;
      }
    }
    if (!conflict_found) {
      if (scratch_arenas[i] == nullptr)
        scratch_arenas[i] =
            Arena::create(ARENA_SCRATCH_RESERVE, ARENA_SCRATCH_COMMIT, false);
      return ArenaTemp(scratch_arenas[i]);
    }
  }
  return ArenaTemp(nullptr);
}
} // namespace gradientcore
