---
sidebar_position: 1
title: "Arena Allocator"
---

# Arena Allocator

The `Arena` is GradCore-Tensor's memory backbone. If you've ever watched a neural network training session grind to a halt because `malloc` decided it was a great time to coalesce free blocks, you'll appreciate what an arena allocator does for you: it doesn't bother with any of that.

## The Core Idea

An arena (also called a "linear allocator" or "bump allocator") works like this:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Reserved Virtual Memory                   │
├──────────────────────────────────┬───────────────────────────────┤
│         Committed Pages          │    Not yet committed (free)   │
├─────────────────────┬────────────┤                               │
│   Arena Header      │  Used data │                               │
│   (metadata)        │            │                               │
│                     ▲            ▲                               │
│                  base_pos       pos                              │
└──────────────────────────────────────────────────────────────────┘
```

To allocate, the arena simply advances `pos` by the requested size. To "free" an entire epoch's worth of computation graphs, it resets `pos` back to where it was before. No per-object tracking, no fragmentation, no surprises.

**Allocation**: O(1) — bump a pointer.  
**Deallocation of a scope**: O(1) — restore a saved position.  
**Deallocation of an individual object**: not supported (and you don't need it).

## Struct Layout

```cpp
struct Arena {
    Arena   *current;       // Points to the most-recently-grown chunk
    Arena   *prev;          // Linked list of grown chunks

    uint64_t reserve_size;  // Total virtual address range reserved
    uint64_t commit_size;   // Chunk size for committing new pages
    bool     growable;      // Can this arena chain new chunks?

    uint64_t base_pos;      // Absolute offset of this chunk's start
    uint64_t pos;           // Current write position within this chunk
    uint64_t commit_pos;    // How far we've committed to the OS
};
```

## Creating an Arena

```cpp
Arena *perm_arena  = Arena::create(MiB(1024), MiB(64), true);
Arena *graph_arena = Arena::create(MiB(512),  MiB(32), true);
```

| Parameter | Meaning |
|---|---|
| `reserve_size` | Virtual address space to reserve (does not consume physical RAM) |
| `commit_size` | Pages committed from the OS in chunks of this size |
| `growable` | If `true`, chains a new chunk when the current one fills up |

`MiB(n)` and `KiB(n)` are `constexpr` helpers:

```cpp
constexpr uint64_t MiB(uint64_t n) { return n << 20; }
constexpr uint64_t KiB(uint64_t n) { return n << 10; }
```

:::tip Two-arena pattern
GradCore-Tensor uses two arenas everywhere:

- **Permanent arena** (`perm_arena`) — model parameters, optimizer state, datasets. Lives for the entire program.
- **Graph arena** (`graph_arena`) — forward-pass activations, autograd graph nodes, batch tensors. Rewound after every batch.

This split is what makes the memory model so clean: you never need to track which intermediate tensors to free.
:::

## Allocating on an Arena

### `push<T>()` — allocate a single object

```cpp
Tensor *t = arena->push<Tensor>();
```

Allocates `sizeof(T)` bytes, zero-initialises them, and returns a typed pointer. The object is placement-new'd into the slab — the arena doesn't call constructors for you, so use placement `new` when you need one:

```cpp
nn::Linear *l = perm_arena->push<nn::Linear>();
new (l) nn::Linear(perm_arena, 8, 128);
```

### `push_array<T>(count)` — allocate an array

```cpp
float *data = arena->push_array<float>(784);
Edge  *edges = arena->push_array<Edge>(num_parents);
```

### `push_raw(size)` — raw bytes

```cpp
void *buf = arena->push_raw(my_size_in_bytes);
```

All three variants zero-initialise by default (`non_zero = false` means "please zero it"). Pass `true` to skip zeroing for a small performance gain when you're about to overwrite immediately anyway.

## Saving and Restoring Position

This is the killer feature. Before a forward pass:

```cpp
uint64_t saved = graph_arena->get_pos();

// ... entire forward pass, loss, backward ...

graph_arena->pop_to(saved);   // All graph tensors gone in O(1)
```

`get_pos()` returns the current absolute write position across all chunks. `pop_to(pos)` rewinds to exactly that position, releasing any chunks that were grown past it.

### `pop(size)` vs `pop_to(pos)`

```cpp
arena->pop(1024);           // Free the last 1024 bytes
arena->pop_to(saved_pos);   // Free everything after saved_pos
```

Prefer `pop_to` — it's less error-prone than remembering how many bytes you pushed.

## ArenaTemp — RAII scope guard

For temporary scratch work inside a function:

```cpp
{
    ArenaTemp temp(scratch_arena);
    // allocate freely on scratch_arena ...
} // temp's destructor calls pop_to automatically
```

`ArenaTemp` stores the position at construction and restores it at destruction. It is move-only (no copies).

## Thread-local Scratch Arenas

```cpp
ArenaTemp scratch = scratch_get(conflicts, num_conflicts);
```

`scratch_get` returns a thread-local scratch arena that does **not** conflict with any arena in the `conflicts` array. This is used internally by backward passes to avoid aliasing issues when multiple arenas share the same thread. There are `ARENA_NUM_SCRATCH` (= 2) scratch arenas per thread.

## Growing Across Chunks

When a growable arena runs out of reserved virtual memory, `push_raw` chains a new chunk:

```
┌──────────────────┐     ┌──────────────────┐
│  Original chunk  │ ←── │   Grown chunk    │
│  (prev = null)   │     │  (prev = orig)   │
│  base_pos = 0    │     │  base_pos = big  │
└──────────────────┘     └──────────────────┘
                              ▲
                           current
```

`get_pos()` returns `current->base_pos + current->pos` — a single global offset that works correctly across all chunks. `pop_to` walks the linked list backwards, releasing chunks that are completely freed.

:::warning Non-growable arenas
If `growable = false` and the arena fills up, `push_raw` calls `platform::exit(1)`. Don't set `growable = false` unless you've sized the arena carefully.
:::

## Alignment

All allocations are aligned to `sizeof(void*)` (8 bytes on 64-bit platforms). This is `ARENA_ALIGN`:

```cpp
constexpr size_t ARENA_ALIGN = sizeof(void *);
```

The arena header itself is placed at the very start of the first reserved chunk, so `sizeof(Arena)` bytes are consumed before any user data.

## Destroying an Arena

```cpp
arena->destroy();
```

Walks the chunk linked list and calls `platform::mem_release` (i.e. `munmap` on Linux) on each. After `destroy()`, the pointer is invalid — don't use it.

## Full Example

```cpp
// Permanent storage: model lives here forever
auto* perm  = Arena::create(MiB(512), MiB(32), true);

// Temporary storage: rewound every batch
auto* graph = Arena::create(MiB(256), MiB(16), true);

// Allocate model parameters on perm
uint32_t shape[2] = {784, 128};
Tensor *weights = tensor_create(perm, 2, shape);

// Train loop
for (int epoch = 0; epoch < 40; epoch++) {
    for (auto& batch : dataloader) {
        uint64_t pos = graph->get_pos();   // Save graph state

        // Forward pass (everything allocated on graph)
        auto* out  = model.forward(graph, x);
        auto* loss = cross_entropy_loss(graph, out, y, REDUCTION_MEAN);

        // Backward pass (also on graph)
        backward(graph, loss);
        optimizer.step(graph);

        graph->pop_to(pos);               // Free entire batch graph
    }
}

perm->destroy();
graph->destroy();
```
