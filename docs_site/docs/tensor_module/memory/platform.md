---
sidebar_position: 2
title: "Platform Layer"
---

# Platform Layer

`platform.hpp` is a thin wrapper around OS-level operations: virtual memory management and entropy generation. All the gritty `mmap`/`munmap`/`mprotect` calls live here so the rest of the codebase stays clean and portable (in theory — in practice it's Linux only, as the [Introduction](/docs/intro) honestly admits).

## API

All functions live in `gradientcore::platform`.

```cpp
namespace gradientcore {
namespace platform {

void     exit(int32_t code);

uint32_t page_size();

void    *mem_reserve(uint64_t size);
bool     mem_commit(void *ptr, uint64_t size);
bool     mem_decommit(void *ptr, uint64_t size);
bool     mem_release(void *ptr, uint64_t size);

void     get_entropy(void *data, uint64_t size);

} // namespace platform
} // namespace gradientcore
```

## Function Reference

### `exit(code)`

```cpp
platform::exit(1);
```

Calls `std::exit`. Used by the arena allocator when an unrecoverable allocation failure occurs. Wrapped here so a future port can do something fancier (e.g. throw an exception, log to a crash reporter).

### `page_size()`

```cpp
uint32_t ps = platform::page_size();  // typically 4096 on x86-64
```

Returns the system page size via `sysconf(_SC_PAGESIZE)`. Used by the arena to align reservation and commit sizes to page boundaries.

### `mem_reserve(size)`

```cpp
void *ptr = platform::mem_reserve(MiB(1024));
```

Reserves a contiguous virtual address range of `size` bytes **without** committing physical memory. On Linux this is:

```cpp
mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
```

`PROT_NONE` means the pages are mapped but not accessible — touching them would segfault. Physical RAM is only consumed after `mem_commit` is called on a sub-range.

This is how you can "reserve" 1 GiB but only actually use 64 MiB.

### `mem_commit(ptr, size)`

```cpp
bool ok = platform::mem_commit(ptr, MiB(64));
```

Makes a previously reserved range accessible by setting `PROT_READ | PROT_WRITE` via `mprotect`. The OS may page-fault individual pages in lazily on first access.

Returns `true` on success.

### `mem_decommit(ptr, size)`

```cpp
bool ok = platform::mem_decommit(ptr, MiB(64));
```

Advises the kernel that the physical backing for this range is no longer needed (`MADV_DONTNEED`). The virtual addresses remain valid but the pages may be reclaimed. Currently not called by the arena (the arena releases full chunks instead), but available for future use.

### `mem_release(ptr, size)`

```cpp
bool ok = platform::mem_release(ptr, size);
```

Fully unmaps a previously reserved range via `munmap`. Called by `Arena::destroy()` on each chunk.

### `get_entropy(data, size)`

```cpp
uint64_t seed[2];
platform::get_entropy(seed, sizeof(seed));
```

Fills `data` with `size` bytes of cryptographically-random entropy from the OS via `getrandom(data, size, 0)`. Used to seed the PRNG from non-deterministic state.

## Why Bother Wrapping This?

Direct `mmap`/`mprotect` calls scattered through the codebase would make porting nightmarish. By isolating them here:

- A Windows port only needs to implement `VirtualAlloc`/`VirtualFree`/`BCryptGenRandom` equivalents in `platform_windows.cpp`.
- The arena and tensor code remain completely untouched.

The `platform.cpp` file dispatches to the right implementation:

```cpp
// platform.cpp
#if defined(__linux__)
#include "platform_linux.cpp"
#endif
```

Adding macOS support is just a matter of writing `platform_macos.cpp` and adding an `#elif defined(__APPLE__)` branch. ([Contributions](https://github.com/spandan11106/GradCore-Tensor?tab=contributing-ov-file#contributing-to-gradcore-tensor) welcome — see the Contributing Guide.)
