#pragma once

#include <cstdint>

namespace gradientcore {
namespace platform {
void exit(int32_t code);

uint32_t page_size();

void *mem_reserve(uint64_t size);
bool mem_commit(void *ptr, uint64_t size);
bool mem_decommit(void *ptr, uint64_t size);
bool mem_release(void *ptr, uint64_t size);

void get_entropy(void *data, uint64_t size);
} // namespace platform
} // namespace gradientcore
