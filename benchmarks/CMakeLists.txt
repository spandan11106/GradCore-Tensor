cmake_minimum_required(VERSION 3.15)
project(GradCoreBenchmarks LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB_RECURSE GC_SOURCES "${GC_ROOT}/src/*.cpp")
list(FILTER GC_SOURCES EXCLUDE REGEX "platform_linux\\.cpp$")

# Two builds of the same library, differing only in the allocator.
set(GC_ARENA_SOURCES ${GC_SOURCES})
set(GC_MALLOC_SOURCES ${GC_SOURCES})
list(FILTER GC_MALLOC_SOURCES EXCLUDE REGEX "memory_cpu/arena\\.cpp$")
list(APPEND GC_MALLOC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/malloc_arena.cpp)

find_package(OpenMP REQUIRED)

set(GC_FLAGS -O3 -mavx2 -mfma)
set(GC_WRAP_FLAGS
    -Wl,--wrap=_ZN12gradientcore5Arena8push_rawEmb
    -Wl,--wrap=_ZN12gradientcore5Arena6pop_toEm
    -Wl,--wrap=_ZN12gradientcore5Arena3popEm)

foreach(variant arena malloc)
    string(TOUPPER ${variant} VARIANT)

    add_library(gc_${variant} STATIC ${GC_${VARIANT}_SOURCES})
    target_include_directories(gc_${variant} PUBLIC ${GC_ROOT}/include)
    target_compile_options(gc_${variant} PRIVATE ${GC_FLAGS})
    target_link_libraries(gc_${variant} PUBLIC OpenMP::OpenMP_CXX)

    # End-to-end timing: no instrumentation in the hot path.
    add_executable(bench_${variant} bench_training.cpp)
    target_compile_definitions(bench_${variant} PRIVATE GC_VARIANT="${variant}")
    target_compile_options(bench_${variant} PRIVATE ${GC_FLAGS})
    target_link_libraries(bench_${variant} PRIVATE gc_${variant})

    # Allocator-only timing: every push/pop wrapped with a clock read.
    add_executable(bench_${variant}_timed bench_training.cpp alloc_timer.cpp)
    target_compile_definitions(bench_${variant}_timed PRIVATE
        GC_VARIANT="${variant}" GC_BENCH_INSTRUMENTED)
    target_compile_options(bench_${variant}_timed PRIVATE ${GC_FLAGS})
    target_link_libraries(bench_${variant}_timed PRIVATE gc_${variant})
    target_link_options(bench_${variant}_timed PRIVATE ${GC_WRAP_FLAGS})
endforeach()
