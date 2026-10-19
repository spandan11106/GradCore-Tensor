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

set(GC_MALLOC_SOURCES ${GC_SOURCES})
list(FILTER GC_MALLOC_SOURCES EXCLUDE REGEX "memory_cpu/arena\\.cpp$")
list(APPEND GC_MALLOC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/malloc_arena.cpp)

find_package(OpenMP REQUIRED)

# Compute variants. "simd_omp" matches the shipped library build.
set(FLAGS_scalar   -O3 -fno-tree-vectorize -fno-tree-slp-vectorize)
set(FLAGS_simd     -O3 -mavx2 -mfma)
set(FLAGS_simd_omp -O3 -mavx2 -mfma)
set(OMP_scalar   OFF)
set(OMP_simd     OFF)
set(OMP_simd_omp ON)

# gc_<name>: a static build of the library with the given sources and flags.
function(gc_library name sources flags use_omp)
    add_library(gc_${name} STATIC ${sources})
    target_include_directories(gc_${name} PUBLIC ${GC_ROOT}/include)
    target_compile_options(gc_${name} PRIVATE ${flags})
    if(use_omp)
        target_link_libraries(gc_${name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

# bench_<exe>: a benchmark linked against gc_<lib>, compiled with its flags.
function(gc_bench exe lib src variant flags)
    add_executable(bench_${exe} ${src} ${ARGN})
    target_compile_definitions(bench_${exe} PRIVATE GC_VARIANT="${variant}")
    target_compile_options(bench_${exe} PRIVATE ${flags})
    target_link_libraries(bench_${exe} PRIVATE gc_${lib})
endfunction()

foreach(variant scalar simd simd_omp)
    gc_library(${variant} "${GC_SOURCES}" "${FLAGS_${variant}}" ${OMP_${variant}})
    gc_bench(kernels_${variant} ${variant} bench_kernels.cpp ${variant} "${FLAGS_${variant}}")
    gc_bench(training_${variant} ${variant} bench_training.cpp ${variant} "${FLAGS_${variant}}")
endforeach()

# Allocator comparison: arena (the shipped build) vs a malloc/free baseline.
gc_library(malloc "${GC_MALLOC_SOURCES}" "${FLAGS_simd_omp}" ON)

set(GC_WRAP_FLAGS
    -Wl,--wrap=_ZN12gradientcore5Arena8push_rawEmb
    -Wl,--wrap=_ZN12gradientcore5Arena6pop_toEm
    -Wl,--wrap=_ZN12gradientcore5Arena3popEm)

foreach(variant arena malloc)
    if(variant STREQUAL "arena")
        set(lib simd_omp)
    else()
        set(lib malloc)
    endif()

    # End-to-end timing: no instrumentation in the hot path.
    gc_bench(${variant} ${lib} bench_training.cpp ${variant} "${FLAGS_simd_omp}")

    # Allocator-only timing: every push/pop wrapped with a clock read.
    gc_bench(${variant}_timed ${lib} bench_training.cpp ${variant} "${FLAGS_simd_omp}"
             alloc_timer.cpp)
    target_compile_definitions(bench_${variant}_timed PRIVATE GC_BENCH_INSTRUMENTED)
    target_link_options(bench_${variant}_timed PRIVATE ${GC_WRAP_FLAGS})
endforeach()
