---
sidebar_position: 2
title: Getting Started
---
## Getting Started

This guide helps you build the library, run the bundled examples, and link GradCore-Tensor into your own C++ projects. GradCore-Tensor is developed and tested on Linux only — see the compatibility note in the [Introduction](/docs/intro).

**Quick overview**
- Clone the repository (with Git LFS if you want dataset files).
- Build the library with CMake.
- Run the provided examples in `examples/`.
- Optionally install the library for downstream CMake `find_package()` use.

## Prerequisites
- Linux (recommended: a recent distribution with glibc)
- C++ compiler with C++17 support (GCC or Clang)
- CMake 3.15+
- `make` or `ninja`
- `git` and `git-lfs` (required only if you want example datasets)

**Important compatibility note**
GradCore-Tensor is developed, tested, and supported on Linux. The memory allocator, threading model, and build pipelines are tuned for Linux toolchains. Building or running on Windows or macOS is not officially supported and may require platform-specific porting and validation.

## Installation

Clone the repository:

```bash
git clone https://github.com/spandan11106/GradCore-Tensor.git
cd GradCore-Tensor
```

If you want to run examples that depend on large dataset files, enable and pull LFS objects:

```bash
git lfs install   # run once per machine
git lfs pull      # fetch large dataset files tracked by LFS
```

If you don't need the LFS datasets, a normal `git clone` is sufficient.

Create a build directory, configure, and compile:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Notes:
- Add `-G Ninja` to the `cmake` command if you prefer Ninja.
- You can pass extra `-D` options to change the install prefix or enable optional features.

## Running the Examples

Each example folder includes a `run.sh` script that compiles the example binaries against the built library and places them in a local `bin/` folder.

```bash
# California Housing regression
cd examples/regression_ex
./run.sh
./bin/train_regression
./bin/inference_regression

# MNIST digit classification
cd ../mnist
./run.sh
./bin/train_mnist
./bin/inference_mnist
```

If an example binary fails to find `libgradientcore.so` at runtime, set the library path before running:

```bash
export LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH"
./bin/train_mnist
```

## Linking GradCore-Tensor into Your Project

### Option A — Direct path (no install step)

Add the library's include directory and built `.so` directly to your `CMakeLists.txt`:

```cmake
set(GRADCORE_ROOT "/path/to/GradCore-Tensor")

target_include_directories(your_target PRIVATE
    ${GRADCORE_ROOT}/include
)

target_link_libraries(your_target PRIVATE
    ${GRADCORE_ROOT}/build/libgradientcore.so
)
```

Then set `LD_LIBRARY_PATH` at runtime as shown above, or use an `RPATH`:

```cmake
set_target_properties(your_target PROPERTIES
    BUILD_RPATH "${GRADCORE_ROOT}/build"
)
```

### Option B — Install and use `find_package`

Install the library to a system or local prefix:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/.local
make -j$(nproc)
make install
```

Then in your downstream project:

```cmake
find_package(GradCoreTensor REQUIRED)
target_link_libraries(your_target PRIVATE GradCoreTensor::gradientcore)
```

### Compiler flags used by the examples

The example `run.sh` scripts use the following flags, which you should replicate in your own build:

```
-O3 -mavx2 -mfma -fopenmp
```

`-mavx2 -mfma` enables the AVX2 SIMD path in the matrix multiply kernel (see `src/tensor/arithmetic/mat_mul.cpp`). `-fopenmp` enables multi-threaded tensor operations. Both are optional — the library falls back to scalar code without them.

## Troubleshooting

- **Compilation errors** — Ensure your compiler supports C++17 (`-std=c++17`). Check `CMakeLists.txt` for required flags.
- **Missing datasets** — Run `git lfs pull` in the repository root after cloning.
- **Runtime linker errors (`libgradientcore.so: not found`)** — Use `LD_LIBRARY_PATH` as shown above, or install the library and run `ldconfig`.
- **AVX2 not available** — Remove `-mavx2 -mfma` from your compile flags. The scalar fallback in `mat_mul.cpp` will be used instead.
