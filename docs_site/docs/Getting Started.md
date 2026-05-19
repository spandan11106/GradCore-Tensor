---
sidebar_position: 2
title: Getting Started
---
## Getting Started

This guide helps you build, run the examples, and link GradCore-Tensor into your C++ projects. GradCore-Tensor is developed and tested on Linux only — see the Important Compatibility note below.

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

### Installation 
If you want to run examples that depend on large dataset files, enable and pull LFS objects:

```bash
git clone https://github.com/your-org/GradCore-Tensor.git
cd GradCore-Tensor
git lfs install         # run once per machine (or omit if already set)
git lfs pull            # fetch large dataset files tracked by LFS
```

If you don't need the LFS datasets, a normal `git clone` is sufficient.

Create a build directory, configure and compile:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Notes:
- Add `-G Ninja` to `cmake` if you prefer Ninja.
- You can pass extra `-D` options to `cmake` to change install prefix or features.

### Testing
Each example folder includes a run script and example code. Example:

```bash
# MNIST example
cd examples/mnist
./run.sh

# Regression example
cd ../regression_ex
./run.sh
```

By running this scripts, the training and inference scripts are compiled inside the `\bin` folder. 

If examples fail to find `libgradientcore.so`, set the library path before running:

```bash
export LD_LIBRARY_PATH="$(pwd)/../../build:$LD_LIBRARY_PATH"
./run.sh
```

### Troubleshooting
- Compilation errors: Ensure your compiler supports C++17. Review `CMakeLists.txt` for required flags.  
- Missing datasets: Run `git lfs pull` in the repo root after cloning.  
- Runtime linker errors: Use `LD_LIBRARY_PATH` temporarily or install and run `ldconfig`.
