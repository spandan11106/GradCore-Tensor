---
sidebar_position: 2
title: Getting Started
slug: /get-started
---

# Getting Started with GradientCore

Welcome to the Quick Start guide for **GradientCore**. This section will walk you through the prerequisites, building the library from source, and setting up your first neural network. 

## Prerequisites

GradientCore is built to be lightweight with minimal external dependencies. Before you begin, ensure your system has the following installed:

* **CMake**: Version 3.15 or higher is required.
* **C++ Compiler**: A compiler with strict support for the C++17 standard.
* **OpenMP**: Required for multi-threading capabilities.

*Note: For maximum performance, the build applies hardware-specific optimizations including AVX2 and FMA SIMD instructions by default.*

---

## Building from Source

GradientCore is compiled into a shared library named `gradientcore` (`libgradientcore.so` on UNIX systems). 

Clone the repository and build the project using CMake:

```bash
# Clone the repository
git clone [https://github.com/spandan11106/GradCore-Tensor.git](https://github.com/spandan11106/GradCore-Tensor.git)
cd GradCore-Tensor

# Create a build directory
mkdir build && cd build

# Configure and compile
cmake ..
make -j$(nproc)

# Install the library, headers, and CMake config files
sudo make install 
```

This will install the library, headers to your system's `include` directory, and generate `GradientCoreConfig.cmake` files for easy linking in your own downstream projects.

