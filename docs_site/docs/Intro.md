---
sidebar_position: 1
title: Introduction
---
# GradCore-Tensor — C++ neural networks, simple and transparent

GradCore-Tensor is a compact, high-performance C++ library for building, training, and running neural networks with an API inspired by PyTorch. It provides a clear, object-oriented implementation of tensors, an autograd engine, neural network building blocks, optimizers, and a lightweight data pipeline — all implemented from first principles so you can read, understand, and extend every part of the stack.

What this project gives you:

- A PyTorch-style C++ API (`nn::Model`, `nn::Linear`, `autograd::backward`) built on arena allocators — no heap fragmentation, no Python runtime, no hidden overhead.

- Example-driven tutorials (MNIST digit classification at **97.77 % test accuracy**, California Housing regression) that run natively in C++.

- A small-footprint runtime with zero external dependencies beyond a C++17 compiler and CMake, suitable for learning, research prototypes, and embedding in native applications.

## System requirements & important compatibility notice

:::warning 
**Important:** GradCore-Tensor is developed, tested, and supported on Linux. The memory allocator, threading model, and build pipelines are tuned for Linux toolchains and glibc-based environments. Building or running on Windows or macOS is not officially supported and may fail or produce incorrect behavior.
:::

If you must target another OS, expect to:

- encounter build and runtime issues,

- need to adapt low-level memory/threading code, and

- write/validate platform-specific tests.

We recommend using a recent Linux distribution and GCC or Clang with at least C++17 support.

## What you'll find in the docs

Use the sidebar to navigate. The main sections are:

- **Getting Started** — prerequisites, build instructions, and how to link GradCore-Tensor into your own CMake project.

- **Tutorials** — step-by-step walkthroughs of the California Housing regression and MNIST classification examples, covering data loading, model definition, the training loop, saving/loading, and inference.

- **Module deep dives** — `tensor`, `autograd`, `nn`, `optim`, and `data` with design rationale and code references.

- **Contributing** — how to set up a dev environment, add new autograd ops or layers, run the examples as validation, and propose changes.

## Project status and goals

GradCore-Tensor is a student-led, research-first project (author: Class of 2028). The goals are:

- clarity over feature bloat — demonstrate how core deep learning primitives are implemented in native C++,

- provide a reproducible learning platform for students and researchers, and

- provide an approachable base for contributions that improve correctness, performance, or API ergonomics.

## Contributions & community

Contributions are welcome — new layers, optimizers, documentation improvements, and portability work especially so. See the Contributing Guide for setup steps, coding standards, and how to run the example suite. When opening issues or PRs, please include reproducible steps and system details (Linux distro, compiler version).

**Acknowledgements:** This project was developed by a student (Class of 2028). Community feedback and pull requests are highly appreciated — they help make the library safer and more useful for others.
