---
sidebar_position: 1
title: Introduction
---
# GradCore-Tensor — C++ neural networks, simple and transparent

GradCore-Tensor is a compact, high-performance C++ library for building, training, and running neural networks with an API inspired by PyTorch. It provides a clear, object-oriented implementation of tensors, an autograd engine, neural network building blocks, optimizers, and a lightweight data pipeline — all implemented from first principles so you can understand and extend every part of the stack.

What this project gives you:

- A minimal, readable C++ API for experimenting with automatic differentiation and model design.

- Example-driven tutorials (MNIST, California Housing regression) that run natively in C++.

- A small footprint runtime without Python dependencies, suitable for learning, research prototypes, and embedding in native applications.

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

- Getting Started: prerequisites, build instructions, and how to link GradCore-Tensor into your CMake project.

- Tutorials: step-by-step C++ tutorials — California Housing regression and MNIST classification (training + inference).

- Module deep dives: `tensor`, `autograd`, `nn` (layers, activations, losses), `optim`, and `data` with design rationale and code references.

- API guidance: examples using the public headers and suggested patterns for model, training, and evaluation code.

- Contributing: how to set up a dev environment, add new autograd ops or layers, run tests, and propose changes.

## Project status and goals

GradCore-Tensor is a student-led, research-first project (author: Class of 2028). The goals are:

- clarity over feature bloat — demonstrate how core deep learning primitives are implemented in native C++,

- provide a reproducible learning platform for students and researchers, and

- provide an approachable base for contributions that improve correctness, performance, or API ergonomics.

## Contributions & community

Contributions are welcome. Useful ways to help:

- add examples and tutorials that demonstrate real use cases,

- implement and test new layers, activation functions, and optimizers,

- improve documentation and API examples, or

- add portability and CI to broaden platform support.

See the Contributing Guide for setup steps, coding standards, and how to run the test suite. When opening issues or PRs, please include reproducible steps and system details (Linux distro, compiler version).

**Acknowledgement:** This project was developed by a student (Class of 2028). Community feedback and pull requests are highly appreciated — they help make the library safer and more useful for others.
