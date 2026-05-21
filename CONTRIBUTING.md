# Contributing to GradCore-Tensor

First off, thank you for considering contributing to **GradCore-Tensor**! Whether you're fixing a bug, adding a new optimization algorithm, or simply improving the documentation, your help is incredibly valuable.

## 📚 Read the Documentation

Before diving into the code, we highly recommend familiarizing yourself with how GradCore-Tensor works. **Please read the official documentation here:**
👉 **[GradCore-Tensor Documentation](https://spandan11106.github.io/GradCore-Tensor/)**

Understanding the core concepts, memory management, and existing API will make your contribution process much smoother and ensure your code aligns with the library's design philosophy.

---

## 🛠 Getting Started

> **⚠️ IMPORTANT:** **GradCore-Tensor is developed, tested, and supported on Linux.** The memory allocator, threading model, and build pipelines are tuned for Linux toolchains and glibc-based environments. Building or running on Windows or macOS is not officially supported and may fail or produce incorrect behavior.

### Prerequisites
To build and develop GradCore-Tensor, you will need a standard C++ development environment tailored for Linux:
* **Linux Environment** (e.g., Arch Linux, Ubuntu)
* **C++ Compiler**: A compiler supporting modern C++ (e.g., GCC or Clang).
* **CMake**: Used as the primary build system.
* **Git**: For version control.
* **Node.js & npm** *(Optional)*: Only required if you plan to contribute to the documentation site (`docs_site/`).

### Local Development Setup

1. **Fork and Clone:**
   Fork the repository on GitHub, then clone it locally:
   ```bash
   git clone https://github.com/spandan11106/GradCore-Tensor.git
   cd GradCore-Tensor
   ```

2. **Build the Project:**
   Create a build directory and compile the library using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

3. **Run the Tests:**
   Before making any changes, verify that the existing tests pass:
   ```bash
   make test
   ```

## 📂 Repository Structure

Familiarize yourself with the layout of the project so you know where to make your changes:

| Directory | Description |
|-----------|-------------|
| `include/` | Public C++ header files. Organized by module (autograd/, nn/, optim/, tensor/). |
| `src/` | The C++ source files implementing the core logic (forward/backward passes, memory arenas, etc.). |
| `test/` | Unit tests validating modules like tensors, memory allocation, and autograd functionality. |
| `examples/` | Example models demonstrating how to use the library (e.g., mnist/, regression_ex/). |
| `docs_site/` | Documentation website powered by Docusaurus. |

## 💻 Development Workflow

### 1. Create a Branch
Always create a new branch for your work. Keep the branch name descriptive:
```bash
git checkout -b feature/add-new-optimizer
```

### 2. Making Changes
- **Core Library:** When adding a new feature, ensure you add the declaration in the corresponding `include/` header and the implementation in `src/`.
- **Autograd:** If adding a new differentiable operation, remember to implement both the forward logic (in `src/tensor/`) and the backward gradient logic (in `src/autograd/ops/` or `src/tensor/loss_grad/`).

### 3. Writing Tests
Any new feature or bug fix should be accompanied by a test in the `test/` directory. GradCore-Tensor relies heavily on these tests to ensure the computational graphs and custom memory allocators remain stable across Linux environments.

### 4. Updating Documentation
If you are adding user-facing features, please update the documentation to reflect your changes.
```bash
cd docs_site
npm install
npm run start
```

## � Module Organization

Understand where different features belong in the codebase:

| Module | Location | Purpose |
|--------|----------|---------|
| **Tensor Operations** | `include/tensor/` → `src/tensor/` | Core tensor creation, reshaping, views, arithmetic |
| **Autograd Engine** | `include/autograd/` → `src/autograd/` | Automatic differentiation, backward pass computation |
| **Neural Network Layers** | `include/nn/layers/` → `src/nn/layers/` | Linear, Conv, Activation, BatchNorm, Dropout |
| **Network Containers** | `include/nn/core/` → `src/nn/core/` | Module base class, Sequential, Parameter registration |
| **Loss Functions** | `include/nn/losses/` → `src/nn/` | MSE, CrossEntropy, etc. (forward + backward) |
| **Optimizers** | `include/optim/` → `src/optim/` | SGD, Adam, AdamW, etc. |
| **Data Handling** | `include/nn/data/` → `src/nn/data/` | Dataset, DataLoader, CSV loading |
| **Memory Management** | `include/tensor/memory_cpu/` → `src/tensor/memory_cpu/` | Arena allocator (don't modify without careful testing) |

**Adding a New Feature:**
1. **Headers First:** Add interface in `include/` (matching the module structure)
2. **Implementation:** Add source in `src/` (mirror the `include/` structure)
3. **Tests:** Add tests in `test/` directory (same subdirectory structure)
4. **Documentation:** Update relevant `.md` files in `docs_site/docs/`

## 🧪 Writing and Running Tests

### Test Framework

GradCore-Tensor uses a lightweight, custom C++ testing framework with no external dependencies:

```cpp
// Pattern: Use report() helper and global counters
static int g_passed = 0;
static int g_failed = 0;

static void report(const std::string &name, bool ok) {
  if (ok) {
    std::cout << "  [PASS] " << name << "\n";
    g_passed++;
  } else {
    std::cerr << "  [FAIL] " << name << "\n";
    g_failed++;
  }
}

// Floating-point comparison with tolerance
auto close = [](float a, float b, float tol = 1e-5) {
  return std::abs(a - b) < tol;
};
```

### Test Organization

```
test/
├── autograd/          # Autograd and computation graph tests
│   ├── test_xor.cpp           # Simple autodiff validation
│   └── test_xor_nn.cpp        # Neural network training test
├── tensors/           # Tensor operations tests
│   ├── test_mat_mul.cpp       # Matrix multiplication correctness
│   └── activation_test.cpp    # Activation function tests
└── memory_cpu/        # Memory allocator tests
    └── check.cpp              # Arena allocator validation
```

### Writing Tests

- **Use reference implementations** to validate correctness against known results
- **Test edge cases:** empty tensors, dimension mismatches, boundary conditions
- **Include both forward and backward passes** for autograd-related features
- **Check memory safety:** Use `ArenaTemp` to validate cleanup behavior
- **Document test purpose** at the top of each test file

**Example Test Structure:**
```cpp
int main() {
  Arena *arena = Arena::create(MiB(16), MiB(1), false);

  // Test 1: Basic functionality
  report("Basic tensor creation", tensor != nullptr && tensor->size > 0);

  // Test 2: Computation
  float result = some_operation();
  report("Correct computation result", close(result, expected, 1e-5));

  // Test 3: Memory cleanup
  uint64_t pos_before = arena->get_pos();
  {
    ArenaTemp temp(arena);
    // Allocate and use temporary memory
  }
  report("Memory freed correctly", arena->get_pos() == pos_before);

  // Print summary
  std::cout << "\nTotal: " << (g_passed + g_failed) << " tests\n";
  std::cout << "Passed: " << g_passed << ", Failed: " << g_failed << "\n";
  
  return g_failed > 0 ? 1 : 0;
}
```

### Running Tests

```bash
cd build
make test  # Builds and runs all tests
```

Tests are automatically discovered and built by CMake. Test output includes:
- `[PASS]` for successful assertions
- `[FAIL]` for failed assertions
- Summary of total passed/failed tests

## 🔨 Build System Details

GradCore-Tensor uses CMake 3.15+ as the build system. Important details for contributors:

### Key Build Configuration

- **C++ Standard:** C++17 (required)
- **Compiler Flags:** 
  - `-O3` (aggressive optimization)
  - `-mavx2 -mfma` (SIMD instructions for tensor operations)
- **Threading:** OpenMP (`OpenMP::OpenMP_CXX` target)
- **Output:** Shared library `libgradientcore.so` on Linux

### Building from Source

```bash
mkdir build
cd build
cmake ..                    # Configure with defaults
cmake -DCMAKE_BUILD_TYPE=Debug ..  # Debug build (without -O3)
make -j$(nproc)            # Build in parallel
make test                  # Run tests
make install               # Install library and headers
```

### CMake Structure

- **CMakeLists.txt:** Main build configuration
- **GradientCoreConfig.cmake.in:** Package configuration template
- **Installed Files:**
  - Library: `lib/libgradientcore.so`
  - Headers: `include/` (mirroring source structure)
  - CMake configs: `lib/cmake/GradientCore/`

### Adding New Source Files

The build system automatically discovers source files via:
```cmake
file(GLOB_RECURSE SOURCES "src/**/*.cpp")
```

Simply add `.cpp` files to appropriate `src/` subdirectories; no manual CMakeLists.txt edits needed (in most cases). If adding a new subdirectory, verify it's properly discovered by the glob pattern.



### Naming Conventions

Follow these naming conventions to maintain consistency across the codebase:

| Category | Pattern | Example |
|----------|---------|---------|
| **Classes** | PascalCase | `Linear`, `Sequential`, `Module` |
| **Functions** | snake_case | `tensor_create()`, `tensor_reshape()` |
| **Variables** | snake_case | `learning_rate`, `num_params` |
| **Member Variables** | Underscore prefix for private/protected | `_training`, `_parameters` |
| **Constants** | UPPER_SNAKE_CASE | `MAX_TENSOR_DIMS`, `ARENA_ALIGN` |
| **Namespaces** | snake_case, hierarchical | `gradientcore::autograd`, `gradientcore::nn` |

### Code Style Guidelines

- **Indentation:** Use 2 spaces (no tabs)
- **Brace Style:** Allman style for blocks
- **Pointer Syntax:** `Type *variable` (asterisk with variable)
- **Include Guards:** Use `#pragma once` in all headers
- **Line Length:** Keep lines under 100 characters where practical
- **Header Includes:** Keep `#include` directives minimal. Prefer forward declarations in headers to reduce compilation overhead.

### Memory Management

**Critical:** GradCore-Tensor uses a custom arena allocator for zero-copy, predictable memory management. Understanding this is essential for any contribution.

**Two-Arena Strategy:**
```cpp
// Long-lived arena for parameters and weights
Arena *perm_arena = Arena::create(MiB(16), MiB(1), false);

// Short-lived arena for computation graphs and temporary data
Arena *graph_arena = Arena::create(MiB(64), MiB(1), true);

// In training loops, reset transient arena after each iteration
uint64_t start_pos = graph_arena->get_pos();
// ... build computation graph ...
graph_arena->pop_to(start_pos);  // Free all temporary allocations
```

**Rules for Memory Usage:**
1. **Always allocate through arenas** - never use `malloc()` or `new()`
2. **Pass arena pointers explicitly** - functions should accept `Arena *` parameters
3. **Use `ArenaTemp` for scoped allocations** - automatic cleanup on scope exit
4. **Validate arena pointers** - check for `nullptr` before use
5. **Understand tensor ownership** - data pointers in tensors reference arena memory, not independent allocations

**Key Arena API:**
- `Arena::create(reserve_size, commit_size, growable)` - Create a new arena
- `arena->push<T>()` - Allocate single object
- `arena->push_array<T>(count)` - Allocate array of objects
- `arena->push_raw(size, non_zero)` - Low-level allocation
- `arena->pop_to(pos)` - Free to specific checkpoint
- `arena->get_pos()` - Get current position for checkpointing

### Error Handling

GradCore-Tensor uses explicit error handling without exceptions:

- **Return `nullptr`** to signal allocation/creation failure
- **Return `bool`** to indicate success/failure of operations
- **Log to `std::cerr`** for error messages: `std::cerr << "Error: " << description`
- **Validate inputs** before processing in every function
- **Never assume preconditions** - validate all pointers and dimensions

**Example Pattern:**
```cpp
if (tensor == nullptr || tensor->ndims == 0) {
  std::cerr << "Error: Invalid tensor" << std::endl;
  return nullptr;
}
```

### Header Organization

All public API is exposed through `include/gradient.hpp`. Internal headers should:
- Use `#pragma once` for include guards
- Include only necessary dependencies
- Use forward declarations where possible
- Keep public and implementation details clearly separated

## � Version Management

GradCore-Tensor follows **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR:** Breaking API changes
- **MINOR:** Backward-compatible feature additions
- **PATCH:** Bug fixes and documentation updates

When updating version numbers:
1. Update `include/gradient_version.hpp` (macros and string)
2. Update `CMakeLists.txt` (`project()` version)
3. Update `SOVERSION` in CMakeLists.txt if making breaking changes

**Current Version:** 1.0.0

## 📖 Documenting Your Changes

### Code Comments

- **Minimize comments** - code should be self-documenting through clear naming
- **Document non-obvious behavior:** complex algorithms, memory management details, performance-critical sections
- **Use `/* */` for multi-line comments within functions**
- **Use `//` for single-line explanations**

### Documentation Website

If adding user-facing features:

1. **Update relevant `.md` files** in `docs_site/docs/`
2. **Follow Docusaurus format:**
   - Use frontmatter for metadata: `---` with title, description
   - Use Markdown with code blocks
   - Include actual working examples with expected output
3. **Update navigation:** Modify `docs_site/sidebars.ts` if adding new pages
4. **Test locally:** Run `npm run start` in `docs_site/` and verify rendering

### Updating Main README

- Keep `README.md` high-level; detailed docs go in `docs_site/`
- Reference the documentation website for in-depth guides
- Update examples if API changes

## ⚠️ Common Pitfalls to Avoid

### Memory Management Issues

- ❌ **Don't** allocate memory with `malloc()` or `new()` - use arena allocators
- ❌ **Don't** hold onto arena pointers after calling `pop_to()` on freed memory
- ❌ **Don't** assume memory alignment - arena handles alignment automatically
- ✅ **Do** use `ArenaTemp` for scoped allocations with guaranteed cleanup

### Autograd Issues

- ❌ **Don't** forget to implement backward functions for new operations
- ❌ **Don't** modify tensors in-place without considering autograd implications
- ❌ **Don't** assume gradient accumulation is automatic - check if it's needed
- ✅ **Do** implement both forward and backward passes for differentiable ops
- ✅ **Do** test gradients numerically against finite differences

### Testing Issues

- ❌ **Don't** use hard-coded floating-point comparisons - use tolerance
- ❌ **Don't** assume SIMD operations produce bit-identical results across platforms
- ❌ **Don't** skip testing edge cases (empty tensors, zero dimensions, etc.)
- ✅ **Do** use reference implementations to validate new operations
- ✅ **Do** include tests for both forward and backward passes

### Platform Issues

- ❌ **Don't** assume Windows or macOS compatibility - this is a Linux library
- ❌ **Don't** use platform-specific system calls without wrapping in `platform.hpp`
- ✅ **Do** keep all platform-specific code in `memory_cpu/platform_linux.cpp`
- ✅ **Do** test your changes on Linux with GCC and Clang

## 🆘 Getting Help

### Before Opening an Issue

1. **Check existing documentation:** https://spandan11106.github.io/GradCore-Tensor/
2. **Review examples:** Look at `examples/mnist/` and `examples/regression_ex/`
3. **Check existing tests:** See how similar features are tested in `test/`
4. **Review recent changes:** Check git history for similar modifications

### Communication Channels

- **Issues:** For bugs, feature requests, and questions
- **Pull Requests:** For code contributions (link to related issue if applicable)
- **Documentation:** Check the official docs and tutorial pages first

### PR Review Process

Your PR will be reviewed for:
- ✅ Code follows GradCore-Tensor standards
- ✅ Memory management is correct (no leaks, proper arena usage)
- ✅ Tests pass and new tests are included
- ✅ Documentation is updated if needed
- ✅ Changes are Linux-compatible

## �🚀 Submitting a Pull Request

Once your feature is complete and locally tested, you are ready to submit a PR!

### Preparation Checklist

Before submitting, verify:
- [ ] Code follows the naming conventions and style guidelines
- [ ] Memory management uses arenas (no `malloc`/`new`)
- [ ] Input validation and error handling are present
- [ ] Tests are added and pass locally (`make test`)
- [ ] Documentation is updated (if adding user-facing features)
- [ ] Changes are Linux-compatible (no platform-specific assumptions)
- [ ] Commit messages are clear and descriptive

### Submitting Your PR

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add new feature: brief description"
   git commit -m "Fix bug: clear explanation of the fix"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request:**
   - Go to the original GradCore-Tensor repository
   - Click "New Pull Request"
   - Select your branch from your fork
   - Fill in the PR template with:
     - Clear title (e.g., "Add AdaGrad optimizer" or "Fix memory leak in arena allocator")
     - Description of changes
     - Related issues (if applicable)
     - Any special testing instructions

4. **Ensure CI passes:**
   - GitHub Actions will automatically test your changes
   - All checks must pass before merging
   - Address any failures reported by CI

### After Submission

- **Respond to feedback:** Be open to suggestions and code review comments
- **Make requested changes:** Push additional commits to your branch (don't force push unless explicitly asked)
- **Test suggestions:** If reviewers suggest changes, test them locally before pushing
- **Ask questions:** If feedback is unclear, ask for clarification

---

## 📚 Additional Resources

- **[Official Documentation](https://spandan11106.github.io/GradCore-Tensor/)** - Complete API reference and tutorials
- **[Example Projects](examples/)** - MNIST training and regression examples
- **[Unit Tests](test/)** - Reference implementation patterns
- **[GitHub Issues](https://github.com/spandan11106/GradCore-Tensor/issues)** - Community discussions and feature requests

---

Thank you for contributing to GradCore-Tensor! Your efforts help make this library better for everyone. 🙏
