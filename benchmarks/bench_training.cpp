// Times an MNIST-shaped training step (MLP + Adam + cross-entropy), mirroring
// Trainer::fit. Built once against the real Arena and once against the malloc
// baseline (malloc_arena.cpp); run_bench.sh compares the two.
//
// Usage: bench_<variant> [--steps N] [--warmup N] [--trials N] [--batch N]
//                        [--hidden N] [--ftz 0|1]
//
// --ftz (default 1) flushes denormals to zero. Without it, steps slow down
// ~4x as training converges and weights/grads go denormal, which swamps the
// allocator difference being measured.
// Lines starting with "RESULT" are parsed by run_bench.sh.

#include "../include/gradient.hpp"
#ifdef GC_BENCH_INSTRUMENTED
#include "alloc_timer.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pmmintrin.h>
#include <random>
#include <sys/resource.h>
#include <vector>
#include <xmmintrin.h>

#ifndef GC_VARIANT
#define GC_VARIANT "unknown"
#endif

using namespace gradientcore;

namespace {
constexpr uint32_t INPUT_DIM = 784;
constexpr uint32_t NUM_CLASSES = 10;
constexpr uint32_t NUM_BATCHES = 32; // distinct synthetic batches, cycled

struct Config {
  uint32_t steps = 500;
  uint32_t warmup = 50;
  uint32_t trials = 7;
  uint32_t batch = 64;
  uint32_t hidden = 128;
  uint32_t ftz = 1;
};

Config parse_args(int argc, char **argv) {
  Config c;
  for (int i = 1; i + 1 < argc; i += 2) {
    uint32_t v = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
    if (!std::strcmp(argv[i], "--steps")) c.steps = v;
    else if (!std::strcmp(argv[i], "--warmup")) c.warmup = v;
    else if (!std::strcmp(argv[i], "--trials")) c.trials = v;
    else if (!std::strcmp(argv[i], "--batch")) c.batch = v;
    else if (!std::strcmp(argv[i], "--hidden")) c.hidden = v;
    else if (!std::strcmp(argv[i], "--ftz")) c.ftz = v;
    else {
      std::fprintf(stderr, "Unknown option %s\n", argv[i]);
      std::exit(1);
    }
  }
  return c;
}

struct Batch {
  std::vector<float> x, y;
};

std::vector<Batch> make_batches(uint32_t batch) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> pixel(0.0f, 1.0f);
  std::uniform_int_distribution<uint32_t> label(0, NUM_CLASSES - 1);

  std::vector<Batch> out(NUM_BATCHES);
  for (auto &b : out) {
    b.x.resize(batch * INPUT_DIM);
    b.y.assign(batch * NUM_CLASSES, 0.0f);
    for (auto &v : b.x) v = pixel(rng);
    for (uint32_t i = 0; i < batch; i++) b.y[i * NUM_CLASSES + label(rng)] = 1.0f;
  }
  return out;
}

struct Rusage {
  long minflt, maxrss_kb;
};

Rusage rusage_now() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return {ru.ru_minflt, ru.ru_maxrss};
}
} // namespace

int main(int argc, char **argv) {
  Config cfg = parse_args(argc, argv);
  if (cfg.ftz) {
    // Set before the first OpenMP region so worker threads inherit it.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
  prng::seed(42, 54);

  Arena *perm_arena = Arena::create(MiB(1024), MiB(64), true);
  Arena *graph_arena = Arena::create(MiB(512), MiB(32), true);

  auto *model = perm_arena->push<nn::Sequential>();
  new (model) nn::Sequential();

  auto *l1 = perm_arena->push<nn::Linear>();
  new (l1) nn::Linear(perm_arena, INPUT_DIM, cfg.hidden);
  auto *relu = perm_arena->push<nn::ReLU>();
  new (relu) nn::ReLU();
  auto *l2 = perm_arena->push<nn::Linear>();
  new (l2) nn::Linear(perm_arena, cfg.hidden, NUM_CLASSES);
  model->add(l1);
  model->add(relu);
  model->add(l2);

  auto *opt = perm_arena->push<optim::Adam>();
  new (opt) optim::Adam(perm_arena, model->parameters(), 0.0005f);
  nn::CrossEntropyLoss criterion;

  std::vector<Batch> batches = make_batches(cfg.batch);
  uint64_t step_idx = 0;
  float last_loss = 0.0f;

  auto train_step = [&]() {
    const Batch &b = batches[step_idx++ % NUM_BATCHES];
    uint64_t start_pos = graph_arena->get_pos();

    uint32_t shape_x[2] = {cfg.batch, INPUT_DIM};
    uint32_t shape_y[2] = {cfg.batch, NUM_CLASSES};
    Tensor *t_x = tensor_create(graph_arena, 2, shape_x);
    Tensor *t_y = tensor_create(graph_arena, 2, shape_y);
    std::memcpy(t_x->storage->data + t_x->offset, b.x.data(),
                b.x.size() * sizeof(float));
    std::memcpy(t_y->storage->data + t_y->offset, b.y.data(),
                b.y.size() * sizeof(float));

    autograd::Variable *x = autograd::create_leaf(graph_arena, t_x, false);
    autograd::Variable *y = autograd::create_leaf(graph_arena, t_y, false);
    autograd::Variable *pred = model->forward(graph_arena, x);
    autograd::Variable *loss = criterion.forward(graph_arena, pred, y);
    last_loss = loss->data->storage->data[loss->data->offset];

    opt->zero_grad();
    autograd::backward(graph_arena, loss);
    opt->step(graph_arena);

    graph_arena->pop_to(start_pos);
  };

  for (uint32_t i = 0; i < cfg.warmup; i++) train_step();

#ifdef GC_BENCH_INSTRUMENTED
  bench::alloc_stats_reset();
#endif
  Rusage ru_start = rusage_now();

  std::vector<double> ms_per_step;
  for (uint32_t t = 0; t < cfg.trials; t++) {
    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < cfg.steps; i++) train_step();
    auto t1 = std::chrono::steady_clock::now();
    ms_per_step.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.steps);
  }

  Rusage ru_end = rusage_now();
  uint64_t measured_steps = uint64_t(cfg.steps) * cfg.trials;

  std::vector<double> sorted = ms_per_step;
  std::sort(sorted.begin(), sorted.end());
  double median = sorted[sorted.size() / 2];

  std::printf("variant=%s batch=%u hidden=%u steps=%u x %u trials ftz=%u\n",
              GC_VARIANT, cfg.batch, cfg.hidden, cfg.steps, cfg.trials,
              cfg.ftz);
  std::printf("  per-trial ms/step:");
  for (double v : ms_per_step) std::printf(" %.4f", v);
  std::printf("\n  final loss: %.6f (should match across variants)\n",
              last_loss);

  std::printf("RESULT ms_per_step %.6f\n", median);
  std::printf("RESULT min_ms_per_step %.6f\n", sorted.front());
  std::printf("RESULT page_faults_per_step %.2f\n",
              double(ru_end.minflt - ru_start.minflt) / measured_steps);
  std::printf("RESULT peak_rss_mb %.1f\n", ru_end.maxrss_kb / 1024.0);

#ifdef GC_BENCH_INSTRUMENTED
  bench::AllocStats s = bench::alloc_stats();
  double overhead = bench::timer_overhead_ns();
  double alloc_ns = std::max(0.0, s.alloc_ns - overhead * s.push_calls);
  double free_ns = std::max(0.0, s.free_ns - overhead * s.pop_calls);
  std::printf("RESULT allocs_per_step %.1f\n", double(s.push_calls) / measured_steps);
  std::printf("RESULT frees_per_step %.1f\n", double(s.pop_calls) / measured_steps);
  std::printf("RESULT alloc_kb_per_step %.1f\n",
              s.push_bytes / 1024.0 / measured_steps);
  std::printf("RESULT alloc_us_per_step %.3f\n", alloc_ns / 1e3 / measured_steps);
  std::printf("RESULT free_us_per_step %.3f\n", free_ns / 1e3 / measured_steps);
  std::printf("RESULT mem_us_per_step %.3f\n",
              (alloc_ns + free_ns) / 1e3 / measured_steps);
  std::printf("RESULT timer_overhead_ns %.1f\n", overhead);
#endif

  perm_arena->destroy();
  graph_arena->destroy();
  return 0;
}
