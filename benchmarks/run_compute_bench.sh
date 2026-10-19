#!/usr/bin/env bash
# Builds the library three ways (scalar, AVX2 SIMD, SIMD + OpenMP) and
# compares kernel and training-step speed. Variants run in alternating rounds
# (ROUNDS, default 3); each metric is the median across rounds.
#   ./run_compute_bench.sh            # all cores
#   OMP_NUM_THREADS=4 ./run_compute_bench.sh
set -euo pipefail

cd "$(dirname "$0")"
BUILD_DIR=build
RESULTS_DIR=results
ROUNDS=${ROUNDS:-3}
VARIANTS=(scalar simd simd_omp)

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)"
mkdir -p "$RESULTS_DIR"
for v in "${VARIANTS[@]}"; do
    : > "$RESULTS_DIR/kernels_$v.txt"
    : > "$RESULTS_DIR/training_$v.txt"
done

for round in $(seq 1 "$ROUNDS"); do
    for v in "${VARIANTS[@]}"; do
        echo "Round $round/$ROUNDS: $v"
        "./$BUILD_DIR/bench_kernels_$v" | tee -a "$RESULTS_DIR/kernels_$v.txt" | grep -v RESULT
        "./$BUILD_DIR/bench_training_$v" --trials 5 --steps 300 |
            tee -a "$RESULTS_DIR/training_$v.txt" | grep -E "per-trial|final loss"
    done
done
echo

# Median of a RESULT key across rounds.
get() {
    awk -v k="$2" '$1 == "RESULT" && $2 == k { print $3 }' "$RESULTS_DIR/$1.txt" |
        sort -g | awk '{ v[NR] = $1 } END { if (NR) print v[int((NR + 1) / 2)] }'
}
speedup() { awk -v b="$1" -v x="$2" 'BEGIN { printf "%.2fx", b / x }'; }

row() { # label file key
    local s m o
    s=$(get "$2_scalar" "$3"); m=$(get "$2_simd" "$3"); o=$(get "$2_simd_omp" "$3")
    printf "%-26s %10.4f %10.4f %10.4f %9s %9s\n" "$1" "$s" "$m" "$o" \
        "$(speedup "$s" "$m")" "$(speedup "$s" "$o")"
}

{
    echo "=== Scalar vs SIMD vs SIMD+OpenMP (median of $ROUNDS rounds, $(get kernels_simd_omp threads) threads) ==="
    printf "%-26s %10s %10s %10s %9s %9s\n" "time (ms)" "scalar" "simd" "simd+omp" "simd" "simd+omp"
    printf "%-26s %10s %10s %10s %9s %9s\n" "" "" "" "" "speedup" "speedup"
    row "matmul 64x784x128 (MNIST)" kernels matmul_mnist_64x784x128_ms
    row "matmul 256^3" kernels matmul_256_ms
    row "matmul 512^3" kernels matmul_512_ms
    row "matmul 1024^3" kernels matmul_1024_ms
    row "relu 64K" kernels relu_64K_ms
    row "relu 4M" kernels relu_4M_ms
    row "sigmoid 64K" kernels sigmoid_64K_ms
    row "sigmoid 4M" kernels sigmoid_4M_ms
    row "add 64K" kernels add_64K_ms
    row "add 4M" kernels add_4M_ms
    row "MNIST train step" training ms_per_step
    echo
    printf "matmul 1024^3 GFLOP/s: scalar %s, simd %s, simd+omp %s\n" \
        "$(get kernels_scalar matmul_1024_gflops)" "$(get kernels_simd matmul_1024_gflops)" \
        "$(get kernels_simd_omp matmul_1024_gflops)"
} | tee "$RESULTS_DIR/compute_summary.txt"
