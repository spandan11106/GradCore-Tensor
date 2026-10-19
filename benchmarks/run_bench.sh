#!/usr/bin/env bash
# Builds the arena and malloc variants and compares them.
# Extra arguments are passed to every benchmark, e.g.
#   ./run_bench.sh --batch 128 --hidden 512 --steps 300
# Variants run in alternating rounds (ROUNDS, default 5) so thermal and
# frequency drift hit both equally; each metric is the median across rounds.
set -euo pipefail

cd "$(dirname "$0")"
BUILD_DIR=build
RESULTS_DIR=results
ROUNDS=${ROUNDS:-5}
BINS=(arena malloc arena_timed malloc_timed)

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)"
mkdir -p "$RESULTS_DIR"
for bin in "${BINS[@]}"; do : > "$RESULTS_DIR/$bin.txt"; done

for round in $(seq 1 "$ROUNDS"); do
    for bin in "${BINS[@]}"; do
        echo "Round $round/$ROUNDS: bench_$bin"
        "./$BUILD_DIR/bench_$bin" "$@" | tee -a "$RESULTS_DIR/$bin.txt" | grep -E "per-trial|final loss"
    done
done
echo

# Median of a RESULT key across rounds.
get() {
    awk -v k="$2" '$1 == "RESULT" && $2 == k { print $3 }' "$RESULTS_DIR/$1.txt" |
        sort -g | awk '{ v[NR] = $1 } END { if (NR) print v[int((NR + 1) / 2)] }'
}
pct() { awk -v a="$1" -v m="$2" 'BEGIN { if (m > 0) printf "%.1f%%", (m - a) / m * 100; else print "n/a" }'; }

a_step=$(get arena ms_per_step);        m_step=$(get malloc ms_per_step)
a_mem=$(get arena_timed mem_us_per_step); m_mem=$(get malloc_timed mem_us_per_step)
a_pf=$(get arena page_faults_per_step); m_pf=$(get malloc page_faults_per_step)

{
    echo "=== Arena vs malloc (median of $ROUNDS rounds) ==="
    printf "%-28s %14s %14s %12s\n" "metric" "arena" "malloc" "reduction"
    printf "%-28s %14s %14s %12s\n" "train step (ms)" "$a_step" "$m_step" "$(pct "$a_step" "$m_step")"
    printf "%-28s %14s %14s %12s\n" "alloc+free time (us/step)" "$a_mem" "$m_mem" "$(pct "$a_mem" "$m_mem")"
    printf "%-28s %14s %14s %12s\n" "page faults / step" "$a_pf" "$m_pf" "$(pct "$a_pf" "$m_pf")"
    printf "%-28s %14s %14s\n" "allocations / step" "$(get arena_timed allocs_per_step)" "$(get malloc_timed allocs_per_step)"
    printf "%-28s %14s %14s\n" "free calls / step" "$(get arena_timed frees_per_step)" "$(get malloc_timed frees_per_step)"
    printf "%-28s %14s %14s\n" "allocated KiB / step" "$(get arena_timed alloc_kb_per_step)" "$(get malloc_timed alloc_kb_per_step)"
    printf "%-28s %14s %14s\n" "peak RSS (MB)" "$(get arena peak_rss_mb)" "$(get malloc peak_rss_mb)"
    echo
    echo "Note: 'free calls' counts pop/pop_to. Under malloc each one frees every"
    echo "block pushed since that position, one free() per block."
} | tee "$RESULTS_DIR/summary.txt"
