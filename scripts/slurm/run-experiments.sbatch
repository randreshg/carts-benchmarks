#!/bin/bash
#SBATCH --job-name=carts-experiments
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=carts-experiments-%j.out
#SBATCH --error=carts-experiments-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Error: size argument required (small, medium, large, extralarge)"
    echo "Usage: sbatch run-experiments.sbatch <size> [experiment]"
    exit 1
fi

SIZE="$1"
EXPERIMENT_FILTER="${2:-all}"
CARTS_ROOT="${CARTS_ROOT:-/opt/carts}"
CARTS="$CARTS_ROOT/tools/carts"

cd "$CARTS_ROOT"

echo "=== CARTS Experiments ==="
echo "Date:       $(date -Iseconds)"
echo "Host:       $(hostname)"
echo "SLURM Job:  ${SLURM_JOB_ID:-local}"
echo "Size:       $SIZE"
echo "Filter:     $EXPERIMENT_FILTER"
echo "========================="

if [[ "$EXPERIMENT_FILTER" == "all" || "$EXPERIMENT_FILTER" == "thread-sweep" ]]; then
    echo ""
    echo ">>> Experiment 1: Single-node thread sweep"
    $CARTS benchmarks run \
        --experiment docker-thread-sweep \
        --size "$SIZE" \
        --timeout 120
fi

if [[ "$EXPERIMENT_FILTER" == "all" || "$EXPERIMENT_FILTER" == "debug" ]]; then
    echo ""
    echo ">>> Experiment 2: Single-node debug + counters"
    $CARTS benchmarks run \
        --experiment docker-debug-counters \
        --size "$SIZE" \
        --timeout 120
fi

if [[ "$EXPERIMENT_FILTER" == "all" || "$EXPERIMENT_FILTER" == "multinode" ]]; then
    echo ""
    echo ">>> Experiment 3: Multi-node sweep (SLURM sub-jobs)"
    $CARTS benchmarks run \
        --slurm \
        --experiment docker-multinode-sweep \
        --size "$SIZE" \
        --time-limit 00:10:00
fi

echo ""
echo "=== All experiments complete ==="
